"""Graph construction, batching, forward pass, and explainer compatibility."""

from __future__ import annotations

import torch
from torch_geometric.loader import DataLoader

from src.model4.graphs import atom_index_table, build_graphs, smiles_to_graph
from src.model4.model import DeepChemStyleGraphConv

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
REFERENCE_SMILES = "CN(C)C(=O)Oc1cccc([N+](C)(C)C)c1"


def make_model(**kwargs) -> DeepChemStyleGraphConv:
    torch.manual_seed(42)
    return DeepChemStyleGraphConv(**kwargs)


# --- graph construction -----------------------------------------------------


def test_graph_has_expected_shapes_and_undirected_edges():
    graph = smiles_to_graph(ASPIRIN, label=1, row_index=7)
    assert graph.x.shape == (13, 75)
    # 13 bonds in aspirin, each stored in both directions.
    assert graph.edge_index.shape == (2, 26)
    assert graph.y.tolist() == [1]
    assert graph.source_row.tolist() == [7]
    assert graph.smiles == ASPIRIN


def test_row_identifiers_survive_batching():
    """PyG increments any attribute whose key contains 'index' by the node
    offset. The row identifier must not be affected by that behaviour, or every
    per-molecule prediction is attributed to the wrong dataset row."""
    graphs = [
        smiles_to_graph("CCO", label=1, row_index=10),
        smiles_to_graph("c1ccccc1", label=0, row_index=20),
        smiles_to_graph(ASPIRIN, label=1, row_index=30),
    ]
    batch = next(iter(DataLoader(graphs, batch_size=3)))
    assert batch.source_row.tolist() == [10, 20, 30]


def test_every_edge_has_its_reverse():
    graph = smiles_to_graph(ASPIRIN)
    edges = {tuple(pair) for pair in graph.edge_index.t().tolist()}
    assert all((end, start) in edges for start, end in edges)


def test_invalid_smiles_returns_none_and_is_reported():
    assert smiles_to_graph("not-a-molecule") is None
    graphs, report = build_graphs([ASPIRIN, "not-a-molecule"], labels=[1, 0])
    assert report.requested == 2
    assert report.built == 1
    assert report.failed_smiles == ("not-a-molecule",)
    assert len(graphs) == 1


def test_single_atom_molecule_has_empty_edge_index():
    graph = smiles_to_graph("C")
    assert graph.x.shape == (1, 75)
    assert graph.edge_index.shape == (2, 0)


def test_atom_index_table_labels_every_atom():
    table = atom_index_table(REFERENCE_SMILES)
    assert len(table) == 16
    assert table[0]["atom_label"] == "C0"
    assert table[1]["atom_symbol"] == "N"
    assert sum(row["is_aromatic"] for row in table) == 6


# --- model ------------------------------------------------------------------


def test_forward_returns_logits_per_graph():
    model = make_model().eval()
    graphs = [smiles_to_graph(ASPIRIN, label=1), smiles_to_graph(REFERENCE_SMILES, label=0)]
    batch = next(iter(DataLoader(graphs, batch_size=2)))
    logits = model(batch.x, batch.edge_index, batch.batch)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_batching_matches_individual_forward_passes():
    """A graph's prediction must not depend on what else is in its batch."""
    model = make_model().eval()
    graphs = [smiles_to_graph(ASPIRIN), smiles_to_graph(REFERENCE_SMILES)]

    with torch.no_grad():
        batched = model(
            *(lambda b: (b.x, b.edge_index, b.batch))(
                next(iter(DataLoader(graphs, batch_size=2)))
            )
        )
        singles = torch.cat([model(g.x, g.edge_index) for g in graphs])

    torch.testing.assert_close(batched, singles, rtol=1e-5, atol=1e-5)


def test_graph_output_is_permutation_invariant():
    """D-011 relies on this: reordering atoms must not change the prediction."""
    model = make_model().eval()
    graph = smiles_to_graph(REFERENCE_SMILES)

    permutation = torch.randperm(graph.x.size(0), generator=torch.Generator().manual_seed(0))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())

    with torch.no_grad():
        original = model(graph.x, graph.edge_index)
        shuffled = model(graph.x[permutation], inverse[graph.edge_index])

    torch.testing.assert_close(original, shuffled, rtol=1e-5, atol=1e-5)


def test_loss_is_finite_and_backpropagates():
    model = make_model()
    graphs = [smiles_to_graph(ASPIRIN, label=1), smiles_to_graph(REFERENCE_SMILES, label=0)]
    batch = next(iter(DataLoader(graphs, batch_size=2)))

    logits = model(batch.x, batch.edge_index, batch.batch)
    loss = torch.nn.CrossEntropyLoss()(logits, batch.y)

    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_eval_mode_is_deterministic():
    """Dropout must be inactive so a reported checkpoint scores reproducibly."""
    model = make_model().eval()
    graph = smiles_to_graph(ASPIRIN)
    with torch.no_grad():
        first = model(graph.x, graph.edge_index)
        second = model(graph.x, graph.edge_index)
    torch.testing.assert_close(first, second)


def test_architecture_matches_recovered_specification():
    model = make_model()
    assert len(model.convs) == 2
    assert model.convs[0].in_channels == 75
    assert model.convs[0].out_channels == 64
    assert model.convs[1].out_channels == 64
    assert model.fc1.in_features == 64 and model.fc1.out_features == 128
    assert model.fc2.out_features == 2
    assert model.dropout_rate == 0.2
    assert len(model.bns) == 2


def test_gnnexplainer_runs_against_the_model():
    """P0.8 acceptance: the explainer must consume this model with no wrapper."""
    from torch_geometric.explain import Explainer, GNNExplainer

    model = make_model().eval()
    graph = smiles_to_graph(REFERENCE_SMILES)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=5),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )
    explanation = explainer(
        x=graph.x,
        edge_index=graph.edge_index,
        batch=torch.zeros(graph.x.size(0), dtype=torch.long),
    )

    assert explanation.node_mask.shape == (16, 75)
    assert explanation.edge_mask.shape == (graph.edge_index.size(1),)
