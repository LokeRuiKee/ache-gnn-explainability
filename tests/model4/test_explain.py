"""Explanation aggregation, motif extraction, and quantitative XAI measures."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model4.explain import (
    _fragment_count,
    ExplainerSettings,
    atom_report,
    connected_motifs,
    edge_fidelity,
    edge_index_without_bonds,
    explain_molecule,
    fidelity,
    sparsity,
    stability,
    top_k_atoms,
)
from src.model4.graphs import smiles_to_graph
from src.model4.model import DeepChemStyleGraphConv

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
FAST = ExplainerSettings(algorithm_epochs=5, seed=42)


def make_model() -> DeepChemStyleGraphConv:
    torch.manual_seed(42)
    return DeepChemStyleGraphConv(hidden_dims=(8, 8), dense_dim=8).eval()


# --- pure helpers -----------------------------------------------------------


def test_top_k_atoms_picks_highest_and_breaks_ties_by_index():
    assert top_k_atoms([0.1, 0.9, 0.5, 0.9], k=2) == [1, 3]
    assert top_k_atoms([0.5, 0.5, 0.5], k=2) == [0, 1]


def test_sparsity_reports_selected_fraction_and_definition():
    result = sparsity([0.0, 0.25, 0.5, 1.0], threshold=0.5)
    assert result["n_atoms"] == 4
    assert result["n_selected"] == 2  # normalized 0.5 and 1.0
    assert result["proportion_selected"] == pytest.approx(0.5)
    assert result["sparsity"] == pytest.approx(0.5)


def test_flat_importance_normalizes_to_zero_without_dividing_by_zero():
    result = sparsity([0.3, 0.3, 0.3])
    assert result["n_selected"] == 0


def test_connected_motifs_separates_disjoint_fragments():
    # Aspirin atoms 0-1 are bonded; atom 5 is in the ring, far from them.
    motifs = connected_motifs(ASPIRIN, [0, 1, 5])
    sizes = sorted(motif["size"] for motif in motifs)
    assert sizes == [1, 2]
    assert all("atom_labels" in motif for motif in motifs)


def test_connected_motifs_joins_a_contiguous_fragment():
    motifs = connected_motifs(ASPIRIN, [4, 5, 6])
    assert len(motifs) == 1
    assert motifs[0]["size"] == 3
    assert motifs[0]["all_aromatic"] is True


def test_atom_report_is_sorted_and_carries_identities():
    importance = list(np.linspace(0, 1, 13))
    table = atom_report(ASPIRIN, importance)
    assert len(table) == 13
    assert table[0]["importance_normalized"] >= table[-1]["importance_normalized"]
    assert {"atom_label", "atom_symbol", "in_ring", "importance_raw"} <= set(table[0])


# --- explainer integration --------------------------------------------------


def test_explain_returns_atom_and_bond_attributions():
    model = make_model()
    result = explain_molecule(model, ASPIRIN, FAST)

    assert result["n_atoms"] == 13
    assert len(result["atom_importance_raw"]) == 13
    assert len(result["feature_importance"]) == 75
    # 13 bonds in aspirin, each combining its two directed edges.
    assert result["n_bonds"] == 13
    assert all(bond["n_directed_edges"] == 2 for bond in result["bond_importance"])
    assert result["settings"]["mode"] == "multiclass_classification"


def test_explanation_settings_are_recorded_for_publication():
    model = make_model()
    result = explain_molecule(model, ASPIRIN, FAST)
    settings = result["settings"]
    assert settings["node_aggregation"] == "sum_over_attribute_mask"
    assert settings["edge_aggregation"] == "mean_over_both_directions"
    assert settings["algorithm"] == "GNNExplainer"


def test_fidelity_compares_top_against_bottom_and_random():
    model = make_model()
    result = explain_molecule(model, ASPIRIN, FAST)
    scores = fidelity(model, ASPIRIN, result["atom_importance_raw"], k=3, n_random=10)

    assert 0.0 <= scores["baseline_probability"] <= 1.0
    assert scores["k"] == 3
    assert scores["removal_semantics"].startswith("node feature rows zeroed")
    assert scores["target_class"] == result["prediction"]["predicted_class"]
    for field in (
        "margin_drop_top_k",
        "margin_drop_bottom_k",
        "margin_drop_random_k_mean",
        "margin_drop_random_k_sd",
        "top_k_beats_random_on_margin",
    ):
        assert field in scores


def test_fidelity_reports_saturation_so_probability_drops_are_not_misread():
    """A probability pinned at 1.0 cannot drop; the flag must say so."""
    model = make_model()
    result = explain_molecule(model, ASPIRIN, FAST)
    scores = fidelity(model, ASPIRIN, result["atom_importance_raw"], k=3, n_random=5)
    assert scores["prediction_is_saturated"] == (scores["baseline_probability"] >= 0.9999)


def test_fidelity_curve_covers_several_k_values():
    model = make_model()
    result = explain_molecule(model, ASPIRIN, FAST)
    scores = fidelity(
        model, ASPIRIN, result["atom_importance_raw"], k=3, n_random=5, k_curve=(1, 3, 5)
    )
    assert [entry["k"] for entry in scores["k_curve"]] == [1, 3, 5]


def test_fidelity_k_is_clamped_to_molecule_size():
    model = make_model()
    result = explain_molecule(model, ASPIRIN, FAST)
    scores = fidelity(model, ASPIRIN, result["atom_importance_raw"], k=999, n_random=5)
    assert scores["k"] == 13


# --- edge-masking probe -----------------------------------------------------


def test_edge_index_without_bonds_drops_both_directed_edges():
    """A bond is two directed columns; deleting one direction only is not a deletion."""
    graph = smiles_to_graph(ASPIRIN)
    reduced = edge_index_without_bonds(
        graph.edge_index, [{"begin_atom": 0, "end_atom": 1}]
    )
    assert reduced.size(1) == graph.edge_index.size(1) - 2
    remaining = {(min(a, b), max(a, b)) for a, b in reduced.t().tolist()}
    assert (0, 1) not in remaining


def test_edge_index_without_bonds_is_order_independent():
    graph = smiles_to_graph(ASPIRIN)
    forward = edge_index_without_bonds(
        graph.edge_index, [{"begin_atom": 1, "end_atom": 0}]
    )
    backward = edge_index_without_bonds(
        graph.edge_index, [{"begin_atom": 0, "end_atom": 1}]
    )
    assert torch.equal(forward, backward)


def test_edge_fidelity_compares_top_bonds_against_bottom_and_random():
    model = make_model()
    result = explain_molecule(model, ASPIRIN, FAST)
    scores = edge_fidelity(model, ASPIRIN, result["bond_importance"], k=3, n_random=10)

    assert scores["k"] == 3
    assert scores["n_bonds"] == 13
    assert scores["target_class"] == result["prediction"]["predicted_class"]
    assert "bond" in scores["removal_semantics"]
    for field in (
        "margin_drop_top_k",
        "margin_drop_bottom_k",
        "margin_drop_random_k_mean",
        "margin_drop_random_k_sd",
        "top_k_beats_random_on_margin",
    ):
        assert field in scores


def test_edge_fidelity_leaves_node_features_untouched():
    """The whole point of this probe: perturb topology, not the feature matrix."""
    model = make_model()
    graph = smiles_to_graph(ASPIRIN)
    before = graph.x.clone()
    result = explain_molecule(model, ASPIRIN, FAST)
    edge_fidelity(model, ASPIRIN, result["bond_importance"], k=3, n_random=5)
    torch.testing.assert_close(smiles_to_graph(ASPIRIN).x, before)


def test_edge_fidelity_removing_every_bond_makes_top_and_bottom_agree():
    """Deleting all 13 bonds is the same graph however the bonds were ranked."""
    model = make_model()
    result = explain_molecule(model, ASPIRIN, FAST)
    scores = edge_fidelity(model, ASPIRIN, result["bond_importance"], k=13, n_random=3)
    assert scores["margin_drop_top_k"] == pytest.approx(scores["margin_drop_bottom_k"])


def test_both_probes_report_the_advantage_in_sd_units():
    """The standardized effect size is what the drafts quote, so it must be saved."""
    model = make_model()
    result = explain_molecule(model, ASPIRIN, FAST)

    node = fidelity(model, ASPIRIN, result["atom_importance_raw"], k=3, n_random=10)
    edge = edge_fidelity(model, ASPIRIN, result["bond_importance"], k=3, n_random=10)

    for scores in (node, edge):
        advantage = scores["margin_drop_top_minus_random"]
        spread = scores["margin_drop_random_k_sd"]
        assert scores["margin_drop_top_minus_random_sd_units"] == pytest.approx(
            advantage / spread
        )


def test_edge_fidelity_reports_the_fragmentation_confound():
    """Bond deletion can split the molecule; that has to be visible, not hidden."""
    model = make_model()
    result = explain_molecule(model, ASPIRIN, FAST)
    scores = edge_fidelity(model, ASPIRIN, result["bond_importance"], k=3, n_random=5)

    fragments = scores["fragmentation"]
    assert fragments["fragments_top_k"] >= 1
    assert fragments["fragments_random_k_mean"] >= 1.0
    # Aspirin has one ring; removing any 3 bonds cannot exceed 4 pieces.
    assert fragments["fragments_top_k"] <= 4


def test_fragment_count_counts_disconnected_pieces():
    graph = smiles_to_graph(ASPIRIN)
    assert _fragment_count(graph.x.size(0), graph.edge_index) == 1
    # Aspirin atom 0 is a terminal methyl: cutting its single bond isolates it.
    cut = edge_index_without_bonds(graph.edge_index, [{"begin_atom": 0, "end_atom": 1}])
    assert _fragment_count(graph.x.size(0), cut) == 2


def test_edge_fidelity_k_is_clamped_to_bond_count():
    model = make_model()
    result = explain_molecule(model, ASPIRIN, FAST)
    scores = edge_fidelity(model, ASPIRIN, result["bond_importance"], k=999, n_random=3)
    assert scores["k"] == 13


def test_edge_fidelity_curve_covers_several_k_values():
    model = make_model()
    result = explain_molecule(model, ASPIRIN, FAST)
    scores = edge_fidelity(
        model, ASPIRIN, result["bond_importance"], k=3, n_random=5, k_curve=(1, 3, 5)
    )
    assert [entry["k"] for entry in scores["k_curve"]] == [1, 3, 5]


def test_edge_fidelity_reports_a_bondless_molecule_as_not_applicable():
    """Methane has no bonds, so an edge probe has nothing to remove."""
    model = make_model()
    scores = edge_fidelity(model, "C", [], k=3, n_random=3)
    assert scores["n_bonds"] == 0
    assert scores["applicable"] is False
    assert scores["margin_drop_top_k"] is None


def test_stability_reports_agreement_across_seeds():
    model = make_model()
    result = stability(model, ASPIRIN, FAST, seeds=[1, 2, 3], k=3)

    assert result["n_pairs"] == 3
    assert 0.0 <= result["jaccard_mean"] <= 1.0
    assert -1.0 <= result["spearman_mean"] <= 1.0
    assert len(result["top_k_sets"]) == 3


def test_explanation_is_reproducible_for_a_fixed_seed():
    model = make_model()
    first = explain_molecule(model, ASPIRIN, FAST, seed=7)
    second = explain_molecule(model, ASPIRIN, FAST, seed=7)
    np.testing.assert_allclose(
        first["atom_importance_raw"], second["atom_importance_raw"], rtol=1e-6
    )


def test_invalid_smiles_is_rejected():
    with pytest.raises(ValueError, match="could not parse"):
        explain_molecule(make_model(), "not-a-molecule", FAST)
