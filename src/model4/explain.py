"""GNNExplainer applied to a frozen Model 4 checkpoint, mapped back to chemistry.

Design rules enforced here:

* The explained model is the exact frozen checkpoint whose performance is
  reported. Nothing re-trains.
* Node indices are RDKit atom indices (see ``docs/discrepancies.md`` D-011), so
  every score can be published with a verifiable atom label.
* Aggregations are explicit: a node's importance is the **sum** of its attribute
  mask, and a bond's importance is the **mean** of its two directed edge masks.
  Both are documented rather than left implicit.
* Attribution is a statement about the model's decision surface, never about
  biochemical causality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from scipy.stats import spearmanr
from torch_geometric.explain import Explainer, GNNExplainer

from src.model4.graphs import atom_index_table, smiles_to_graph

NODE_AGGREGATION = "sum_over_attribute_mask"
EDGE_AGGREGATION = "mean_over_both_directions"


@dataclass(frozen=True)
class ExplainerSettings:
    """Every explainer knob, recorded alongside each saved explanation."""

    algorithm_epochs: int = 200
    explanation_type: str = "model"
    node_mask_type: str = "attributes"
    edge_mask_type: str = "object"
    mode: str = "multiclass_classification"
    task_level: str = "graph"
    return_type: str = "raw"
    lr: float = 0.01
    seed: int = 42

    def as_dict(self) -> dict:
        return {
            "algorithm": "GNNExplainer",
            "algorithm_epochs": self.algorithm_epochs,
            "explanation_type": self.explanation_type,
            "node_mask_type": self.node_mask_type,
            "edge_mask_type": self.edge_mask_type,
            "mode": self.mode,
            "task_level": self.task_level,
            "return_type": self.return_type,
            "lr": self.lr,
            "seed": self.seed,
            "node_aggregation": NODE_AGGREGATION,
            "edge_aggregation": EDGE_AGGREGATION,
        }


def build_explainer(model: torch.nn.Module, settings: ExplainerSettings) -> Explainer:
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=settings.algorithm_epochs, lr=settings.lr),
        explanation_type=settings.explanation_type,
        node_mask_type=settings.node_mask_type,
        edge_mask_type=settings.edge_mask_type,
        model_config=dict(
            mode=settings.mode,
            task_level=settings.task_level,
            return_type=settings.return_type,
        ),
    )


@torch.no_grad()
def predict_one(model: torch.nn.Module, graph) -> dict:
    logits = model(graph.x, graph.edge_index)
    probabilities = torch.softmax(logits, dim=1)[0]
    return {
        "predicted_class": int(logits.argmax(dim=1).item()),
        "probability_class_0": float(probabilities[0]),
        "probability_class_1": float(probabilities[1]),
    }


def explain_molecule(
    model: torch.nn.Module,
    smiles: str,
    settings: ExplainerSettings,
    seed: int | None = None,
) -> dict:
    """Explain one molecule and return atom- and bond-level attributions."""
    torch.manual_seed(settings.seed if seed is None else seed)

    graph = smiles_to_graph(smiles)
    if graph is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    explainer = build_explainer(model, settings)
    explanation = explainer(
        x=graph.x,
        edge_index=graph.edge_index,
        batch=torch.zeros(graph.x.size(0), dtype=torch.long),
    )

    node_mask = explanation.node_mask.detach().cpu().numpy()
    edge_mask = explanation.edge_mask.detach().cpu().numpy()

    atom_importance = node_mask.sum(axis=1)
    bond_importance = _bond_importance(smiles, graph, edge_mask)

    return {
        "smiles": smiles,
        "n_atoms": int(graph.x.size(0)),
        "n_bonds": len(bond_importance),
        "prediction": predict_one(model, graph),
        "atom_importance_raw": atom_importance.tolist(),
        "atom_importance_normalized": _normalize(atom_importance).tolist(),
        "feature_importance": node_mask.sum(axis=0).tolist(),
        "bond_importance": bond_importance,
        "settings": settings.as_dict(),
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a flat mask maps to all zeros rather than dividing by zero."""
    span = values.max() - values.min()
    if span <= 0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def _bond_importance(smiles: str, graph, edge_mask: np.ndarray) -> list[dict]:
    """Combine each bond's two directed edge scores into one value."""
    mol = Chem.MolFromSmiles(smiles)
    edges = graph.edge_index.t().tolist()
    lookup: dict[tuple[int, int], list[float]] = {}
    for position, (start, end) in enumerate(edges):
        lookup.setdefault((min(start, end), max(start, end)), []).append(
            float(edge_mask[position])
        )

    bonds = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        scores = lookup.get((min(begin, end), max(begin, end)), [])
        bonds.append(
            {
                "bond_index": bond.GetIdx(),
                "begin_atom": begin,
                "end_atom": end,
                "bond_type": str(bond.GetBondType()),
                "is_aromatic": bool(bond.GetIsAromatic()),
                "importance": float(np.mean(scores)) if scores else 0.0,
                "n_directed_edges": len(scores),
            }
        )
    return bonds


def top_k_atoms(atom_importance: list[float] | np.ndarray, k: int) -> list[int]:
    """Indices of the ``k`` highest-scoring atoms, ties broken by lower index."""
    values = np.asarray(atom_importance)
    order = np.lexsort((np.arange(values.size), -values))
    return sorted(int(index) for index in order[:k])


def connected_motifs(smiles: str, atom_indices: list[int]) -> list[dict]:
    """Group selected atoms into connected substructures of the molecule.

    A scattered set of high-scoring atoms and one contiguous fragment mean very
    different things chemically, so connectivity is reported explicitly.
    """
    mol = Chem.MolFromSmiles(smiles)
    selected = set(atom_indices)
    neighbours = {
        index: {
            neighbour.GetIdx()
            for neighbour in mol.GetAtomWithIdx(index).GetNeighbors()
            if neighbour.GetIdx() in selected
        }
        for index in selected
    }

    motifs: list[dict] = []
    unvisited = set(selected)
    while unvisited:
        stack = [unvisited.pop()]
        component = set(stack)
        while stack:
            current = stack.pop()
            for neighbour in neighbours[current] - component:
                component.add(neighbour)
                unvisited.discard(neighbour)
                stack.append(neighbour)
        atoms = sorted(component)
        motifs.append(
            {
                "atom_indices": atoms,
                "atom_labels": [
                    f"{mol.GetAtomWithIdx(index).GetSymbol()}{index}" for index in atoms
                ],
                "size": len(atoms),
                "smarts": Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, canonical=True),
                "all_aromatic": all(
                    mol.GetAtomWithIdx(index).GetIsAromatic() for index in atoms
                ),
                "all_in_ring": all(
                    mol.GetAtomWithIdx(index).IsInRing() for index in atoms
                ),
            }
        )
    motifs.sort(key=lambda motif: (-motif["size"], motif["atom_indices"]))
    return motifs


# --- quantitative evaluation ------------------------------------------------


@torch.no_grad()
def _scores_of(
    model: torch.nn.Module, x: torch.Tensor, edge_index: torch.Tensor, class_index: int
) -> tuple[float, float]:
    """Return ``(probability, logit_margin)`` for ``class_index``.

    The margin is ``logit[target] - max(other logits)``. Probability saturates at
    1.0 for confident graphs, which makes a probability drop unmeasurable; the
    margin does not saturate and stays informative in that regime.
    """
    logits = model(x, edge_index)[0]
    probability = float(torch.softmax(logits, dim=0)[class_index])
    other = torch.cat([logits[:class_index], logits[class_index + 1 :]])
    return probability, float(logits[class_index] - other.max())


def _drop_statistics(
    masked,
    order,
    n_items: int,
    size: int,
    n_random: int,
    rng: np.random.Generator,
    baseline_probability: float,
    baseline_margin: float,
) -> dict:
    """Top-k versus bottom-k versus random-k drops for one perturbation size.

    Shared by both probes so their numbers are computed identically and stay
    comparable; only ``masked`` differs between them.
    """
    size = min(size, n_items)
    top_probability, top_margin = masked(order[:size])
    bottom_probability, bottom_margin = masked(order[-size:])
    draws = [masked(rng.choice(n_items, size=size, replace=False)) for _ in range(n_random)]
    random_probabilities = np.array([value for value, _ in draws])
    random_margins = np.array([value for _, value in draws])
    return {
        "k": int(size),
        "probability_drop_top_k": baseline_probability - top_probability,
        "probability_drop_bottom_k": baseline_probability - bottom_probability,
        "probability_drop_random_k_mean": baseline_probability - float(random_probabilities.mean()),
        "probability_drop_random_k_sd": float(random_probabilities.std(ddof=1)),
        "margin_drop_top_k": baseline_margin - top_margin,
        "margin_drop_bottom_k": baseline_margin - bottom_margin,
        "margin_drop_random_k_mean": baseline_margin - float(random_margins.mean()),
        "margin_drop_random_k_sd": float(random_margins.std(ddof=1)),
        # The headline judgement: does the explanation beat random selection?
        "margin_drop_top_minus_random": (baseline_margin - top_margin)
        - (baseline_margin - float(random_margins.mean())),
        # The same advantage in units of the random baseline's own spread, which
        # is what makes results comparable across molecules and across probes
        # whose margins live on very different scales.
        "margin_drop_top_minus_random_sd_units": (
            float(
                ((baseline_margin - top_margin) - (baseline_margin - float(random_margins.mean())))
                / float(random_margins.std(ddof=1))
            )
            if float(random_margins.std(ddof=1)) > 0
            else None
        ),
        "top_k_beats_random_on_margin": bool(top_margin < float(random_margins.mean())),
    }


def fidelity(
    model: torch.nn.Module,
    smiles: str,
    atom_importance: list[float],
    k: int,
    seed: int = 42,
    n_random: int = 50,
    k_curve: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Compare masking the top-k atoms against low-importance and random atoms.

    Atoms are "removed" by zeroing their feature rows while keeping the bond
    topology intact. This is a soft removal: it does not produce a chemically
    valid molecule, and the resulting numbers describe model behaviour only.
    See :func:`edge_fidelity` for the topological probe that avoids this.

    An explanation is faithful only if masking its top atoms degrades the
    prediction *more than masking random atoms does*. The random baseline is
    therefore reported with its spread, not just its mean.
    """
    graph = smiles_to_graph(smiles)
    values = np.asarray(atom_importance)
    n_atoms = values.size
    k = min(k, n_atoms)

    target = predict_one(model, graph)["predicted_class"]
    baseline_probability, baseline_margin = _scores_of(
        model, graph.x, graph.edge_index, target
    )

    def masked(indices) -> tuple[float, float]:
        x = graph.x.clone()
        x[list(indices)] = 0.0
        return _scores_of(model, x, graph.edge_index, target)

    order = np.lexsort((np.arange(n_atoms), -values))
    rng = np.random.default_rng(seed)

    def at_k(size: int) -> dict:
        return _drop_statistics(
            masked, order, n_atoms, size, n_random, rng,
            baseline_probability, baseline_margin,
        )

    primary = at_k(k)
    return {
        "target_class": target,
        "k": int(k),
        "baseline_probability": baseline_probability,
        "baseline_logit_margin": baseline_margin,
        "prediction_is_saturated": bool(baseline_probability >= 0.9999),
        **primary,
        "k_curve": [at_k(size) for size in k_curve if size <= n_atoms],
        "n_random_draws": n_random,
        "removal_semantics": "node feature rows zeroed; bond topology unchanged",
        "interpretation": (
            "Faithfulness requires margin_drop_top_k to exceed "
            "margin_drop_random_k_mean by more than its spread. Probability drops "
            "are uninformative when prediction_is_saturated is true."
        ),
    }


def edge_index_without_bonds(edge_index: torch.Tensor, bonds: list[dict]) -> torch.Tensor:
    """Drop both directed edges of every listed bond.

    The graph stores each bond twice, once per direction. Removing a single
    column would leave a one-way bond, which is not a molecule the model was
    ever trained on, so deletion is always done as a pair.
    """
    targets = {
        (min(bond["begin_atom"], bond["end_atom"]), max(bond["begin_atom"], bond["end_atom"]))
        for bond in bonds
    }
    if not targets:
        return edge_index
    keep = [
        position
        for position, (start, end) in enumerate(edge_index.t().tolist())
        if (min(start, end), max(start, end)) not in targets
    ]
    return edge_index[:, torch.tensor(keep, dtype=torch.long)]


def _fragment_count(n_atoms: int, edge_index: torch.Tensor) -> int:
    """Number of connected components, by union-find over the remaining edges."""
    parent = list(range(n_atoms))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for start, end in edge_index.t().tolist():
        left, right = find(start), find(end)
        if left != right:
            parent[left] = right
    return len({find(node) for node in range(n_atoms)})


def _fragmentation_diagnostic(
    graph, bonds: list[dict], order, size: int, seed: int, n_random: int
) -> dict:
    """How badly each bond selection breaks the molecule apart.

    Deleting a bridging bond splits the graph and changes every downstream
    message path, while deleting a ring bond leaves the molecule connected. If
    the top-ranked bonds fragment the molecule more than random bonds do, a
    larger margin drop is partly a topological artefact rather than evidence
    that the explanation found what the model uses. This is reported so the
    edge-probe result can be read with that confound visible.

    The random draws here are an independent sample from the same distribution;
    this statistic describes which bonds get selected, not the model's output,
    so it does not need to reuse the margin probe's draws.
    """
    n_bonds = len(bonds)
    size = min(size, n_bonds)

    def fragments(positions) -> int:
        reduced = edge_index_without_bonds(
            graph.edge_index, [bonds[int(index)] for index in positions]
        )
        return _fragment_count(graph.x.size(0), reduced)

    rng = np.random.default_rng(seed)
    random_counts = np.array(
        [fragments(rng.choice(n_bonds, size=size, replace=False)) for _ in range(n_random)],
        dtype=float,
    )
    return {
        "fragments_top_k": fragments(order[:size]),
        "fragments_bottom_k": fragments(order[-size:]),
        "fragments_random_k_mean": float(random_counts.mean()),
        "fragments_random_k_sd": float(random_counts.std(ddof=1)),
        "note": (
            "1 means the molecule stayed connected. If fragments_top_k exceeds "
            "fragments_random_k_mean, part of the top-k margin drop is explained "
            "by topology rather than by attribution quality."
        ),
    }


def edge_fidelity(
    model: torch.nn.Module,
    smiles: str,
    bond_importance: list[dict],
    k: int,
    seed: int = 42,
    n_random: int = 50,
    k_curve: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Faithfulness probe that deletes bonds instead of blanking atom features.

    This exists because :func:`fidelity` has a real weakness: zeroing a node's
    feature row produces an atom with no element, no degree, and no
    hybridization, which is far outside anything the model saw in training, so
    an erratic response there may say more about the perturbation than about the
    explanation. Deleting a bond keeps every remaining atom's features intact and
    asks a question the architecture is built to answer, since message passing is
    defined over exactly these edges. It also scores the explainer's *edge* mask,
    which the node probe never examines.

    A molecule with a bond removed is still not a real molecule, so this probe
    corroborates rather than replaces the node-level result: agreement between
    two perturbations with different failure modes is what makes the finding hard
    to dismiss.
    """
    graph = smiles_to_graph(smiles)
    bonds = list(bond_importance)
    n_bonds = len(bonds)

    target = predict_one(model, graph)["predicted_class"]
    baseline_probability, baseline_margin = _scores_of(
        model, graph.x, graph.edge_index, target
    )
    header = {
        "target_class": target,
        "n_bonds": n_bonds,
        "baseline_probability": baseline_probability,
        "baseline_logit_margin": baseline_margin,
        "prediction_is_saturated": bool(baseline_probability >= 0.9999),
        "n_random_draws": n_random,
        "removal_semantics": "bonds deleted in both directions; node features unchanged",
        "ranked_by": "GNNExplainer edge mask, averaged over both directions",
    }

    if n_bonds == 0:
        return {
            **header,
            "applicable": False,
            "k": 0,
            "margin_drop_top_k": None,
            "margin_drop_bottom_k": None,
            "margin_drop_random_k_mean": None,
            "margin_drop_random_k_sd": None,
            "margin_drop_top_minus_random": None,
            "top_k_beats_random_on_margin": None,
            "k_curve": [],
            "interpretation": "No bonds to remove, so the edge probe does not apply.",
        }

    k = min(k, n_bonds)
    importance = np.array([bond["importance"] for bond in bonds], dtype=float)
    order = np.lexsort((np.arange(n_bonds), -importance))
    rng = np.random.default_rng(seed)

    def masked(positions) -> tuple[float, float]:
        reduced = edge_index_without_bonds(
            graph.edge_index, [bonds[int(index)] for index in positions]
        )
        return _scores_of(model, graph.x, reduced, target)

    def at_k(size: int) -> dict:
        return _drop_statistics(
            masked, order, n_bonds, size, n_random, rng,
            baseline_probability, baseline_margin,
        )

    primary = at_k(k)
    return {
        **header,
        "applicable": True,
        **primary,
        "fragmentation": _fragmentation_diagnostic(
            graph, bonds, order, k, seed, n_random
        ),
        "k_curve": [at_k(size) for size in k_curve if size <= n_bonds],
        "interpretation": (
            "Faithfulness requires margin_drop_top_k to exceed "
            "margin_drop_random_k_mean by more than its spread. Node features are "
            "untouched here, so this probe is not subject to the out-of-"
            "distribution objection that limits the node-masking result."
        ),
    }


def stability(
    model: torch.nn.Module,
    smiles: str,
    settings: ExplainerSettings,
    seeds: list[int],
    k: int,
) -> dict:
    """Re-run the explainer under different seeds and measure agreement."""
    runs = [explain_molecule(model, smiles, settings, seed=seed) for seed in seeds]
    scores = [np.asarray(run["atom_importance_raw"]) for run in runs]
    top_sets = [set(top_k_atoms(score, k)) for score in scores]

    jaccards, correlations = [], []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            union = top_sets[i] | top_sets[j]
            jaccards.append(len(top_sets[i] & top_sets[j]) / len(union) if union else 1.0)
            correlation = spearmanr(scores[i], scores[j]).statistic
            if correlation == correlation:  # skip NaN from a constant mask
                correlations.append(float(correlation))

    return {
        "seeds": list(seeds),
        "k": int(k),
        "n_pairs": len(jaccards),
        "jaccard_mean": float(np.mean(jaccards)) if jaccards else None,
        "jaccard_sd": float(np.std(jaccards, ddof=1)) if len(jaccards) > 1 else 0.0,
        "spearman_mean": float(np.mean(correlations)) if correlations else None,
        "spearman_sd": float(np.std(correlations, ddof=1)) if len(correlations) > 1 else 0.0,
        "top_k_sets": [sorted(indices) for indices in top_sets],
    }


def sparsity(atom_importance: list[float], threshold: float = 0.5) -> dict:
    """Proportion of atoms retained at a stated normalized-importance threshold."""
    normalized = _normalize(np.asarray(atom_importance))
    selected = int((normalized >= threshold).sum())
    return {
        "threshold": threshold,
        "n_atoms": int(normalized.size),
        "n_selected": selected,
        "proportion_selected": float(selected / normalized.size),
        "sparsity": float(1.0 - selected / normalized.size),
        "definition": "fraction of atoms below the normalized-importance threshold",
    }


def atom_report(smiles: str, atom_importance: list[float]) -> list[dict]:
    """Join atom identities with their attribution scores for publication tables."""
    normalized = _normalize(np.asarray(atom_importance))
    table = atom_index_table(smiles)
    for row, raw, scaled in zip(table, atom_importance, normalized):
        row["importance_raw"] = float(raw)
        row["importance_normalized"] = float(scaled)
    return sorted(table, key=lambda row: -row["importance_normalized"])
