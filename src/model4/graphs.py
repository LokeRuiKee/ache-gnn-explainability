"""Convert SMILES into PyTorch Geometric graphs for the Model 4 reconstruction.

Atoms are emitted in RDKit atom order so that node index *k* always denotes
RDKit atom *k*. This is what makes GNNExplainer output interpretable as
chemistry (see ``docs/discrepancies.md`` D-011).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from rdkit import Chem
from torch_geometric.data import Data

from src.model4.featurizer import NODE_FEATURE_DIM, molecule_features


@dataclass(frozen=True)
class GraphBuildReport:
    """Counts describing a batch conversion, so losses are never silent."""

    requested: int
    built: int
    failed_smiles: tuple[str, ...]


def smiles_to_graph(smiles: str, label: int | None = None, row_index: int | None = None) -> Data | None:
    """Build one PyG graph, or return ``None`` when RDKit cannot parse ``smiles``."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.from_numpy(molecule_features(mol))

    # Undirected: every bond contributes both directions, matching the
    # historical adjacency-list construction.
    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append((begin, end))
        edges.append((end, begin))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, smiles=smiles)
    if label is not None:
        data.y = torch.tensor([int(label)], dtype=torch.long)
    if row_index is not None:
        # Keeps every graph traceable back to its manifest row.
        #
        # The attribute is deliberately NOT called `row_index`: PyG's default
        # `Data.__inc__` adds the running node offset to any attribute whose key
        # contains "index", which silently corrupts the identifier during
        # batching. `source_row` is left untouched.
        data.source_row = torch.tensor([int(row_index)], dtype=torch.long)
    return data


def build_graphs(
    smiles_list: list[str],
    labels: list[int] | None = None,
    row_indices: list[int] | None = None,
) -> tuple[list[Data], GraphBuildReport]:
    """Convert many SMILES, reporting any that failed rather than dropping them silently."""
    graphs: list[Data] = []
    failed: list[str] = []
    for position, smiles in enumerate(smiles_list):
        label = labels[position] if labels is not None else None
        row_index = row_indices[position] if row_indices is not None else None
        graph = smiles_to_graph(smiles, label=label, row_index=row_index)
        if graph is None:
            failed.append(smiles)
            continue
        graphs.append(graph)
    report = GraphBuildReport(
        requested=len(smiles_list),
        built=len(graphs),
        failed_smiles=tuple(failed),
    )
    return graphs, report


def atom_index_table(smiles: str) -> list[dict]:
    """Map every node index to its atom identity.

    Exported alongside each explanation so no atom label is ever published
    without a verifiable mapping back to the structure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return [
        {
            "node_index": atom.GetIdx(),
            "atom_symbol": atom.GetSymbol(),
            "atom_label": f"{atom.GetSymbol()}{atom.GetIdx()}",
            "degree": atom.GetDegree(),
            "formal_charge": atom.GetFormalCharge(),
            "is_aromatic": bool(atom.GetIsAromatic()),
            "hybridization": str(atom.GetHybridization()),
            "total_num_hs": atom.GetTotalNumHs(),
            "in_ring": bool(atom.IsInRing()),
        }
        for atom in mol.GetAtoms()
    ]


__all__ = [
    "NODE_FEATURE_DIM",
    "GraphBuildReport",
    "smiles_to_graph",
    "build_graphs",
    "atom_index_table",
]
