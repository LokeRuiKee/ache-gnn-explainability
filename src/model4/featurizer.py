"""75-dimensional atom featurization matching DeepChem's ``ConvMolFeaturizer``.

The historical Model 4 pipeline featurized molecules with
``deepchem.feat.ConvMolFeaturizer``. This module reproduces that feature layout
exactly using RDKit alone, with one deliberate difference recorded in
``configs/model4_pyg.yaml`` and ``docs/discrepancies.md`` D-011:

    ``ConvMolFeaturizer`` emits atom rows sorted by degree. This module emits
    rows in RDKit atom order so that explanation node indices map to named
    atoms. Graph-level predictions are unaffected because the model is
    permutation-invariant at the graph level.

The layout is verified against a feature tensor saved in the historical
notebooks; see ``tests/model4/test_featurizer.py``.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType

# DeepChem's atom symbol vocabulary, in order. The final entry absorbs any
# symbol outside the list.
ATOM_SYMBOLS: tuple[str, ...] = (
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg",
    "Na", "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl",
    "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H",
    "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr",
    "Pt", "Hg", "Pb", "Unknown",
)
DEGREES: tuple[int, ...] = tuple(range(11))
IMPLICIT_VALENCES: tuple[int, ...] = tuple(range(7))
HYBRIDIZATIONS: tuple[HybridizationType, ...] = (
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
)
TOTAL_HS: tuple[int, ...] = tuple(range(5))

NODE_FEATURE_DIM = (
    len(ATOM_SYMBOLS)
    + len(DEGREES)
    + len(IMPLICIT_VALENCES)
    + 2
    + len(HYBRIDIZATIONS)
    + 1
    + len(TOTAL_HS)
)
assert NODE_FEATURE_DIM == 75, NODE_FEATURE_DIM

# Block offsets, exposed so downstream code can name a feature index instead of
# hard-coding an integer.
OFFSET_SYMBOL = 0
OFFSET_DEGREE = OFFSET_SYMBOL + len(ATOM_SYMBOLS)          # 44
OFFSET_IMPLICIT_VALENCE = OFFSET_DEGREE + len(DEGREES)     # 55
INDEX_FORMAL_CHARGE = OFFSET_IMPLICIT_VALENCE + len(IMPLICIT_VALENCES)  # 62
INDEX_RADICAL_ELECTRONS = INDEX_FORMAL_CHARGE + 1          # 63
OFFSET_HYBRIDIZATION = INDEX_RADICAL_ELECTRONS + 1         # 64
INDEX_AROMATIC = OFFSET_HYBRIDIZATION + len(HYBRIDIZATIONS)  # 69
OFFSET_TOTAL_HS = INDEX_AROMATIC + 1                       # 70


def _implicit_valence(atom: Chem.Atom) -> int:
    """Implicit valence via whichever RDKit API this build exposes.

    RDKit 2026.03 deprecated ``GetImplicitValence`` in favour of
    ``GetValence(ValenceType.IMPLICIT)``. Both return the same quantity, so the
    feature value is unchanged; this only avoids a per-atom deprecation warning.
    """
    valence_type = getattr(Chem, "ValenceType", None)
    if valence_type is not None:
        return atom.GetValence(valence_type.IMPLICIT)
    return atom.GetImplicitValence()


def _one_of_k_unk(value, vocabulary: tuple) -> list[float]:
    """One-hot encode ``value``, folding anything unknown into the last slot."""
    encoding = [0.0] * len(vocabulary)
    try:
        encoding[vocabulary.index(value)] = 1.0
    except ValueError:
        encoding[-1] = 1.0
    return encoding


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """Return the 75-dimensional feature vector for one RDKit atom."""
    features: list[float] = []
    features += _one_of_k_unk(atom.GetSymbol(), ATOM_SYMBOLS)
    features += _one_of_k_unk(atom.GetDegree(), DEGREES)
    features += _one_of_k_unk(_implicit_valence(atom), IMPLICIT_VALENCES)
    features.append(float(atom.GetFormalCharge()))
    features.append(float(atom.GetNumRadicalElectrons()))
    features += _one_of_k_unk(atom.GetHybridization(), HYBRIDIZATIONS)
    features.append(float(atom.GetIsAromatic()))
    features += _one_of_k_unk(atom.GetTotalNumHs(), TOTAL_HS)
    return np.asarray(features, dtype=np.float32)


def molecule_features(mol: Chem.Mol) -> np.ndarray:
    """Return an ``(n_atoms, 75)`` matrix in RDKit atom order."""
    if mol.GetNumAtoms() == 0:
        return np.zeros((0, NODE_FEATURE_DIM), dtype=np.float32)
    return np.stack([atom_features(atom) for atom in mol.GetAtoms()])


def describe_feature_index(index: int) -> str:
    """Return a human-readable name for a node-feature column.

    Used when reporting GNNExplainer attribute masks so that a feature index in
    a figure or table is never published without its meaning.
    """
    if not 0 <= index < NODE_FEATURE_DIM:
        raise IndexError(f"feature index {index} outside 0..{NODE_FEATURE_DIM - 1}")
    if index < OFFSET_DEGREE:
        return f"atom_symbol={ATOM_SYMBOLS[index - OFFSET_SYMBOL]}"
    if index < OFFSET_IMPLICIT_VALENCE:
        return f"degree={DEGREES[index - OFFSET_DEGREE]}"
    if index < INDEX_FORMAL_CHARGE:
        return f"implicit_valence={IMPLICIT_VALENCES[index - OFFSET_IMPLICIT_VALENCE]}"
    if index == INDEX_FORMAL_CHARGE:
        return "formal_charge"
    if index == INDEX_RADICAL_ELECTRONS:
        return "num_radical_electrons"
    if index < INDEX_AROMATIC:
        return f"hybridization={HYBRIDIZATIONS[index - OFFSET_HYBRIDIZATION].name}"
    if index == INDEX_AROMATIC:
        return "is_aromatic"
    return f"total_num_hs={TOTAL_HS[index - OFFSET_TOTAL_HS]}"
