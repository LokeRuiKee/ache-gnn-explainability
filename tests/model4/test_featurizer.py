"""Featurizer parity against the historical DeepChem ConvMol layout."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from src.model4.featurizer import (
    NODE_FEATURE_DIM,
    atom_features,
    describe_feature_index,
    molecule_features,
)

# First molecule of cleanData.xlsx. Its saved feature tensor appears in
# FYP2/FYP2exp13.5_best hyperparameter700epoch.ipynb cell 10 output.
REFERENCE_SMILES = "CN(C)C(=O)Oc1cccc([N+](C)(C)C)c1"
REFERENCE_N_ATOMS = 16
# Active indices of the first atom row, read directly from that saved tensor.
REFERENCE_ATOM_0_ACTIVE = {0, 45, 58, 66, 73}


def test_feature_dimension_is_seventy_five():
    assert NODE_FEATURE_DIM == 75


def test_reference_molecule_matches_saved_notebook_tensor():
    """The historical saved tensor is the ground truth for feature parity."""
    mol = Chem.MolFromSmiles(REFERENCE_SMILES)
    features = molecule_features(mol)

    assert features.shape == (REFERENCE_N_ATOMS, 75)
    active = {index for index, value in enumerate(features[0]) if value == 1.0}
    assert active == REFERENCE_ATOM_0_ACTIVE


def test_reference_atom_zero_decodes_to_a_methyl_carbon():
    """Guards against a layout that matches by coincidence rather than meaning."""
    names = sorted(describe_feature_index(i) for i in REFERENCE_ATOM_0_ACTIVE)
    assert names == [
        "atom_symbol=C",
        "degree=1",
        "hybridization=SP3",
        "implicit_valence=3",
        "total_num_hs=3",
    ]


def test_atoms_are_emitted_in_rdkit_order_not_degree_order():
    """D-011: the historical featurizer sorted by degree; this one must not."""
    mol = Chem.MolFromSmiles(REFERENCE_SMILES)
    features = molecule_features(mol)

    # RDKit atom 1 of this SMILES is nitrogen, so row 1 must encode N, not a
    # third methyl carbon as the degree-sorted historical output did.
    assert mol.GetAtomWithIdx(1).GetSymbol() == "N"
    nitrogen_index = describe_feature_index(
        int(np.argmax(features[1][:44]))
    )
    assert nitrogen_index == "atom_symbol=N"


def test_unknown_symbol_folds_into_last_slot():
    mol = Chem.MolFromSmiles("[U]")
    features = molecule_features(mol)
    assert features[0][43] == 1.0  # the "Unknown" slot
    assert describe_feature_index(43) == "atom_symbol=Unknown"


def test_formal_charge_and_aromaticity_are_recorded():
    mol = Chem.MolFromSmiles(REFERENCE_SMILES)
    features = molecule_features(mol)
    charges = [row[62] for row in features]
    aromatics = [row[69] for row in features]
    assert max(charges) == 1.0  # the quaternary ammonium nitrogen
    assert sum(aromatics) == 6.0  # one benzene ring


def test_describe_feature_index_rejects_out_of_range():
    with pytest.raises(IndexError):
        describe_feature_index(75)


def test_empty_molecule_returns_empty_matrix():
    mol = Chem.MolFromSmiles("")
    assert molecule_features(mol).shape == (0, 75)


def test_atom_features_returns_float32_vector():
    mol = Chem.MolFromSmiles("CC")
    vector = atom_features(mol.GetAtomWithIdx(0))
    assert vector.dtype == np.float32
    assert vector.shape == (75,)
