import pytest

from src.chemistry.identities import derive_identities


def test_stereo_insensitive_identity_collapses_enantiomers():
    left = derive_identities("N[C@@H](C)C(=O)O")
    right = derive_identities("N[C@H](C)C(=O)O")

    assert left["canonical_isomeric"] != right["canonical_isomeric"]
    assert left["stereo_insensitive"] == right["stereo_insensitive"]


def test_largest_fragment_identity_collapses_salt_form():
    salt = derive_identities("CCN.Cl")
    parent = derive_identities("CCN")

    assert salt["largest_fragment"] == parent["largest_fragment"]


def test_generic_scaffold_identity_ignores_ring_substituents_and_atom_types():
    phenol = derive_identities("Oc1ccccc1")
    methylpyridine = derive_identities("Cc1ccncc1")

    assert phenol["murcko_scaffold"] == "C1CCCCC1"
    assert phenol["murcko_scaffold"] == methylpyridine["murcko_scaffold"]


def test_acyclic_molecule_records_empty_scaffold():
    identities = derive_identities("CCO")

    assert identities["murcko_scaffold"] == ""
    assert identities["empty_scaffold"] is True
    assert identities["parse_valid"] is True


def test_invalid_smiles_is_rejected():
    with pytest.raises(ValueError, match="invalid SMILES"):
        derive_identities("not-smiles")
