"""Derive explicitly defined molecular identity representations."""

from __future__ import annotations

from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


def _canonical(mol: Chem.Mol, *, isomeric: bool = True) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def derive_identities(smiles: str) -> dict[str, str | bool]:
    """Return four diagnostic identities for one valid SMILES string."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")

    canonical_isomeric = _canonical(mol)

    stereo_free = Chem.Mol(mol)
    Chem.RemoveStereochemistry(stereo_free)
    stereo_insensitive = _canonical(stereo_free, isomeric=False)

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    ranked_fragments = sorted(
        fragments,
        key=lambda fragment: (-fragment.GetNumHeavyAtoms(), _canonical(fragment)),
    )
    largest_fragment = _canonical(ranked_fragments[0])

    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold_mol.GetNumAtoms():
        generic_scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
        Chem.RemoveStereochemistry(generic_scaffold)
        murcko_scaffold = _canonical(generic_scaffold, isomeric=False)
    else:
        murcko_scaffold = ""

    return {
        "parse_valid": True,
        "canonical_isomeric": canonical_isomeric,
        "stereo_insensitive": stereo_insensitive,
        "largest_fragment": largest_fragment,
        "murcko_scaffold": murcko_scaffold,
        "empty_scaffold": not bool(murcko_scaffold),
    }
