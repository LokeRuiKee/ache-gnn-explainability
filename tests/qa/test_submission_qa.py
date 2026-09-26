"""Tests for the artifact inventory.

The inventory is what lets a reader confirm two things without rerunning
anything: that every result file in this repository has a documented command
that regenerates it, and that no file has changed since it was recorded. Both
claims fail silently when they break — a wrong producer still looks attributed,
and a stale hash still looks like a hash — so they are asserted here rather than
trusted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = REPOSITORY_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inventory = load_script("build_artifact_inventory")


# --- producer attribution ---------------------------------------------------


def test_longest_matching_prefix_wins():
    assert (
        inventory.producer_for("results/model4/reproduction/metrics.json")
        == "scripts/train_model4.py --protocol reproduction"
    )


def test_exact_file_match_is_attributed():
    assert (
        inventory.producer_for("results/model4/model4_spec_evidence.json")
        == "scripts/recover_model4_config.py"
    )


def test_unknown_path_is_reported_not_guessed():
    assert inventory.producer_for("results/mystery/output.json") == "unattributed"


# --- end-to-end on the real repository --------------------------------------


def test_every_committed_artifact_has_a_producer():
    report = json.loads(
        (REPOSITORY_ROOT / "results/artifact_inventory.json").read_text(encoding="utf-8")
    )
    assert report["unattributed"] == []
    assert report["artifact_count"] > 0


# --- standalone plot scripts must be attributed individually -----------------

STEM_PATTERN = re.compile(r"""["']--stem["'],\s*default=["']([A-Za-z0-9_]+)["']""")


def _plot_script_stems() -> list:
    """Every `scripts/plot_*.py` paired with the figure stem it defaults to."""
    found = []
    for path in sorted((REPOSITORY_ROOT / "scripts").glob("plot_*.py")):
        match = STEM_PATTERN.search(path.read_text(encoding="utf-8"))
        if match:
            found.append((f"scripts/{path.name}", match.group(1)))
    return found


def test_the_plot_script_scan_actually_finds_scripts():
    """Guard the guard: a broken pattern would make the next test vacuous."""
    assert len(_plot_script_stems()) >= 3


@pytest.mark.parametrize("script,stem", _plot_script_stems())
def test_each_plot_script_figure_is_attributed_to_that_script(script, stem):
    """`results/figures` maps to the figure generator as a directory rule.

    A standalone plot script's output inherits that rule unless an exact-path
    entry overrides it, so the inventory reports the file as attributed while
    naming a script that cannot regenerate it. A clean pass hiding a wrong
    answer is worse than a reported gap, so require the exact entry.
    """
    for extension in ("png", "svg"):
        key = f"results/figures/{stem}.{extension}"
        assert key in inventory.PRODUCERS, (
            f"{key} has no exact PRODUCERS entry, so build_artifact_inventory.py "
            f"would attribute it to the results/figures directory rule rather "
            f"than to {script}."
        )
        assert inventory.PRODUCERS[key] == script


# --- the case-study molecule's atom 29 --------------------------------------


def test_case_study_atom_29_is_chlorine_not_carbon():
    """The atom labelled 29 in the explanation figure is chlorine, not carbon.

    Lowercase `l` and uppercase `I` are hard to distinguish in some fonts, so
    `Cl29` can be misread as a mistyped carbon index. Relabelling it `C29` would
    introduce an error rather than fix one.

    This pins the fact to the committed SMILES: if `CASE_STUDY` is edited the
    atom indices shift, and this fails rather than the figure quietly going
    wrong.
    """
    Chem = pytest.importorskip("rdkit.Chem")
    module = load_script("plot_case_study_attribution")
    molecule = Chem.MolFromSmiles(module.CASE_STUDY)
    assert molecule is not None
    assert molecule.GetAtomWithIdx(29).GetSymbol() == "Cl"
    # The highly attributed bond is C28-Cl29, so pin its endpoints too.
    assert molecule.GetAtomWithIdx(28).GetSymbol() == "C"
    assert molecule.GetBondBetweenAtoms(28, 29) is not None


def test_case_study_molecule_has_exactly_one_chlorine():
    Chem = pytest.importorskip("rdkit.Chem")
    module = load_script("plot_case_study_attribution")
    molecule = Chem.MolFromSmiles(module.CASE_STUDY)
    chlorines = [a.GetIdx() for a in molecule.GetAtoms() if a.GetSymbol() == "Cl"]
    assert chlorines == [29]


# --- the inventory's own hashes must be true ---------------------------------


def _committed_inventory() -> dict:
    path = REPOSITORY_ROOT / "results/artifact_inventory.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_recorded_hash_matches_its_file():
    """The inventory exists so a reader can confirm a file has not changed.

    An entry whose SHA-256 does not match its own file defeats that, and it
    fails silently, so it is checked here. The usual cause is editing a result
    file without regenerating the inventory afterwards.
    """
    mismatches = []
    for entry in _committed_inventory()["artifacts"]:
        path = REPOSITORY_ROOT / entry["path"]
        if not path.exists():
            mismatches.append(f"{entry['path']}: recorded but missing from disk")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            mismatches.append(
                f"{entry['path']}: recorded {entry['sha256'][:12]}, "
                f"actual {actual[:12]}"
            )
    assert not mismatches, (
        "Inventory is out of date. Regenerate it with:\n"
        "  python scripts/build_artifact_inventory.py\n" + "\n".join(mismatches)
    )


def test_every_recorded_size_matches_its_file():
    for entry in _committed_inventory()["artifacts"]:
        path = REPOSITORY_ROOT / entry["path"]
        if path.exists():
            assert path.stat().st_size == entry["bytes"], entry["path"]
