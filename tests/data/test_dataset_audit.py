import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.dataset_audit import audit_dataframe


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_audit_script_runs_by_file_path():
    result = subprocess.run(
        [sys.executable, "scripts/audit_dataset.py", "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_requires_smiles_and_label_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        audit_dataframe(pd.DataFrame({"SMILES": ["CC"]}))


def test_counts_rows_classes_invalid_and_missing_values():
    frame = pd.DataFrame(
        {
            "SMILES": ["CC", "not-smiles", None, "O"],
            "single-class-label": [1, 0, 1, 0],
        }
    )

    summary, duplicates = audit_dataframe(frame)

    assert summary["raw_rows"] == 4
    assert summary["missing_smiles"] == 1
    assert summary["missing_labels"] == 0
    assert summary["invalid_smiles"] == 1
    assert summary["valid_molecules"] == 2
    assert summary["class_counts_raw"] == {"0": 2, "1": 2}
    assert summary["unique_valid_canonical_smiles"] == 2
    assert duplicates.empty


def test_detects_canonical_duplicates_and_conflicting_labels():
    frame = pd.DataFrame(
        {
            "SMILES": ["CCO", "OCC", "CCO"],
            "single-class-label": [1, 1, 0],
        }
    )

    summary, duplicates = audit_dataframe(frame)

    assert summary["duplicate_rows"] == 3
    assert summary["duplicate_groups"] == 1
    assert summary["conflicting_label_groups"] == 1
    assert set(duplicates["row_index"]) == {0, 1, 2}
    assert set(duplicates["canonical_smiles"]) == {"CCO"}


def test_rejects_nonbinary_labels():
    frame = pd.DataFrame(
        {
            "SMILES": ["CC", "O"],
            "single-class-label": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="binary labels 0/1"):
        audit_dataframe(frame)


def test_class_percentages_use_raw_nonmissing_labels():
    frame = pd.DataFrame(
        {
            "SMILES": ["CC", "O", "N", "C"],
            "single-class-label": [1, 1, 1, 0],
        }
    )

    summary, _ = audit_dataframe(frame)

    assert summary["class_percentages_raw"] == {"0": 25.0, "1": 75.0}
