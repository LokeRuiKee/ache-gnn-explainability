import json
import subprocess
import sys
import warnings
from pathlib import Path

from src.protocol.notebook_protocol import extract_split_protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

HOLDOUT_CODE = """
from sklearn.model_selection import train_test_split
smiles_train, smiles_temp, labels_train, labels_temp = train_test_split(
    filtered_smiles_list, filtered_labels, test_size=0.2, train_size=0.8,
    random_state=42, stratify=filtered_labels)
smiles_test, smiles_val, labels_test, labels_val = train_test_split(
    smiles_temp, labels_temp, test_size=0.5, random_state=42,
    stratify=labels_temp)
"""

KFOLD_CODE = """
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, val_index) in enumerate(skf.split(smiles_list, labels)):
    pass
fold_metrics.append({
    'best_f1': max(history['val_f1']),
    'best_auc': max(history['val_auc']),
    'best_acc': max(history['val_acc']),
})
"""


def write_notebook(tmp_path: Path, *code_cells: str) -> Path:
    path = tmp_path / "evidence.ipynb"
    notebook = {
        "cells": [
            {"cell_type": "code", "source": code.splitlines(keepends=True), "outputs": []}
            for code in code_cells
        ]
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return path


def test_extracts_two_stage_stratified_holdout(tmp_path):
    record = extract_split_protocol(write_notebook(tmp_path, HOLDOUT_CODE))

    assert record["holdout_splits"] == [
        {
            "cell_index": 0,
            "stages": [
                {
                    "train_size": 0.8,
                    "test_size": 0.2,
                    "random_state": 42,
                    "stratified": True,
                },
                {
                    "train_size": None,
                    "test_size": 0.5,
                    "random_state": 42,
                    "stratified": True,
                },
            ],
            "effective_partition_fractions": {
                "train": 0.8,
                "validation": 0.1,
                "test": 0.1,
            },
        }
    ]


def test_extracts_full_dataset_stratified_kfold(tmp_path):
    record = extract_split_protocol(write_notebook(tmp_path, KFOLD_CODE))

    assert record["cross_validation"] == [
        {
            "cell_index": 0,
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "split_inputs": ["smiles_list", "labels"],
            "independent_metric_maxima": True,
        }
    ]


def test_resolves_same_cell_literal_used_for_fold_count(tmp_path):
    code = KFOLD_CODE.replace(
        "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "n_splits = 5\nskf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)",
    )

    record = extract_split_protocol(write_notebook(tmp_path, code))

    assert record["cross_validation"][0]["n_splits"] == 5


def test_keeps_cell_level_provenance(tmp_path):
    record = extract_split_protocol(write_notebook(tmp_path, "x = 1", HOLDOUT_CODE))

    assert record["holdout_splits"][0]["cell_index"] == 1


def test_saved_invalid_escape_does_not_emit_syntax_warning(tmp_path):
    path = write_notebook(tmp_path, "pattern = '\\C'")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        extract_split_protocol(path)

    assert not [item for item in caught if issubclass(item.category, SyntaxWarning)]


def test_extract_protocol_script_runs_from_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/extract_split_protocol.py", "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
