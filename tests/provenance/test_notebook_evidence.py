import json
from pathlib import Path
import subprocess
import sys

from src.provenance.notebook_evidence import (
    extract_notebook_evidence,
    rank_model4_candidates,
)


def _write_notebook(tmp_path: Path, code: str, output: str) -> Path:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "source": code.splitlines(keepends=True),
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": output}
                ]
                if output
                else [],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = tmp_path / "candidate.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return path


def _candidate(path: str, accuracy: float, f1: float, auc: float, metric: str):
    return {
        "path": path,
        "selection_metrics": [metric],
        "metric_triplets": [
            {
                "cell_index": 0,
                "accuracy": accuracy,
                "f1": f1,
                "roc_auc": auc,
            }
        ],
    }


def test_extracts_metric_triplet_and_cell_index(tmp_path):
    path = _write_notebook(
        tmp_path,
        code="history = run_training_loop(monitor_metric='f1')",
        output="Accuracy: 0.8725\nF1 Score: 0.8556\nAUC-ROC: 0.9308",
    )

    record = extract_notebook_evidence(path)

    assert record["metric_triplets"] == [
        {"cell_index": 0, "accuracy": 0.8725, "f1": 0.8556, "roc_auc": 0.9308}
    ]


def test_extracts_model_selection_and_hyperparameters(tmp_path):
    path = _write_notebook(
        tmp_path,
        code=(
            "model = DeepChemStyleGraphConv(hidden_dims=[128, 128], "
            "dropout_rate=0.25)\n"
            "optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)\n"
            "run_training_loop(model=model, num_epochs=1000, "
            "monitor_metric='f1')"
        ),
        output="",
    )

    record = extract_notebook_evidence(path)

    assert record["selection_metrics"] == ["f1"]
    assert record["declared_epochs"] == [1000]
    assert record["learning_rates"] == [0.0005]
    assert record["hidden_dimensions"] == [[128, 128]]
    assert record["dropout_rates"] == [0.25]


def test_extracts_saved_notebook_errors(tmp_path):
    path = _write_notebook(tmp_path, code="broken_call()", output="")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    notebook["cells"][0]["outputs"] = [
        {
            "output_type": "error",
            "ename": "NameError",
            "evalue": "name 'broken_call' is not defined",
            "traceback": [],
        }
    ]
    path.write_text(json.dumps(notebook), encoding="utf-8")

    record = extract_notebook_evidence(path)

    assert record["errors"] == [
        {
            "cell_index": 0,
            "type": "NameError",
            "message": "name 'broken_call' is not defined",
        }
    ]


def test_deepchem_style_class_is_not_evidence_of_deepchem_runtime(tmp_path):
    path = _write_notebook(
        tmp_path,
        code=(
            "from torch_geometric.nn import GraphConv\n"
            "class DeepChemStyleGraphConv: pass"
        ),
        output="",
    )

    record = extract_notebook_evidence(path)

    assert record["framework_evidence"] == ["PyTorch Geometric"]


def test_extracts_cross_validation_average_metrics(tmp_path):
    path = _write_notebook(
        tmp_path,
        code="print(cv_results.mean())",
        output=(
            "Average Scores:\n"
            "best_f1         0.806676\n"
            "best_auc        0.898190\n"
            "best_acc        0.822331\n"
            "min_val_loss    0.404623\n"
            "dtype: float64"
        ),
    )

    record = extract_notebook_evidence(path)

    assert record["cv_average_metrics"] == [
        {
            "cell_index": 0,
            "accuracy": 0.822331,
            "f1": 0.806676,
            "roc_auc": 0.898190,
            "minimum_validation_loss": 0.404623,
        }
    ]


def test_ranking_prefers_f1_selected_result_nearest_reported_model4():
    records = [
        _candidate("surrogate.ipynb", 0.8223, 0.8067, 0.8982, "f1"),
        _candidate("model4.ipynb", 0.8725, 0.8556, 0.9308, "f1"),
        _candidate("other.ipynb", 0.8799, 0.8679, 0.9342, "accuracy"),
    ]

    ranked = rank_model4_candidates(records)

    assert ranked[0]["path"] == "model4.ipynb"
    assert ranked[0]["candidate_score"] < ranked[1]["candidate_score"]


def test_recovery_script_runs_by_file_path_from_repository_root():
    repository_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "scripts/recover_model4_spec.py", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--markdown" in completed.stdout
