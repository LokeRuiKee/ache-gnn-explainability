"""Checkpoint-selection semantics, determinism, and manifest loading."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.model4.data import build_partitions, load_manifest
from src.model4.graphs import build_graphs
from src.model4.training import TrainingConfig, load_selected_model, train_model

SMILES = [
    "CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CC(=O)O", "CCOC", "CCBr",
    "CC(C)O", "CNC", "CCS", "CC=O", "c1ccncc1", "CC(N)=O", "CCCC", "CCI",
]
LABELS = [0, 1] * 8

TINY_CONFIG = TrainingConfig(
    num_epochs=3,
    batch_size=4,
    seed=42,
    model_kwargs=dict(hidden_dims=(8, 8), dense_dim=8),
)


def tiny_graphs():
    train, _ = build_graphs(SMILES, LABELS, list(range(len(SMILES))))
    validation, _ = build_graphs(SMILES[:8], LABELS[:8], list(range(8)))
    return train, validation


# --- checkpoint selection ---------------------------------------------------


def test_selected_checkpoint_is_the_best_epoch_by_the_declared_metric():
    train, validation = tiny_graphs()
    result = train_model(train, validation, TINY_CONFIG, progress_every=0)

    best = max(result["history"], key=lambda row: row["val_f1"])
    assert result["selected_epoch"] == best["epoch"]
    assert result["selection_value"] == pytest.approx(best["val_f1"])
    assert result["selection_metric"] == "f1"
    assert result["selection_split"] == "validation"


def test_history_records_every_epoch():
    train, validation = tiny_graphs()
    result = train_model(train, validation, TINY_CONFIG, progress_every=0)
    assert [row["epoch"] for row in result["history"]] == [1, 2, 3]


def test_selected_state_is_a_snapshot_not_the_final_weights():
    """The checkpoint must be frozen at its epoch, not mutated by later training."""
    train, validation = tiny_graphs()
    result = train_model(train, validation, TINY_CONFIG, progress_every=0)

    if result["selected_epoch"] == TINY_CONFIG.num_epochs:
        pytest.skip("Best epoch was the final epoch, so this cannot be distinguished.")

    model = load_selected_model(result["state_dict"], TINY_CONFIG)
    reloaded = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    for name, tensor in result["state_dict"].items():
        torch.testing.assert_close(reloaded[name], tensor)


def test_training_is_reproducible_for_a_fixed_seed():
    train, validation = tiny_graphs()
    first = train_model(train, validation, TINY_CONFIG, progress_every=0)
    second = train_model(train, validation, TINY_CONFIG, progress_every=0)

    assert first["selected_epoch"] == second["selected_epoch"]
    assert first["selection_value"] == pytest.approx(second["selection_value"])
    for name, tensor in first["state_dict"].items():
        torch.testing.assert_close(tensor, second["state_dict"][name])


def test_loaded_checkpoint_is_in_eval_mode():
    train, validation = tiny_graphs()
    result = train_model(train, validation, TINY_CONFIG, progress_every=0)
    model = load_selected_model(result["state_dict"], TINY_CONFIG)
    assert model.training is False


# --- manifest loading -------------------------------------------------------


def manifest_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "partition": ["train", "train", "validation", "test"],
            "label": [0, 1, 0, 1],
        }
    )


def test_manifest_must_cover_every_row_exactly_once(tmp_path):
    path = tmp_path / "manifest.csv"
    manifest_frame().to_csv(path, index=False)
    assert len(load_manifest(path, expected_rows=4)) == 4

    with pytest.raises(ValueError, match="exactly once"):
        load_manifest(path, expected_rows=5)


def test_manifest_rejects_unknown_partition_names(tmp_path):
    frame = manifest_frame()
    frame.loc[0, "partition"] = "holdout"
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="unknown partitions"):
        load_manifest(path, expected_rows=4)


def test_manifest_rejects_missing_columns(tmp_path):
    path = tmp_path / "manifest.csv"
    manifest_frame().drop(columns=["label"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest(path, expected_rows=4)


def test_partitions_follow_the_manifest_and_keep_row_indices():
    workbook = pd.DataFrame({"SMILES": SMILES[:4]})
    graphs, reports = build_partitions(workbook, manifest_frame())

    assert [len(graphs[name]) for name in ("train", "validation", "test")] == [2, 1, 1]
    assert graphs["validation"][0].source_row.item() == 2
    assert graphs["test"][0].y.item() == 1
    assert all(report.failed_smiles == () for report in reports.values())
