"""Metric computation, bootstrap intervals, and fold summaries."""

from __future__ import annotations

import numpy as np
import pytest

from src.model4.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
    summarize_folds,
)


def test_perfect_prediction_scores_one():
    y_true = np.array([0, 0, 1, 1])
    metrics = classification_metrics(y_true, y_true, np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == {
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 2,
    }


def test_confusion_matrix_orientation_is_explicit():
    """Guards against silently transposing the matrix in a manuscript table."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    metrics = classification_metrics(y_true, y_pred, np.array([0.6, 0.2, 0.4, 0.9]))
    assert metrics["confusion_matrix"]["false_positive"] == 1
    assert metrics["confusion_matrix"]["false_negative"] == 1
    assert metrics["confusion_matrix"]["true_positive"] == 1
    assert metrics["confusion_matrix"]["true_negative"] == 1


def test_class_counts_are_reported():
    metrics = classification_metrics(
        np.array([0, 1, 1]), np.array([0, 1, 1]), np.array([0.1, 0.9, 0.8])
    )
    assert metrics["n"] == 3
    assert metrics["n_positive"] == 2
    assert metrics["n_negative"] == 1


def test_single_class_evaluation_returns_none_for_auc():
    """AUC is undefined with one class present; it must not be invented."""
    metrics = classification_metrics(
        np.array([1, 1]), np.array([1, 1]), np.array([0.8, 0.9])
    )
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["accuracy"] == 1.0


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_prob = np.where(y_true == 1, rng.uniform(0.5, 1, 200), rng.uniform(0, 0.5, 200))
    y_pred = (y_prob >= 0.5).astype(int)

    point = classification_metrics(y_true, y_pred, y_prob)
    intervals = bootstrap_confidence_intervals(y_true, y_pred, y_prob, n_resamples=200)

    for name in ("accuracy", "f1", "roc_auc"):
        assert intervals[name]["lower"] <= point[name] <= intervals[name]["upper"]
        assert intervals[name]["confidence"] == 0.95


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    y_true = np.array([0, 1] * 50)
    y_prob = np.linspace(0, 1, 100)
    y_pred = (y_prob >= 0.5).astype(int)

    first = bootstrap_confidence_intervals(y_true, y_pred, y_prob, n_resamples=100, seed=7)
    second = bootstrap_confidence_intervals(y_true, y_pred, y_prob, n_resamples=100, seed=7)
    assert first == second


def test_fold_summary_reports_mean_sd_and_interval():
    folds = [
        {"accuracy": 0.80, "f1": 0.70, "roc_auc": 0.90, "precision": 0.7, "recall": 0.7, "pr_auc": 0.8},
        {"accuracy": 0.90, "f1": 0.80, "roc_auc": 0.92, "precision": 0.8, "recall": 0.8, "pr_auc": 0.9},
    ]
    summary = summarize_folds(folds)
    assert summary["accuracy"]["mean"] == pytest.approx(0.85)
    # Sample standard deviation (ddof=1), not the population value.
    assert summary["accuracy"]["sd"] == pytest.approx(0.0707106, rel=1e-4)
    assert summary["accuracy"]["n_folds"] == 2
    assert summary["accuracy"]["values"] == [0.80, 0.90]
    assert summary["accuracy"]["ci95_lower"] < 0.85 < summary["accuracy"]["ci95_upper"]


def test_fold_summary_skips_undefined_metrics():
    summary = summarize_folds([{"accuracy": 0.8, "roc_auc": None}])
    assert summary["roc_auc"] is None
    assert summary["accuracy"]["mean"] == 0.8
