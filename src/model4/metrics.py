"""Classification metrics computed from one checkpoint's saved predictions.

Every metric here is a pure function of ``(y_true, y_pred, y_prob)``. Nothing in
this module touches a model or a training history, which is what makes it
impossible to report metrics that came from different epochs (see
``docs/discrepancies.md`` D-004).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    """Return every reported metric for the positive class (label 1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    single_class = len(np.unique(y_true)) < 2
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    return {
        "n": int(y_true.size),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # ROC-AUC and PR-AUC are undefined when only one class is present.
        "roc_auc": None if single_class else float(roc_auc_score(y_true, y_prob)),
        "pr_auc": None if single_class else float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap intervals over the evaluated samples.

    This quantifies uncertainty of a *single* checkpoint on a finite test set.
    It is not a substitute for cross-validation variability, and the two must
    never be presented as the same quantity.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    rng = np.random.default_rng(seed)
    n = y_true.size
    collected: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}

    for _ in range(n_resamples):
        index = rng.integers(0, n, size=n)
        resampled = classification_metrics(y_true[index], y_pred[index], y_prob[index])
        for name in METRIC_NAMES:
            value = resampled[name]
            if value is not None:
                collected[name].append(value)

    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, dict] = {}
    for name, values in collected.items():
        if not values:
            intervals[name] = None
            continue
        array = np.asarray(values)
        intervals[name] = {
            "lower": float(np.quantile(array, alpha)),
            "upper": float(np.quantile(array, 1.0 - alpha)),
            "n_resamples": int(array.size),
            "confidence": confidence,
        }
    return intervals


def summarize_folds(fold_metrics: list[dict]) -> dict:
    """Mean, SD, and a normal-approximation 95% interval across folds.

    ``ddof=1`` because folds are a sample. With few folds the interval is wide
    and must be reported as such rather than as evidence of significance.
    """
    summary: dict[str, dict] = {}
    for name in METRIC_NAMES:
        values = [
            fold[name] for fold in fold_metrics if fold.get(name) is not None
        ]
        if not values:
            summary[name] = None
            continue
        array = np.asarray(values, dtype=float)
        mean = float(array.mean())
        sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
        half_width = 1.96 * sd / np.sqrt(array.size) if array.size > 1 else 0.0
        summary[name] = {
            "mean": mean,
            "sd": sd,
            "n_folds": int(array.size),
            "ci95_lower": mean - half_width,
            "ci95_upper": mean + half_width,
            "values": [float(value) for value in array],
        }
    return summary
