"""Bootstrap interval estimation for classification metrics.

Reviewer 5 (essential revision 1) objected to one-sample t-tests that treat a
single performance number copied from another publication as a fixed population
mean. That design ignores the literature model's own sampling uncertainty, its
different split, and its different preprocessing, and it applies several
uncorrected significance tests to strongly correlated metrics.

This module replaces the hypothesis tests with interval estimates:

``bootstrap_metric_intervals``
    Percentile and bias-corrected-and-accelerated (BCa) intervals for one
    model's metrics on one evaluated partition. This is the interval that a
    literature point estimate is *described* against — never tested against.

``paired_cluster_bootstrap``
    Interval for the difference between two models evaluated on identical
    rows. Resampling happens within fold, so the fold structure is preserved
    and both models always see the same resampled rows, which keeps the
    comparison paired. With 3,668 pooled holdout rows this is far better
    powered than the five fold-level observations available to a paired t-test.

The metric implementations below are vectorised reimplementations of the
scikit-learn metrics used elsewhere in this project; ``tests/model4/
test_bootstrap.py`` asserts they agree with scikit-learn, including on tied
scores. They exist only because a bootstrap needs tens of thousands of metric
evaluations and the scikit-learn call overhead dominates at that scale.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


# --------------------------------------------------------------------------
# Fast metric implementations
# --------------------------------------------------------------------------


def _counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    true_positive = int(np.count_nonzero((y_true == 1) & (y_pred == 1)))
    false_positive = int(np.count_nonzero((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.count_nonzero((y_true == 1) & (y_pred == 0)))
    true_negative = int(np.count_nonzero((y_true == 0) & (y_pred == 0)))
    return true_negative, false_positive, false_negative, true_positive


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.count_nonzero(y_true == y_pred) / y_true.size)


def precision(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    _, false_positive, _, true_positive = _counts(y_true, y_pred)
    denominator = true_positive + false_positive
    # scikit-learn's ``zero_division=0`` convention.
    return 0.0 if denominator == 0 else true_positive / denominator


def recall(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    _, _, false_negative, true_positive = _counts(y_true, y_pred)
    denominator = true_positive + false_negative
    return 0.0 if denominator == 0 else true_positive / denominator


def f1(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    _, false_positive, false_negative, true_positive = _counts(y_true, y_pred)
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def roc_auc(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    """Mann-Whitney U form of ROC-AUC, with mid-ranks for tied scores."""
    n_positive = int(np.count_nonzero(y_true == 1))
    n_negative = y_true.size - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(y_prob, kind="mergesort")
    sorted_scores = y_prob[order]
    ranks = np.empty(y_prob.size, dtype=np.float64)

    # Average ranks within each run of equal scores (1-based).
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop + 1)
        start = stop

    positive_rank_sum = float(ranks[y_true == 1].sum())
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )


def pr_auc(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    """Average precision, matching ``sklearn.metrics.average_precision_score``.

    Tied scores are collapsed into a single operating point, which is what
    ``precision_recall_curve`` does, so a model that cannot separate two
    molecules is not credited for an arbitrary ordering between them.
    """
    n_positive = int(np.count_nonzero(y_true == 1))
    if n_positive == 0 or n_positive == y_true.size:
        return float("nan")

    order = np.argsort(-y_prob, kind="mergesort")
    sorted_true = y_true[order]
    sorted_scores = y_prob[order]

    cumulative_tp = np.cumsum(sorted_true == 1)
    cumulative_fp = np.cumsum(sorted_true == 0)

    # Keep only the last index of each run of equal scores.
    distinct = np.nonzero(np.diff(sorted_scores))[0]
    boundary = np.append(distinct, sorted_scores.size - 1)

    true_positive = cumulative_tp[boundary]
    false_positive = cumulative_fp[boundary]

    precision_at = true_positive / (true_positive + false_positive)
    recall_at = true_positive / n_positive
    recall_delta = np.diff(np.concatenate(([0.0], recall_at)))
    return float(np.sum(precision_at * recall_delta))


METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
}


def point_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metrics: Sequence[str] = METRIC_NAMES,
) -> dict[str, float]:
    """Evaluate ``metrics`` once on the supplied arrays."""
    return {name: METRIC_FUNCTIONS[name](y_true, y_pred, y_prob) for name in metrics}


# --------------------------------------------------------------------------
# Interval construction
# --------------------------------------------------------------------------


def _percentile_interval(replicates: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(replicates, alpha)),
        float(np.quantile(replicates, 1.0 - alpha)),
    )


def _bca_interval(
    replicates: np.ndarray,
    observed: float,
    jackknife: np.ndarray,
    confidence: float,
) -> tuple[float, float] | None:
    """Bias-corrected and accelerated interval (Efron 1987).

    Returns ``None`` when the correction is undefined, which happens when every
    replicate lies on one side of the observed value or when the jackknife
    distribution is degenerate. Falling back to the percentile interval is then
    the caller's decision, made explicitly rather than silently.
    """
    from scipy import stats  # imported lazily; the fast path does not need scipy

    proportion_below = float(np.mean(replicates < observed))
    if proportion_below <= 0.0 or proportion_below >= 1.0:
        return None
    bias = float(stats.norm.ppf(proportion_below))

    deviation = jackknife.mean() - jackknife
    numerator = float(np.sum(deviation**3))
    denominator = float(np.sum(deviation**2))
    if denominator <= 0.0:
        return None
    acceleration = numerator / (6.0 * denominator**1.5)

    alpha = (1.0 - confidence) / 2.0
    quantiles = []
    for probability in (alpha, 1.0 - alpha):
        z = float(stats.norm.ppf(probability))
        adjusted = bias + (bias + z) / (1.0 - acceleration * (bias + z))
        quantiles.append(float(stats.norm.cdf(adjusted)))
    if not all(0.0 < q < 1.0 for q in quantiles):
        return None
    return (
        float(np.quantile(replicates, quantiles[0])),
        float(np.quantile(replicates, quantiles[1])),
    )


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
    metrics: Sequence[str] = METRIC_NAMES,
    compute_bca: bool = True,
) -> dict:
    """Bootstrap intervals for one model on one evaluated partition.

    The interval describes sampling uncertainty of this checkpoint's score on a
    finite set of molecules. It is not a cross-validation spread and the two
    must never be combined or presented as the same quantity.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)
    n = y_true.size

    observed = point_metrics(y_true, y_pred, y_prob, metrics)

    rng = np.random.default_rng(seed)
    replicates = {name: np.empty(n_resamples, dtype=np.float64) for name in metrics}
    for replicate in range(n_resamples):
        index = rng.integers(0, n, size=n)
        resampled_true = y_true[index]
        resampled_pred = y_pred[index]
        resampled_prob = y_prob[index]
        for name in metrics:
            replicates[name][replicate] = METRIC_FUNCTIONS[name](
                resampled_true, resampled_pred, resampled_prob
            )

    jackknife: dict[str, np.ndarray] = {}
    if compute_bca:
        keep = np.ones(n, dtype=bool)
        jackknife = {name: np.empty(n, dtype=np.float64) for name in metrics}
        for left_out in range(n):
            keep[left_out] = False
            for name in metrics:
                jackknife[name][left_out] = METRIC_FUNCTIONS[name](
                    y_true[keep], y_pred[keep], y_prob[keep]
                )
            keep[left_out] = True

    results: dict[str, dict] = {}
    for name in metrics:
        finite = replicates[name][np.isfinite(replicates[name])]
        n_degenerate = int(n_resamples - finite.size)
        if finite.size == 0:
            results[name] = None
            continue
        lower, upper = _percentile_interval(finite, confidence)
        entry = {
            "point_estimate": float(observed[name]),
            "percentile_ci_lower": lower,
            "percentile_ci_upper": upper,
            "bootstrap_se": float(finite.std(ddof=1)),
            "n_resamples": int(n_resamples),
            "n_degenerate_resamples": n_degenerate,
            "confidence": confidence,
            "seed": seed,
        }
        if compute_bca:
            finite_jackknife = jackknife[name][np.isfinite(jackknife[name])]
            bca = (
                _bca_interval(finite, float(observed[name]), finite_jackknife, confidence)
                if finite_jackknife.size > 1
                else None
            )
            if bca is None:
                entry["bca_ci_lower"] = None
                entry["bca_ci_upper"] = None
                entry["bca_note"] = (
                    "BCa correction undefined for this metric on this partition; "
                    "use the percentile interval."
                )
            else:
                entry["bca_ci_lower"], entry["bca_ci_upper"] = bca
        results[name] = entry
    return results


def locate_reference_value(interval: dict, reference: float) -> dict:
    """Describe where an external point estimate falls relative to an interval.

    This is deliberately *descriptive*. A literature figure carries its own
    unreported sampling uncertainty and was produced under a different split and
    preprocessing, so containment in this interval is not a hypothesis test and
    must never be reported as one.
    """
    lower = interval["percentile_ci_lower"]
    upper = interval["percentile_ci_upper"]
    if reference < lower:
        position = "below interval"
    elif reference > upper:
        position = "above interval"
    else:
        position = "within interval"
    return {
        "reference_value": float(reference),
        "position": position,
        "difference_from_point_estimate": float(interval["point_estimate"] - reference),
        "interpretation": (
            "Descriptive only: the reference is a single published number whose "
            "own sampling uncertainty is unknown and which was obtained under a "
            "different split and preprocessing pipeline. No significance is claimed."
        ),
    }


def paired_cluster_bootstrap(
    folds: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    first_model: str,
    second_model: str,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
    metrics: Sequence[str] = METRIC_NAMES,
) -> dict:
    """Interval for ``first_model`` minus ``second_model`` on identical rows.

    ``predictions`` maps a model name to ``(y_pred, y_prob)`` arrays aligned
    with ``folds`` and ``y_true``. Rows are resampled *within* fold, so every
    replicate keeps the original fold sizes and applies the same resampled rows
    to both models. Both a macro summary (the mean over per-fold scores, which
    is the quantity the manuscript reports as "mean ± SD") and a pooled summary
    (one score over all holdout rows) are returned, because they answer slightly
    different questions and disagree when folds differ in difficulty.
    """
    folds = np.asarray(folds)
    y_true = np.asarray(y_true).astype(np.int64)
    unique_folds = np.unique(folds)
    fold_index = {fold: np.nonzero(folds == fold)[0] for fold in unique_folds}

    def evaluate(model: str, rows: np.ndarray) -> dict[str, float]:
        y_pred, y_prob = predictions[model]
        return point_metrics(y_true[rows], y_pred[rows], y_prob[rows], metrics)

    def summarise(model: str, per_fold_rows: dict) -> tuple[dict, dict]:
        per_fold = {
            fold: evaluate(model, rows) for fold, rows in per_fold_rows.items()
        }
        macro = {
            name: float(np.mean([per_fold[fold][name] for fold in unique_folds]))
            for name in metrics
        }
        pooled_rows = np.concatenate([per_fold_rows[fold] for fold in unique_folds])
        pooled = evaluate(model, pooled_rows)
        return macro, pooled

    observed_first_macro, observed_first_pooled = summarise(first_model, fold_index)
    observed_second_macro, observed_second_pooled = summarise(second_model, fold_index)

    rng = np.random.default_rng(seed)
    replicates = {
        "macro": {name: np.empty(n_resamples) for name in metrics},
        "pooled": {name: np.empty(n_resamples) for name in metrics},
    }
    for replicate in range(n_resamples):
        resampled_rows = {
            fold: rng.choice(rows, size=rows.size, replace=True)
            for fold, rows in fold_index.items()
        }
        first_macro, first_pooled = summarise(first_model, resampled_rows)
        second_macro, second_pooled = summarise(second_model, resampled_rows)
        for name in metrics:
            replicates["macro"][name][replicate] = first_macro[name] - second_macro[name]
            replicates["pooled"][name][replicate] = (
                first_pooled[name] - second_pooled[name]
            )

    observed = {
        "macro": {
            name: observed_first_macro[name] - observed_second_macro[name]
            for name in metrics
        },
        "pooled": {
            name: observed_first_pooled[name] - observed_second_pooled[name]
            for name in metrics
        },
    }

    results: dict[str, dict] = {
        "comparison": f"{first_model} minus {second_model}",
        "n_rows": int(y_true.size),
        "n_folds": int(unique_folds.size),
        "n_resamples": int(n_resamples),
        "confidence": confidence,
        "seed": seed,
        "resampling": "with replacement within each fold; identical rows for both models",
        "aggregations": {},
    }
    for aggregation in ("macro", "pooled"):
        entries: dict[str, dict] = {}
        for name in metrics:
            finite = replicates[aggregation][name]
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                entries[name] = None
                continue
            lower, upper = _percentile_interval(finite, confidence)
            entries[name] = {
                "observed_difference": float(observed[aggregation][name]),
                "percentile_ci_lower": lower,
                "percentile_ci_upper": upper,
                "bootstrap_se": float(finite.std(ddof=1)),
                "interval_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                "n_degenerate_resamples": int(n_resamples - finite.size),
            }
        results["aggregations"][aggregation] = {
            f"{first_model}": (
                observed_first_macro if aggregation == "macro" else observed_first_pooled
            ),
            f"{second_model}": (
                observed_second_macro
                if aggregation == "macro"
                else observed_second_pooled
            ),
            "difference": entries,
        }
    return results
