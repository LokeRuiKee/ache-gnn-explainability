"""Bootstrap intervals and the vectorised metrics they depend on.

The fast metric implementations exist only for speed, so the first duty of
these tests is to prove they are numerically identical to the scikit-learn
metrics used everywhere else in the project — including on tied scores, where
average precision and ROC-AUC are easy to get subtly wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model4.bootstrap import (
    METRIC_NAMES,
    bootstrap_metric_intervals,
    locate_reference_value,
    paired_cluster_bootstrap,
    point_metrics,
)


def _random_case(rng: np.random.Generator, n: int, tied: bool):
    y_true = rng.integers(0, 2, size=n)
    if tied:
        # Coarse probabilities force many exact ties.
        y_prob = rng.integers(0, 5, size=n).astype(float) / 4.0
    else:
        y_prob = rng.random(n)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("n", [20, 137, 407])
def test_fast_metrics_match_sklearn(n, tied):
    rng = np.random.default_rng(20260912 + n + int(tied))
    for _ in range(25):
        y_true, y_pred, y_prob = _random_case(rng, n, tied)
        if len(np.unique(y_true)) < 2:
            continue
        fast = point_metrics(y_true, y_pred, y_prob)
        assert fast["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
        assert fast["precision"] == pytest.approx(
            precision_score(y_true, y_pred, zero_division=0)
        )
        assert fast["recall"] == pytest.approx(
            recall_score(y_true, y_pred, zero_division=0)
        )
        assert fast["f1"] == pytest.approx(f1_score(y_true, y_pred, zero_division=0))
        assert fast["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_prob))
        assert fast["pr_auc"] == pytest.approx(average_precision_score(y_true, y_prob))


def test_roc_auc_is_one_half_when_every_score_is_tied():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.full(4, 0.5)
    y_pred = np.ones(4, dtype=int)
    assert point_metrics(y_true, y_pred, y_prob)["roc_auc"] == pytest.approx(0.5)


def test_single_class_auc_is_nan_not_an_exception():
    y_true = np.ones(10, dtype=int)
    y_prob = np.linspace(0.1, 0.9, 10)
    y_pred = np.ones(10, dtype=int)
    computed = point_metrics(y_true, y_pred, y_prob)
    assert np.isnan(computed["roc_auc"])
    assert np.isnan(computed["pr_auc"])
    assert computed["accuracy"] == 1.0


def test_interval_brackets_the_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(7)
    y_true, y_pred, y_prob = _random_case(rng, 300, tied=False)
    first = bootstrap_metric_intervals(
        y_true, y_pred, y_prob, n_resamples=400, seed=11, compute_bca=False
    )
    second = bootstrap_metric_intervals(
        y_true, y_pred, y_prob, n_resamples=400, seed=11, compute_bca=False
    )
    assert first == second
    for name in METRIC_NAMES:
        entry = first[name]
        assert entry["percentile_ci_lower"] <= entry["point_estimate"]
        assert entry["point_estimate"] <= entry["percentile_ci_upper"]
        assert entry["n_resamples"] == 400


def test_bca_interval_is_reported_or_explicitly_declined():
    rng = np.random.default_rng(3)
    y_true, y_pred, y_prob = _random_case(rng, 150, tied=False)
    intervals = bootstrap_metric_intervals(
        y_true, y_pred, y_prob, n_resamples=500, seed=5, compute_bca=True
    )
    for name in METRIC_NAMES:
        entry = intervals[name]
        has_bca = entry.get("bca_ci_lower") is not None
        # Either a usable BCa interval, or a stated reason it was not computed.
        assert has_bca or "bca_note" in entry


def test_wider_interval_for_smaller_partition():
    rng = np.random.default_rng(19)
    big_true, big_pred, big_prob = _random_case(rng, 2000, tied=False)
    small_true, small_pred, small_prob = big_true[:100], big_pred[:100], big_prob[:100]
    big = bootstrap_metric_intervals(
        big_true, big_pred, big_prob, n_resamples=400, seed=1, compute_bca=False
    )
    small = bootstrap_metric_intervals(
        small_true, small_pred, small_prob, n_resamples=400, seed=1, compute_bca=False
    )
    big_width = big["accuracy"]["percentile_ci_upper"] - big["accuracy"]["percentile_ci_lower"]
    small_width = (
        small["accuracy"]["percentile_ci_upper"] - small["accuracy"]["percentile_ci_lower"]
    )
    assert small_width > big_width


def test_reference_value_position_is_described_not_tested():
    interval = {
        "point_estimate": 0.88,
        "percentile_ci_lower": 0.85,
        "percentile_ci_upper": 0.91,
    }
    assert locate_reference_value(interval, 0.87)["position"] == "within interval"
    assert locate_reference_value(interval, 0.70)["position"] == "below interval"
    assert locate_reference_value(interval, 0.99)["position"] == "above interval"
    # The wording must not promise significance.
    assert "No significance is claimed" in locate_reference_value(interval, 0.87)[
        "interpretation"
    ]


def test_paired_bootstrap_of_a_model_against_itself_is_centred_on_zero():
    rng = np.random.default_rng(23)
    n = 400
    folds = np.repeat([1, 2, 3, 4], n // 4)
    y_true = rng.integers(0, 2, size=n)
    y_prob = rng.random(n)
    y_pred = (y_prob >= 0.5).astype(int)
    result = paired_cluster_bootstrap(
        folds,
        y_true,
        {"a": (y_pred, y_prob), "b": (y_pred, y_prob)},
        "a",
        "b",
        n_resamples=200,
        seed=2,
    )
    for aggregation in ("macro", "pooled"):
        for name in METRIC_NAMES:
            entry = result["aggregations"][aggregation]["difference"][name]
            assert entry["observed_difference"] == pytest.approx(0.0)
            assert entry["percentile_ci_lower"] == pytest.approx(0.0)
            assert entry["percentile_ci_upper"] == pytest.approx(0.0)
            assert entry["interval_excludes_zero"] is False


def test_paired_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(29)
    n = 600
    folds = np.repeat([1, 2, 3], n // 3)
    y_true = rng.integers(0, 2, size=n)
    # ``strong`` is right 95% of the time; ``weak`` is right 55% of the time.
    strong_pred = np.where(rng.random(n) < 0.95, y_true, 1 - y_true)
    weak_pred = np.where(rng.random(n) < 0.55, y_true, 1 - y_true)
    result = paired_cluster_bootstrap(
        folds,
        y_true,
        {
            "strong": (strong_pred, strong_pred.astype(float)),
            "weak": (weak_pred, weak_pred.astype(float)),
        },
        "strong",
        "weak",
        n_resamples=300,
        seed=4,
    )
    accuracy = result["aggregations"]["pooled"]["difference"]["accuracy"]
    assert accuracy["observed_difference"] > 0.3
    assert accuracy["interval_excludes_zero"] is True
    assert accuracy["percentile_ci_lower"] > 0.0


def test_paired_bootstrap_preserves_fold_sizes():
    rng = np.random.default_rng(31)
    n = 90
    folds = np.repeat([1, 2, 3], 30)
    y_true = rng.integers(0, 2, size=n)
    y_prob = rng.random(n)
    y_pred = (y_prob >= 0.5).astype(int)
    result = paired_cluster_bootstrap(
        folds,
        y_true,
        {"a": (y_pred, y_prob), "b": (1 - y_pred, 1.0 - y_prob)},
        "a",
        "b",
        n_resamples=50,
        seed=6,
    )
    assert result["n_rows"] == n
    assert result["n_folds"] == 3
    assert "within each fold" in result["resampling"]
