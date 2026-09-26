"""Bootstrap interval estimates replacing the one-sample tests against literature.

Reviewer 5, essential revision 1: the manuscript's "Statistical Comparison with
Baseline Models" section applied one-sample t-tests that treated a performance
figure copied from another publication as a fixed population mean, ran several
uncorrected tests over strongly correlated metrics, and ignored the literature
model's own sampling uncertainty. That design cannot be repaired by a
correction factor, because the quantity it tests against is not an estimate
with a known distribution.

This script produces the replacement the reviewer asked for:

1. Bootstrap confidence intervals, with numerical bounds, for every reported
   metric on each held-out test partition. A literature figure is then simply
   *located* relative to that interval and explicitly labelled as descriptive.
2. A paired bootstrap interval for the graph model minus the ECFP4 support
   vector classifier, resampling molecules within fold so both models always
   see identical rows. With 3,668 pooled holdout molecules this is far better
   powered than the five fold-level observations available to the paired
   t-test, and it reports an interval rather than a p-value.

Nothing here is invented: every number is computed from committed prediction
files, and the literature values are read from ``configs/literature_baselines.json``
where they are flagged as unverified.

Example:

    python scripts/bootstrap_intervals.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.model4.bootstrap import (  # noqa: E402
    METRIC_NAMES,
    bootstrap_metric_intervals,
    locate_reference_value,
    paired_cluster_bootstrap,
)

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


def display_path(path: Path) -> str:
    """Repository-relative when possible, absolute otherwise.

    Output paths are configurable, so they are not guaranteed to sit inside the
    repository; printing must not fail after the artifacts are already written.
    """
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def software_versions() -> dict:
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def partition_intervals(
    predictions_path: Path,
    partition: str,
    n_resamples: int,
    seed: int,
    confidence: float,
) -> dict:
    frame = pd.read_csv(predictions_path)
    subset = frame[frame["partition"] == partition]
    if subset.empty:
        raise SystemExit(f"No rows with partition '{partition}' in {predictions_path}")
    intervals = bootstrap_metric_intervals(
        subset["y_true"].to_numpy(),
        subset["y_pred"].to_numpy(),
        subset["y_prob_positive"].to_numpy(),
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "source": display_path(predictions_path),
        "partition": partition,
        "n": int(len(subset)),
        "n_positive": int((subset["y_true"] == 1).sum()),
        "n_negative": int((subset["y_true"] == 0).sum()),
        "metrics": intervals,
    }


def attach_literature_comparison(protocols: dict, baselines: dict) -> dict:
    """Locate each external figure relative to each protocol's interval."""
    comparisons = []
    for baseline in baselines["baselines"]:
        for protocol_name, protocol in protocols.items():
            for metric, reference in baseline["metrics"].items():
                interval = protocol["metrics"].get(metric)
                if interval is None:
                    continue
                located = locate_reference_value(interval, reference)
                comparisons.append(
                    {
                        "baseline": baseline["name"],
                        "baseline_citation": baseline["reported_by"],
                        "baseline_protocol": baseline["evaluation_protocol"],
                        "this_study_protocol": protocol_name,
                        "metric": metric,
                        "this_study_point_estimate": interval["point_estimate"],
                        "this_study_ci_lower": interval["percentile_ci_lower"],
                        "this_study_ci_upper": interval["percentile_ci_upper"],
                        **located,
                    }
                )
    return {
        "rule": baselines["usage_rule"],
        "baseline_status": baselines["status"],
        "comparisons": comparisons,
    }


def paired_comparison(
    fold_predictions_path: Path,
    first_model: str,
    second_model: str,
    n_resamples: int,
    seed: int,
    confidence: float,
) -> dict:
    frame = pd.read_csv(fold_predictions_path)
    models = sorted(frame["model"].unique())
    if first_model not in models or second_model not in models:
        raise SystemExit(f"Expected both models in {models}")

    first = frame[frame["model"] == first_model].sort_values(["fold", "row_index"])
    second = frame[frame["model"] == second_model].sort_values(["fold", "row_index"])
    if first["row_index"].tolist() != second["row_index"].tolist():
        raise SystemExit(
            "Fold holdout rows differ between models; the comparison is not paired."
        )
    if not (first["y_true"].to_numpy() == second["y_true"].to_numpy()).all():
        raise SystemExit("Labels differ between models for the same rows.")

    result = paired_cluster_bootstrap(
        first["fold"].to_numpy(),
        first["y_true"].to_numpy(),
        {
            first_model: (
                first["y_pred"].to_numpy(),
                first["y_prob_positive"].to_numpy(),
            ),
            second_model: (
                second["y_pred"].to_numpy(),
                second["y_prob_positive"].to_numpy(),
            ),
        },
        first_model,
        second_model,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    result["source"] = display_path(fold_predictions_path)
    return result


def write_interval_table(protocols: dict, csv_path: Path, markdown_path: Path) -> None:
    rows = []
    for protocol_name, protocol in protocols.items():
        for metric in METRIC_NAMES:
            interval = protocol["metrics"].get(metric)
            if interval is None:
                continue
            rows.append(
                {
                    "Protocol": protocol_name,
                    "Partition": protocol["partition"],
                    "N": protocol["n"],
                    "Metric": METRIC_LABELS[metric],
                    "Estimate": round(interval["point_estimate"], 4),
                    "95% CI (percentile)": (
                        f"[{interval['percentile_ci_lower']:.4f}, "
                        f"{interval['percentile_ci_upper']:.4f}]"
                    ),
                    "95% CI (BCa)": (
                        "not computable"
                        if interval.get("bca_ci_lower") is None
                        else f"[{interval['bca_ci_lower']:.4f}, {interval['bca_ci_upper']:.4f}]"
                    ),
                    "Bootstrap SE": round(interval["bootstrap_se"], 4),
                }
            )
    frame = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")


def write_paired_table(paired: dict, csv_path: Path, markdown_path: Path) -> None:
    rows = []
    for aggregation, block in paired["aggregations"].items():
        for metric in METRIC_NAMES:
            entry = block["difference"].get(metric)
            if entry is None:
                continue
            rows.append(
                {
                    "Aggregation": aggregation,
                    "Metric": METRIC_LABELS[metric],
                    "Difference (graph - SVC)": round(entry["observed_difference"], 4),
                    "95% CI": (
                        f"[{entry['percentile_ci_lower']:.4f}, "
                        f"{entry['percentile_ci_upper']:.4f}]"
                    ),
                    "Excludes zero": "yes" if entry["interval_excludes_zero"] else "no",
                }
            )
    frame = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reproduction-predictions",
        type=Path,
        default=REPOSITORY_ROOT / "results/model4/reproduction/predictions.csv",
    )
    parser.add_argument(
        "--generalization-predictions",
        type=Path,
        default=REPOSITORY_ROOT / "results/model4/generalization/predictions.csv",
    )
    parser.add_argument(
        "--fold-predictions",
        type=Path,
        default=REPOSITORY_ROOT / "results/model4/cross_validation/fold_predictions.csv",
    )
    parser.add_argument(
        "--literature-baselines",
        type=Path,
        default=REPOSITORY_ROOT / "configs/literature_baselines.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "results/model4/bootstrap_intervals.json",
    )
    parser.add_argument(
        "--table-prefix",
        type=Path,
        default=REPOSITORY_ROOT / "results/tables",
    )
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--paired-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    protocols = {
        "Historical random split (reproduction)": partition_intervals(
            args.reproduction_predictions,
            "test",
            args.n_resamples,
            args.seed,
            args.confidence,
        ),
        "Prospective scaffold split (generalization)": partition_intervals(
            args.generalization_predictions,
            "test",
            args.n_resamples,
            args.seed,
            args.confidence,
        ),
    }

    baselines = json.loads(args.literature_baselines.read_text(encoding="utf-8"))
    literature = attach_literature_comparison(protocols, baselines)

    paired = paired_comparison(
        args.fold_predictions,
        "model4_pyg",
        "svc_ecfp4",
        args.paired_resamples,
        args.seed,
        args.confidence,
    )

    payload = {
        "schema_version": 1,
        "purpose": (
            "Interval estimates replacing one-sample hypothesis tests against "
            "literature point values (Reviewer 5, essential revision 1)."
        ),
        "method": {
            "single_model_intervals": (
                "Nonparametric bootstrap over the evaluated molecules, "
                f"{args.n_resamples} resamples, percentile and BCa intervals."
            ),
            "paired_comparison": (
                "Cluster bootstrap resampling molecules within fold, "
                f"{args.paired_resamples} resamples, identical rows for both models."
            ),
            "what_the_intervals_do_not_cover": [
                "Uncertainty from the choice of split; see the seed-sensitivity and cross-validation results.",
                "Uncertainty in the underlying activity labels and the pIC50 threshold.",
                "Any uncertainty in the externally reported literature figures, which is unknown.",
            ],
        },
        "confidence": args.confidence,
        "seed": args.seed,
        "software": software_versions(),
        "protocols": protocols,
        "literature_comparison": literature,
        "paired_within_study_comparison": paired,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    write_interval_table(
        protocols,
        args.table_prefix / "table_bootstrap_intervals.csv",
        args.table_prefix / "table_bootstrap_intervals.md",
    )
    write_paired_table(
        paired,
        args.table_prefix / "table_paired_bootstrap.csv",
        args.table_prefix / "table_paired_bootstrap.md",
    )

    print(f"Wrote {display_path(args.output)}")
    for protocol_name, protocol in protocols.items():
        accuracy = protocol["metrics"]["accuracy"]
        print(
            f"  {protocol_name}: accuracy {accuracy['point_estimate']:.4f} "
            f"95% CI [{accuracy['percentile_ci_lower']:.4f}, "
            f"{accuracy['percentile_ci_upper']:.4f}] (n={protocol['n']})"
        )
    pooled = paired["aggregations"]["pooled"]["difference"]["f1"]
    print(
        f"  graph minus SVC, pooled F1: {pooled['observed_difference']:+.4f} "
        f"95% CI [{pooled['percentile_ci_lower']:.4f}, "
        f"{pooled['percentile_ci_upper']:.4f}]"
    )


if __name__ == "__main__":
    main()
