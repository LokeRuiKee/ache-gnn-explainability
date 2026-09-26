"""Paired statistical comparison of within-study models on identical folds.

Only models rerun inside this study on the same folds are compared. Literature
metrics for SVC, AttentiveFP, and CNN reported in other publications are
excluded by construction: a p-value against a single number copied from another
paper is not meaningful (``TODO_MASTER_REVISION.md`` P1.13).

A power caveat is computed and reported rather than left implicit: with five
paired observations the two-sided Wilcoxon signed-rank test cannot return a
p-value below 0.0625, so it can never reach alpha = 0.05 no matter how large
the effect.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.model4.metrics import METRIC_NAMES

ALPHA = 0.05


def minimum_two_sided_wilcoxon_p(n: int) -> float:
    """Smallest achievable two-sided signed-rank p-value for ``n`` pairs.

    The most extreme outcome is every difference sharing one sign, giving
    ``W- = 0`` with exact two-sided probability ``2 / 2**n``. For n = 5 that
    floor is 0.0625, which already exceeds alpha = 0.05.
    """
    if n < 1:
        return 1.0
    return min(1.0, 2.0 ** (1 - n))


def paired_comparison(
    first: np.ndarray, second: np.ndarray, first_name: str, second_name: str
) -> dict:
    """Paired t-test with Wilcoxon as a non-parametric sensitivity check."""
    differences = first - second
    n = differences.size

    if np.allclose(differences, 0):
        return {
            "n_pairs": int(n),
            "mean_difference": 0.0,
            "note": "Identical fold-level values; no test performed.",
        }

    t_statistic, t_p = stats.ttest_rel(first, second)
    try:
        w_statistic, w_p = stats.wilcoxon(first, second)
    except ValueError as error:  # too few non-zero differences
        w_statistic, w_p = float("nan"), float("nan")
        wilcoxon_note = str(error)
    else:
        wilcoxon_note = None

    sd = differences.std(ddof=1)
    cohens_dz = float(differences.mean() / sd) if sd > 0 else float("nan")
    shapiro_p = float(stats.shapiro(differences).pvalue) if n >= 3 else float("nan")

    return {
        "comparison": f"{first_name} minus {second_name}",
        "n_pairs": int(n),
        "mean_difference": float(differences.mean()),
        "sd_difference": float(sd),
        "effect_direction": (
            f"{first_name} higher" if differences.mean() > 0 else f"{second_name} higher"
        ),
        "cohens_dz": cohens_dz,
        "normality_shapiro_p": shapiro_p,
        "paired_t_test": {
            "statistic": float(t_statistic),
            "p_value": float(t_p),
            "significant_at_alpha": bool(t_p < ALPHA),
        },
        "wilcoxon_signed_rank": {
            "statistic": float(w_statistic),
            "p_value": float(w_p),
            "significant_at_alpha": (not math.isnan(w_p)) and bool(w_p < ALPHA),
            "note": wilcoxon_note,
        },
        "minimum_achievable_wilcoxon_p": minimum_two_sided_wilcoxon_p(n),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-metrics",
        type=Path,
        default=REPOSITORY_ROOT / "results/model4/cross_validation/fold_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "results/model4/statistical_tests.json",
    )
    args = parser.parse_args()

    folds = pd.read_csv(args.fold_metrics)
    models = sorted(folds["model"].unique())
    if len(models) < 2:
        raise SystemExit(
            f"Only one within-study model present ({models}); no valid paired "
            "comparison exists. Literature metrics must not be substituted."
        )

    results: dict[str, dict] = {}
    n_pairs = 0
    for index, first_name in enumerate(models):
        for second_name in models[index + 1 :]:
            first = folds[folds["model"] == first_name].sort_values("fold")
            second = folds[folds["model"] == second_name].sort_values("fold")
            if first["fold"].tolist() != second["fold"].tolist():
                raise SystemExit("Fold identifiers differ; the observations are not paired.")

            per_metric = {}
            for metric in METRIC_NAMES:
                if first[metric].isna().any() or second[metric].isna().any():
                    per_metric[metric] = None
                    continue
                per_metric[metric] = paired_comparison(
                    first[metric].to_numpy(),
                    second[metric].to_numpy(),
                    first_name,
                    second_name,
                )
                n_pairs = per_metric[metric]["n_pairs"]
            results[f"{first_name}__vs__{second_name}"] = per_metric

    n_tests = sum(
        1 for metrics in results.values() for value in metrics.values() if value
    )
    report = {
        "schema_version": 1,
        "alpha": ALPHA,
        "source": args.fold_metrics.resolve().relative_to(REPOSITORY_ROOT).as_posix(),
        "design": {
            "pairing": "Both models were rerun in this study on identical folds, so fold-level scores are paired.",
            "excluded": "Literature metrics (SVC, AttentiveFP, CNN taken from other publications) are excluded; they are not paired observations.",
            "primary_test": "Paired t-test on fold-level differences.",
            "sensitivity_test": "Wilcoxon signed-rank.",
        },
        "power_caveat": {
            "n_pairs": n_pairs,
            "minimum_achievable_two_sided_wilcoxon_p": minimum_two_sided_wilcoxon_p(n_pairs),
            "statement": (
                f"With {n_pairs} paired folds the two-sided Wilcoxon signed-rank test "
                f"cannot fall below {minimum_two_sided_wilcoxon_p(n_pairs):.4f}, so it "
                f"cannot reach alpha={ALPHA} regardless of effect size. A non-significant "
                "Wilcoxon result here is therefore uninformative about the absence of an effect."
            ),
        },
        "multiple_comparisons": {
            "n_tests": n_tests,
            "bonferroni_alpha": ALPHA / n_tests if n_tests else None,
            "statement": (
                "Several metrics are tested on the same folds and are strongly correlated. "
                "Treat individual p-values as descriptive; prefer effect sizes and intervals."
            ),
        },
        "comparisons": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output.resolve().relative_to(REPOSITORY_ROOT).as_posix()}")
    for name, metrics in results.items():
        print(f"\n{name}")
        for metric, value in metrics.items():
            if not value or "paired_t_test" not in value:
                continue
            print(
                f"  {metric:10s} diff={value['mean_difference']:+.4f} "
                f"t_p={value['paired_t_test']['p_value']:.4f} "
                f"w_p={value['wilcoxon_signed_rank']['p_value']:.4f} "
                f"dz={value['cohens_dz']:+.2f}"
            )


if __name__ == "__main__":
    main()
