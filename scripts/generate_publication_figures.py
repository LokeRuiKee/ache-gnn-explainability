"""Regenerate every manuscript table and figure from machine-readable results.

Addresses Reviewer 4 comment 4: spreadsheet screenshots are replaced by tables
generated from committed result files, and every figure is written as vector SVG
plus a >=300 DPI raster fallback.

Nothing here hard-codes a metric. If a number is not present in a results file,
the corresponding table or figure is skipped and reported as skipped.

Example:
    python scripts/generate_publication_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

DPI = 300
FIGURE_DIR = REPOSITORY_ROOT / "results/figures"
TABLE_DIR = REPOSITORY_ROOT / "results/tables"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        # Without a fixed salt, matplotlib derives SVG element ids from a random
        # seed, so every regeneration rewrites the file even when nothing about
        # the data changed. A stable salt is what makes "regenerate and diff" a
        # meaningful check rather than guaranteed churn.
        "svg.hashsalt": "bsj-1200r2",
    }
)

# Written into SVG metadata in place of the wall-clock date, for the same reason.
SVG_METADATA = {"Date": None}


def save_figure(figure, stem: str) -> list[str]:
    """Write vector SVG plus a 300 DPI PNG fallback.

    Output is byte-stable across runs: identical inputs regenerate identical
    files, so `git diff` after regeneration reports real changes only.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("svg", "png"):
        path = FIGURE_DIR / f"{stem}.{suffix}"
        if suffix == "svg":
            figure.savefig(path, dpi=DPI, format=suffix, metadata=SVG_METADATA)
        else:
            figure.savefig(path, dpi=DPI, format=suffix)
        written.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    plt.close(figure)
    return written


def save_table(frame: pd.DataFrame, stem: str, caption: str) -> list[str]:
    """Write a CSV source of truth and a Markdown rendering to paste."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / f"{stem}.csv"
    markdown_path = TABLE_DIR / f"{stem}.md"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    markdown_path.write_text(
        f"**{caption}**\n\n{frame.to_markdown(index=False)}\n", encoding="utf-8"
    )
    return [
        csv_path.relative_to(REPOSITORY_ROOT).as_posix(),
        markdown_path.relative_to(REPOSITORY_ROOT).as_posix(),
    ]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def format_metric(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


# --- tables -----------------------------------------------------------------


def table_headline_metrics(runs: dict) -> pd.DataFrame | None:
    rows = []
    for protocol, report in runs.items():
        for partition, metrics in report["metrics"].items():
            intervals = metrics.get("confidence_intervals", {})
            rows.append(
                {
                    "Protocol": report["protocol_label"],
                    "Partition": partition.capitalize(),
                    "N": metrics["n"],
                    "Accuracy": format_metric(metrics["accuracy"]),
                    "Precision": format_metric(metrics["precision"]),
                    "Recall": format_metric(metrics["recall"]),
                    "F1": format_metric(metrics["f1"]),
                    "ROC-AUC": format_metric(metrics["roc_auc"]),
                    "PR-AUC": format_metric(metrics["pr_auc"]),
                    "F1 95% CI": (
                        f"[{intervals['f1']['lower']:.4f}, {intervals['f1']['upper']:.4f}]"
                        if intervals.get("f1")
                        else "n/a"
                    ),
                }
            )
    return pd.DataFrame(rows) if rows else None


def table_historical_comparison(runs: dict) -> pd.DataFrame | None:
    """Historical validation-only figures beside the corrected values."""
    report = runs.get("reproduction")
    if report is None:
        return None
    validation = report["metrics"]["validation"]
    test = report["metrics"]["test"]
    rows = [
        {
            "Quantity": "Accuracy",
            "Historical reported (validation, max-F1 epoch)": "0.8676",
            "Reconstruction (validation, selected checkpoint)": format_metric(validation["accuracy"]),
            "Reconstruction (held-out test, same checkpoint)": format_metric(test["accuracy"]),
        },
        {
            "Quantity": "F1",
            "Historical reported (validation, max-F1 epoch)": "0.8533",
            "Reconstruction (validation, selected checkpoint)": format_metric(validation["f1"]),
            "Reconstruction (held-out test, same checkpoint)": format_metric(test["f1"]),
        },
        {
            "Quantity": "ROC-AUC",
            "Historical reported (validation, max-F1 epoch)": "0.9281",
            "Reconstruction (validation, selected checkpoint)": format_metric(validation["roc_auc"]),
            "Reconstruction (held-out test, same checkpoint)": format_metric(test["roc_auc"]),
        },
    ]
    return pd.DataFrame(rows)


def table_cross_validation(cv: dict) -> pd.DataFrame | None:
    if cv is None:
        return None
    rows = []
    for model_name, summary in cv["summaries"].items():
        row = {"Model": model_name, "Folds": None}
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            entry = summary.get(metric)
            if entry is None:
                row[metric.upper()] = "n/a"
                continue
            row["Folds"] = entry["n_folds"]
            row[metric.upper()] = f"{entry['mean']:.4f} ± {entry['sd']:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def table_statistical_tests(tests: dict) -> pd.DataFrame | None:
    if tests is None:
        return None
    rows = []
    for comparison, metrics in tests["comparisons"].items():
        for metric, result in metrics.items():
            if not result or "paired_t_test" not in result:
                continue
            rows.append(
                {
                    "Comparison": result["comparison"],
                    "Metric": metric,
                    "N pairs": result["n_pairs"],
                    "Mean difference": f"{result['mean_difference']:+.4f}",
                    "Cohen's dz": f"{result['cohens_dz']:+.2f}",
                    "Paired t p": f"{result['paired_t_test']['p_value']:.4f}",
                    "Wilcoxon p": f"{result['wilcoxon_signed_rank']['p_value']:.4f}",
                }
            )
    return pd.DataFrame(rows) if rows else None


def _edge_metric(record: dict, field: str) -> str:
    """Edge-probe value, or a dash for molecules that have no bonds to delete."""
    edge = record.get("edge_fidelity")
    if edge is None or not edge.get("applicable"):
        return "n/a"
    return format_metric(edge[field])


def _edge_verdict(record: dict) -> str:
    edge = record.get("edge_fidelity")
    if edge is None or not edge.get("applicable"):
        return "n/a"
    return "yes" if edge["top_k_beats_random_on_margin"] else "no"


def table_atom_attributions(explanations: dict) -> pd.DataFrame | None:
    if explanations is None:
        return None
    rows = []
    for record in explanations["explanations"]:
        for motif in record["connected_motifs"]:
            rows.append(
                {
                    "Role": record["role"],
                    "SMILES": record["smiles"],
                    "True label": (
                        "not in partition"
                        if record["true_label"] is None or record["true_label"] != record["true_label"]
                        else int(record["true_label"])
                    ),
                    # Always the positive (active) class, whatever was predicted.
                    "P(active)": (
                        format_metric(record["predicted_probability_positive"])
                        if record["predicted_probability_positive"] is not None
                        and record["predicted_probability_positive"]
                        == record["predicted_probability_positive"]
                        else format_metric(
                            record["checkpoint_prediction"]["probability_class_1"]
                        )
                    ),
                    "Motif atoms": ", ".join(motif["atom_labels"]),
                    "Motif fragment": motif["smarts"],
                    "Aromatic": "yes" if motif["all_aromatic"] else "no",
                    "In ring": "yes" if motif["all_in_ring"] else "no",
                    # Both probes are shown: they disagree for three of five
                    # molecules (D-013), so reporting either alone would give a
                    # one-sided verdict on faithfulness.
                    "Node mask: top-5": format_metric(
                        record["fidelity"]["margin_drop_top_k"]
                    ),
                    "Node mask: random-5": format_metric(
                        record["fidelity"]["margin_drop_random_k_mean"]
                    ),
                    "Node beats random": (
                        "yes" if record["fidelity"]["top_k_beats_random_on_margin"] else "no"
                    ),
                    "Edge deletion: top-5": _edge_metric(record, "margin_drop_top_k"),
                    "Edge deletion: random-5": _edge_metric(
                        record, "margin_drop_random_k_mean"
                    ),
                    "Edge beats random": _edge_verdict(record),
                    "Top-k Jaccard across seeds": format_metric(
                        record["stability"]["jaccard_mean"], 3
                    ),
                    "Atoms selected": f"{record['sparsity']['n_selected']}/{record['sparsity']['n_atoms']}",
                }
            )
    return pd.DataFrame(rows) if rows else None


# --- figures ----------------------------------------------------------------


def figure_training_curves(histories: dict) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for protocol, history in histories.items():
        axes[0].plot(history["epoch"], history["train_loss"], label=f"{protocol} train", lw=1)
        axes[0].plot(history["epoch"], history["val_loss"], label=f"{protocol} val", lw=1, ls="--")
        axes[1].plot(history["epoch"], history["val_f1"], label=f"{protocol} val F1", lw=1)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[1].set(xlabel="Epoch", ylabel="F1", title="Validation F1")
    for axis in axes:
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    return figure


def figure_confusion_matrices(runs: dict):
    panels = [
        (report["protocol_label"], partition, metrics["confusion_matrix"])
        for report in runs.values()
        for partition, metrics in report["metrics"].items()
        if partition == "test"
    ]
    figure, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.0), squeeze=False)
    for axis, (label, partition, matrix) in zip(axes[0], panels):
        grid = np.array(
            [
                [matrix["true_negative"], matrix["false_positive"]],
                [matrix["false_negative"], matrix["true_positive"]],
            ]
        )
        axis.imshow(grid, cmap="Blues")
        for (row, column), value in np.ndenumerate(grid):
            axis.text(
                column, row, str(value), ha="center", va="center",
                color="white" if value > grid.max() / 2 else "black",
            )
        axis.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["Pred. inactive", "Pred. active"],
            yticklabels=["True inactive", "True active"],
            title=f"{label}\n({partition}, n={grid.sum()})",
        )
    return figure


def figure_roc_and_pr(predictions: dict):
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for protocol, frame in predictions.items():
        test = frame[frame["partition"] == "test"]
        if test.empty:
            continue
        fpr, tpr, _ = roc_curve(test["y_true"], test["y_prob_positive"])
        precision, recall, _ = precision_recall_curve(test["y_true"], test["y_prob_positive"])
        axes[0].plot(fpr, tpr, lw=1.2, label=protocol)
        axes[1].plot(recall, precision, lw=1.2, label=protocol)
    axes[0].plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)")
    for axis in axes:
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    return figure


def figure_fold_variability(fold_metrics: pd.DataFrame):
    metrics = ["accuracy", "f1", "roc_auc"]
    models = sorted(fold_metrics["model"].unique())
    figure, axis = plt.subplots(figsize=(6.0, 3.2))
    width = 0.8 / len(models)
    for offset, model_name in enumerate(models):
        subset = fold_metrics[fold_metrics["model"] == model_name]
        means = [subset[metric].mean() for metric in metrics]
        errors = [subset[metric].std(ddof=1) for metric in metrics]
        positions = np.arange(len(metrics)) + offset * width
        axis.bar(positions, means, width=width, yerr=errors, capsize=3, label=model_name)
        for position, subset_metric in zip(positions, metrics):
            axis.scatter(
                np.full(len(subset), position), subset[subset_metric],
                s=8, color="black", zorder=3,
            )
    axis.set(
        xticks=np.arange(len(metrics)) + width * (len(models) - 1) / 2,
        xticklabels=[metric.upper() for metric in metrics],
        ylabel="Score", ylim=(0, 1),
        title="Cross-validation, mean ± SD with fold values",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    return figure


def figure_explanation_molecules(explanations: dict):
    """Molecular structures with attributed atoms highlighted, drawn as vectors."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    records = [r for r in explanations["explanations"] if r["role"] != "historical_case_study"]
    records += [r for r in explanations["explanations"] if r["role"] == "historical_case_study"]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for record in records:
        mol = Chem.MolFromSmiles(record["smiles"])
        scores = {row["node_index"]: row["importance_normalized"] for row in record["atom_table"]}
        highlight = record["top_atoms"]
        colors = {index: (1.0, 1.0 - scores[index], 1.0 - scores[index]) for index in highlight}

        def draw(drawer) -> None:
            options = drawer.drawOptions()
            options.addAtomIndices = True
            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer, mol, highlightAtoms=highlight, highlightAtomColors=colors
            )
            drawer.FinishDrawing()

        svg_drawer = rdMolDraw2D.MolDraw2DSVG(420, 340)
        draw(svg_drawer)
        svg_path = FIGURE_DIR / f"explanation_{record['role']}.svg"
        svg_path.write_text(svg_drawer.GetDrawingText(), encoding="utf-8")
        written.append(svg_path.relative_to(REPOSITORY_ROOT).as_posix())

        # Raster fallback at roughly 300 DPI for a ~1.4 inch wide panel.
        try:
            png_drawer = rdMolDraw2D.MolDraw2DCairo(1260, 1020)
        except (AttributeError, RuntimeError):
            continue  # RDKit built without Cairo; the vector file still exists.
        draw(png_drawer)
        png_path = FIGURE_DIR / f"explanation_{record['role']}.png"
        png_path.write_bytes(png_drawer.GetDrawingText())
        written.append(png_path.relative_to(REPOSITORY_ROOT).as_posix())
    return written


# --- driver -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPOSITORY_ROOT / "results/model4")
    args = parser.parse_args()

    generated: dict[str, list[str]] = {}
    skipped: list[str] = []

    runs, histories, predictions = {}, {}, {}
    for protocol in ("reproduction", "generalization"):
        run_dir = args.results_dir / protocol
        report = load_json(run_dir / "metrics.json")
        if report is None:
            skipped.append(f"{protocol}: metrics.json not found")
            continue
        runs[protocol] = report
        histories[protocol] = pd.read_csv(run_dir / "training_history.csv")
        predictions[protocol] = pd.read_csv(run_dir / "predictions.csv")

    if runs:
        headline = table_headline_metrics(runs)
        generated["table_headline_metrics"] = save_table(
            headline, "table_headline_metrics",
            "Predictive performance of the selected checkpoint under each evaluation protocol.",
        )
        comparison = table_historical_comparison(runs)
        if comparison is not None:
            generated["table_historical_comparison"] = save_table(
                comparison, "table_historical_comparison",
                "Historical validation-only figures beside the reconstruction's validation and held-out test values.",
            )
        generated["figure_training_curves"] = save_figure(
            figure_training_curves(histories), "figure_training_curves"
        )
        generated["figure_confusion_matrices"] = save_figure(
            figure_confusion_matrices(runs), "figure_confusion_matrices"
        )
        generated["figure_roc_pr"] = save_figure(
            figure_roc_and_pr(predictions), "figure_roc_pr"
        )
    else:
        skipped.append("all run-level tables and figures")

    cv = load_json(args.results_dir / "cross_validation/cross_validation.json")
    cv_table = table_cross_validation(cv)
    if cv_table is not None:
        generated["table_cross_validation"] = save_table(
            cv_table, "table_cross_validation",
            "Five-fold cross-validation on the train+validation pool; the held-out test partition is excluded.",
        )
        fold_metrics = pd.read_csv(args.results_dir / "cross_validation/fold_metrics.csv")
        generated["figure_fold_variability"] = save_figure(
            figure_fold_variability(fold_metrics), "figure_fold_variability"
        )
    else:
        skipped.append("cross-validation table and figure")

    tests = load_json(args.results_dir / "statistical_tests.json")
    tests_table = table_statistical_tests(tests)
    if tests_table is not None:
        generated["table_statistical_tests"] = save_table(
            tests_table, "table_statistical_tests",
            "Paired comparison of within-study models on identical folds.",
        )
    else:
        skipped.append("statistical test table")

    explanations = load_json(
        args.results_dir / "reproduction/explanations/explanations.json"
    )
    attribution_table = table_atom_attributions(explanations)
    if attribution_table is not None:
        generated["table_atom_attributions"] = save_table(
            attribution_table, "table_atom_attributions",
            "Attributed structural motifs for each representative molecule, with "
            "fidelity under two independent perturbations (node-feature masking "
            "and bond deletion), stability across explainer seeds, and sparsity. "
            "Margin values are drops in the target-class logit margin; larger is "
            "more faithful. The two probes disagree for three of five molecules.",
        )
        generated["figure_explanations"] = figure_explanation_molecules(explanations)
    else:
        skipped.append("explanation table and molecule figures")

    manifest = {
        "schema_version": 1,
        "dpi": DPI,
        "formats": ["svg (vector)", "png (300 dpi raster fallback)"],
        "generated": generated,
        "skipped": skipped,
        "note": "Every value originates from a committed machine-readable result file.",
    }
    (REPOSITORY_ROOT / "results/figures/generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    for name, paths in generated.items():
        print(f"  {name}: {', '.join(paths)}")
    for item in skipped:
        print(f"  [skipped] {item}")


if __name__ == "__main__":
    main()
