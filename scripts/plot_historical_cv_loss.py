"""Rebuild the Model 6 cross-validation loss figure from saved notebook output.

The original checkpoints for the historical Model 6 run were not preserved, but
the notebook's *saved cell output* records train and validation loss for every
epoch of every fold. That text is sufficient to redraw the loss curve without
retraining anything, and it keeps the figure traceable to committed evidence.

Source: ``FYP2 demo with best hyper & cross val.ipynb`` cell 18 — the 100-epoch,
early-stopping, five-fold run whose averages appear as Model 6 in the manuscript
(accuracy 0.8223, F1 0.8067, ROC-AUC 0.8982).

Usage:
    python scripts/plot_historical_cv_loss.py --fold last
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPOSITORY_ROOT / "FYP2 demo with best hyper & cross val.ipynb"
OUTPUT_DIR = REPOSITORY_ROOT / "results/figures"
DPI = 300

EPOCH_LINE = re.compile(
    r"\[Epoch (\d+)\] Train Loss: ([0-9.]+) \| Val Loss: ([0-9.]+)"
)
FOLD_LINE = re.compile(r"=+ Fold (\d+) =+")

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        "svg.hashsalt": "bsj-1200r2",
    }
)


def cell_output_text(notebook: Path, cell_index: int) -> str:
    data = json.loads(notebook.read_text(encoding="utf-8"))
    cell = data["cells"][cell_index]
    parts = []
    for output in cell.get("outputs", []):
        text = output.get("text")
        if text:
            parts.append("".join(text))
    return "".join(parts)


def parse_folds(text: str) -> dict[int, dict[str, list[float]]]:
    """Split the log into folds and collect per-epoch train/validation loss."""
    folds: dict[int, dict[str, list[float]]] = {}
    current = None
    for line in text.split("\n"):
        fold = FOLD_LINE.search(line)
        if fold:
            current = int(fold.group(1))
            folds[current] = {"epoch": [], "train": [], "validation": []}
            continue
        match = EPOCH_LINE.search(line)
        if match and current is not None:
            folds[current]["epoch"].append(int(match.group(1)))
            folds[current]["train"].append(float(match.group(2)))
            folds[current]["validation"].append(float(match.group(3)))
    return folds


def plot_fold(fold_number: int, curves: dict[str, list[float]], stem: str) -> list[str]:
    figure, axis = plt.subplots(figsize=(5.0, 3.2))
    axis.plot(curves["epoch"], curves["train"], label="Training loss", linewidth=1.6)
    axis.plot(
        curves["epoch"], curves["validation"], label="Validation loss", linewidth=1.6
    )

    best = min(range(len(curves["validation"])), key=lambda i: curves["validation"][i])
    axis.axvline(curves["epoch"][best], linestyle="--", linewidth=1.0, color="grey")
    axis.annotate(
        f"minimum validation loss\nepoch {curves['epoch'][best]}, {curves['validation'][best]:.4f}",
        xy=(curves["epoch"][best], curves["validation"][best]),
        xytext=(0.42, 0.72),
        textcoords="axes fraction",
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", linewidth=0.8, color="grey"),
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title(
        f"Fold {fold_number} of five-fold cross-validation\n"
        f"100 epochs with early stopping (stopped at epoch {curves['epoch'][-1]})"
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("svg", "png"):
        path = OUTPUT_DIR / f"{stem}.{suffix}"
        if suffix == "svg":
            figure.savefig(path, dpi=DPI, format=suffix, metadata={"Date": None})
        else:
            figure.savefig(path, dpi=DPI, format=suffix)
        written.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    plt.close(figure)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=int, default=18)
    parser.add_argument("--fold", default="last", help='fold number, or "last"')
    parser.add_argument("--stem", default="figure_model6_cv_loss")
    args = parser.parse_args()

    if not NOTEBOOK.exists():
        sys.exit(f"Notebook not found: {NOTEBOOK}")

    folds = parse_folds(cell_output_text(NOTEBOOK, args.cell))
    if not folds:
        sys.exit("No fold/epoch lines were found in that cell's saved output.")

    print(f"Parsed {len(folds)} folds from {NOTEBOOK.name} cell {args.cell}:")
    for number, curves in sorted(folds.items()):
        best = min(curves["validation"])
        print(
            f"  fold {number}: {len(curves['epoch'])} epochs recorded, "
            f"stopped at epoch {curves['epoch'][-1]}, "
            f"minimum validation loss {best:.4f}"
        )

    chosen = max(folds) if args.fold == "last" else int(args.fold)
    written = plot_fold(chosen, folds[chosen], args.stem)
    print(f"\nPlotted fold {chosen} -> " + ", ".join(written))


if __name__ == "__main__":
    main()
