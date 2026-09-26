"""Rebuild the 1000-epoch overfitting figure from saved notebook output.

Companion to ``plot_historical_cv_loss.py``, which handles the fold-structured
cross-validation log. This one handles a single uninterrupted training run.

Source: ``FYP2/FYP2exp13.1_best hyperparameter1000epoch.ipynb`` cell 16 — the
1000-epoch run reported as Model 4. No checkpoint survives, but the cell's saved
output records train and validation loss for all 1000 epochs, which is enough to
redraw the curve without retraining.

The divergence point is computed from the data rather than asserted: the epoch of
minimum validation loss is where the run stops generalizing and begins fitting
noise, and everything after it is the overfitting region.

Usage:
    python scripts/plot_overfitting_curve.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from plot_historical_cv_loss import EPOCH_LINE, cell_output_text  # noqa: E402

NOTEBOOK = REPOSITORY_ROOT / "FYP2/FYP2exp13.1_best hyperparameter1000epoch.ipynb"
OUTPUT_DIR = REPOSITORY_ROOT / "results/figures"
DPI = 300

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


def parse_run(text: str) -> dict[str, list[float]]:
    """Collect per-epoch train and validation loss from a single training log."""
    curves: dict[str, list[float]] = {"epoch": [], "train": [], "validation": []}
    for line in text.split("\n"):
        match = EPOCH_LINE.search(line)
        if match:
            curves["epoch"].append(int(match.group(1)))
            curves["train"].append(float(match.group(2)))
            curves["validation"].append(float(match.group(3)))
    return curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--cell", type=int, default=16)
    parser.add_argument("--stem", default="figure_overfitting_1000_epochs")
    args = parser.parse_args()

    if not args.notebook.exists():
        sys.exit(f"Notebook not found: {args.notebook}")

    curves = parse_run(cell_output_text(args.notebook, args.cell))
    if not curves["epoch"]:
        sys.exit("No epoch lines were found in that cell's saved output.")

    best = min(range(len(curves["validation"])), key=lambda i: curves["validation"][i])
    best_epoch = curves["epoch"][best]
    final_gap = curves["validation"][-1] - curves["train"][-1]

    print(f"epochs recorded: {len(curves['epoch'])} (to epoch {curves['epoch'][-1]})")
    print(f"minimum validation loss: {curves['validation'][best]:.4f} at epoch {best_epoch}")
    print(f"final train loss {curves['train'][-1]:.4f} | final validation loss {curves['validation'][-1]:.4f}")
    print(f"final generalization gap: {final_gap:+.4f}")

    figure, axis = plt.subplots(figsize=(5.6, 3.4))
    axis.plot(curves["epoch"], curves["train"], label="Training loss", linewidth=1.3)
    axis.plot(curves["epoch"], curves["validation"], label="Validation loss", linewidth=1.3)

    axis.axvspan(best_epoch, curves["epoch"][-1], color="grey", alpha=0.12, zorder=0)
    axis.axvline(best_epoch, linestyle="--", linewidth=1.0, color="grey")
    axis.annotate(
        f"minimum validation loss\nepoch {best_epoch} ({curves['validation'][best]:.4f})",
        xy=(best_epoch, curves["validation"][best]),
        xytext=(0.17, 0.80),
        textcoords="axes fraction",
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", linewidth=0.8, color="grey"),
    )
    # Placed in the empty band between the two curves so it never sits on data.
    axis.annotate(
        "validation loss rises while training loss falls:\nthe model is fitting noise",
        xy=(0.50, 0.26),
        xycoords="axes fraction",
        fontsize=7.5,
        color="0.25",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Training and validation loss over 1000 epochs")
    axis.legend(frameon=False, loc="lower left")
    axis.spines[["top", "right"]].set_visible(False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("svg", "png"):
        path = OUTPUT_DIR / f"{args.stem}.{suffix}"
        if suffix == "svg":
            figure.savefig(path, dpi=DPI, format=suffix, metadata={"Date": None})
        else:
            figure.savefig(path, dpi=DPI, format=suffix)
        written.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    plt.close(figure)
    print("\nWrote " + ", ".join(written))


if __name__ == "__main__":
    main()
