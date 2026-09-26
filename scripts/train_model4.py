"""Train the Model 4 reconstruction under one immutable split protocol.

Saves the selected checkpoint, the full training history, per-molecule
predictions, and machine-readable metrics. Every reported metric is computed
from the single checkpoint selected by the declared validation criterion, and
the held-out test partition is scored from that same checkpoint.

Example:
    python scripts/train_model4.py --protocol reproduction --epochs 700
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.model4.data import build_partitions, load_manifest, load_workbook
from src.model4.metrics import bootstrap_confidence_intervals, classification_metrics
from src.model4.training import (
    TrainingConfig,
    load_selected_model,
    predict,
    set_global_seed,
)

PROTOCOLS = {
    "reproduction": {
        "manifest_key": "reproduction",
        "label": "Historical random split (reproduction)",
        "comparable_with_history": True,
    },
    "generalization": {
        "manifest_key": "generalization",
        "label": "Prospective scaffold split (generalization)",
        "comparable_with_history": False,
    },
}


def _display_path(path: Path) -> str:
    """Repository-relative when possible, absolute otherwise."""
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def environment_record() -> dict:
    import numpy
    import rdkit
    import sklearn
    import torch_geometric

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "rdkit": rdkit.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/model4_pyg.yaml")
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--epochs", type=int, default=None, help="Overrides the config epoch budget.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help=(
            "Skip training and re-score the checkpoint already saved in the output "
            "directory. Metrics are recomputed from the same frozen weights, so the "
            "reported values are unchanged; only the derived files are rewritten."
        ),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    protocol = PROTOCOLS[args.protocol]

    seed = args.seed if args.seed is not None else config["reproducibility"]["seed"]
    epochs = args.epochs if args.epochs is not None else config["training"]["num_epochs"]
    output_dir = args.output_dir or (REPOSITORY_ROOT / "results/model4" / args.protocol)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config["reproducibility"].get("device", "cpu"))
    set_global_seed(seed)

    # --- data -------------------------------------------------------------
    data_config = config["data"]
    workbook = load_workbook(
        REPOSITORY_ROOT / data_config["workbook"],
        sheet=data_config["sheet"],
        expected_sha256=data_config["expected_sha256"],
    )
    manifest_path = (
        REPOSITORY_ROOT / config["evaluation"]["splits"][protocol["manifest_key"]]
    )
    manifest = load_manifest(manifest_path, expected_rows=len(workbook))
    manifest_display = _display_path(manifest_path)
    graphs, reports = build_partitions(workbook, manifest, data_config["smiles_column"])

    for partition, report in reports.items():
        if report.failed_smiles:
            raise RuntimeError(
                f"{len(report.failed_smiles)} SMILES in the {partition} partition "
                f"could not be parsed: {report.failed_smiles[:3]}"
            )
    print(
        f"Protocol: {protocol['label']}\n"
        f"Manifest: {manifest_display}\n"
        + "\n".join(
            f"  {name:11s} {len(items):5d} graphs" for name, items in graphs.items()
        )
    )

    # --- train ------------------------------------------------------------
    model_config = config["model"]
    training_config = TrainingConfig(
        num_epochs=epochs,
        batch_size=config["training"]["batch_size"],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"] or 0.0,
        monitor_metric=config["training"]["checkpoint_selection"]["monitor"],
        seed=seed,
        model_kwargs=dict(
            node_feat_dim=model_config["node_feat_dim"],
            hidden_dims=tuple(model_config["hidden_dims"]),
            dense_dim=model_config["dense_dim"],
            dropout_rate=model_config["dropout_rate"],
            n_classes=model_config["n_classes"],
            batch_norm=model_config["batch_norm"],
        ),
    )

    from src.model4.training import train_model

    started = time.time()
    if args.predict_only:
        previous = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        result = {
            "state_dict": torch.load(output_dir / "checkpoint.pt", map_location=device),
            "selected_epoch": previous["selected_epoch"],
            "selection_metric": previous["selection_rule"]["metric"],
            "selection_split": previous["selection_rule"]["split"],
            "selection_value": previous["selection_rule"]["value"],
            "history": pd.read_csv(output_dir / "training_history.csv").to_dict("records"),
        }
        elapsed = previous.get("training_minutes", 0.0) * 60
        print(
            f"Re-scoring the saved checkpoint from epoch {result['selected_epoch']} "
            "without retraining."
        )
    else:
        result = train_model(
            graphs["train"],
            graphs["validation"],
            training_config,
            device=device,
            progress_every=args.progress_every,
        )
        elapsed = time.time() - started
        print(
            f"Selected epoch {result['selected_epoch']} by maximum validation "
            f"{result['selection_metric']} = {result['selection_value']:.4f} "
            f"({elapsed / 60:.1f} min)"
        )

    # --- evaluate the ONE selected checkpoint -----------------------------
    model = load_selected_model(result["state_dict"], training_config, device)

    evaluations: dict[str, dict] = {}
    prediction_frames = []
    for partition in ("validation", "test"):
        loader = DataLoader(graphs[partition], batch_size=training_config.batch_size)
        outputs = predict(model, loader, device)
        metrics = classification_metrics(
            outputs["y_true"], outputs["y_pred"], outputs["y_prob"]
        )
        metrics["confidence_intervals"] = bootstrap_confidence_intervals(
            outputs["y_true"], outputs["y_pred"], outputs["y_prob"], seed=seed
        )
        evaluations[partition] = metrics
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_index": outputs["row_index"],
                    "partition": partition,
                    "y_true": outputs["y_true"],
                    "y_pred": outputs["y_pred"],
                    "y_prob_positive": outputs["y_prob"],
                }
            )
        )

    # --- persist ----------------------------------------------------------
    torch.save(result["state_dict"], output_dir / "checkpoint.pt")

    pd.DataFrame(result["history"]).to_csv(
        output_dir / "training_history.csv", index=False, lineterminator="\n"
    )
    predictions = pd.concat(prediction_frames).sort_values(["partition", "row_index"])
    predictions.to_csv(output_dir / "predictions.csv", index=False, lineterminator="\n")

    report = {
        "schema_version": 1,
        "protocol": args.protocol,
        "protocol_label": protocol["label"],
        "comparable_with_historical_value": protocol["comparable_with_history"],
        "config_id": config["config_id"],
        "manifest": manifest_display,
        "source_sha256": data_config["expected_sha256"],
        "seed": seed,
        "num_epochs": epochs,
        "selected_epoch": result["selected_epoch"],
        "selection_rule": {
            "metric": result["selection_metric"],
            "split": result["selection_split"],
            "mode": "max",
            "value": result["selection_value"],
        },
        "partition_sizes": {name: len(items) for name, items in graphs.items()},
        "metrics": evaluations,
        "environment": environment_record(),
        "training_minutes": round(elapsed / 60, 2),
        "notes": [
            "All reported metrics come from the single checkpoint selected by the declared validation criterion.",
            "Confidence intervals are percentile bootstrap over evaluated samples, not cross-validation variability.",
        ],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    for partition, metrics in evaluations.items():
        print(
            f"  {partition:11s} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f} "
            f"(n={metrics['n']})"
        )
    print(f"Wrote artifacts to {_display_path(output_dir)}")


if __name__ == "__main__":
    main()
