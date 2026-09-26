"""Repeat the reproduction protocol across seeds on one fixed split.

This isolates **initialisation and training-order variability**. The split is
held constant (the committed reproduction manifest), so nothing here mixes with
the fold variability reported by ``cross_validate_model4.py``. The two
quantities answer different questions and are reported separately, as required
by ``TODO_MASTER_REVISION.md`` P1.11.

Example:
    python scripts/seed_sensitivity.py --seeds 42 43 44
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
from src.model4.metrics import METRIC_NAMES, classification_metrics, summarize_folds
from src.model4.training import (
    TrainingConfig,
    load_selected_model,
    predict,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/model4_pyg.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_ROOT / "results/model4/seed_sensitivity"
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    epochs = args.epochs if args.epochs is not None else config["training"]["num_epochs"]
    device = torch.device(config["reproducibility"].get("device", "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_config = config["data"]
    workbook = load_workbook(
        REPOSITORY_ROOT / data_config["workbook"],
        sheet=data_config["sheet"],
        expected_sha256=data_config["expected_sha256"],
    )
    manifest = load_manifest(
        REPOSITORY_ROOT / config["evaluation"]["splits"]["reproduction"],
        expected_rows=len(workbook),
    )
    graphs, _ = build_partitions(workbook, manifest, data_config["smiles_column"])

    model_config = config["model"]
    model_kwargs = dict(
        node_feat_dim=model_config["node_feat_dim"],
        hidden_dims=tuple(model_config["hidden_dims"]),
        dense_dim=model_config["dense_dim"],
        dropout_rate=model_config["dropout_rate"],
        n_classes=model_config["n_classes"],
        batch_norm=model_config["batch_norm"],
    )

    records = []
    started = time.time()
    for seed in args.seeds:
        training_config = TrainingConfig(
            num_epochs=epochs,
            batch_size=config["training"]["batch_size"],
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"] or 0.0,
            monitor_metric=config["training"]["checkpoint_selection"]["monitor"],
            seed=seed,
            model_kwargs=model_kwargs,
        )
        result = train_model(
            graphs["train"], graphs["validation"], training_config,
            device=device, progress_every=0,
        )
        model = load_selected_model(result["state_dict"], training_config, device)
        outputs = predict(
            model, DataLoader(graphs["test"], batch_size=training_config.batch_size), device
        )
        metrics = classification_metrics(
            outputs["y_true"], outputs["y_pred"], outputs["y_prob"]
        )
        records.append(
            {
                "seed": seed,
                "selected_epoch": result["selected_epoch"],
                "selection_value": result["selection_value"],
                **{name: metrics[name] for name in METRIC_NAMES},
            }
        )
        print(
            f"[seed {seed}] epoch={result['selected_epoch']:4d} "
            f"test acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "seed_metrics.csv", index=False, lineterminator="\n")
    summary = summarize_folds(records)

    report = {
        "schema_version": 1,
        "variability_source": "model initialisation and training order (seed)",
        "held_constant": "the committed reproduction split manifest",
        "distinct_from": (
            "results/model4/cross_validation/cross_validation.json, which varies the "
            "data partition instead. Seed variability and fold variability must not be "
            "pooled or presented as one figure."
        ),
        "seeds": list(args.seeds),
        "num_epochs": epochs,
        "evaluated_partition": "test",
        "summary": summary,
        "runtime_minutes": round((time.time() - started) / 60, 2),
    }
    (args.output_dir / "seed_sensitivity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    for metric in ("accuracy", "f1", "roc_auc"):
        entry = summary[metric]
        print(f"  {metric:9s} {entry['mean']:.4f} +/- {entry['sd']:.4f} (n={entry['n_folds']} seeds)")


if __name__ == "__main__":
    main()
