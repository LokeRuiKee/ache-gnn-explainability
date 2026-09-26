"""Five-fold cross-validation of the Model 4 reconstruction, with a paired baseline.

Design notes that matter scientifically:

* The 407-row held-out test partition of the reproduction manifest is **never**
  touched here. Folds are drawn only from the train+validation pool. This is the
  correction for ``docs/discrepancies.md`` D-005, where the historical
  cross-validation ran over the full dataset after a nominal holdout.
* Each fold carves an inner validation set out of its own training rows for
  checkpoint selection, so the fold's reported holdout is never used to choose
  the epoch. This is the correction for D-010.
* The SVC baseline is fitted on exactly the same fold training rows and scored
  on exactly the same fold holdout rows, which is what makes a paired test
  legitimate (P1.13).

Example:
    python scripts/cross_validate_model4.py --folds 5 --epochs 700
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.model4.baselines import fit_predict_svc
from src.model4.data import load_manifest, load_workbook
from src.model4.graphs import build_graphs
from src.model4.metrics import METRIC_NAMES, classification_metrics, summarize_folds
from src.model4.training import (
    TrainingConfig,
    load_selected_model,
    predict,
    set_global_seed,
    train_model,
)

INNER_VALIDATION_FRACTION = 1 / 9  # keeps the 80/10 train:validation ratio per fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/model4_pyg.yaml")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "results/model4/cross_validation")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else config["reproducibility"]["seed"]
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

    pool = manifest[manifest["partition"].isin(["train", "validation"])].sort_values("row_index")
    held_out_test = manifest[manifest["partition"] == "test"]
    pool_rows = pool["row_index"].to_numpy()
    pool_labels = pool["label"].to_numpy()
    pool_smiles = workbook.loc[pool_rows, data_config["smiles_column"]].tolist()

    print(
        f"Cross-validation pool: {len(pool_rows)} molecules "
        f"(train+validation). Held-out test partition of "
        f"{len(held_out_test)} molecules is excluded entirely."
    )

    model_config = config["model"]
    model_kwargs = dict(
        node_feat_dim=model_config["node_feat_dim"],
        hidden_dims=tuple(model_config["hidden_dims"]),
        dense_dim=model_config["dense_dim"],
        dropout_rate=model_config["dropout_rate"],
        n_classes=model_config["n_classes"],
        batch_norm=model_config["batch_norm"],
    )

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
    fold_records: list[dict] = []
    fold_assignments: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    started = time.time()

    for fold_number, (train_positions, holdout_positions) in enumerate(
        splitter.split(pool_rows, pool_labels), start=1
    ):
        # Inner split for checkpoint selection; the fold holdout stays untouched.
        inner_train_positions, inner_val_positions = train_test_split(
            train_positions,
            test_size=INNER_VALIDATION_FRACTION,
            random_state=seed,
            stratify=pool_labels[train_positions],
        )

        def rows_for(positions):
            return (
                pool_rows[positions].tolist(),
                [pool_smiles[position] for position in positions],
                pool_labels[positions].tolist(),
            )

        train_ids, train_smiles, train_labels = rows_for(inner_train_positions)
        val_ids, val_smiles, val_labels = rows_for(inner_val_positions)
        holdout_ids, holdout_smiles, holdout_labels = rows_for(holdout_positions)

        fold_assignments.append(
            pd.DataFrame(
                {
                    "fold": fold_number,
                    "row_index": train_ids + val_ids + holdout_ids,
                    "fold_role": (
                        ["inner_train"] * len(train_ids)
                        + ["inner_validation"] * len(val_ids)
                        + ["fold_holdout"] * len(holdout_ids)
                    ),
                }
            )
        )

        set_global_seed(seed + fold_number)
        train_graphs, _ = build_graphs(train_smiles, train_labels, train_ids)
        val_graphs, _ = build_graphs(val_smiles, val_labels, val_ids)
        holdout_graphs, _ = build_graphs(holdout_smiles, holdout_labels, holdout_ids)

        training_config = TrainingConfig(
            num_epochs=epochs,
            batch_size=config["training"]["batch_size"],
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"] or 0.0,
            monitor_metric=config["training"]["checkpoint_selection"]["monitor"],
            seed=seed + fold_number,
            model_kwargs=model_kwargs,
        )
        result = train_model(
            train_graphs, val_graphs, training_config, device=device, progress_every=0
        )
        model = load_selected_model(result["state_dict"], training_config, device)
        outputs = predict(
            model,
            DataLoader(holdout_graphs, batch_size=training_config.batch_size),
            device,
        )
        gnn_metrics = classification_metrics(
            outputs["y_true"], outputs["y_pred"], outputs["y_prob"]
        )

        record = {
            "fold": fold_number,
            "model": "model4_pyg",
            "selected_epoch": result["selected_epoch"],
            "selection_value": result["selection_value"],
            **{name: gnn_metrics[name] for name in METRIC_NAMES},
            "n": gnn_metrics["n"],
        }
        fold_records.append(record)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold_number,
                    "model": "model4_pyg",
                    "row_index": outputs["row_index"],
                    "y_true": outputs["y_true"],
                    "y_pred": outputs["y_pred"],
                    "y_prob_positive": outputs["y_prob"],
                }
            )
        )
        print(
            f"[fold {fold_number}] model4_pyg epoch={result['selected_epoch']:4d} "
            f"acc={gnn_metrics['accuracy']:.4f} f1={gnn_metrics['f1']:.4f} "
            f"roc_auc={gnn_metrics['roc_auc']:.4f}"
        )

        if not args.skip_baseline:
            # Identical fold training rows (inner train + inner validation, since
            # the SVC needs no epoch selection) and identical holdout rows.
            baseline_smiles = train_smiles + val_smiles
            baseline_labels = np.asarray(train_labels + val_labels)
            svc_pred, svc_prob = fit_predict_svc(
                baseline_smiles, baseline_labels, holdout_smiles, seed=seed
            )
            svc_metrics = classification_metrics(
                np.asarray(holdout_labels), svc_pred, svc_prob
            )
            fold_records.append(
                {
                    "fold": fold_number,
                    "model": "svc_ecfp4",
                    "selected_epoch": None,
                    "selection_value": None,
                    **{name: svc_metrics[name] for name in METRIC_NAMES},
                    "n": svc_metrics["n"],
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "fold": fold_number,
                        "model": "svc_ecfp4",
                        "row_index": holdout_ids,
                        "y_true": holdout_labels,
                        "y_pred": svc_pred,
                        "y_prob_positive": svc_prob,
                    }
                )
            )
            print(
                f"[fold {fold_number}] svc_ecfp4              "
                f"acc={svc_metrics['accuracy']:.4f} f1={svc_metrics['f1']:.4f} "
                f"roc_auc={svc_metrics['roc_auc']:.4f}"
            )

    elapsed = time.time() - started
    folds_frame = pd.DataFrame(fold_records)
    folds_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    pd.concat(fold_assignments).sort_values(["fold", "row_index"]).to_csv(
        args.output_dir / "fold_assignments.csv", index=False, lineterminator="\n"
    )
    pd.concat(prediction_frames).sort_values(["model", "fold", "row_index"]).to_csv(
        args.output_dir / "fold_predictions.csv", index=False, lineterminator="\n"
    )

    summaries = {
        model_name: summarize_folds(group.to_dict("records"))
        for model_name, group in folds_frame.groupby("model")
    }

    report = {
        "schema_version": 1,
        "design": {
            "pool": "train+validation partitions of the reproduction manifest",
            "pool_size": int(len(pool_rows)),
            "excluded_test_partition_size": int(len(held_out_test)),
            "n_folds": args.folds,
            "inner_validation_fraction": INNER_VALIDATION_FRACTION,
            "checkpoint_selection": "maximum inner-validation F1, per fold",
            "corrections": [
                "The held-out test partition is excluded from cross-validation (D-005).",
                "Each fold selects its checkpoint on an inner validation split, never on its reported holdout (D-010).",
                "The baseline uses identical fold training and holdout rows, so fold-level differences are paired.",
            ],
        },
        "seed": seed,
        "num_epochs": epochs,
        "summaries": summaries,
        "runtime_minutes": round(elapsed / 60, 2),
    }
    (args.output_dir / "cross_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nCompleted in {elapsed / 60:.1f} min")
    for model_name, summary in summaries.items():
        line = "  ".join(
            f"{name}={summary[name]['mean']:.4f}+/-{summary[name]['sd']:.4f}"
            for name in ("accuracy", "f1", "roc_auc")
            if summary.get(name)
        )
        print(f"  {model_name:12s} {line}")


if __name__ == "__main__":
    main()
