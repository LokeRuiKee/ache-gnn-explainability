"""Run GNNExplainer on the exact frozen checkpoint whose performance is reported.

Representative molecules are chosen by a stated rule, not by inspection, so the
selection cannot be cherry-picked:

* confident true positive  - highest positive-class probability among TPs
* confident true negative  - lowest positive-class probability among TNs
* false positive           - highest positive-class probability among FPs
* false negative           - lowest positive-class probability among FNs

The historical case-study molecule is additionally explained when it is present
in the evaluated partition, so the revision can be compared with the old figure.

Example:
    python scripts/run_gnnexplainer.py --protocol reproduction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml
from rdkit import RDLogger

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.model4.data import load_workbook
from src.model4.explain import (
    ExplainerSettings,
    atom_report,
    connected_motifs,
    edge_fidelity,
    explain_molecule,
    fidelity,
    sparsity,
    stability,
    top_k_atoms,
)
from src.model4.training import TrainingConfig, load_selected_model

RDLogger.DisableLog("rdApp.*")

HISTORICAL_CASE_STUDY = "COc1cccc(CN2CCC(/C=C3\\Cc4cccc(OC)c4C3=O)CC2)c1"

SELECTION_RULES = {
    "confident_true_positive": ("true_positive", "max"),
    "confident_true_negative": ("true_negative", "min"),
    "false_positive": ("false_positive", "max"),
    "false_negative": ("false_negative", "min"),
}


def outcome_of(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "true_positive"
    if y_true == 0 and y_pred == 0:
        return "true_negative"
    if y_true == 0 and y_pred == 1:
        return "false_positive"
    return "false_negative"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/model4_pyg.yaml")
    parser.add_argument("--protocol", choices=("reproduction", "generalization"), default="reproduction")
    parser.add_argument("--partition", default="test")
    parser.add_argument("--top-k", type=int, default=5, help="Atoms treated as the explanation.")
    parser.add_argument("--stability-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = REPOSITORY_ROOT / "results/model4" / args.protocol
    output_dir = run_dir / "explanations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- the exact reported checkpoint ------------------------------------
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    model_config = config["model"]
    training_config = TrainingConfig(
        model_kwargs=dict(
            node_feat_dim=model_config["node_feat_dim"],
            hidden_dims=tuple(model_config["hidden_dims"]),
            dense_dim=model_config["dense_dim"],
            dropout_rate=model_config["dropout_rate"],
            n_classes=model_config["n_classes"],
            batch_norm=model_config["batch_norm"],
        )
    )
    state_dict = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    model = load_selected_model(state_dict, training_config)
    print(
        f"Explaining the checkpoint selected at epoch {metrics['selected_epoch']} "
        f"({metrics['protocol_label']})"
    )

    explainer_config = config["explainer"]
    settings = ExplainerSettings(
        algorithm_epochs=explainer_config["algorithm_epochs"],
        explanation_type=explainer_config["explanation_type"],
        node_mask_type=explainer_config["node_mask_type"],
        edge_mask_type=explainer_config["edge_mask_type"],
        mode=explainer_config["mode"],
        task_level=explainer_config["task_level"],
        return_type=explainer_config["return_type"],
        seed=config["reproducibility"]["seed"],
    )

    # --- choose molecules by the stated rule ------------------------------
    data_config = config["data"]
    workbook = load_workbook(
        REPOSITORY_ROOT / data_config["workbook"],
        sheet=data_config["sheet"],
        expected_sha256=data_config["expected_sha256"],
    )
    predictions = pd.read_csv(run_dir / "predictions.csv")
    predictions = predictions[predictions["partition"] == args.partition].copy()
    predictions["smiles"] = workbook.loc[
        predictions["row_index"], data_config["smiles_column"]
    ].to_numpy()
    predictions["outcome"] = [
        outcome_of(int(row.y_true), int(row.y_pred)) for row in predictions.itertuples()
    ]

    selected: list[dict] = []
    for role, (outcome, direction) in SELECTION_RULES.items():
        group = predictions[predictions["outcome"] == outcome]
        if group.empty:
            print(f"  [skip] no {outcome} in the {args.partition} partition")
            continue
        row = (
            group.loc[group["y_prob_positive"].idxmax()]
            if direction == "max"
            else group.loc[group["y_prob_positive"].idxmin()]
        )
        selected.append({"role": role, "selection_rule": f"{direction} y_prob_positive among {outcome}", **row.to_dict()})

    historical = predictions[predictions["smiles"] == HISTORICAL_CASE_STUDY]
    if not historical.empty:
        row = historical.iloc[0]
        selected.append(
            {
                "role": "historical_case_study",
                "selection_rule": "molecule used in the historical XAI figure",
                **row.to_dict(),
            }
        )
    else:
        print(
            "  [note] the historical case-study molecule is not in this partition; "
            "it is explained separately for continuity."
        )
        selected.append(
            {
                "role": "historical_case_study",
                "selection_rule": "molecule used in the historical XAI figure (outside this partition)",
                "row_index": None,
                "smiles": HISTORICAL_CASE_STUDY,
                "y_true": None,
                "y_pred": None,
                "y_prob_positive": None,
                "outcome": None,
            }
        )

    # --- explain -----------------------------------------------------------
    records = []
    for entry in selected:
        smiles = entry["smiles"]
        explanation = explain_molecule(model, smiles, settings)
        importance = explanation["atom_importance_raw"]
        k = min(args.top_k, explanation["n_atoms"])
        top_atoms = top_k_atoms(importance, k)

        record = {
            "role": entry["role"],
            "selection_rule": entry["selection_rule"],
            "row_index": entry["row_index"],
            "smiles": smiles,
            "true_label": entry["y_true"],
            "predicted_label": entry["y_pred"],
            "predicted_probability_positive": entry["y_prob_positive"],
            "checkpoint_prediction": explanation["prediction"],
            "n_atoms": explanation["n_atoms"],
            "n_bonds": explanation["n_bonds"],
            "top_k": k,
            "top_atoms": top_atoms,
            "top_atom_labels": [
                row["atom_label"]
                for row in atom_report(smiles, importance)
                if row["node_index"] in top_atoms
            ],
            "connected_motifs": connected_motifs(smiles, top_atoms),
            "atom_table": atom_report(smiles, importance),
            "bond_importance": explanation["bond_importance"],
            "fidelity": fidelity(model, smiles, importance, k),
            "edge_fidelity": edge_fidelity(
                model, smiles, explanation["bond_importance"], k
            ),
            "sparsity": sparsity(importance),
            "stability": stability(model, smiles, settings, args.stability_seeds, k),
            "settings": explanation["settings"],
        }
        records.append(record)
        fidelity_scores = record["fidelity"]
        edge_scores = record["edge_fidelity"]
        print(
            f"  {entry['role']:26s} atoms={record['n_atoms']:3d} "
            f"top={record['top_atom_labels']}\n"
            f"      node-mask  margin drop: top={fidelity_scores['margin_drop_top_k']:+.4f} "
            f"random={fidelity_scores['margin_drop_random_k_mean']:+.4f}"
            f"+/-{fidelity_scores['margin_drop_random_k_sd']:.4f} "
            f"bottom={fidelity_scores['margin_drop_bottom_k']:+.4f} "
            f"| beats random: {fidelity_scores['top_k_beats_random_on_margin']}"
        )
        if edge_scores["applicable"]:
            print(
                f"      edge-mask  margin drop: top={edge_scores['margin_drop_top_k']:+.4f} "
                f"random={edge_scores['margin_drop_random_k_mean']:+.4f}"
                f"+/-{edge_scores['margin_drop_random_k_sd']:.4f} "
                f"bottom={edge_scores['margin_drop_bottom_k']:+.4f} "
                f"| beats random: {edge_scores['top_k_beats_random_on_margin']}"
            )
        print(f"      jaccard={record['stability']['jaccard_mean']:.3f}")

    report = {
        "schema_version": 1,
        "protocol": args.protocol,
        "partition": args.partition,
        "checkpoint": {
            "path": "results/model4/" + args.protocol + "/checkpoint.pt",
            "selected_epoch": metrics["selected_epoch"],
            "selection_rule": metrics["selection_rule"],
        },
        "explainer_settings": settings.as_dict(),
        "selection_rules": {
            role: f"{direction} positive-class probability among {outcome}"
            for role, (outcome, direction) in SELECTION_RULES.items()
        },
        "explanations": records,
        "interpretation_limits": [
            "Attribution describes the model's decision surface, not biochemical causality.",
            "Node indices are RDKit atom indices; historical indices were degree-sorted and are not comparable (D-011).",
            "Fidelity uses soft removal by zeroing node features; the perturbed graph is not a valid molecule.",
            "edge_fidelity is the independent topological probe: it deletes bonds, leaves node features intact, and scores the explainer's edge mask.",
            "Explainer mode was corrected to multiclass_classification to match the two-logit head (D-012).",
        ],
    }
    (output_dir / "explanations.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    flat = pd.DataFrame(
        [
            {
                "role": record["role"],
                "smiles": record["smiles"],
                "node_index": row["node_index"],
                "atom_label": row["atom_label"],
                "atom_symbol": row["atom_symbol"],
                "is_aromatic": row["is_aromatic"],
                "in_ring": row["in_ring"],
                "importance_raw": row["importance_raw"],
                "importance_normalized": row["importance_normalized"],
            }
            for record in records
            for row in record["atom_table"]
        ]
    )
    flat.to_csv(output_dir / "atom_attributions.csv", index=False, lineterminator="\n")
    print(f"Wrote {output_dir.relative_to(REPOSITORY_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
