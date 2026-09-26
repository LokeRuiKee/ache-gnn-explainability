"""Error analysis of one selected checkpoint's held-out predictions.

Reports where the model fails and what those molecules look like, without
asserting a cause. Descriptors are computed with RDKit; scaffold novelty is
measured against the training partition of the same manifest, so "unseen
scaffold" means unseen by that specific run.

Example:
    python scripts/error_analysis.py --protocol reproduction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.model4.data import load_manifest, load_workbook

RDLogger.DisableLog("rdApp.*")

OUTCOMES = ("true_positive", "true_negative", "false_positive", "false_negative")


def outcome_of(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "true_positive"
    if y_true == 0 and y_pred == 0:
        return "true_negative"
    if y_true == 0 and y_pred == 1:
        return "false_positive"
    return "false_negative"


def describe(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    scaffold = MurckoScaffold.MakeScaffoldGeneric(
        MurckoScaffold.GetScaffoldForMol(mol)
    )
    return {
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "num_rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "generic_scaffold": Chem.MolToSmiles(scaffold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/model4_pyg.yaml")
    parser.add_argument("--protocol", choices=("reproduction", "generalization"), required=True)
    parser.add_argument("--partition", default="test")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = REPOSITORY_ROOT / "results/model4" / args.protocol
    predictions = pd.read_csv(run_dir / "predictions.csv")
    predictions = predictions[predictions["partition"] == args.partition].copy()

    data_config = config["data"]
    workbook = load_workbook(
        REPOSITORY_ROOT / data_config["workbook"],
        sheet=data_config["sheet"],
        expected_sha256=data_config["expected_sha256"],
    )
    manifest_key = "reproduction" if args.protocol == "reproduction" else "generalization"
    manifest = load_manifest(
        REPOSITORY_ROOT / config["evaluation"]["splits"][manifest_key],
        expected_rows=len(workbook),
    )

    smiles_column = data_config["smiles_column"]
    predictions["smiles"] = workbook.loc[predictions["row_index"], smiles_column].to_numpy()
    predictions["outcome"] = [
        outcome_of(int(row.y_true), int(row.y_pred)) for row in predictions.itertuples()
    ]
    predictions["confidence"] = predictions.apply(
        lambda row: row["y_prob_positive"] if row["y_pred"] == 1 else 1 - row["y_prob_positive"],
        axis=1,
    )

    descriptors = pd.DataFrame(
        [describe(smiles) for smiles in predictions["smiles"]],
        index=predictions.index,
    )
    predictions = pd.concat([predictions, descriptors], axis=1)

    # Scaffold novelty relative to this run's own training partition.
    training_rows = manifest[manifest["partition"] == "train"]["row_index"]
    training_scaffolds = {
        describe(smiles)["generic_scaffold"]
        for smiles in workbook.loc[training_rows, smiles_column]
    }
    predictions["scaffold_seen_in_training"] = predictions["generic_scaffold"].isin(
        training_scaffolds
    )

    predictions.sort_values("row_index").to_csv(
        run_dir / "error_analysis.csv", index=False, lineterminator="\n"
    )

    numeric_columns = [
        "molecular_weight", "heavy_atoms", "logp", "tpsa",
        "num_rings", "rotatable_bonds", "confidence",
    ]
    by_outcome = {
        outcome: {
            "count": int(len(group)),
            "share_of_partition": float(len(group) / len(predictions)),
            "scaffold_seen_in_training": float(group["scaffold_seen_in_training"].mean()),
            **{
                column: {
                    "mean": float(group[column].mean()),
                    "sd": float(group[column].std(ddof=1)) if len(group) > 1 else 0.0,
                }
                for column in numeric_columns
            },
        }
        for outcome, group in predictions.groupby("outcome")
    }

    errors = predictions[predictions["outcome"].str.startswith("false")]
    correct = predictions[predictions["outcome"].str.startswith("true")]

    report = {
        "schema_version": 1,
        "protocol": args.protocol,
        "partition": args.partition,
        "n": int(len(predictions)),
        "by_outcome": by_outcome,
        "seen_versus_unseen_scaffold": {
            "seen": {
                "n": int(predictions["scaffold_seen_in_training"].sum()),
                "accuracy": float(
                    predictions[predictions["scaffold_seen_in_training"]]["outcome"]
                    .str.startswith("true").mean()
                ) if predictions["scaffold_seen_in_training"].any() else None,
            },
            "unseen": {
                "n": int((~predictions["scaffold_seen_in_training"]).sum()),
                "accuracy": float(
                    predictions[~predictions["scaffold_seen_in_training"]]["outcome"]
                    .str.startswith("true").mean()
                ) if (~predictions["scaffold_seen_in_training"]).any() else None,
            },
        },
        "confidence": {
            "correct_mean": float(correct["confidence"].mean()) if len(correct) else None,
            "error_mean": float(errors["confidence"].mean()) if len(errors) else None,
        },
        "least_confident_errors": (
            errors.nsmallest(5, "confidence")[
                ["row_index", "smiles", "outcome", "y_prob_positive", "confidence"]
            ].to_dict("records")
        ),
        "most_confident_errors": (
            errors.nlargest(5, "confidence")[
                ["row_index", "smiles", "outcome", "y_prob_positive", "confidence"]
            ].to_dict("records")
        ),
        "interpretation_limits": [
            "Descriptor differences between outcome groups are descriptive, not causal.",
            "The workbook has no pIC50 values, so threshold-borderline compounds cannot be identified (D-007).",
            "Scaffold novelty is measured against this run's training partition only.",
        ],
    }
    (run_dir / "error_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(f"{args.protocol} / {args.partition}: n={report['n']}")
    for outcome in OUTCOMES:
        if outcome in by_outcome:
            entry = by_outcome[outcome]
            print(
                f"  {outcome:15s} {entry['count']:4d} "
                f"({entry['share_of_partition']:.1%})  "
                f"mean confidence {entry['confidence']['mean']:.3f}"
            )
    seen = report["seen_versus_unseen_scaffold"]
    print(
        f"  scaffold seen in training: n={seen['seen']['n']} acc={seen['seen']['accuracy']}\n"
        f"  scaffold unseen:           n={seen['unseen']['n']} acc={seen['unseen']['accuracy']}"
    )


if __name__ == "__main__":
    main()
