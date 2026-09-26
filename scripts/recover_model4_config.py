"""Recover and reconcile the historical Model 4 specification across notebooks.

Writes a deterministic evidence report. Fields on which the candidate notebooks
agree are marked ``agreed``; fields that differ are listed with every observed
value and its source notebook so no ambiguity is silently resolved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.model4.spec import extract_model4_evidence

# The provenance ranking (results/provenance/model_provenance.md) places these
# notebooks closest to the reported Model 4 signature. They are the epoch study
# that shares one architecture, featurizer, and training loop.
CANDIDATE_GLOB = "FYP2/FYP2exp13.*_best hyperparameter*.ipynb"


def _hashable(value: object) -> object:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    return value


def reconcile(records: list[dict], fields: list[str]) -> dict:
    """Group each field's observed values by the notebooks that produced them."""
    reconciliation: dict[str, dict] = {}
    for field in fields:
        observed: dict[object, list[str]] = {}
        for record in records:
            value = record.get(field)
            observed.setdefault(_hashable(value), []).append(record["path"])
        entries = [
            {
                "value": next(
                    record.get(field)
                    for record in records
                    if _hashable(record.get(field)) == key
                ),
                "notebooks": sorted(paths),
            }
            for key, paths in observed.items()
        ]
        entries.sort(key=lambda entry: entry["notebooks"][0])
        reconciliation[field] = {
            "agreed": len(entries) == 1,
            "values": entries,
        }
    return reconciliation


def build_report(root: Path) -> dict:
    paths = sorted(root.glob(CANDIDATE_GLOB), key=lambda path: path.as_posix())
    records = []
    for path in paths:
        record = extract_model4_evidence(path)
        record["path"] = path.resolve().relative_to(root.resolve()).as_posix()
        records.append(record)

    fields = [
        "featurizer",
        "architecture",
        "instantiation",
        "optimizer",
        "criterion",
        "batch_sizes",
        "test_loader_defined",
        "test_set_evaluated",
        "explainer",
    ]
    # `training` is deliberately excluded from agreement: the epoch study varies
    # num_epochs by design, which is the whole point of those notebooks.
    return {
        "schema_version": 1,
        "candidate_glob": CANDIDATE_GLOB,
        "notebook_count": len(records),
        "reconciliation": reconcile(records, fields),
        "training_by_notebook": [
            {"path": record["path"], "training": record["training"]}
            for record in records
        ],
        "notebooks": records,
        "limitations": [
            "No original trained Model 4 checkpoint exists; these records describe source code, not a run.",
            "Notebook outputs may reflect non-linear execution order.",
            "DeepChemStyleGraphConv is PyTorch Geometric code, not DeepChem GraphConvModel.",
            "Every candidate notebook builds a test loader but never evaluates it; the reported metrics are validation-set values at the epoch selected by maximum validation F1.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} for {report['notebook_count']} notebooks.")


if __name__ == "__main__":
    main()
