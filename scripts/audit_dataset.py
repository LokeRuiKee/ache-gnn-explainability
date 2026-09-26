"""Audit the committed molecular workbook without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import rdkit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.dataset_audit import audit_dataframe


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--duplicates", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_excel(args.input, sheet_name=args.sheet)
    summary, duplicates = audit_dataframe(frame)
    report = {
        "schema_version": 1,
        "source_path": args.input.as_posix(),
        "source_sha256": sha256(args.input),
        "sheet": args.sheet,
        "columns": list(frame.columns),
        **summary,
        "canonicalization": {
            "tool": "RDKit",
            "version": rdkit.__version__,
            "method": "MolFromSmiles followed by canonical isomeric MolToSmiles",
            "salt_removal": "not performed",
            "charge_standardization": "not performed",
        },
        "pic50_threshold_verifiable": False,
        "pic50_threshold_reason": "No pIC50 column exists in cleanData.xlsx",
        "original_curation_verifiable": False,
        "original_curation_reason": "The workbook contains only SMILES and single-class-label columns",
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.duplicates.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    duplicates.to_csv(args.duplicates, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
