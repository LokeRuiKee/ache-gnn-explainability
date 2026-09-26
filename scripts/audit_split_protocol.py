"""Generate deterministic split manifests and molecular leakage reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import rdkit
import sklearn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.protocol.leakage import GROUP_COLUMNS, audit_leakage
from src.protocol.splits import historical_random_split, scaffold_split, validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--random-manifest", type=Path, required=True)
    parser.add_argument("--scaffold-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    return parser.parse_args()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> None:
    args = parse_args()
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if source_hash.lower() != args.expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch: expected {args.expected_sha256.lower()}, observed {source_hash}"
        )

    frame = pd.read_excel(args.input, sheet_name=args.sheet)
    random_manifest = historical_random_split(frame, seed=42)
    scaffold_manifest = scaffold_split(frame, seed=42)
    random_validation = validate_manifest(random_manifest, len(frame))
    scaffold_validation = validate_manifest(
        scaffold_manifest, len(frame), group_column="assignment_group"
    )

    random_leakage, random_groups = audit_leakage(random_manifest)
    scaffold_leakage, scaffold_groups = audit_leakage(scaffold_manifest)
    random_groups.insert(0, "split_protocol", "historical_random")
    scaffold_groups.insert(0, "split_protocol", "prospective_scaffold")
    group_columns = ["split_protocol", *GROUP_COLUMNS]
    groups = pd.concat([random_groups, scaffold_groups], ignore_index=True)
    groups = groups.reindex(columns=group_columns).sort_values(
        ["split_protocol", "identity_level", "identity", "row_index"],
        kind="stable",
    ).reset_index(drop=True)

    summary = {
        "schema_version": 1,
        "source_path": _display_path(args.input),
        "source_sha256": source_hash,
        "sheet": args.sheet,
        "seed": 42,
        "software": {
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "splits": {
            "historical_random": random_validation,
            "prospective_scaffold": scaffold_validation,
        },
        "leakage": {
            "historical_random": random_leakage["identity_levels"],
            "prospective_scaffold": scaffold_leakage["identity_levels"],
        },
    }

    payloads = {
        args.random_manifest: _csv_bytes(random_manifest),
        args.scaffold_manifest: _csv_bytes(scaffold_manifest),
        args.summary: (json.dumps(summary, indent=2) + "\n").encode("utf-8"),
        args.groups: _csv_bytes(groups),
    }
    for path, payload in payloads.items():
        _atomic_bytes(path, payload)


if __name__ == "__main__":
    main()
