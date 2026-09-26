"""Extract historical split-protocol evidence from repository notebooks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.protocol.notebook_protocol import extract_split_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    records = []
    for path in sorted(repo.rglob("*.ipynb"), key=lambda item: item.as_posix().lower()):
        record = extract_split_protocol(path)
        record["path"] = path.relative_to(repo).as_posix()
        if record["holdout_splits"] or record["cross_validation"]:
            records.append(record)
    report = {"schema_version": 1, "notebooks": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
