"""Inventory every committed result artifact with its hash, size, and producer.

Supports the submission checklist: a reviewer or editor can confirm that each
reported number has a corresponding file, and that the file has not changed since
it was recorded.

Example:
    python scripts/build_artifact_inventory.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Which command produces which directory. Kept explicit so an artifact can never
# appear in the inventory without a documented way to regenerate it.
PRODUCERS = {
    "results/dataset_audit": "scripts/audit_dataset.py",
    "results/protocol": "scripts/extract_split_protocol.py, scripts/audit_split_protocol.py",
    "results/leakage": "scripts/audit_split_protocol.py",
    "results/provenance": "scripts/recover_model4_spec.py",
    "results/model4/model4_spec_evidence.json": "scripts/recover_model4_config.py",
    "results/model4/statistical_tests.json": "scripts/statistical_tests.py",
    "results/model4/bootstrap_intervals.json": "scripts/bootstrap_intervals.py",
    "results/model4/reproduction": "scripts/train_model4.py --protocol reproduction",
    "results/model4/generalization": "scripts/train_model4.py --protocol generalization",
    "results/model4/cross_validation": "scripts/cross_validate_model4.py",
    "results/model4/seed_sensitivity": "scripts/seed_sensitivity.py",
    "results/tables": "scripts/generate_publication_figures.py",
    # Standalone plot scripts. Exact-path entries, because the `results/figures`
    # directory rule below would otherwise attribute these to the figure
    # generator, which cannot produce them. `tests/qa/test_submission_qa.py`
    # asserts that every `scripts/plot_*.py` stem is listed here.
    "results/figures/figure_case_study_attribution.png": "scripts/plot_case_study_attribution.py",
    "results/figures/figure_case_study_attribution.svg": "scripts/plot_case_study_attribution.py",
    "results/figures/figure_overfitting_1000_epochs.png": "scripts/plot_overfitting_curve.py",
    "results/figures/figure_overfitting_1000_epochs.svg": "scripts/plot_overfitting_curve.py",
    "results/figures/figure_model6_cv_loss.png": "scripts/plot_historical_cv_loss.py",
    "results/figures/figure_model6_cv_loss.svg": "scripts/plot_historical_cv_loss.py",
    "results/tables/table_bootstrap_intervals.csv": "scripts/bootstrap_intervals.py",
    "results/tables/table_bootstrap_intervals.md": "scripts/bootstrap_intervals.py",
    "results/tables/table_paired_bootstrap.csv": "scripts/bootstrap_intervals.py",
    "results/tables/table_paired_bootstrap.md": "scripts/bootstrap_intervals.py",
    "results/figures": "scripts/generate_publication_figures.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def producer_for(relative: str) -> str:
    matches = [
        key
        for key in PRODUCERS
        if relative == key or relative.startswith(key + "/")
    ]
    if not matches:
        return "unattributed"
    # Longest match wins, so results/model4/reproduction beats results/model4.
    return PRODUCERS[max(matches, key=len)]


def tracked_files(root: Path, prefix: str) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", prefix],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [root / line for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--output", type=Path, default=REPOSITORY_ROOT / "results/artifact_inventory.json"
    )
    parser.add_argument(
        "--markdown", type=Path, default=REPOSITORY_ROOT / "docs/artifact_inventory.md"
    )
    args = parser.parse_args()

    entries = []
    for path in sorted(tracked_files(args.root, "results"), key=lambda p: p.as_posix()):
        if not path.exists() or path.resolve() == args.output.resolve():
            continue
        relative = path.resolve().relative_to(args.root.resolve()).as_posix()
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "produced_by": producer_for(relative),
            }
        )

    unattributed = [entry["path"] for entry in entries if entry["produced_by"] == "unattributed"]
    report = {
        "schema_version": 1,
        "artifact_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "unattributed": unattributed,
        "artifacts": entries,
        "note": (
            "Hashes describe the committed files. Regenerating with the listed "
            "command reproduces byte-identical content except for recorded "
            "software versions and training runtime, which are machine dependent."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Artifact Inventory",
        "",
        "Generated by `scripts/build_artifact_inventory.py`. Every reported number",
        "in the revision traces to one of these files.",
        "",
        f"**{len(entries)} artifacts, {report['total_bytes'] / 1024:.0f} KiB total.**",
        "",
        "| Artifact | Size (bytes) | Produced by | SHA-256 (first 16) |",
        "|---|---:|---|---|",
    ]
    lines += [
        f"| `{entry['path']}` | {entry['bytes']} | `{entry['produced_by']}` | `{entry['sha256'][:16]}` |"
        for entry in entries
    ]
    if unattributed:
        lines += ["", "## Unattributed artifacts", ""]
        lines += [f"- `{path}`" for path in unattributed]
    lines.append("")
    args.markdown.write_text("\n".join(lines), encoding="utf-8")

    print(f"{len(entries)} artifacts inventoried; {len(unattributed)} unattributed.")
    if unattributed:
        for path in unattributed:
            print(f"  unattributed: {path}")


if __name__ == "__main__":
    main()
