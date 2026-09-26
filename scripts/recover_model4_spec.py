"""Generate machine-readable and human-readable historical model provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.provenance.notebook_evidence import (
    extract_notebook_evidence,
    rank_model4_candidates,
)


def _relative_record(record: dict, root: Path) -> dict:
    result = dict(record)
    result["path"] = Path(record["path"]).resolve().relative_to(root.resolve()).as_posix()
    return result


def build_report(root: Path) -> dict:
    notebooks = sorted(root.rglob("*.ipynb"), key=lambda path: path.as_posix())
    records = [_relative_record(extract_notebook_evidence(path), root) for path in notebooks]
    candidates = rank_model4_candidates(records)
    return {
        "schema_version": 1,
        "notebook_count": len(records),
        "reported_model4_target": {
            "accuracy": 0.87,
            "f1": 0.86,
            "roc_auc_range": [0.92, 0.93],
        },
        "model4_candidates": candidates[:10],
        "notebooks": records,
        "limitations": [
            "No original trained Model 4 checkpoint is present.",
            "No immutable historical train/validation/test indices are present.",
            "Notebook outputs are historical evidence and may reflect non-linear execution order.",
            "DeepChemStyleGraphConv is PyTorch Geometric code, not the original DeepChem GraphConvModel implementation.",
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Historical Model Provenance Audit",
        "",
        f"Notebooks inspected: **{report['notebook_count']}**.",
        "",
        "## Ranked Model 4 Candidates",
        "",
        "| Rank | Notebook | Cell | Accuracy | F1 | ROC-AUC | Selection | Score |",
        "|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for index, candidate in enumerate(report["model4_candidates"], start=1):
        metric = candidate["selected_triplet"]
        selection = ", ".join(candidate["selection_metrics"]) or "not extracted"
        lines.append(
            f"| {index} | `{candidate['path']}` | {metric['cell_index']} | "
            f"{metric['accuracy']:.4f} | {metric['f1']:.4f} | {metric['roc_auc']:.4f} | "
            f"{selection} | {candidate['candidate_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The ranking is an evidence-discovery aid, not proof that the first row is the original Model 4 artifact.",
            "",
            "## Cross-Validation Average Blocks",
            "",
            "| Notebook | Cell | Accuracy | F1 | ROC-AUC | Minimum validation loss |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for record in report["notebooks"]:
        for metric in record["cv_average_metrics"]:
            lines.append(
                f"| `{record['path']}` | {metric['cell_index']} | "
                f"{metric['accuracy']:.6f} | {metric['f1']:.6f} | "
                f"{metric['roc_auc']:.6f} | {metric['minimum_validation_loss']:.6f} |"
            )
    lines.extend(
        [
            "",
            "These historical summaries independently maximize each metric across epochs and therefore require correction in the final evaluation pipeline.",
            "",
            "## Framework and Error Evidence",
            "",
        ]
    )
    for record in report["notebooks"]:
        if record["framework_evidence"] or record["errors"]:
            frameworks = ", ".join(record["framework_evidence"]) or "not detected"
            lines.append(f"### `{record['path']}`")
            lines.append("")
            lines.append(f"- Framework evidence: {frameworks}")
            if record["errors"]:
                for error in record["errors"]:
                    lines.append(
                        f"- Saved error, cell {error['cell_index']}: "
                        f"{error['type']}: {error['message']}"
                    )
            else:
                lines.append("- Saved errors: none detected")
            lines.append("")
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.repo)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
