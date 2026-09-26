"""Extract auditable evidence from historical Jupyter notebooks."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


METRIC_PATTERN = re.compile(
    r"Accuracy:\s*(0\.\d+).*?"
    r"F1(?: Score|-score)?:\s*(0\.\d+).*?"
    r"(?:AUC-ROC|ROC-AUC|AUC):\s*(0\.\d+)",
    re.IGNORECASE | re.DOTALL,
)


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return "" if value is None else str(value)


def _cell_output_text(cell: dict[str, Any]) -> str:
    chunks: list[str] = []
    for output in cell.get("outputs", []):
        chunks.append(_as_text(output.get("text")))
        data = output.get("data", {})
        chunks.append(_as_text(data.get("text/plain")))
    return "\n".join(chunk for chunk in chunks if chunk)


def _unique_in_order(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def extract_notebook_evidence(path: Path) -> dict[str, Any]:
    """Return cell-indexed metrics, declarations, frameworks, and saved errors."""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    record: dict[str, Any] = {
        "path": path.as_posix(),
        "metric_triplets": [],
        "cv_average_metrics": [],
        "selection_metrics": [],
        "declared_epochs": [],
        "learning_rates": [],
        "hidden_dimensions": [],
        "dropout_rates": [],
        "framework_evidence": [],
        "errors": [],
    }

    all_source: list[str] = []
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        source = _as_text(cell.get("source"))
        output_text = _cell_output_text(cell)
        all_source.append(source)

        for match in METRIC_PATTERN.finditer(output_text):
            record["metric_triplets"].append(
                {
                    "cell_index": cell_index,
                    "accuracy": float(match.group(1)),
                    "f1": float(match.group(2)),
                    "roc_auc": float(match.group(3)),
                }
            )

        average_patterns = {
            "f1": r"best_f1\s+(0\.\d+)",
            "roc_auc": r"best_auc\s+(0\.\d+)",
            "accuracy": r"best_acc\s+(0\.\d+)",
            "minimum_validation_loss": r"min_val_loss\s+(0\.\d+)",
        }
        average_values = {
            name: re.search(pattern, output_text, flags=re.IGNORECASE)
            for name, pattern in average_patterns.items()
        }
        if all(match is not None for match in average_values.values()):
            record["cv_average_metrics"].append(
                {
                    "cell_index": cell_index,
                    **{
                        name: float(match.group(1))
                        for name, match in average_values.items()
                        if match is not None
                    },
                }
            )

        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                record["errors"].append(
                    {
                        "cell_index": cell_index,
                        "type": str(output.get("ename", "UnknownError")),
                        "message": str(output.get("evalue", "")),
                    }
                )

    source_text = "\n".join(all_source)
    record["selection_metrics"] = _unique_in_order(
        re.findall(r"monitor_metric\s*=\s*['\"]([^'\"]+)['\"]", source_text)
    )
    record["declared_epochs"] = _unique_in_order(
        [int(value) for value in re.findall(r"num_epochs\s*=\s*(\d+)", source_text)]
    )
    record["learning_rates"] = _unique_in_order(
        [
            float(value)
            for value in re.findall(
                r"(?:learning_rate|\blr)\s*=\s*(\d+(?:\.\d+)?(?:e-?\d+)?)",
                source_text,
                flags=re.IGNORECASE,
            )
        ]
    )
    record["dropout_rates"] = _unique_in_order(
        [
            float(value)
            for value in re.findall(
                r"dropout_rate\s*=\s*(0?\.\d+)", source_text, flags=re.IGNORECASE
            )
        ]
    )
    hidden_matches = re.findall(r"hidden_dims\s*=\s*\[([^\]]+)\]", source_text)
    record["hidden_dimensions"] = _unique_in_order(
        [
            [int(value) for value in re.findall(r"\d+", match)]
            for match in hidden_matches
        ]
    )

    frameworks: list[str] = []
    if re.search(r"(?:^|\n)\s*(?:import|from)\s+deepchem\b", source_text) or re.search(
        r"\bdc\.models\.GraphConvModel\b", source_text
    ):
        frameworks.append("DeepChem")
    if "torch_geometric" in source_text or "DeepChemStyleGraphConv" in source_text:
        frameworks.append("PyTorch Geometric")
    record["framework_evidence"] = frameworks
    return record


def rank_model4_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank complete metric triplets by distance to the historical Model 4 claim.

    The target uses the midpoint of the documented AUC range. Explicit F1-based
    checkpoint selection receives a small transparent preference.
    """

    target = {"accuracy": 0.87, "f1": 0.86, "roc_auc": 0.925}
    ranked: list[dict[str, Any]] = []
    for record in records:
        for triplet in record.get("metric_triplets", []):
            distance = sum(abs(float(triplet[key]) - value) for key, value in target.items())
            f1_bonus = 0.005 if "f1" in record.get("selection_metrics", []) else 0.0
            candidate = copy.deepcopy(record)
            candidate["selected_triplet"] = triplet
            candidate["distance_to_reported_model4"] = round(distance, 8)
            candidate["f1_selection_bonus"] = f1_bonus
            candidate["candidate_score"] = round(distance - f1_bonus, 8)
            ranked.append(candidate)
    return sorted(
        ranked,
        key=lambda item: (item["candidate_score"], item["path"], item["selected_triplet"]["cell_index"]),
    )
