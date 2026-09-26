"""Extract split declarations from saved notebook code without executing it."""

from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path
from typing import Any


def _literal(keyword: ast.keyword, literals: dict[str, Any]) -> Any:
    if isinstance(keyword.value, ast.Name):
        return literals.get(keyword.value.id)
    try:
        return ast.literal_eval(keyword.value)
    except (ValueError, TypeError):
        return None


def _keywords(call: ast.Call, literals: dict[str, Any]) -> dict[str, Any]:
    return {item.arg: _literal(item, literals) for item in call.keywords if item.arg}


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _source_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    try:
        return ast.unparse(node)
    except AttributeError:
        return ""


def _parse_cell(source: str, cell_index: int) -> tuple[list[dict], list[dict]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return [], []

    literals: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                literals[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    split_calls = [call for call in calls if _call_name(call) == "train_test_split"]
    holdouts: list[dict] = []
    if len(split_calls) >= 2:
        stages = []
        for call in split_calls[:2]:
            kwargs = _keywords(call, literals)
            stages.append(
                {
                    "train_size": kwargs.get("train_size"),
                    "test_size": kwargs.get("test_size"),
                    "random_state": kwargs.get("random_state"),
                    "stratified": any(item.arg == "stratify" for item in call.keywords),
                }
            )
        first_test = stages[0]["test_size"]
        second_test = stages[1]["test_size"]
        if isinstance(first_test, (int, float)) and isinstance(second_test, (int, float)):
            holdouts.append(
                {
                    "cell_index": cell_index,
                    "stages": stages,
                    "effective_partition_fractions": {
                        "train": round(1 - first_test, 12),
                        "validation": round(first_test * (1 - second_test), 12),
                        "test": round(first_test * second_test, 12),
                    },
                }
            )

    cross_validation: list[dict] = []
    kfold_calls = [call for call in calls if _call_name(call) == "StratifiedKFold"]
    for kfold_call in kfold_calls:
        kwargs = _keywords(kfold_call, literals)
        split_inputs: list[str] = []
        for call in calls:
            if _call_name(call) == "split" and len(call.args) >= 2:
                split_inputs = [_source_name(call.args[0]), _source_name(call.args[1])]
                break
        metric_maxima = {
            token
            for token in ("val_f1", "val_auc", "val_acc")
            if f"max(history['{token}'])" in source or f'max(history["{token}"])' in source
        }
        cross_validation.append(
            {
                "cell_index": cell_index,
                "n_splits": kwargs.get("n_splits"),
                "shuffle": kwargs.get("shuffle"),
                "random_state": kwargs.get("random_state"),
                "split_inputs": split_inputs,
                "independent_metric_maxima": metric_maxima
                == {"val_f1", "val_auc", "val_acc"},
            }
        )
    return holdouts, cross_validation


def extract_split_protocol(path: Path) -> dict:
    """Return cell-indexed holdout and cross-validation evidence."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    holdouts: list[dict] = []
    cross_validation: list[dict] = []
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        cell_holdouts, cell_cv = _parse_cell(source, cell_index)
        holdouts.extend(cell_holdouts)
        cross_validation.extend(cell_cv)
    return {
        "path": path.as_posix(),
        "holdout_splits": holdouts,
        "cross_validation": cross_validation,
    }
