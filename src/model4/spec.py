"""Recover the historical Model 4 specification from notebook source code.

The notebooks are immutable evidence. This module parses their code cells with
``ast`` and never executes them. Every field is either supported by an explicit
literal in the source or reported as ``None``; nothing is inferred or defaulted
on the caller's behalf.
"""

from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path
from typing import Any, Iterator

CONV_LAYER_NAMES = {"GraphConv", "GCNConv", "GraphConvLayer"}
FEATURIZER_NAME = "ConvMolFeaturizer"
FEATURIZER_QUALIFIED = "deepchem.feat.ConvMolFeaturizer"
OPTIMIZER_NAMES = {"Adam", "AdamW", "SGD", "RMSprop"}
CRITERION_NAMES = {"CrossEntropyLoss", "BCEWithLogitsLoss", "NLLLoss", "BCELoss"}
BUILTIN_NAMES = frozenset(dir(builtins))


def _literal(node: ast.AST | None) -> Any:
    """Return a Python literal for ``node`` or ``None`` when it is not literal."""
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _func_name(node: ast.Call) -> str:
    """Return the final attribute/name of a call target (``a.b.C()`` -> ``C``)."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _keyword(node: ast.Call, name: str) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _has_keyword(node: ast.Call, name: str) -> bool:
    return any(keyword.arg == name for keyword in node.keywords)


def _iter_code_cells(notebook: dict) -> Iterator[tuple[int, str]]:
    """Yield ``(cell_index, source)`` for code cells only.

    ``cell_index`` is the raw notebook cell position, matching the convention in
    ``src/protocol/notebook_protocol.py`` and ``results/provenance/``, so indices
    are comparable across every evidence artifact in this repository.
    """
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        yield cell_index, source


def _dict_literal(node: ast.AST | None) -> dict[str, Any]:
    """Read ``dict(a=1)`` or ``{'a': 1}`` into a plain mapping of literals."""
    if isinstance(node, ast.Call) and _func_name(node) == "dict":
        return {kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg}
    literal = _literal(node)
    return literal if isinstance(literal, dict) else {}


def _extract_architecture(tree: ast.AST, cell_index: int) -> dict | None:
    """Find a graph-convolution model class and read its ``__init__`` defaults."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        uses_conv = any(
            isinstance(inner, ast.Call) and _func_name(inner) in CONV_LAYER_NAMES
            for inner in ast.walk(node)
        )
        if not uses_conv:
            continue
        init = next(
            (
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__"
            ),
            None,
        )
        defaults: dict[str, Any] = {}
        if init is not None:
            args = [arg.arg for arg in init.args.args if arg.arg != "self"]
            # Defaults bind to the trailing parameters.
            offset = len(args) - len(init.args.defaults)
            for position, default in enumerate(init.args.defaults):
                defaults[args[offset + position]] = _literal(default)
        return {
            "class_name": node.name,
            "cell_index": cell_index,
            "defaults": defaults,
        }
    return None


def extract_model4_evidence(path: Path | str) -> dict:
    """Extract architecture, training, and explainer evidence from one notebook."""
    path = Path(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))

    record: dict[str, Any] = {
        "path": path.as_posix(),
        "featurizer": None,
        "architecture": None,
        "instantiation": None,
        "optimizer": None,
        "criterion": None,
        "batch_sizes": [],
        "training": None,
        "test_loader_defined": False,
        "test_set_evaluated": False,
        "explainer": None,
        "unparsed_cells": [],
    }
    batch_sizes: set[int] = set()

    for cell_index, source in _iter_code_cells(notebook):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            record["unparsed_cells"].append(cell_index)
            continue

        if record["architecture"] is None:
            architecture = _extract_architecture(tree, cell_index)
            if architecture is not None:
                record["architecture"] = architecture

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "test_loader":
                if isinstance(getattr(node, "ctx", None), ast.Store):
                    record["test_loader_defined"] = True

            if not isinstance(node, ast.Call):
                continue
            name = _func_name(node)

            if name == FEATURIZER_NAME:
                record["featurizer"] = FEATURIZER_QUALIFIED

            elif name == "DataLoader":
                size = _literal(_keyword(node, "batch_size"))
                if isinstance(size, int):
                    batch_sizes.add(size)

            elif name in OPTIMIZER_NAMES and record["optimizer"] is None:
                record["optimizer"] = {
                    "name": name,
                    "lr": _literal(_keyword(node, "lr")),
                    "weight_decay": _literal(_keyword(node, "weight_decay")),
                }

            elif name in CRITERION_NAMES and record["criterion"] is None:
                record["criterion"] = {
                    "name": name,
                    "class_weighted": _has_keyword(node, "weight"),
                }

            elif name == "run_training_loop" and record["training"] is None:
                record["training"] = {
                    "cell_index": cell_index,
                    "num_epochs": _literal(_keyword(node, "num_epochs")),
                    "monitor_metric": _literal(_keyword(node, "monitor_metric")),
                    "save_path": _literal(_keyword(node, "save_path")),
                    "is_binary": _literal(_keyword(node, "is_binary")),
                }

            elif name == "Explainer" and record["explainer"] is None:
                algorithm = _keyword(node, "algorithm")
                algorithm_epochs = None
                algorithm_name = None
                if isinstance(algorithm, ast.Call):
                    algorithm_name = _func_name(algorithm)
                    algorithm_epochs = _literal(_keyword(algorithm, "epochs"))
                model_config = _dict_literal(_keyword(node, "model_config"))
                record["explainer"] = {
                    "cell_index": cell_index,
                    "algorithm": algorithm_name,
                    "algorithm_epochs": algorithm_epochs,
                    "explanation_type": _literal(_keyword(node, "explanation_type")),
                    "node_mask_type": _literal(_keyword(node, "node_mask_type")),
                    "edge_mask_type": _literal(_keyword(node, "edge_mask_type")),
                    "mode": model_config.get("mode"),
                    "task_level": model_config.get("task_level"),
                    "return_type": model_config.get("return_type"),
                }

            # A test loader consumed by a non-builtin call is the only evidence
            # that the held-out partition was scored. Builtins are excluded
            # because `len(test_loader)` appears in progress messages.
            if name != "DataLoader" and name not in BUILTIN_NAMES:
                arguments = list(node.args) + [kw.value for kw in node.keywords]
                if any(
                    isinstance(argument, ast.Name) and argument.id == "test_loader"
                    for argument in arguments
                ):
                    record["test_set_evaluated"] = True

        if record["architecture"] is not None and record["instantiation"] is None:
            class_name = record["architecture"]["class_name"]
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _func_name(node) == class_name:
                    record["instantiation"] = {
                        "cell_index": cell_index,
                        "overrides": {
                            kw.arg: _literal(kw.value)
                            for kw in node.keywords
                            if kw.arg
                        },
                    }
                    break

    record["batch_sizes"] = sorted(batch_sizes)
    return record
