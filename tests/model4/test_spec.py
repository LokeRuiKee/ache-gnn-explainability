"""Tests for Model 4 specification recovery from historical notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.model4.spec import extract_model4_evidence

MODEL_CODE = """
import torch.nn as nn
from torch_geometric.nn import GraphConv, global_mean_pool


class DeepChemStyleGraphConv(nn.Module):
    def __init__(self,
                 node_feat_dim=75,
                 hidden_dims=[64, 64],
                 dense_dim=128,
                 dropout_rate=0.2,
                 n_classes=2,
                 batch_norm=True):
        super().__init__()
        self.convs = nn.ModuleList()
        in_channels = node_feat_dim
        for hidden_dim in hidden_dims:
            self.convs.append(GraphConv(in_channels, hidden_dim))
            in_channels = hidden_dim
        self.fc1 = nn.Linear(in_channels, dense_dim)
"""

CLASS_WITHOUT_CONV_CODE = """
import torch.nn as nn


class NotAGraphModel(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.fc = nn.Linear(dim, 2)
"""

FEATURIZER_CODE = """
import deepchem as dc


def convmol_to_pyg(smiles_list, labels=None):
    featurizer = dc.feat.ConvMolFeaturizer()
    convmols = featurizer.featurize(smiles_list)
"""

LOADER_CODE = """
from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32)
print(f"Number of test set batches: {len(test_loader)}")
"""

TEST_EVAL_CODE = """
acc, f1, roc_auc, cm, report, loss = evaluate_model(model, test_loader, device)
"""

TRAIN_CODE = """
model = DeepChemStyleGraphConv(node_feat_dim=75, n_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
history = run_training_loop(
    model=model,
    optimizer=optimizer,
    criterion=torch.nn.CrossEntropyLoss(),
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=700,
    save_path='best_model.pt',
    is_binary=True,
    monitor_metric='f1',
)
"""

EXPLAINER_CODE = """
from torch_geometric.explain import Explainer, GNNExplainer

explainer = Explainer(
    model=wrapped_model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    ),
)
"""

MAGIC_CODE = "%%capture\n!pip install torch torch-geometric\n"


def write_notebook(tmp_path: Path, sources: list[str], name: str = "nb.ipynb") -> Path:
    cells = [
        {"cell_type": "code", "source": source.splitlines(keepends=True), "outputs": []}
        for source in sources
    ]
    path = tmp_path / name
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    return path


def test_extracts_architecture_defaults(tmp_path):
    record = extract_model4_evidence(write_notebook(tmp_path, [MODEL_CODE]))
    architecture = record["architecture"]
    assert architecture["class_name"] == "DeepChemStyleGraphConv"
    assert architecture["defaults"] == {
        "node_feat_dim": 75,
        "hidden_dims": [64, 64],
        "dense_dim": 128,
        "dropout_rate": 0.2,
        "n_classes": 2,
        "batch_norm": True,
    }


def test_class_without_graph_convolution_is_not_the_architecture(tmp_path):
    """Only a class that actually builds graph-convolution layers qualifies."""
    record = extract_model4_evidence(
        write_notebook(tmp_path, [CLASS_WITHOUT_CONV_CODE])
    )
    assert record["architecture"] is None


def test_extracts_featurizer_and_training_hyperparameters(tmp_path):
    record = extract_model4_evidence(
        write_notebook(tmp_path, [FEATURIZER_CODE, LOADER_CODE, TRAIN_CODE])
    )
    assert record["featurizer"] == "deepchem.feat.ConvMolFeaturizer"
    assert record["batch_sizes"] == [32]
    assert record["optimizer"] == {"name": "Adam", "lr": 0.001, "weight_decay": None}
    assert record["criterion"] == {"name": "CrossEntropyLoss", "class_weighted": False}
    assert record["training"]["num_epochs"] == 700
    assert record["training"]["monitor_metric"] == "f1"


def test_reports_absent_test_evaluation(tmp_path):
    """The historical notebooks build a test loader but never evaluate it."""
    record = extract_model4_evidence(
        write_notebook(tmp_path, [LOADER_CODE, TRAIN_CODE])
    )
    assert record["test_loader_defined"] is True
    assert record["test_set_evaluated"] is False


def test_builtin_calls_do_not_count_as_test_evaluation(tmp_path):
    """`len(test_loader)` in a progress message is not a held-out evaluation."""
    record = extract_model4_evidence(write_notebook(tmp_path, [LOADER_CODE]))
    assert record["test_set_evaluated"] is False


def test_detects_a_genuine_test_evaluation(tmp_path):
    record = extract_model4_evidence(
        write_notebook(tmp_path, [LOADER_CODE, TEST_EVAL_CODE])
    )
    assert record["test_set_evaluated"] is True


def test_extracts_explainer_configuration(tmp_path):
    record = extract_model4_evidence(write_notebook(tmp_path, [EXPLAINER_CODE]))
    explainer = record["explainer"]
    assert explainer["algorithm_epochs"] == 200
    assert explainer["node_mask_type"] == "attributes"
    assert explainer["edge_mask_type"] == "object"
    assert explainer["mode"] == "binary_classification"
    assert explainer["task_level"] == "graph"
    assert explainer["return_type"] == "raw"


def test_unparsable_magic_cells_are_recorded_not_fatal(tmp_path):
    record = extract_model4_evidence(write_notebook(tmp_path, [MAGIC_CODE, MODEL_CODE]))
    assert record["unparsed_cells"] == [0]
    assert record["architecture"]["class_name"] == "DeepChemStyleGraphConv"


def test_missing_evidence_is_null_rather_than_invented(tmp_path):
    record = extract_model4_evidence(write_notebook(tmp_path, [FEATURIZER_CODE]))
    assert record["architecture"] is None
    assert record["optimizer"] is None
    assert record["training"] is None
    assert record["explainer"] is None
