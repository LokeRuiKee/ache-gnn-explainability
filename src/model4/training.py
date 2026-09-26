"""Training and single-checkpoint evaluation for the Model 4 reconstruction.

The rule this module enforces: exactly one checkpoint is selected, using the
declared validation criterion, and every reported metric is then computed from
that one checkpoint's predictions. Metrics are never independently maximized
across epochs (contrast ``docs/discrepancies.md`` D-004), and the held-out test
partition is scored from the selected checkpoint rather than left unused
(contrast D-010).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.model4.metrics import classification_metrics
from src.model4.model import DeepChemStyleGraphConv


@dataclass
class TrainingConfig:
    """Training hyperparameters, all recovered from the historical notebooks."""

    num_epochs: int = 700
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    monitor_metric: str = "f1"
    seed: int = 42
    model_kwargs: dict = field(default_factory=dict)


def set_global_seed(seed: int) -> None:
    """Seed every source of randomness the training loop consumes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Return labels, hard predictions, positive-class probabilities, and row ids."""
    model.eval()
    labels, predictions, probabilities, row_indices = [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        probability = F.softmax(logits, dim=1)[:, 1]
        predictions.append(logits.argmax(dim=1).cpu())
        probabilities.append(probability.cpu())
        labels.append(batch.y.cpu())
        if hasattr(batch, "source_row"):
            row_indices.append(batch.source_row.cpu())

    return {
        "y_true": torch.cat(labels).numpy(),
        "y_pred": torch.cat(predictions).numpy(),
        "y_prob": torch.cat(probabilities).numpy(),
        "row_index": torch.cat(row_indices).numpy() if row_indices else None,
    }


def train_one_epoch(model, optimizer, criterion, loader, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def validation_loss(model, criterion, loader, device) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        total += criterion(logits, batch.y).item() * batch.num_graphs
    return total / len(loader.dataset)


def train_model(
    train_graphs: list,
    val_graphs: list,
    config: TrainingConfig,
    device: torch.device | None = None,
    progress_every: int = 50,
    log: callable = print,
) -> dict:
    """Train, then return the single checkpoint selected on validation.

    The returned ``state_dict`` is a deep copy taken at the selected epoch, so
    later training steps cannot silently mutate the reported model.
    """
    device = device or torch.device("cpu")
    set_global_seed(config.seed)

    model = DeepChemStyleGraphConv(**config.model_kwargs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_graphs, batch_size=config.batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_graphs, batch_size=config.batch_size)

    history: list[dict] = []
    best_score = -float("inf")
    best_state = None
    best_epoch = None

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        outputs = predict(model, val_loader, device)
        metrics = classification_metrics(
            outputs["y_true"], outputs["y_pred"], outputs["y_prob"]
        )
        val_loss = validation_loss(model, criterion, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": metrics["accuracy"],
                "val_f1": metrics["f1"],
                "val_roc_auc": metrics["roc_auc"],
            }
        )

        score = metrics[config.monitor_metric]
        if score is not None and score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if progress_every and (epoch % progress_every == 0 or epoch == 1):
            log(
                f"[epoch {epoch:4d}/{config.num_epochs}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_f1={metrics['f1']:.4f} (best epoch {best_epoch}, "
                f"{config.monitor_metric}={best_score:.4f})"
            )

    if best_state is None:
        raise RuntimeError("No checkpoint was selected; the monitored metric was never defined.")

    return {
        "state_dict": best_state,
        "selected_epoch": best_epoch,
        "selection_metric": config.monitor_metric,
        "selection_split": "validation",
        "selection_value": best_score,
        "history": history,
    }


def load_selected_model(
    state_dict: dict, config: TrainingConfig, device: torch.device | None = None
) -> DeepChemStyleGraphConv:
    """Rebuild the exact selected checkpoint in evaluation mode."""
    device = device or torch.device("cpu")
    model = DeepChemStyleGraphConv(**config.model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
