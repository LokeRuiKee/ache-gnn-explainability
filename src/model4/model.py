"""The Model 4 architecture, reconstructed in PyTorch Geometric.

Recovered verbatim from the seven candidate notebooks; see
``docs/model4_specification.md`` and ``results/model4/model4_spec_evidence.json``.
The forward signature is split into tensors rather than a ``Data`` object so
that PyG's ``Explainer`` can call the model directly without a wrapper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool


class DeepChemStyleGraphConv(nn.Module):
    """GraphConv stack with batch norm, mean-pool readout, and a dense head."""

    def __init__(
        self,
        node_feat_dim: int = 75,
        hidden_dims: tuple[int, ...] = (64, 64),
        dense_dim: int = 128,
        dropout_rate: float = 0.2,
        n_classes: int = 2,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if batch_norm else None

        in_channels = node_feat_dim
        for hidden_dim in hidden_dims:
            self.convs.append(GraphConv(in_channels, hidden_dim))
            if batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_channels = hidden_dim

        self.fc1 = nn.Linear(in_channels, dense_dim)
        self.fc2 = nn.Linear(dense_dim, n_classes)

        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return raw class logits of shape ``(n_graphs, n_classes)``."""
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.batch_norm:
                x = self.bns[index](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return self.fc2(x)

    def forward_data(self, data) -> torch.Tensor:
        """Convenience wrapper for a PyG ``Data``/``Batch`` object."""
        return self.forward(data.x, data.edge_index, getattr(data, "batch", None))


def build_model(config: dict) -> DeepChemStyleGraphConv:
    """Instantiate the model from the ``model`` block of ``configs/model4_pyg.yaml``."""
    return DeepChemStyleGraphConv(
        node_feat_dim=config["node_feat_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        dense_dim=config["dense_dim"],
        dropout_rate=config["dropout_rate"],
        n_classes=config["n_classes"],
        batch_norm=config["batch_norm"],
    )
