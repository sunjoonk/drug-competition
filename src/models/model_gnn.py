# src/models/model_gnn.py

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GraphNorm, global_mean_pool


class GINERegressionModel(nn.Module):
    def __init__(self,
                 node_input_dim: int,
                 edge_input_dim: int,
                 graph_feat_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0.2):
        super().__init__()

        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            nn_fn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(nn_fn)
            self.convs.append(conv)
            self.norms.append(GraphNorm(hidden_dim))

        self.pool = global_mean_pool

        # Jumping Knowledge (concat)
        jk_dim = hidden_dim * (num_layers + 1)
        total_input_dim = jk_dim + graph_feat_dim

        self.mlp = MLPHead(input_dim=total_input_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x, edge_index, edge_attr, batch, graph_feat):
        h = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        h_list = [h]
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h, batch)
            h = F.relu(h)
            h = h + h_in  # Residual
            h_list.append(h)

        h_jk = torch.cat(h_list, dim=1)
        h_graph = self.pool(h_jk, batch)
        h_total = torch.cat([h_graph, graph_feat], dim=1)
        out = self.mlp(h_total)  # [batch_size, 1]

        return out.view(-1)  # [batch_size]


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.layers(x).view(-1)
