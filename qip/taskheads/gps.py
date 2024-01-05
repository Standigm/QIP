# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import List, Optional

import torch
import torch.nn as nn
import torch_geometric

class SANGraphHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension. For binary prediction, out_features=1.
        L (int): Number of hidden layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        additional_feature_dim: int = 0,
        features: Optional[List[int]] = None,
        bias: bool = True,
        dropout: float = 0.0,
        L: int = 2,
        pooling = torch_geometric.nn.pool.global_mean_pool,
        ):
        super().__init__()

        self.pooling_fun = pooling
        list_FC_layers = [
            nn.Linear(in_features // 2 ** l, in_features // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(in_features // 2 ** L, out_features, bias=bias))

        list_FC_layers[0] = nn.Linear(in_features+additional_feature_dim, in_features//2, bias = True)
        
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = nn.SiLU()



    def forward(self, x, batch, features = None):
        graph_emb = self.pooling_fun(x, batch)
        
        if isinstance(features, torch.Tensor):
            features = features.reshape(graph_emb.shape[0], -1)
            graph_emb = torch.cat([graph_emb, features], dim = 1)

        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        return graph_emb
