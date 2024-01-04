# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any, Dict, List, Mapping, Optional

import hydra
import lightning as L
import torch
import torch.nn as nn
import torch_geometric

class FFN(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        features: Optional[List[int]] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        in_featdim = in_features
        if features is not None:
            for out_featdim in features:
                layers.append(nn.Linear(in_featdim, out_featdim, bias=bias))
                layers.append(nn.SiLU())
                in_featdim = out_featdim
        layers.append(nn.Dropout(dropout)) if dropout != 0.0 else None
        layers.append(nn.Linear(in_featdim, out_features, bias=bias))

        for idx, module in enumerate(layers):
            self.add_module(str(idx), module)


class GPSTaskTokenHead(L.LightningModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        features: Optional[List[int]] = None,
        bias: bool = True,
        dropout: float = 0.0,
        pooling = torch_geometric.nn.pool.global_mean_pool
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.layers = FFN(in_features, out_features, features, bias, dropout)

        MAX_NUM_NODES = 256
        BATCH_SIZE = 8
        self.example_input_array = {
            "x": torch.zeros([BATCH_SIZE, 1 + MAX_NUM_NODES, in_features], dtype=torch.float, device=self.device),
        }

        self.pooling = pooling

    def forward(self, x, batch):

        x = self.pooling(x, batch)

        return self.layers(x)


# class SANGraphHead(L.LightningModule):
#     """
#     SAN prediction head for graph prediction tasks.

#     Args:
#         in_features (int): Input dimension.
#         out_features (int): Output dimension. For binary prediction, out_features=1.
#         L (int): Number of hidden layers.
#     """

#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         features: Optional[List[int]] = None,
#         bias: bool = True,
#         dropout: float = 0.0,
#         L: int = 2,
#         pooling = torch_geometric.nn.pool.global_mean_pool,
#         ):
#         super().__init__()

#         self.pooling_fun = pooling
#         list_FC_layers = [
#             nn.Linear(in_features // 2 ** l, in_features // 2 ** (l + 1), bias=True)
#             for l in range(L)]
#         list_FC_layers.append(
#             nn.Linear(in_features // 2 ** L, out_features, bias=True))
#         self.FC_layers = nn.ModuleList(list_FC_layers)
#         self.L = L
#         self.activation = nn.SiLU()


#     def forward(self, x, batch):
#         graph_emb = self.pooling_fun(x, batch)
#         for l in range(self.L):
#             graph_emb = self.FC_layers[l](graph_emb)
#             graph_emb = self.activation(graph_emb)
#         graph_emb = self.FC_layers[self.L](graph_emb)

#         return graph_emb

class SANGraphHead_sig(L.LightningModule):
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
            nn.Linear(in_features // 2 ** L, out_features, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = nn.SiLU()



    def forward(self, x, batch):
        graph_emb = self.pooling_fun(x, batch)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        graph_emb = nn.functional.sigmoid(graph_emb)
        return graph_emb

class SANGraphHead(L.LightningModule):
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
            nn.Linear(in_features // 2 ** L, out_features, bias=True))

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
