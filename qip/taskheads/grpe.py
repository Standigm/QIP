# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any, Dict, List, Mapping, Optional

import hydra
import lightning as L
import torch
import torch.nn as nn


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


class GRPETaskTokenHead(L.LightningModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        features: Optional[List[int]] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._setup_modules()

        MAX_NUM_NODES = 256
        BATCH_SIZE = 8
        self.example_input_array = {
            "x": torch.zeros([BATCH_SIZE, 1 + MAX_NUM_NODES, in_features], dtype=torch.float, device=self.device),
        }

    def _setup_modules(self) -> None:
        self.layers = FFN(
            self.hparams.in_features,
            self.hparams.out_features,
            self.hparams.features,
            self.hparams.bias,
            self.hparams.dropout,
        )

    def forward(self, x):
        x = x[:, 0, ...]  # pool task token from input
        return self.layers(x)
