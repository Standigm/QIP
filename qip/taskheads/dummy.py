# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any, Dict, List, Mapping, Optional

import lightning as L
import torch
import torch.nn as nn


class DummyHead(L.LightningModule):
    def __init__(
        self,
        in_features: int = 20,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.layers = nn.Linear(in_features, out_features)

        MAX_NUM_NODES = 256
        BATCH_SIZE = 8
        self.example_input_array = {
            "x": torch.zeros([BATCH_SIZE, 1 + MAX_NUM_NODES, in_features], dtype=torch.float, device=self.device),
        }

    def forward(self, x):
        x = x[:, 0, ...]  # pool task token from input
        return self.layers(x)
