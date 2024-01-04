# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
from typing import Any, Dict, List, Mapping, Optional

import lightning as L
import torch
import torch.nn as nn
from admet_prediction.typing import Data


class DummyEncoder(L.LightningModule):
    """This is a class for integration test. `DummyEncoder` inherits from `L.LightningModule`.
    It takes in two optional arguments, `num_layer` which determines the number of layers to be used in the neural network,
    and `d_model` which determines the number of input features to the neural network.

    Attributes:
        - num_layer (int): number of layers in the neural network
        - d_model (int): number of input features to the neural network

    Methods:
        - __init__(self, num_layer: int = 3, d_model: int = 20): Initializes the neural network with the given number of layers and input features
        - forward(self, x: torch.Tensor, **kwargs): Performs a forward pass of the input tensor `x` through the neural network and returns the output tensor

    Args:
        - num_layer: An integer representing the number of layers in the neural network. Default is 3.
        - d_model: An integer representing the number of input features to the neural network. Default is 20.

    Returns:
        - An output tensor that results from a forward pass through the neural network.

    """

    def __init__(
        self,
        num_layer: int = 4,
        d_model: int = 20,
        num_node_type: int = 11,  # number of node_types
        node_offset: int = 128,
    ):
        super().__init__()
        self.node_emb = nn.Embedding(num_node_type * node_offset, d_model, padding_idx=-1)
        layers = []
        for _ in range(num_layer):
            layers.extend([nn.Linear(d_model, d_model), nn.ReLU()])
        layers.pop()
        self.layers = nn.Sequential(*layers)

        # set example input array
        MAX_NUM_NODES = 256
        BATCH_SIZE = 8
        self.example_input_array = {
            "x": torch.zeros([BATCH_SIZE, MAX_NUM_NODES, abs(num_node_type)], dtype=torch.long, device=self.device),
        }

    def forward(self, x: torch.Tensor):
        x = self.node_emb.weight[x].sum(dim=2)
        x = self.layers(x)
        return x
