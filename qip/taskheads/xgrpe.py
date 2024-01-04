# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any, Dict, List, Mapping, Optional

import hydra
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from admet_prediction.taskheads.grpe import GRPETaskTokenHead
from admet_prediction.modules.xai import xnn


class XLinear(xnn.Linear):
    def forward(self, input):
        # decompose single channel output to 2 channel
        if self.weight.shape[0] == 1:  # number of output is 1 == single output
            # decompose pos and neg activations
            sum_elems = input * self.weight
            pos_sum = F.relu(sum_elems).sum(1, keepdim=True)
            if getattr(self, "bias", None) is not None:
                pos_sum += self.bias.unsqueeze(1)
            neg_sum = F.relu(-sum_elems).sum(1, keepdim=True)
            return torch.cat([neg_sum, pos_sum], dim=-1)
        else:
            return super().forward(input)


class XFFN(nn.Sequential, xnn.RelPropSimple):
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
                layers.append(xnn.Linear(in_featdim, out_featdim, bias=bias))
                layers.append(xnn.SiLU())
                in_featdim = out_featdim
        layers.append(xnn.Dropout(dropout)) if dropout != 0.0 else None
        # layers.append(xnn.Linear(in_featdim, out_features, bias=bias))
        layers.append(XLinear(in_featdim, out_features, bias=bias))

        for idx, module in enumerate(layers):
            self.add_module(str(idx), module)

    def relprop(self, R, **kwargs):
        for layer_module in reversed(self):
            R = layer_module.relprop(R, **kwargs)
        return R


class XGRPETaskTokenHead(GRPETaskTokenHead):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        features: Optional[List[int]] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            features,
            bias,
            dropout,
        )

    def _setup_modules(self) -> None:
        self.layers = XFFN(
            self.hparams.in_features,
            self.hparams.out_features,
            self.hparams.features,
            self.hparams.bias,
            self.hparams.dropout,
        )
        self.pool = xnn.IndexSelect()

    def forward(self, x):
        # x = x[:, 0, ...]  # pool task token from input
        x = self.pool(x, 1, torch.tensor(0, device=x.device, dtype=torch.long)).squeeze(1)
        return self.layers(x)

    def relprop(self, R, **kwargs):
        R = self.layers.relprop(R, **kwargs)
        R = self.pool.relprop(R.unsqueeze(1), **kwargs)
        return R
