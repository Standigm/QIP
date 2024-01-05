"""
The GROVER models for pretraining, finetuning and fingerprint generating.
"""

from torch import nn as nn
from torch_geometric.nn.pool import global_mean_pool


class AtomVocabHead(nn.Module):

    def __init__(
        self,
        dim_emb,
        vocab_size: int = 11688,
    ):
        super().__init__()

        self.projection = nn.Linear(dim_emb, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim = 1)

    def forward(self, x, batch, features=None):
        x = self.projection(x)
        x = self.logsoftmax(x)
        return x


class FunctionalGroupHead(nn.Module):

    def __init__(
        self,
        dim_emb,
        fg_size: int = 85,
        ):
        super().__init__()
        self.readout = global_mean_pool
        self.projection = nn.Linear(dim_emb, fg_size)


    def forward(self, x, batch, features=None):
        x = self.readout(x, batch)
        x = self.projection(x)
        return x
