"""
The GROVER models for pretraining, finetuning and fingerprint generating.
"""
from argparse import Namespace
from typing import List, Dict, Callable

import numpy as np
import torch
from torch import nn as nn
from torch_geometric.nn.pool import global_mean_pool


class AtomVocabHead(nn.Module):

    def __init__(
        self,
        dim_emb,
        vocab_size: int = 11688,
    ):
        super(AtomVocabHead, self).__init__()

        self.projection = nn.Linear(dim_emb, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim = 1)

    def forward(self, x, batch, features=None):
        x = self.projection(x)
        x = self.logsoftmax(x)
        return x

class BondVocabHead(nn.Module):

    def __init__(
        self,
        dim_emb,
        vocab_size: int = 7493,
    ):
        super(BondVocabHead, self).__init__()

        self.TWO_FC_4_BOND_VOCAB = True
        if self.TWO_FC_4_BOND_VOCAB:
            self.projection_reverse = nn.Linear(dim_emb, vocab_size)
        self.projection = nn.Linear(dim_emb, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim = 1)

    def forward(self, edge_attr):
        nm_bonds = edge_attr.shape[0]
        ids1 = list(range(1, nm_bonds, 2))
        ids2 = list(range(0, nm_bonds, 2))

        if self.TWO_FC_4_BOND_VOCAB:
            logits = self.projection(edge_attr[ids1]) + self.projection_reverse(edge_attr[ids2])
        else:
            logits = self.projection(edge_attr)

        return self.logsoftmax(logits)

class FunctionalGroupHead(nn.Module):

    def __init__(
        self,
        dim_emb,
        fg_size: int = 85,
        ):
        super(FunctionalGroupHead, self).__init__()
        self.readout = global_mean_pool
        self.projection = nn.Linear(dim_emb, fg_size)


    def forward(self, x, batch, features=None):
        x = self.readout(x, batch)
        x = self.projection(x)
        return x
