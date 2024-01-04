import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch_scatter import scatter
import lightning as L

class GPSLayer(nn.Module):
    """
    Local MPNN + full graph attention x-former layer.
    """

    def __init__(
        self,
        d_model: int = 300,
        nhead: int = 5,
        dropout: float = 0.2,
        attention_dropout: float = 0.2,
        layer_norm: bool = False,
        batch_norm: bool = True,
        momentum: float = 0.1,
        log_attention_weights: bool = False,
        last: bool = False
        ):
        super(GPSLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.activation = nn.ReLU()

        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.log_attention_weights = log_attention_weights
        # GatedGCN
        self.local_model = GatedGCNLayer(
            self.d_model,
            self.d_model,
            dropout = dropout,
            residual = True,
            last = last,
            batch_norm=self.batch_norm
        )
        self.dropout_local = nn.Dropout(dropout)
        # Self Attention
        self.global_self_attention = nn.MultiheadAttention(
            self.d_model,
            self.nhead,
            dropout = attention_dropout,
            batch_first = True,
        )
        self.dropout_attn = nn.Dropout(dropout)
        # Feed Forward network
        self.feedforward = FeedForwardBlock(
            d_model = self.d_model,
            dropout = dropout,
        )

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        if self.layer_norm:
            self.local_norm = pyg_nn.norm.LayerNorm(d_model)
            self.attention_norm = pyg_nn.norm.LayerNorm(d_model)
            self.ff_norm = pyg_nn.norm.LayerNorm(d_model)

        if self.batch_norm:
            self.local_norm = nn.BatchNorm1d(d_model, momentum=momentum)
            self.attention_norm = nn.BatchNorm1d(d_model, momentum=momentum)
            self.ff_norm = nn.BatchNorm1d(d_model, momentum=momentum)

        

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        last: bool
    ):

        h = x
        h_in1 = h

        # GatedGCNLayer -> dropout internally. edge_attr은 안씀.
        h_local, edge_attr_local = self.local_model(x, edge_attr, edge_index, last)

        if self.layer_norm:
            h_local = self.local_norm(h_local, batch)
        if self.batch_norm:
            h_local = self.local_norm(h_local)

        #MultiheadAttention
        h_dense, mask = to_dense_batch(h, batch)
        if not self.log_attention_weights:
            h_attn = self.global_self_attention(
                h_dense,
                h_dense,
                h_dense,
                attn_mask = None,
                key_padding_mask = ~mask,
                need_weights = False
            )[0]
        else:
            h_attn, A = self.global_self_attention(
                h_dense,
                h_dense,
                h_dense,
                attn_mask = None,
                key_padding_mask = ~mask,
                need_weights = True,
                average_attn_weights = False
            )
            self.attn_weights = A.detach().cpu()
        h_attn = h_attn[mask]
        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn
        if self.layer_norm:
            h_attn = self.attention_norm(h_attn, batch)
        if self.batch_norm:
            h_attn = self.attention_norm(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim = -1)
        h = sum([h_local, h_attn])

        # FeedForward
        h = h + self.feedforward(h)
        if self.layer_norm:
            h = self.ff_norm(h, batch)
        if self.batch_norm:
            h = self.ff_norm(h)

        x = h

        return x, edge_attr_local

    def extra_repr(self):
        s = f'summary: d_model={self.d_model}, ' \
            f'heads={self.nhead}'
        return s

class FeedForwardBlock(nn.Module):

    def __init__(
        self,
        d_model,
        dropout,
    ):
        super(FeedForwardBlock, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ff_linear1 = nn.Linear(
            d_model,
            d_model * 2
        )
        self.ff_linear2 = nn.Linear(
            d_model * 2,
            d_model
        )
        self.activation = nn.ReLU()
        self.ff_dropout1 = nn.Dropout(self.dropout)
        self.ff_dropout2 = nn.Dropout(self.dropout)

    def forward(self, x):

        x = self.ff_linear1(x)
        x = self.activation(x)
        x = self.ff_dropout1(x)

        x = self.ff_linear2(x)
        x = self.activation(x)
        x = self.ff_dropout2(x)

        return x


# set in_dim, out_dim, dropout, residual

class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
        residual,
        last: bool=False,
        batch_norm: bool = False,
        **kwargs
        ):
        super(GatedGCNLayer, self).__init__(**kwargs)
        self.activation = nn.ReLU()
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm_node = nn.BatchNorm1d(out_dim)
            
        self.activation_node = self.activation
        if not last:
            if self.batch_norm:
                self.batch_norm_edge = nn.BatchNorm1d(out_dim)
            self.activation_edge = self.activation
        self.dropout = dropout
        self.residual = residual

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        last: bool = False,
        ):
        x, e, edge_index = x, edge_attr, edge_index

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        x, e = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                              e=e, Ax=Ax)

        
        if self.batch_norm:
            x = self.batch_norm_node(x)
            
        x = self.activation_node(x)
        x = F.dropout(x, self.dropout, training=self.training)
        if not last:
            if self.batch_norm:
                e = self.batch_norm_edge(e)
            e = self.activation_edge(e)
            e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            if not last:
                e = e_in + e


        return x, e

    def message(self, Dx_i, Ex_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)


        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out
