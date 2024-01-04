from typing import Final, List, Optional, Tuple, Union

from torch_geometric.typing import OptTensor, Tensor
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import (
    GATv2Conv,    
    GINEConv,
    MessagePassing,
)
import torch_geometric.nn as pyg_nn
from admet_prediction.encoders.gps.encoders import AtomEncoder, EdgeEncoder
import torch.nn as nn
import torch

        
class GATv2(BasicGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)
        
        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        
        return GATv2Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout, **kwargs)

class GINE(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINEConv(mlp, **kwargs)



class GNNEncoder(nn.Module):

    def __init__(
        self,
        model,
        d_model,
        nhead,
        dropout,
        layer_norm,
        num_layer,
    ):  
        super(GNNEncoder, self).__init__()
        self.atom = AtomEncoder(
            dim_emb=d_model
        )
        self.bond = EdgeEncoder(
            dim_emb=d_model,
            batch_norm=False,
        )
        if layer_norm==True:
            self.layernorm = 'LayerNorm'
        else:
            self.layernorm = None


        if model == 'gatv2':
            self.model = GATv2(
                in_channels = d_model,
                hidden_channels = d_model,
                heads = nhead,
                dropout = dropout,
                num_layers = num_layer,
                act = nn.SiLU(),
                norm = self.layernorm,
                edge_dim = d_model,
            )
        
        elif model == 'gine':
            self.model = GINE(
                in_channels = d_model,
                hidden_channels = d_model,
                dropout = dropout,
                num_layers = num_layer,
                act = nn.SiLU(),
                norm = self.layernorm,
                edge_dim = d_model,
            )
        else:
            raise NotImplementedError("only support gatv2 or gine")

            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        features: torch.Tensor = None,
    ):
        x, edge_attr = self.atom(x), self.bond(edge_attr)

        x = self.model(x = x, edge_index = edge_index, edge_attr = edge_attr)
        

        return x, batch, features