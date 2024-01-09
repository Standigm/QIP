import torch
import torch.nn as nn
from typing import List
from qip.datamodules.featurizers import ATOM_FEATURES_DIM, BOND_FEATURES_DIM
from omegaconf import DictConfig
import lightning as L

# ogb_feat = OGBFeaturizer()
# full_atom_feature_dims = ogb_feat.get_atom_feature_dims()
# full_bond_feature_dims = ogb_feat.get_bond_feature_dims()


class AtomEncoder(nn.Module):

    def __init__(self, dim_emb):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(ATOM_FEATURES_DIM):
            emb = torch.nn.Embedding(dim+1, dim_emb)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding

class RWSENodeEncoder(nn.Module):
    """
    Random Walk Structural Encoding node encoder.
    input feature -> [dim_emb-dim_posenc]
    posec -> [dim_posenc]
    return [dim_emb] = [dim_feat, dim_posenc]
    """
    def __init__(
        self,
        dim_in,
        dim_posenc,
        dim_emb,
        ksteps: List,
        expand_x = False,
        batch_norm = True,
        ):
        
        super(RWSENodeEncoder, self).__init__()
        self.dim_in = dim_in
        self.dim_emb = dim_emb
        self.num_random_walk_steps = len(list(range(ksteps[0], ksteps[1]+1)))
        self.batch_norm = batch_norm
        self.dim_posenc = dim_posenc

        if self.dim_emb - self.dim_posenc < 0:
            raise ValueError(f"PE dim size {self.dim_posenc} is too large for "
                             f"desired embedding size of {self.dim_emb}.")
        # node embedding
        if expand_x and self.dim_emb - self.dim_posenc > 0:
            self.linear_x = nn.Linear(self.dim_in, self.dim_emb - self.dim_posenc)
        self.expand_x = (expand_x) and (self.dim_emb - self.dim_posenc > 0)
        # positional encoding embedding
        self.pe_encoder = nn.Linear(self.num_random_walk_steps, self.dim_posenc)
        if self.batch_norm:
            self.posenc_norm = nn.BatchNorm1d(self.num_random_walk_steps)
        else:
            self.posenc_norm = None

    def forward(
        self,
        x: torch.Tensor,
        rwse: torch.Tensor,
        ):
        if self.expand_x:
            x = self.linear_x(x)
        pos_enc = rwse
        if self.batch_norm:
            pos_enc = self.posenc_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)
        x = torch.cat((x, pos_enc), 1)
        return x


class NodeEncoder(nn.Module):

    def __init__(
        self,
        dim_in: int = 1,
        dim_posenc: int = 20,
        dim_emb: int = 100,
        ksteps: List = [1,17],
        expand_x: bool = False,
        batch_norm: bool = True,
    ):
        super(NodeEncoder, self).__init__()
        self.dim_in = dim_in
        self.dim_posenc = dim_posenc
        self.dim_emb = dim_emb
        self.ksteps = ksteps
        self.expand_x = expand_x
        self.batch_norm = batch_norm

        self.encoder1 = AtomEncoder(self.dim_emb-self.dim_posenc)
        self.encoder2 = RWSENodeEncoder(
            dim_in = self.dim_in,
            dim_posenc = self.dim_posenc,
            dim_emb = self.dim_emb,
            ksteps = self.ksteps,
            expand_x = self.expand_x,
            batch_norm = self.batch_norm,
        )
        if self.batch_norm:
            self.batch_normlayer = nn.BatchNorm1d(dim_emb)
    def forward(
        self, 
        x: torch.Tensor,
        rwse: torch.Tensor
        ):

        x = self.encoder1(x)
        x = self.encoder2(x, rwse)
        if self.batch_norm:
            x = self.batch_normlayer(x)
        return x

class BondEncoder(nn.Module):

    def __init__(self, dim_emb):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(BOND_FEATURES_DIM):
            emb = torch.nn.Embedding(dim+1, dim_emb)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding

class EdgeEncoder(nn.Module):

    def __init__(
        self,
        dim_emb,
        batch_norm=True
        ):
        super(EdgeEncoder, self).__init__()
        self.batch_norm = batch_norm
        self.encoder = BondEncoder(dim_emb)
        if self.batch_norm:
            self.batch_normlayer = nn.BatchNorm1d(dim_emb)

    def forward(
        self, 
        edge_attr: torch.Tensor
        ):
        edge_attr = self.encoder(edge_attr)
        if self.batch_norm:
            edge_attr = self.batch_normlayer(edge_attr)
        return edge_attr

class FeatureEncoder(nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(
        self,
        node_config: DictConfig,
        edge_config: DictConfig,
        ):

        super(FeatureEncoder, self).__init__()
        self.node_config = node_config
        self.edge_config = edge_config

        self.node_encoder = NodeEncoder(
             dim_in = self.node_config.dim_in,
             dim_posenc = self.node_config.dim_posenc,
             dim_emb = self.node_config.dim_emb,
             ksteps = self.node_config.ksteps,
             batch_norm = self.node_config.batch_norm,
        )
        self.edge_encoder = EdgeEncoder(
            dim_emb = self.edge_config.dim_emb,
            batch_norm = self.edge_config.batch_norm
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        rwse: torch.Tensor,
        ):
        
        x = self.node_encoder(x, rwse)
        edge_attr = self.edge_encoder(edge_attr)
        return x, edge_attr
