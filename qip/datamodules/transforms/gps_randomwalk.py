import numpy as np
import torch
from qip.datamodules.transforms.base import TransformBase
from qip.typing import Data
from typing import List
from copy import deepcopy


import torch.nn.functional as F
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, scatter)
from torch_geometric.utils.num_nodes import maybe_num_nodes


class RandomWalkGenerator(TransformBase):
    """
    Get RandomWalk landing probs.
    """

    def __init__(
        self,
        ksteps: List,
        space_dim: int = 0
    ):
        self.ksteps = ksteps
        self.space_dim = space_dim

    @staticmethod
    def _get_random_walk_landing_probs(
        ksteps: List,
        edge_index,
        edge_weight = None,
        num_nodes = None,
        space_dim = 0
    ):
        """Compute Random Walk landing probabilities for given list of K steps.
        Args:
            ksteps: List of k-steps for which to compute the RW landings
            edge_index: PyG sparse representation of the graph
            edge_weight: (optional) Edge weights
            num_nodes: (optional) Number of nodes in the graph
            space_dim: (optional) Estimated dimensionality of the space. Used to
                correct the random-walk diagonal by a factor `k^(space_dim/2)`.
                In euclidean space, this correction means that the height of
                the gaussian distribution stays almost constant across the number of
                steps, if `space_dim` is the dimension of the euclidean space.

        Returns:
            2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
        """
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device = edge_index.device)
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        source, dest = edge_index[0], edge_index[1]
        deg = scatter(
            edge_weight,
            source,
            dim = 0,
            dim_size = num_nodes,
            reduce = 'sum'
        ) # scatter
        deg_inv = deg.pow(-1.) # 역수
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)

        if edge_index.numel() == 0:
            P = edge_index.new_zeros((1, num_nodes, num_nodes))
        else:
            P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes = num_nodes) # 대각성분 @ adj_matrix
        rws = []
        steps = list(range(ksteps[0], ksteps[1]+1))
        Pk = P.clone().detach().matrix_power(steps[0])
        for k in steps:
            rws.append(torch.diagonal(Pk, dim1 = -2, dim2 = -1) * (k**(space_dim / 2)))
            Pk = Pk @ P

        rw_landing = torch.cat(rws, dim = 0).transpose(0, 1)

        return rw_landing

    def transform_(self, data: Data) -> Data:
        if hasattr(data, 'num_nodes'):
            N = data.num_nodes
        else:
            N = data.x.shape[0]
        rw_landing = self._get_random_walk_landing_probs(
            ksteps = self.ksteps,
            edge_index = data.edge_index,
            num_nodes = N,
            space_dim = self.space_dim
        )
        data['rwse'] = rw_landing
        return data