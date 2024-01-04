import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from admet_prediction.datamodules.transforms.base import TransformBase
from admet_prediction.typing import Data


class ShortestPathGenerator(TransformBase):
    """
    Get ShortestPath
    """

    def __init__(self, directed=False) -> None:
        self.directed = directed

    def transform_(self, data: Data) -> Data:
        row = data.edge_index[0].numpy()
        col = data.edge_index[1].numpy()
        weight = np.ones_like(row)

        graph = csr_matrix((weight, (row, col)), shape=(len(data.x), len(data.x)))
        dist_matrix, _ = shortest_path(csgraph=graph, directed=self.directed, return_predecessors=True)

        data["hop"] = torch.from_numpy(dist_matrix)
        return data


class OneHotEdgeAttr(TransformBase):
    def __init__(self, max_range=4) -> Data:
        self.max_range = max_range

    def transform_(self, data: Data) -> Data:
        x = data["edge_attr"]
        if len(x.shape) == 1:
            return data

        offset = torch.ones((1, x.shape[1]), dtype=torch.long)
        offset[:, 1:] = self.max_range
        offset = torch.cumprod(offset, dim=1)
        x = (x * offset).sum(dim=1)
        data["edge_attr"] = x
        return data
