from typing import Optional

import numpy as np
import torch

from qip.datamodules.transforms.base import TransformBase
from qip.typing import Data


class Cal3DdistanceGenerator(TransformBase):
    """
    Get pairwise atom 3D distance matrix
    """

    def __init__(self, bin: Optional[dict] = None):
        self.bin = bin

    def transform_(self, data: Data) -> Data:
        coords = data["coordinates"]
        N = len(coords.keys())
        coords_arr = np.array(list(coords.values()))
        diff = coords_arr[:, None, :] - coords_arr[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_matrix, 0)
        data["3d_dist"] = torch.from_numpy(dist_matrix)
        return data
