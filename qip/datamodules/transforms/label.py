import torch

from admet_prediction.datamodules.transforms.base import TransformBase
from admet_prediction.typing import Data


class NanToNum(TransformBase):
    """replace nan value from output y"""

    def __init__(self, nan=0.0) -> None:
        self.nan = nan

    def transform_(self, data: Data) -> Data:
        data.y = torch.nan_to_num(data.y, nan=self.nan)

        return data
