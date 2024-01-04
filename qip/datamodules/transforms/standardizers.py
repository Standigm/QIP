from numbers import Number
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import torch

from admet_prediction.datamodules.transforms.base import TransformBase
from admet_prediction.typing import Data


class LabelStandardizer(TransformBase):
    """standardize output y"""

    def __init__(
        self, means: Iterable[Number], stds: Iterable[Number], targets: Optional[Iterable[Number]] = None
    ) -> None:
        if not len(means) == len(stds):
            raise ValueError(f"Invalid len(mean) !=  len(std): {len(means)} != {len(stds)}")
        if targets is not None:
            if not len(targets) == len(means):
                raise ValueError(f"Invalid len(targets) !=  len(mean): {len(targets)} != {len(means)}")
        self.means = torch.FloatTensor(means)
        self.stds = torch.FloatTensor(stds)
        self.targets = targets

    def transform_(self, data: Data):
        data.y = (data.y.to(torch.float) - self.means.reshape(data.y.shape).to(data.y.device)) / (
            self.stds.reshape(data.y.shape).to(data.y.device) + 1e-8
        )

        return data


class NablaDFTStandardizer(LabelStandardizer):
    def __init__(self) -> None:
        super().__init__(
            means=[-0.2997553798488207, 0.01676176596583252, 0.3158373039507147, -6.582392139913874, 4.663864915505554],
            stds=[0.06032327156938616, 0.06343349130471096, 0.027712618921288207, 6.705450151394074, 3.913140406055191],
            targets=("DFT HOMO", "DFT LUMO", "DFT HOMO-LUMO GAP", "DFT FORMATION ENERGY", "DFT TOTAL DIPOLE"),
        )


class StandardizerFromStatFile(LabelStandardizer):
    def __init__(self, statistics_file) -> None:
        sdict = torch.load(statistics_file)
        super().__init__(means=sdict["means"], stds=sdict["stds"], targets=sdict["targets"])



class RescaleFromStatFile:

    def __init__(self, statistics_file) -> None:
        sdict = torch.load(statistics_file)
        self.means = torch.FloatTensor(sdict["means"])
        self.stds = torch.FloatTensor(sdict["stds"])
        self.targets = sdict["targets"]

    def __call__(self, target):
        target = target.to('cpu')
        return self.means + target*self.stds
