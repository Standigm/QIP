from typing import AbstractSet, Any, Mapping, Sequence, Tuple, Union

import torch
from torch.utils.data import default_collate

from qip.typing import BaseData, Data, Batch, DATACOLLECTIONS


class DefaultCollater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: Sequence[Any]):
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, BaseData):
            return self.collate_data(batch)
        elif isinstance(elem, (torch.Tensor, float, int, str)):
            return default_collate(batch)
        elif isinstance(elem, Mapping):
            return elem_type({key: self([data[key] for data in batch]) for key in elem})
        elif isinstance(elem, (Sequence, AbstractSet)):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError("each element in list of batch should be of equal size")
            transposed = list(zip(*batch))

            if isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
                return elem_type(*[self(samples) for samples in transposed])
            else:
                try:
                    return elem_type([self(samples) for samples in transposed])
                except TypeError:
                    # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                    return [self(samples) for samples in transposed]

        raise TypeError(f"DataLoader found invalid type: {type(elem)}")

    def collate_data(self, batch: Sequence[DATACOLLECTIONS]) -> DATACOLLECTIONS:
        return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
