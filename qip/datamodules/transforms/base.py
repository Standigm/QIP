from typing import AbstractSet, Mapping, Sequence, Union, Iterable

import torch

from qip.typing import DATACOLLECTIONS, Data


class TransformBase:
    """
    Base class for data transforms.

    Attributes:
        None

    Methods:
        transform(data: Data) -> Data:
            Abstract method that must be implemented in subclasses. This is an inplace method.

        __call__(data: Union[Data, Sequence[Data], Mapping[str, Data]]) -> Union[Data, Sequence[Data], Mapping[str, Data]]:
            Applies the transform to the input data. Accepts a single Data object, a Sequence of Data objects, or a Mapping of keys to Data objects.
    """

    def transform_(self, data: Data) -> Data:
        raise NotImplementedError

    def __call__(self, data: DATACOLLECTIONS):
        if isinstance(data, Data):
            newdata = self.transform_(data)

        elif isinstance(data, Sequence) and not isinstance(data, str):
            if isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
                newdata = data.__class__(*[self(d) for d in data])
            else:
                newdata = data.__class__([self(d) for d in data])

        elif isinstance(data, Mapping):
            newdata = data.__class__()
            for key, value in data.items():
                newdata[key] = self(value)
        elif isinstance(data, AbstractSet):
            newdata = data.__class__(self(d) for d in data)
        else:
            # not transform if data is not Data type
            newdata = data
        return newdata


class Compose(TransformBase):
    """
    Class for composing multiple transforms.

    Attributes:
        transforms (List[TransformBase]): A list of TransformBase objects to compose.
    """

    def __init__(self, transforms: Sequence[TransformBase]):
        if not isinstance(transforms, Sequence):
            raise ValueError(f"Invalid transforms type: {type(transforms)}")

        for transform_idx, transform in enumerate(transforms):
            if not isinstance(transform, TransformBase):
                raise ValueError(f"Invalid transforms[{transform_idx}] type: {type(transform)}")

        self.transforms = transforms

    def transform_(self, data: Data) -> Data:
        return self(data)

    def __call__(self, data: DATACOLLECTIONS) -> DATACOLLECTIONS:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
