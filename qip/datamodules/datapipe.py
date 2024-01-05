# base classes
from qip.datamodules.collaters import DefaultCollater
from qip.datamodules.transforms import TransformBase
from qip.datamodules.featurizers import FeaturizerBase

from collections import namedtuple
from qip.utils.misc import _type_error_print_format
from qip.typing import Optional, Any, Batch, Callable, Union, Iterable


class DataPipeline(namedtuple("DataPipeline", ["featurizer", "collater", "pre_transform", "transform"])):
    """Data pipeline that featurizes, pre_transform, transforms, and collates data for batching.

    Args:
        featurizer (FeaturizerBase): The featurizer to convert input data to features.
        collater (DefaultCollater): The collater to convert featurized data to batched data.
        pre_transform (Optional[TransformBase]): Optional pre-processing transform to apply to featurized data.
        transform (Optional[TransformBase]): Optional transform to apply to featurized data.

    Returns:
        DataPipeline: An instance of DataPipeline.

    Raises:
        TypeError: If any of the input arguments are of incorrect types.

    """

    __slots__ = ()

    def __new__(
        cls,
        featurizer: FeaturizerBase,
        collater: Union[DefaultCollater, Callable],
        pre_transform: Optional[TransformBase] = None,
        transform: Optional[TransformBase] = None,
    ) -> "DataPipeline":
        if not isinstance(featurizer, FeaturizerBase):
            raise TypeError(_type_error_print_format(featurizer, "featurizer", "FeaturizerBase"))
        if not isinstance(collater, Union[DefaultCollater, Callable]):
            raise TypeError(_type_error_print_format(collater, "collater", "DefaultCollater"))
        if not isinstance(transform, Optional[TransformBase]):
            raise TypeError(_type_error_print_format(transform, "transform", "Optional[TransformBase]"))
        if not isinstance(pre_transform, Optional[TransformBase]):
            raise TypeError(_type_error_print_format(pre_transform, "pre_transform", "Optional[TransformBase]"))
        
        return super().__new__(cls, featurizer, collater, pre_transform, transform)

    def __call__(self, input: Any) -> Batch:
        """Invoke the data pipeline on input data.

        Args:
            input (Any): The input data to process.

        Returns:
            Batch: The batched data output.

        """
        if isinstance(input, Iterable):
            data = [self.featurizer(inp) for inp in input]
        else:
            data = [self.featurizer]
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        if self.transform is not None:
            data = self.transform(data)
        batch_data = self.collater(data)
        return batch_data
