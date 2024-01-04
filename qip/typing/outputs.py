import abc
from collections import namedtuple
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from admet_prediction.typing import OptTensor, Batch, Data

from admet_prediction.utils.misc import _type_error_print_format


class Output(object):
    __metaclass__ = abc.ABCMeta


class EncoderTaskOutput(Output, namedtuple("EncoderTaskOutput", ["encoder_outputs", "task_outputs", "batch_idx"])):
    __slots__ = ()

    def __new__(
        cls,
        encoder_outputs: Mapping[str, torch.Tensor],
        task_outputs: Mapping[str, torch.Tensor],
        batch_idx: Optional[int],
    ) -> "EncoderTaskOutput":
        if isinstance(encoder_outputs, Mapping):
            pass
            # for title, output in encoder_outputs.items():
            #     if not isinstance(output, torch.Tensor):
            #         raise TypeError(
            #             _type_error_print_format(task_outputs, f"encoder_outputs[{title}]", "torch.Tensor")
            #         )
        else:
            raise TypeError(_type_error_print_format(task_outputs, "encoder_outputs", "Mapping"))
        
        if isinstance(task_outputs, Mapping):
            for title, output in task_outputs.items():
                if not isinstance(output, torch.Tensor):
                    raise TypeError(
                        _type_error_print_format(task_outputs, f"task_outputs[{title}]", "torch.Tensor")
                    )
        else:
            raise TypeError(_type_error_print_format(task_outputs, "task_outputs", "Mapping"))
        if not isinstance(batch_idx, (int, None)):
            raise TypeError(_type_error_print_format(batch_idx, "batch_idx", "int | None"))
        return super().__new__(cls, encoder_outputs, task_outputs, batch_idx)
