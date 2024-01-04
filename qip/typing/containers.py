from collections import namedtuple
from typing import Any

from admet_prediction.typing import Data

from admet_prediction.utils.misc import _type_error_print_format


class TripletDataContainer(namedtuple("TripletData", ["anc", "pos", "neg"])):
    __slots__ = ()

    def __new__(cls, anc: Data, pos: Data, neg: Data) -> "TripletDataContainer":
        if not isinstance(anc, Data):
            raise TypeError(_type_error_print_format(anc, "anc", "admet_prediction.typing.Data"))
        if not isinstance(pos, Data):
            raise TypeError(_type_error_print_format(pos, "pos", "admet_prediction.typing.Data"))
        if not isinstance(neg, Data):
            raise TypeError(_type_error_print_format(neg, "neg", "admet_prediction.typing.Data"))
        return super().__new__(cls, anc, pos, neg)
