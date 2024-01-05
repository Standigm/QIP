from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Iterable

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.batch import BaseData, Batch
from torch_geometric.typing import OptTensor

# molecule types
from rdkit.Chem.rdchem import Mol as rdMol
from rdkit.Chem.rdchem import Atom as rdAtom
from rdkit.Chem.rdchem import Bond as rdBond
from openeye.oechem import OEAtomBase, OEBondBase, OEMolBase

# containers
from qip.typing.containers import TripletDataContainer
from qip.typing.outputs import EncoderTaskOutput

from torchmetrics.classification import BinaryAveragePrecision, MulticlassAUROC, MulticlassAveragePrecision, MulticlassAccuracy



PATH = Union[str, Path]
DATATYPE = Union[Tensor, int, str, float, Data, TripletDataContainer, Batch]
DATACOLLECTIONS = Union[DATATYPE, Sequence[DATATYPE], Mapping[str, DATATYPE]]
MOLTYPE = Union[rdMol, OEMolBase]
ATOMTYPE = Union[rdAtom, OEAtomBase]
BONDTYPE = Union[rdBond, OEBondBase]
MOLCOLLECTIONS = Union[MOLTYPE, Sequence[MOLTYPE], Mapping[str, MOLTYPE]]

AVAILABLE_TASKS = tuple(
    [
        "cyp-1a2",
        "cyp-2c9",
        "cyp-2c19",
        "cyp-2d6",
        "cyp-3a4",
        "dili",
        "hepatox",
        "herg",
        "lipophilicity",
        "livertox",
        "ms-human",
        "ms-mouse",
        "permeability",
        "acidic",
        "basic",
        "solubility-pbs",
        "solubility-water",
        "solubility-dmso",
        "tox21-nr-ahr",
        "tox21-nr-ar-lbd",
        "tox21-nr-ar",
        "tox21-nr-aromatase",
        "tox21-nr-er-lbd",
        "tox21-nr-er",
        "tox21-nr-ppar-gamma",
        "tox21-sr-are",
        "tox21-sr-atad5",
        "tox21-sr-hse",
        "tox21-sr-mmp",
        "tox21-sr-p53",
    ]
)

LONGTARGETMETRICS = Union[MulticlassAUROC, BinaryAveragePrecision, MulticlassAccuracy, MulticlassAveragePrecision]
