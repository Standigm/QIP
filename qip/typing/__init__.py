from pathlib import Path
from typing import Any, Callable, Dict, List, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Iterable

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
from qip.typing.outputs import EncoderTaskOutput

from torchmetrics.classification import BinaryAveragePrecision, MulticlassAUROC, MulticlassAveragePrecision, MulticlassAccuracy



PATH = Union[str, Path]
DATATYPE = Union[Tensor, int, str, float, Data, Batch]
DATACOLLECTIONS = Union[DATATYPE, Sequence[DATATYPE], Mapping[str, DATATYPE]]
MOLTYPE = Union[rdMol, OEMolBase]
ATOMTYPE = Union[rdAtom, OEAtomBase]
BONDTYPE = Union[rdBond, OEBondBase]
MOLCOLLECTIONS = Union[MOLTYPE, Sequence[MOLTYPE], Mapping[str, MOLTYPE]]

LONGTARGETMETRICS = Union[MulticlassAUROC, BinaryAveragePrecision, MulticlassAccuracy, MulticlassAveragePrecision]
