from collections import defaultdict
from pathlib import Path

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from qip.typing import Data
import numpy as np
# periodic table mapping
PERIODIC_TABLE = pd.read_csv(Path(__file__).parent / "periodic_table.csv")

ATOMNUM2GROUP = defaultdict(lambda: -1, {k: v for k, v in PERIODIC_TABLE[["atomic_num", "group"]].values})
ATOMNUM2PERIOD = defaultdict(lambda: -1, {k: v for k, v in PERIODIC_TABLE[["atomic_num", "period"]].values})


# non = np.array(['kappa3', 'MaxPartialCharge', 'MinPartialCharge',
#        'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'BCUT2D_MWHI',
#        'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI',
#        'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'Ipc', 'Kappa3'])

# descriptor_list = np.array([x[0] for x in Chem.Descriptors._descList])
# descriptor_list = np.setdiff1d(descriptor_list, non)
# descriptor_names = np.array(rdMolDescriptors.Properties.GetAvailableProperties())
# descriptor_names = np.setdiff1d(descriptor_names, non)
    
# Get RDkit descriptors
def get_descriptors_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptor_list = np.array([x[0] for x in Chem.Descriptors._descList])
    calculator = MolecularDescriptorCalculator(descriptor_list)
    return {k: v for k, v in zip(descriptor_list, calculator.CalcDescriptors(mol))}


def get_rdmol_descriptors_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptors = {}
    descriptor_names = np.array(rdMolDescriptors.Properties.GetAvailableProperties())
    get_descriptors = rdMolDescriptors.Properties(descriptor_names)
    if mol:
        descriptors.update({k: v for k, v in zip(descriptor_names, get_descriptors.ComputeProperties(mol))})
    return descriptors


def get_all_descriptors_from_smiles(smiles):
    d1 = get_descriptors_from_smiles(smiles)
    d2 = get_rdmol_descriptors_from_smiles(smiles)
    d2.update(d1)
    return d2


def validate_smiles_rdkit(smiles: str, sanitize=True):
    if smiles == "":
        return False
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        return False if mol is None else True
    except Exception as e:
        return False


if __name__ == "__main__":
    smiles = "O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5"
    desc1 = get_descriptors_from_smiles(smiles)
    desc2 = get_rdmol_descriptors_from_smiles(smiles)
