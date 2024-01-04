import torch

# periodic table mapping
from admet_prediction.utils.molecule import ATOMNUM2GROUP, ATOMNUM2PERIOD
from admet_prediction.datamodules.featurizers.base import Featurizer, FeaturizerBase
from admet_prediction.typing import MOLTYPE, OEMolBase, rdMol, Data, Optional
from openeye import oechem
from pathlib import Path


def _default_oechem_input_parser(smiles_or_path: str):
    if isinstance(smiles_or_path, str):
        path = Path(smiles_or_path)
        if path.is_file():
            raise NotImplementedError
            # if given input is file
            raw_data_reader = oechem.OEMolDatabase()
            raw_data_reader.Open(smiles_or_path)
            mol = next(raw_data_reader.GetOEGraphMols())
        else:
            mol = oechem.OEMol()
            if not oechem.OESmilesToMol(mol, smiles_or_path):
                raise ValueError(f"Invalid smiles({smiles_or_path}): cannot create OEMol object")
    else:
        raise ValueError(f"Invalid smiles_or_path type: {type(smiles_or_path)}")
    return mol


def oechem_preprocess(mol: OEMolBase, options={"SuppressHydrogens": False}):
    oechem.OEFindRingAtomsAndBonds(mol)
    oechem.OEAssignAromaticFlags(mol, oechem.OEAroModel_MDL)
    oechem.OEPerceiveChiral(mol)
    oechem.OE3DToAtomStereo(mol)
    oechem.OE3DToBondStereo(mol)
    oechem.OEAssignZap9Radii(mol)
    oechem.OEAssignHybridization(mol)
    # TODO: check this if state is correct
    if not options.get("SuppressHydrogens", False):
        oechem.OESuppressHydrogens(mol)
    return mol


# TODO: finish this
class OEOGBFeaturizer(Featurizer):
    def __init__(self, mode: Optional[str] = None) -> None:
        self.mode = mode
        if mode == "atomic_num":
            atom_feats = [
                "atomic_num",
                # "chirality",
                "degree",
                "formal_charge",
                "numH",
                # "num_radical_e",
                # "hybridization",
                "is_aromatic",
                "is_in_ring",
            ]
        elif mode == "group_period":
            atom_feats = [
                "atomic_group",
                "atomic_period",
                # "chirality",
                "degree",
                "formal_charge",
                "numH",
                # "num_radical_e",
                # "hybridization",
                "is_aromatic",
                "is_in_ring",
            ]
        else:
            atom_feats = [
                "atomic_num",
                "atomic_group",
                "atomic_period",
                # "chirality",
                "degree",
                "formal_charge",
                "numH",
                # "num_radical_e",
                # "hybridization",
                "is_aromatic",
                "is_in_ring",
            ]

        super().__init__(
            atom_features=atom_feats,
            # bond_features=["bond_type", "bond_stereo", "is_conjugated"],
            bond_features=["bond_type"],
        )
        self.preprocess = oechem_preprocess
        self.input_parser = _default_oechem_input_parser
