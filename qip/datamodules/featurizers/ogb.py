import torch

# periodic table mapping
from admet_prediction.utils.molecule import ATOMNUM2GROUP, ATOMNUM2PERIOD
from admet_prediction.datamodules.featurizers.base import Featurizer, FeaturizerBase
from admet_prediction.typing import MOLTYPE, rdMol, Data, Optional


from ogb.utils import smiles2graph
from rdkit import Chem


class OGBOriginalFeaturizer(FeaturizerBase):
    def featurize(self, mol: MOLTYPE) -> Data:
        smiles = Chem.MolToSmiles(mol)
        graph = smiles2graph(smiles)

        return Data(
            x=torch.from_numpy(graph["node_feat"]),
            edge_index=torch.from_numpy(graph["edge_index"]),
            edge_attr=torch.from_numpy(graph["edge_feat"]),
        )


class OGBFeaturizer(Featurizer):
    def __init__(self, mode: Optional[str] = None) -> None:
        self.mode = mode
        if mode == "atomic_num":
            atom_feats = [
                "atomic_num",
            ]
        elif mode == "group_period":
            atom_feats = [
                "atomic_group",
                "atomic_period",
            ]
        else:
            atom_feats = [
                "atomic_num",
                "atomic_group",
                "atomic_period",
            ]
        atom_feats += [
            "chirality",
            "degree",
            "formal_charge",
            "numH",
            "num_radical_e",
            "hybridization",
            "is_aromatic",
            "is_in_ring",
        ]
        super().__init__(
            atom_features=atom_feats,
            bond_features=["bond_type", "bond_stereo", "is_conjugated"],
        )
