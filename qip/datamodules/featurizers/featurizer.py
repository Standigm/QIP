from rdkit import Chem
from rdkit.Chem.rdchem import Atom as rdAtom
from rdkit.Chem.rdchem import Bond as rdBond
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# periodic table mapping
PERIODIC_TABLE = pd.read_csv("./periodic_table.csv")

ATOMNUM2GROUP = defaultdict(lambda: -1, {k: v for k, v in PERIODIC_TABLE[["atomic_num", "group"]].values})
ATOMNUM2PERIOD = defaultdict(lambda: -1, {k: v for k, v in PERIODIC_TABLE[["atomic_num", "period"]].values})


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

VALID_FEATURES_atom_num = tuple(list(range(1, 119)))
VALID_FEATURES_group = tuple(list(range(1, 19)))
VALID_FEATURES_period = tuple(list(range(1, 7)))
VALID_FEATURES_chirality = tuple(["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
VALID_FEATURES_degree = tuple([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
VALID_FEATURES_formalcharge = tuple([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
VALID_FEATURES_numH = tuple([0, 1, 2, 3, 4, 5, 6, 7, 8])
VALID_FEATURES_numradical = tuple([0, 1, 2, 3, 4])
VALID_FEATURES_hybridization = tuple(["SP", "SP2", "SP3", "SP3D", "SP3D2"])
VALID_FEATURES_isaromatic = tuple([False, True])
VALID_FEATURES_isinring = tuple([False, True])

ATOM_FEATURES = [
    VALID_FEATURES_atom_num,
    VALID_FEATURES_group,
    VALID_FEATURES_period,
    VALID_FEATURES_chirality,
    VALID_FEATURES_degree,
    VALID_FEATURES_formalcharge,
    VALID_FEATURES_numH,
    VALID_FEATURES_numradical,
    VALID_FEATURES_hybridization,
    VALID_FEATURES_isaromatic,
    VALID_FEATURES_isinring
]

def atom_featurizer(atom: rdAtom):
    atomic_num = atom.GetAtomicNum()
    atom_num = safe_index(ATOM_FEATURES[0], atomic_num)
    atom_group = safe_index(ATOM_FEATURES[1], ATOMNUM2GROUP[atomic_num])
    atom_period = safe_index(ATOM_FEATURES[2], ATOMNUM2PERIOD[atomic_num])
    atom_chirality = ATOM_FEATURES[3].index(str(atom.GetChiralTag()))
    atom_degree = safe_index(ATOM_FEATURES[4], atom.GetTotalDegree())
    atom_formalcharge = safe_index(ATOM_FEATURES[5], atom.GetFormalCharge())
    atom_numH = safe_index(ATOM_FEATURES[6], atom.GetTotalNumHs())
    atom_numradical_e = safe_index(ATOM_FEATURES[7], atom.GetNumRadicalElectrons())
    atom_hybridization = safe_index(ATOM_FEATURES[8], str(atom.GetHybridization()))
    atom_isaromatic = safe_index(ATOM_FEATURES[9], atom.GetIsAromatic())
    atom_isinring = safe_index(ATOM_FEATURES[10], atom.IsInRing())
    return [atom_num, atom_group, atom_period, atom_chirality, atom_degree, atom_formalcharge, atom_numH, atom_numradical_e, atom_hybridization, atom_isaromatic, atom_isinring]


VALID_FEATURES_bondtype = tuple(["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"])
VALID_FEATURES_stereo = tuple(
        [
            "STEREONONE",  # no special style
            "STEREOZ",  # Z double bond
            "STEREOE",  # E double bond
            "STEREOCIS",  # cis double bond
            "STEREOTRANS",  # trans double bond
            "STEREOANY",  # intentionally unspecified
        ]
    )
VALID_FEATURES_isconjugated = tuple([False, True])

BOND_FEATURES = [
    VALID_FEATURES_bondtype,
    VALID_FEATURES_stereo,
    VALID_FEATURES_isconjugated
]
def bond_featurizer(bond: rdBond):
    bond_type = safe_index(BOND_FEATURES[0], str(bond.GetBondType()))
    bond_stereo = BOND_FEATURES[1].index(str(bond.GetStereo()))
    bond_isconjugated = BOND_FEATURES[2].index(bond.GetIsConjugated())
    return [bond_type, bond_stereo, bond_isconjugated]


def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_features_list = []
    _atom_idx_mapper = {}  # in case when some index is missing
    for new_atom_idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append(atom_featurizer(atom))
        _atom_idx_mapper[int(atom.GetIdx())] = new_atom_idx
    x = np.array(atom_features_list, dtype=np.int64)
    # bonds
    num_edge_features = 3
    bonds = list(mol.GetBonds())
    if len(bonds) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in bonds:
            if isinstance(bond, rdBond):
                i = _atom_idx_mapper[bond.GetBeginAtomIdx()]
                j = _atom_idx_mapper[bond.GetEndAtomIdx()]
            else:
                raise NotImplementedError(f"Unknown bond type: {type(bond)}")

            edge_feature = bond_featurizer(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_edge_features), dtype=np.int64)

    # convert graph to data
    data = Data(
        x=torch.from_numpy(x).to(torch.int64),
        edge_index=torch.from_numpy(edge_index).to(torch.int64),
        edge_attr=torch.from_numpy(edge_attr).to(torch.int64),
    )
    return data


if __name__=="__main__":
    mol = Chem.MolFromSmiles('c1ccccc1')
    for atom in mol.GetAtoms():
        break
    for bond in mol.GetBonds():
        break
    atom_feats = atom_featurizer(atom)
    bond_feats = bond_featurizer(bond)
    print(atom_feats, len(atom_feats))
    print(bond_feats, len(bond_feats))

    
    