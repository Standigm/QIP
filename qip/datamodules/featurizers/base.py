# from typing import AbstractSet, Mapping, Sequence, Union, Callable, Optional, Any, List, Container
# import abc

# import torch
# import numpy as np

# from pathlib import Path
# from rdkit import Chem

# from qip.typing import ATOMTYPE, BONDTYPE, MOLTYPE, rdMol, OEMolBase, rdBond, OEBondBase, Data
# from collections import OrderedDict

# # TODO: this defaults is not used. but will be used for oechem
# _defaults = {
#     "Symbol": {
#         "H",
#         "He",
#         "Li",
#         "Be",
#         "B",
#         "C",
#         "N",
#         "O",
#         "F",
#         "Ne",
#         "Na",
#         "Mg",
#         "Al",
#         "Si",
#         "P",
#         "S",
#         "Cl",
#         "Ar",
#         "K",
#         "Ca",
#         "Sc",
#         "Ti",
#         "V",
#         "Cr",
#         "Mn",
#         "Fe",
#         "Co",
#         "Ni",
#         "Cu",
#         "Zn",
#         "Ga",
#         "Ge",
#         "As",
#         "Se",
#         "Br",
#         "Kr",
#         "Rb",
#         "Sr",
#         "Y",
#         "Zr",
#         "Nb",
#         "Mo",
#         "Tc",
#         "Ru",
#         "Rh",
#         "Pd",
#         "Ag",
#         "Cd",
#         "In",
#         "Sn",
#         "Sb",
#         "Te",
#         "I",
#         "Xe",
#         "Cs",
#         "Ba",
#         "La",
#         "Ce",
#         "Pr",
#         "Nd",
#         "Pm",
#         "Sm",
#         "Eu",
#         "Gd",
#         "Tb",
#         "Dy",
#         "Ho",
#         "Er",
#         "Tm",
#         "Yb",
#         "Lu",
#         "Hf",
#         "Ta",
#         "W",
#         "Re",
#         "Os",
#         "Ir",
#         "Pt",
#         "Au",
#         "Hg",
#         "Tl",
#         "Pb",
#         "Bi",
#         "Po",
#         "At",
#         "Rn",
#         "Fr",
#         "Ra",
#         "Ac",
#         "Th",
#         "Pa",
#         "U",
#         "Np",
#         "Pu",
#         "Am",
#         "Cm",
#         "Bk",
#         "Cf",
#         "Es",
#         "Fm",
#         "Md",
#         "No",
#         "Lr",
#         "Rf",
#         "Db",
#         "Sg",
#         "Bh",
#         "Hs",
#         "Mt",
#         "Ds",
#         "Rg",
#         "Cn",
#     },
#     "Hybridization": {"S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED"},
#     "CIPCode": {"R", "S", None},
#     "FormalCharge": {-3, -2, -1, 0, 1, 2, 3, 4},
#     "TotalNumHs": {0, 1, 2, 3, 4},
#     "TotalValence": {0, 1, 2, 3, 4, 5, 6, 7, 8},
#     "NumRadicalElectrons": {0, 1, 2, 3},
#     "Degree": {0, 1, 2, 3, 4, 5, 6, 7, 8},
#     "RingSize": {0, 3, 4, 5, 6, 7, 8},
#     "BondType": {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"},
#     "Stereo": {"STEREOE", "STEREOZ", "STEREOANY", "STEREONONE"},
# }


# def safe_index(l, e):
#     """
#     Return index of element e in list l. If e is not present, return the last index
#     """
#     try:
#         return l.index(e)
#     except:
#         return len(l) - 1


# class Feature:
#     @abc.abstractmethod
#     def __call__(self, inputs: Any) -> Any:
#         """Obtain feature of atom(s) or bond(s) using oechem or rdkit."""
#         pass

#     def __repr__(self) -> str:
#         _attributes = sorted([(k, v) for k, v in self.__dict__.items() if not k.startswith("_")], key=lambda x: x[0])
#         fields = [f"{k}={v}" for (k, v) in _attributes]
#         return self.name + "(" + ", ".join(fields) + ")"

#     @property
#     def name(self) -> str:
#         return self.__class__.__name__

#     @classmethod
#     @property
#     @abc.abstractmethod
#     def VALID_FEATURES(self) -> Container:
#         raise NotImplementedError


# class AtomFeature(Feature):
#     @abc.abstractmethod
#     def __call__(self, atom: ATOMTYPE) -> Any:
#         """Obtain feature of atom(s) using oechem or rdkit."""
#         pass


# class BondFeature(Feature):
#     @abc.abstractmethod
#     def __call__(self, bond: BONDTYPE) -> Any:
#         """Obtain feature of bond(s) using oechem or rdkit."""
#         pass


# class MolFeature(Feature):
#     @abc.abstractmethod
#     def __call__(self, mol: MOLTYPE) -> Any:
#         """Obtain feature of molecule(s) using oechem or rdkit."""
#         pass


# class FeatureRegistry:
#     "A factory for features."

#     def __init__(self, feature_type):
#         self._feature_type = feature_type
#         self._features = OrderedDict()

#     def register(self, name: str) -> Callable[[Feature], Feature]:
#         def wrapper(feature: Feature) -> Feature:
#             self._add_docs(feature, self._feature_type)
#             self._features[name] = feature
#             return feature

#         return wrapper

#     def get(self, name):
#         """
#         Get feature by name
#         """
#         return self._features[name]

#     def get_feature_list(self, include: Union[str, Sequence[str]]) -> List[Feature]:
#         if isinstance(include, str):
#             include = [include]
#         features_list = []
#         for feature in include:
#             if feature in self._features.keys():
#                 features_list.append(self.get(feature)())
#             else:
#                 raise ValueError(f"{feature} is not registered.")
#         return features_list

#     def registered_features(self) -> List[str]:
#         return list(self._features.keys())

#     @staticmethod
#     def _add_docs(feature: Feature, string: str) -> None:
#         feature.__doc__ = f"""{string.capitalize()} feature."""
#         feature.__call__.__doc__ = f"""Transforms an ``oechem.OE{string.capitalize()}Base`` to a feature.
#             Args:
#                 {string.lower()} (oechem.OE{string.capitalize()}Base):
#                     The input to be transformed to a feature.
#             """

#     def __repr__(self):
#         class_name = self.__class__.__name__
#         return class_name + f"(registered_features={self.registered_features()})"


# class AtomFeatureRegistry(FeatureRegistry):
#     def __init__(self):
#         super().__init__("Atom")


# class BondFeatureRegistry(FeatureRegistry):
#     def __init__(self):
#         super().__init__("Bond")


# class MolFeatureRegistry(FeatureRegistry):
#     def __init__(self):
#         super().__init__("Mol")


# atom_feature_registry = AtomFeatureRegistry()
# bond_feature_registry = BondFeatureRegistry()
# mol_feature_registry = MolFeatureRegistry()


# def _default_rdkit_input_parser(smiles_or_path: str):
#     if isinstance(smiles_or_path, str):
#         try:
#             path = Path(smiles_or_path)
#             is_path = path.is_dir() or path.is_file()
#         except OSError:
#             is_path = False

#         if is_path:
#             # if given input is file
#             raise NotImplementedError(f"Input_parser for {str(path)} is not implemented")
#         else:
#             mol = Chem.MolFromSmiles(smiles_or_path)
#     else:
#         raise ValueError(f"Invalid smiles_or_path type: {type(smiles_or_path)}")
#     return mol


# class FeaturizerBase(abc.ABC):
#     @abc.abstractmethod
#     def featurize(self, mol: MOLTYPE) -> Data:
#         """convert mol object to Data object"""

#     @property
#     def input_parser(self) -> Callable[[str], MOLTYPE]:
#         return getattr(self, "_input_parser", _default_rdkit_input_parser)

#     @input_parser.setter
#     def input_parser(self, parser_func: Optional[Callable]):
#         if parser_func is None:
#             return _default_rdkit_input_parser
#         elif isinstance(parser_func, Callable):
#             self._input_parser = parser_func
#         else:
#             raise ValueError(f"input_parser should be Callable Object but got {type(parser_func)}")

#     @property
#     def preprocess(self) -> Optional[Callable[[MOLTYPE], MOLTYPE]]:
#         return getattr(self, "_preprocess", None)

#     @preprocess.setter
#     def preprocess(self, preprocess_fn: Optional[Callable]):
#         if preprocess_fn is None:
#             return None
#         elif isinstance(preprocess_fn, Callable):
#             self._preprocess = preprocess_fn
#         else:
#             raise ValueError(f"preprocess should be Callable Object but got {type(preprocess_fn)}")

#     def __call__(self, smiles_or_path: str) -> Data:
#         mol = self.input_parser(smiles_or_path)
#         mol = self.preprocess(mol) if self.preprocess is not None else mol
#         data = self.featurize(mol)
#         return data

#     def __repr__(self) -> str:
#         _attributes = sorted([(k, v) for k, v in self.__dict__.items() if not k.startswith("_")], key=lambda x: x[0])
#         fields = [f"{k}={v}" for (k, v) in _attributes]
#         return self.__class__.__name__ + "(" + ", ".join(fields) + ")"


# class Featurizer(FeaturizerBase):
#     def __init__(
#         self,
#         atom_features: Sequence[str],
#         bond_features: Sequence[str],
#         mol_features: Optional[Sequence[str]] = None,
#         input_parser: Optional[Callable[[str], MOLTYPE]] = None,
#     ) -> None:
#         self.atom_features = atom_feature_registry.get_feature_list(include=atom_features)
#         self.bond_features = bond_feature_registry.get_feature_list(include=bond_features)
#         # currently mol feature is not used
#         if mol_features is not None:
#             self.mol_features = mol_feature_registry.get_feature_list(include=mol_features)
#         else:
#             self.mol_features = []
#         self.input_parser = input_parser

#     # get feature dimensions
#     def get_atom_feature_dims(self):
#         return list(map(len, [feat.VALID_FEATURES for feat in self.atom_features]))

#     def get_bond_feature_dims(self):
#         return list(map(len, [feat.VALID_FEATURES for feat in self.bond_features]))

#     def get_mol_feature_dims(self):
#         return list(map(len, [feat.VALID_FEATURES for feat in self.mol_features]))

#     def featurize(self, mol: MOLTYPE) -> Data:
#         if not isinstance(mol, MOLTYPE):
#             raise ValueError(f"Input type({type(mol)}) is not valid for {self.__class__.__name__}.featurize(mol)")

#         atom_features_list = []
#         _atom_idx_mapper = {}  # in case when some index is missing
#         for new_atom_idx, atom in enumerate(mol.GetAtoms()):
#             atom_features_list.append([feature(atom) for feature in self.atom_features])
#             _atom_idx_mapper[int(atom.GetIdx())] = new_atom_idx
#         x = np.array(atom_features_list, dtype=np.int64)
#         # bonds
#         num_edge_features = len(self.bond_features)
#         bonds = list(mol.GetBonds())
#         if len(bonds) > 0:  # mol has bonds
#             edges_list = []
#             edge_features_list = []
#             for bond in bonds:
#                 if isinstance(bond, rdBond):
#                     i = _atom_idx_mapper[bond.GetBeginAtomIdx()]
#                     j = _atom_idx_mapper[bond.GetEndAtomIdx()]
#                 elif isinstance(bond, OEBondBase):
#                     i = _atom_idx_mapper[bond.GetBgnIdx()]
#                     j = _atom_idx_mapper[bond.GetEndIdx()]
#                 else:
#                     raise NotImplementedError(f"Unknown bond type: {type(bond)}")

#                 edge_feature = [func(bond) for func in self.bond_features]
#                 # add edges in both directions
#                 edges_list.append((i, j))
#                 edge_features_list.append(edge_feature)
#                 edges_list.append((j, i))
#                 edge_features_list.append(edge_feature)

#             # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
#             edge_index = np.array(edges_list, dtype=np.int64).T

#             # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
#             edge_attr = np.array(edge_features_list, dtype=np.int64)
#         else:
#             edge_index = np.empty((2, 0), dtype=np.int64)
#             edge_attr = np.empty((0, num_edge_features), dtype=np.int64)

#         # TODO: add Mol features, position(coordinate)
#         # pos = ...
#         # Data(pos=pos)

#         # convert graph to data
#         data = Data(
#             x=torch.from_numpy(x).to(torch.int64),
#             edge_index=torch.from_numpy(edge_index).to(torch.int64),
#             edge_attr=torch.from_numpy(edge_attr).to(torch.int64),
#         )
#         return data


# class FeaturizerMixin:
#     @property
#     def featurizer(self) -> Optional[FeaturizerBase]:
#         return getattr(self, "_featurizer", None)

#     @featurizer.setter
#     def featurizer(self, featurizer_obj: FeaturizerBase):
#         if isinstance(featurizer_obj, FeaturizerBase):
#             self._featurizer = featurizer_obj
#         else:
#             raise ValueError(f"Invalid featurizer type: {type(featurizer_obj)}")
