import glob
import itertools
import os
import os.path as osp
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import torch
from lightning.pytorch.utilities.types import _METRIC, STEP_OUTPUT
from sklearn.model_selection import train_test_split

from torch_geometric.data.dataset import Dataset

# from qip.datamodules.transforms.base import Compose

from tqdm import tqdm

from qip.utils.misc import get_logger
from qip.datamodules.featurizers import (
    QIPFeaturizer,
    FeaturizerMixin,
    FeaturizerBase,
)
from qip.datamodules.transforms import Compose, TransformBase
from qip.typing import PATH, Data

log = get_logger(__name__)


def _repr(obj: Any) -> str:
    if obj is None:
        return "None"
    return re.sub("(<.*?)\\s.*(>)", r"\1\2", obj.__repr__())


class MoleculeGraphFromSMILESDataset(Dataset, FeaturizerMixin):
    def __init__(
        self,
        root: str,
        raw_file_names: Union[Iterable[str], str],
        raw_label_file_names: Union[Iterable[str], str, None] = None,
        transform: Optional[TransformBase] = None,
        pre_transform: Optional[TransformBase] = None,
        pre_filter: Optional[Callable] = None,
        chunksize: int = 10000,
        featurizer: FeaturizerBase = QIPFeaturizer(),
        # featurizer: FeaturizerBase = OGBOriginalFeaturizer(),
        check: bool = False,
    ):
        self.root = root
        self.raw_file_names = raw_file_names
        self.raw_label_file_names = raw_label_file_names
        self.chunksize = int(chunksize)
        self.featurizer = featurizer
        self.check = check

        if isinstance(transform, list):
            transform = Compose(transforms=transform)

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading."""
        return getattr(self, "_raw_file_names", [])

    @raw_file_names.setter
    def raw_file_names(self, file_names):
        if isinstance(file_names, str):
            if osp.isfile(Path(self.raw_dir) / file_names):
                self._raw_file_names = [file_names]
            else:
                raise ValueError(f"File not found. raw_file_names: {str(Path(self.raw_dir) / file_names)}")
        elif isinstance(file_names, Iterable):
            for file_name in iter(file_names):
                if isinstance(file_name, str):
                    if not osp.isfile(Path(self.raw_dir) / file_name):
                        raise ValueError(f"File not found. raw_file_names: {str(Path(self.raw_dir) / file_name)}")
            self._raw_file_names = list(file_names)
        else:
            raise ValueError(f"Invalid input type({type(file_names)}). raw_file_names: {file_names}")

    @property
    def raw_label_file_names(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading."""
        return getattr(self, "_raw_label_file_names", [])

    @raw_label_file_names.setter
    def raw_label_file_names(self, file_names):
        if file_names is None:
            pass
        elif isinstance(file_names, str):
            if osp.isfile(Path(self.raw_dir) / file_names):
                self._raw_label_file_names = [file_names]
            else:
                ValueError(f"File not found. raw_label_file_names: {str(Path(self.raw_dir) / file_names)}")
        elif isinstance(file_names, Iterable):
            for file_name in iter(file_names):
                if isinstance(file_name, str):
                    if not osp.isfile(Path(self.raw_dir) / file_name):
                        raise ValueError(f"File not found. raw_label_file_names: {str(Path(self.raw_dir) / file_name)}")
            self._raw_label_file_names = list(file_names)
        else:
            raise ValueError(f"Invalid input type. raw_label_file_names: {file_names}")

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.
        """
        if getattr(self, "_processed_file_names", None) is None:
            self._samples_per_raw_file = []
            self._processed_file_names = []

            if self.raw_label_file_names == []:
                raw_label_file_names = [None] * len(self.raw_file_names)
            else:
                raw_label_file_names = self.raw_label_file_names

            for raw_file_name, raw_label_file_name in zip(self.raw_file_names, raw_label_file_names):
                # get total number of samples
                total_samples_per_raw_file = sum(
                    [len(df) for df in pd.read_csv(Path(self.raw_dir) / raw_file_name, chunksize=self.chunksize)]
                )
                self._samples_per_raw_file.append(total_samples_per_raw_file)

                # make directory
                processed_file_dirname, _ = raw_file_name.split(os.extsep, 1)
                if raw_label_file_name is not None:
                    processed_label_file_dirname, _ = raw_label_file_name.split(os.extsep, 1)
                    processed_file_dirname = osp.join(processed_file_dirname, processed_label_file_dirname)

                os.makedirs(Path(self.processed_dir) / processed_file_dirname, exist_ok=True)
                self._processed_file_names.extend(
                    [osp.join(processed_file_dirname, f"{i}.pt") for i in range(total_samples_per_raw_file)]
                )

        return getattr(self, "_processed_file_names", [])

    def download(self) -> bool:
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder.
        Currently downloading is not supported. Just check whether the raw file exists
        """

        unknown_files = [
            str(raw_file) for raw_file in self.raw_file_names if not osp.isfile(Path(self.root) / raw_file)
        ]
        if len(unknown_files) > 0:
            raise ValueError(f"raw_file_names not exist on root({self.root}): {unknown_files}")

        # check labelfile and rawfile
        if self.raw_label_file_names != []:
            if not osp.isfile(Path(self.processed_dir) / "targets_list.pt"):
                raise ValueError(f"Target list doesn't exist. Remove {str(Path(self.processed_dir))} and restart.")

            if len(self.raw_file_names) != len(self.raw_label_file_names):
                raise ValueError(
                    f"The number of file are different: " "raw_file_names != raw_label_file_names): {unknown_files}"
                )

            else:
                different_num_sample_files = []
                for raw_idx in range(len(self.raw_file_names)):
                    total_samples = sum(
                        [
                            len(df)
                            for df in pd.read_csv(
                                osp.join(self.raw_dir, self.raw_file_names[raw_idx]),
                                chunksize=self.chunksize,
                            )
                        ]
                    )
                    label_total_samples = sum(
                        [
                            len(df)
                            for df in pd.read_csv(
                                osp.join(self.raw_dir, self.raw_label_file_names[raw_idx]),
                                chunksize=self.chunksize,
                            )
                        ]
                    )
                    if total_samples != label_total_samples:
                        different_num_sample_files.append(
                            (
                                self.raw_file_names[raw_idx],
                                self.raw_label_file_names[raw_idx],
                            )
                        )

                if len(different_num_sample_files) > 0:
                    raise ValueError(
                        "The number of samples are different: \n"
                        + "\n".join(["{} != {}".format(data, label) for data, label in different_num_sample_files])
                    )
        return True

    @property
    def num_tasks(self):
        """number of tasks"""
        return len(self.get(0).y)

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        # return len(sum(self._samples_per_raw_file))
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        r"""Gets the data object at index :obj:`idx`."""
        sample_data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))

        return sample_data

    def _process(self):
        f = osp.join(self.processed_dir, "raw_file_names.pt")
        if osp.exists(f) and torch.load(f) != _repr(self.raw_file_names):
            warnings.warn(
                f"The `raw_file_names` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first"
            )

        f = osp.join(self.processed_dir, "raw_label_file_names.pt")
        if osp.exists(f) and torch.load(f) != _repr(self.raw_label_file_names):
            warnings.warn(
                f"The `raw_label_file_names` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first"
            )

        f = osp.join(self.processed_dir, "pre_transform.pt")
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first"
            )

        f = osp.join(self.processed_dir, "pre_filter.pt")
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                f"The `pre_filter` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-fitering technique, make sure to "
                f"delete '{self.processed_dir}' first"
            )

        f = osp.join(self.processed_dir, "chunksize.pt")
        if osp.exists(f) and torch.load(f) != _repr(self.chunksize):
            warnings.warn(
                f"The `chunksize` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-fitering technique, make sure to "
                f"delete '{self.processed_dir}' first"
            )

        f = osp.join(self.processed_dir, "featurizer.pt")
        if osp.exists(f) and torch.load(f) != _repr(self.featurizer):
            warnings.warn(
                f"The `featurizer` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-fitering technique, make sure to "
                f"delete '{self.processed_dir}' first"
            )

   
        if not self.check:
            if osp.isdir(Path(self.processed_dir)) and any(
                Path(self.processed_dir).iterdir()
            ):  # only check processed_dir exists
                return
        else:
            # only check number of files
            exists_files = sorted(glob.glob(f"{Path(self.processed_dir)/'*/*.pt'}"))
            if len(exists_files) == len(self.processed_paths):
                return

        print("Processing...", file=sys.stderr)

        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        path = osp.join(self.processed_dir, "raw_file_names.pt")
        torch.save(_repr(self.raw_file_names), path)
        path = osp.join(self.processed_dir, "raw_label_file_names.pt")
        torch.save(_repr(self.raw_label_file_names), path)

        path = osp.join(self.processed_dir, "pre_transform.pt")
        torch.save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, "pre_filter.pt")
        torch.save(_repr(self.pre_filter), path)
        path = osp.join(self.processed_dir, "chunksize.pt")
        torch.save(_repr(self.chunksize), path)
        path = osp.join(self.processed_dir, "featurizer.pt")
        torch.save(_repr(self.featurizer), path)

        print("Done!", file=sys.stderr)

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder.
        each samples are saved as raw_file_name/{index}.pt"""

        print("Converting SMILES strings into graphs...", file=sys.stderr)
        processed_idx = 0
        with tqdm(total=self.len(), desc="Processing") as pbar:
            for raw_idx, raw_file_name in enumerate(self.raw_file_names):
                raw_data_reader = pd.read_csv(osp.join(self.raw_dir, raw_file_name), chunksize=self.chunksize)
                if self.raw_label_file_names != []:
                    raw_label_reader = pd.read_csv(
                        osp.join(self.raw_dir, self.raw_label_file_names[raw_idx]),
                        chunksize=self.chunksize,
                    )
                else:
                    raw_label_reader = itertools.repeat(None)

                for data_chunk_df, label_chunk_df in zip(raw_data_reader, raw_label_reader):
                    smiles_list = data_chunk_df["smiles"] if "smiles" in data_chunk_df else data_chunk_df["SMILES"]

                    for df_idx, smiles in enumerate(smiles_list):
                        smiles = smiles_list.iloc[df_idx]
                        data = self.featurizer(smiles)

                        if label_chunk_df is not None:
                            # remove smiles column if exists..
                            label_chunk_df = label_chunk_df[
                                [
                                    col_name
                                    for col_name in label_chunk_df.columns
                                    if col_name not in ("smiles", "Smiles", "SMILES")
                                ]
                            ]
                            target_values = label_chunk_df.iloc[df_idx].values.tolist()
                            data.y = torch.Tensor([target_values])
                            data.y_mask = ~torch.isnan(torch.Tensor([target_values]))

                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue

                        if self.pre_transform is not None:
                            data = self.pre_transform(data)

                        torch.save(
                            data,
                            Path(self.processed_dir) / self.processed_file_names[processed_idx],
                        )
                        processed_idx += 1
                        pbar.update(1)

        if label_chunk_df is not None:
            torch.save(
                [label_chunk_df.columns.tolist()],
                osp.join(self.processed_dir, "targets_list.pt"),
            )

    def get_idx_split(
        self, split_dict_name: Optional[PATH] = None, split_val: bool = False
    ) -> Dict[str, Union[torch.Tensor, slice, List[int]]]:
        if split_dict_name is not None and osp.exists(osp.join(self.raw_dir, split_dict_name)):
            log.info(f"Get split from {osp.join(self.raw_dir, split_dict_name)}")
            split_dict = torch.load(osp.join(self.raw_dir, split_dict_name), map_location="cpu")

        else:
            # TODO: KFOLD, Random, predefined set
            # randomly split unlabeled set
            train_ids, test_ids = train_test_split(
                list(range(len(self))), test_size=min(5000, int(len(self) * 0.1)), random_state=123
            )  # 10% of totalset upto 5000
            split_dict = {"train": train_ids, "test": test_ids}

        if "train" not in split_dict.keys() or "test" not in split_dict.keys():
            raise ValueError("split_dict doesn't contain train, test")

        # if val is not contained.
        if "val" not in split_dict.keys():
            if split_val:
                train_ids = split_dict["train"]
                if isinstance(train_ids, slice):
                    train_ids = range(train_ids.start, train_ids.stop)
                train_ids, val_ids = train_test_split(
                    train_ids, test_size=min(5000, int(len(train_ids) * 0.1)), random_state=123
                )  # 10% of trainset upto 5000
                split_dict["train"] = train_ids
                split_dict["val"] = val_ids
            else:
                # no validation. set 'val' as the same as 'test'
                split_dict["val"] = split_dict["test"]

        return split_dict


from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

class GroverGraphFromSMILESDataset(MoleculeGraphFromSMILESDataset):
    def __init__(
        self,
        root: str,
        raw_file_names: Union[Iterable[str], str],
        raw_label_file_names: Union[Iterable[str], str, None] = None,
        transform: Optional[TransformBase] = None,
        pre_transform: Optional[TransformBase] = None,
        pre_filter: Optional[Callable] = None,
        chunksize: int = 10000,
        featurizer: FeaturizerBase = QIPFeaturizer(),
        # featurizer: FeaturizerBase = OGBOriginalFeaturizer(),
        check: bool = False,
        atom_vocab = '/db2/data/ADMET/data_final/data_refined/grover/raw/vocab/atom_vocab.pkl',
        bond_vocab = '/db2/data/ADMET/data_final/data_refined/grover/raw/vocab/bond_vocab.pkl',
        target: str = 'atom'

    ):
        super(GroverGraphFromSMILESDataset, self).__init__(root, raw_file_names, raw_label_file_names, transform, pre_transform, pre_filter)
        self.target = target
        with open(atom_vocab, 'rb') as f:
            self.atom_vocab = pickle.load(atom_vocab)
        with open(bond_vocab, 'rb') as f:
            self.bond_vocab = pickle.load(bond_vocab)
        # self.atom_vocab = atom_vocab
        # self.bond_vocab = bond_vocab

        descriptor_list = [x[0] for x in Chem.Descriptors._descList]
        RDKIT_PROPS = descriptor_list[-85:]
        self.calculator = MolecularDescriptorCalculator(RDKIT_PROPS)

    @staticmethod
    def atom2vocab(mol, atom):
        """
        Convert atom to vocabulary. The convention is based on atom type and bond type.
        :param mol: the molecular.
        :param atom: the target atom.
        :return: the generated atom vocabulary with its contexts.
        """
        nei = Counter()
        for a in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
            nei[str(a.GetSymbol()) + "-" + str(bond.GetBondType())] += 1
        keys = nei.keys()
        keys = list(keys)
        keys.sort()
        output = atom.GetSymbol()
        for k in keys:
            output = "%s_%s%d" % (output, k, nei[k])

        # The generated atom_vocab is too long?
        return output

    @staticmethod
    def bond2vocab(mol, begin_idx, end_idx):
        """
        Convert bond to vocabulary. The convention is based on atom type and bond type.
        Considering one-hop neighbor atoms
        :param mol: the molecular.
        :param atom: the target atom.
        :return: the generated bond vocabulary with its contexts.
        """

        def _get_bond_feature_name(bond):
            """
            Return the string format of bond features.
            Bond features are surrounded with ()

            """
            ret = []
            for bond_feature in ['BondType', 'Stereo', 'BondDir']:
                fea = eval(f"bond.Get{bond_feature}")()
                ret.append(str(fea))

            return '(' + '-'.join(ret) + ')'
        nei = Counter()
        two_neighbors = (mol.GetAtomWithIdx(int(begin_idx)), mol.GetAtomWithIdx(int(end_idx)))
        two_indices = [a.GetIdx() for a in two_neighbors]
        for nei_atom in two_neighbors:
            for a in nei_atom.GetNeighbors():
                a_idx = a.GetIdx()
                if a_idx in two_indices:
                    continue
                tmp_bond = mol.GetBondBetweenAtoms(nei_atom.GetIdx(), a_idx)
                nei[str(nei_atom.GetSymbol()) + '-' + _get_bond_feature_name(tmp_bond)] += 1
        keys = list(nei.keys())
        keys.sort()
        output = get_bond_feature_name(bond)
        for k in keys:
            output = "%s_%s%d" % (output, k, nei[k])
        return output


    def random_mask(self, labels):
        percent = 0.85
        n_mask = math.ceil(len(labels)*percent)
        perm = np.random.permutation(len(labels))[:n_mask]
        for p in perm:
            labels[p] = 0
        return labels

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder.
        each samples are saved as raw_file_name/{index}.pt"""

        print("Converting SMILES strings into graphs...", file=sys.stderr)
        processed_idx = 0
        with tqdm(total=self.len(), desc="Processing") as pbar:
            for raw_idx, raw_file_name in enumerate(self.raw_file_names):
                raw_data_reader = pd.read_csv(osp.join(self.raw_dir, raw_file_name), chunksize=self.chunksize)
                if self.raw_label_file_names != []:
                    raw_label_reader = pd.read_csv(
                        osp.join(self.raw_dir, self.raw_label_file_names[raw_idx]),
                        chunksize=self.chunksize,
                    )
                else:
                    raw_label_reader = itertools.repeat(None)

                for data_chunk_df, label_chunk_df in zip(raw_data_reader, raw_label_reader):
                    smiles_list = data_chunk_df["smiles"] if "smiles" in data_chunk_df else data_chunk_df["SMILES"]

                    for df_idx, smiles in enumerate(smiles_list):
                        smiles = smiles_list.iloc[df_idx]

                        data = self.featurizer(smiles)

                        if label_chunk_df is not None:
                            # remove smiles column if exists..
                            label_chunk_df = label_chunk_df[
                                [
                                    col_name
                                    for col_name in label_chunk_df.columns
                                    if col_name not in ("smiles", "Smiles", "SMILES")
                                ]
                            ]
                            target_values = label_chunk_df.iloc[df_idx].values.tolist()
                            data.y = torch.Tensor([target_values])
                            data.y_mask = ~torch.isnan(torch.Tensor([target_values]))

                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue

                        if self.pre_transform is not None:
                            data = self.pre_transform(data)

                        mol = Chem.MolFromSmiles(smiles)

                        if self.target == 'atom':
                            target_values = [self.atom_vocab.stoi.get(atom2vocab(mol, atom), self.atom_vocab.other_index) for atom in mol.GetAtoms()]
                            data.y = torch.Tensor(target_values)
                            data.y_mask = ~torch.isnan(torch.Tensor(target_values))

                        elif self.target == 'bond':
                            target_values = [self.bond_vocab.stoi.get(bond2vocab(mol, data.edge_index[0][i], data.edge_index[1][i])) for i in range(len(data.edge_index[0]))]
                            data.y = torch.Tensor(target_values)
                            data.y_mask = ~torch.isnan(torch.Tensor(target_values))

                        elif self.target == 'fg':
                            fg_label = np.array([v for v in self.calculator.CalcDescriptors(mol)])
                            fg_label[fg_label != 0] = 1
                            target_values = fg_label.tolist()
                            data.y = torch.Tensor(target_values)
                            data.y_mask = ~torch.isnan(torch.Tensor(target_values))
                          

                        torch.save(
                            data,
                            Path(self.processed_dir) / self.processed_file_names[processed_idx],
                        )
                        processed_idx += 1
                        pbar.update(1)

        if label_chunk_df is not None:
            torch.save(
                [label_chunk_df.columns.tolist()],
                osp.join(self.processed_dir, "targets_list.pt"),
            )

    def get(self, idx: int) -> Data:
        r"""Gets the data object at index :obj:`idx`."""
        sample_data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        sample_data.av_targets = self.random_mask(sample_data.av_targets)
        sample_data.bv_tagrtes = self.random_mask(sample_data.bv_targets)
        return sample_data



if __name__ == "__main__":
    print("devtest")
    dataset = MoleculeGraphFromSMILESDataset(
        root="data/dummies/task1",
        raw_file_names=["test_smiles.csv", "val_smiles.csv", "train_smiles.csv", "train_added_smiles.csv"],
        raw_label_file_names=["test_labels.csv", "val_labels.csv", "train_labels.csv", "train_added_labels.csv"],
    )
    print(dataset[-1])

    # print("herg_herg_central_cdd_cai2019")
    # dataset = MoleculeGraphFromSMILESDataset(
    #     root="/db2/data/ADMET/data_final/herg/herg_cdd_central_cai2019/",
    #     raw_file_names=[
    #         "herg_cdd_smiles.csv",
    #         "herg_cai2019_smiles.csv",
    #         "herg_central_smiles.csv",
    #     ],
    #     raw_label_file_names=[
    #         "herg_cdd_labels.csv",
    #         "herg_cai2019_labels.csv",
    #         "herg_central_labels.csv",
    #     ],
    # )
    # print(dataset[-1])

    # # print(dataset.get_idx_split())

    # print("nablaDFT")
    # dataset = MoleculeGraphFromSMILESDataset(
    #     root="/db2/data/benchmarks/nablaDFT/",
    #     raw_file_names="nabladft_min_formation_smiles.csv",
    #     raw_label_file_names="nabladft_min_formation_label.csv",
    # )
    # print(dataset.get_idx_split())

    # print("chemdb unlabeled")
    # dataset = MoleculeGraphFromSMILESDataset(
    #     root="/db2/data/unlabeled/",
    #     raw_file_names="unlabeled_smiles.csv.gz",
    #     raw_label_file_names="unlabeled_desc.csv.gz",
    # )
    # print(dataset.get_idx_split())