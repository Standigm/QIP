from collections import OrderedDict
from typing import Callable, Dict, List, Mapping, Optional

import hydra
import lightning as L
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
from torch.utils.data import DataLoader, Dataset

from qip.utils.misc import get_logger, warn_once
from qip.datamodules.datapipe import DataPipeline

log = get_logger(__name__)


class MultiDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_configs: OrderedDict[str, Dict],
        batch_size: int = 1,
        num_workers: int = 0,
        split_val: bool = False,
        collate_fn: Optional[Dict] = None,
        mode: str = "max_size_cycle",
    ) -> None:
        super().__init__()

        self.num_workers = num_workers
        self._default_collate_fn = hydra.utils.instantiate(collate_fn) if collate_fn is not None else None
        self._default_batch_size = batch_size
        self._default_split_val = split_val
        self.mode = mode

        # parser datasets
        self.dataset_dict = dict()
        self.split_dict_names = dict()
        self.collate_fns = dict()
        self.batch_sizes = dict()
        self.split_vals = dict()

        self._dataset_names = []
        self._task_names = []

        for dataset_name, dataset_config in dataset_configs.items():
            self._dataset_names.append(dataset_name)  # add dataset_name
            task_name = dataset_config.get("task_name", None)
            if task_name is None:
                log.warn(
                    f"{dataset_name} dataset doesn't contain task_name attribute. " f"set task_name to {dataset_name}."
                )
                task_name = dataset_name
            # if task_name is not provided, task_name == dataset_name
            self._task_names.append(task_name)
            self.dataset_dict[dataset_name] = dataset_config.get("dataset")
            if (
                not isinstance(self.dataset_dict[dataset_name], Mapping)
                and self.dataset_dict[dataset_name].get("_target_", None) is None
            ):
                raise ValueError(f"Invalid dataset config:\n " f"{dataset_name}:{dataset_config}")

            self.split_dict_names[dataset_name] = dataset_config.get("split_dict_name", None)
            self.collate_fns[dataset_name] = (
                hydra.utils.instantiate(dataset_config.get("collate_fn"))
                if dataset_config.get("collate_fn", None)
                else self._default_collate_fn
            )

            self.batch_sizes[dataset_name] = int(dataset_config.get("batch_size", self._default_batch_size))
            self.split_vals[dataset_name] = dataset_config.get("split_val", self._default_split_val)
        if len(set(self._dataset_names)) != len(self._dataset_names):
            raise ValueError("dataset_name is not unique!")
        self._dataset_names = tuple(self._dataset_names)
        self._task_names = tuple(self._task_names)
        self._datasets = None

    @property
    def data_pipelines(self) -> Dict[str, DataPipeline]:
        pipelines = dict()
        if self._datasets is None:
            self.setup()
        for task_name, dataset_name in zip(self.task_names, self.dataset_names):
            if task_name in pipelines.keys():
                warn_once(f"data_pipelines[{task_name}] is already setup.")
                continue
            featurizer = self._datasets[dataset_name].featurizer
            pre_transform = self._datasets[dataset_name].pre_transform
            transform = self._datasets[dataset_name].transform
            collater = self.collate_fns[dataset_name]
            pipelines[task_name] = DataPipeline(
                featurizer=featurizer, collater=collater, pre_transform=pre_transform, transform=transform
            )
        return pipelines

    @property
    def task_names(self):
        return self._task_names

    @property
    def dataset_names(self):
        return self._dataset_names

    def prepare_data(self) -> None:
        # download process by instantiation
        for dataset_name, dataset_dict in self.dataset_dict.items():
            dataset_downloaded = hydra.utils.instantiate(dataset_dict, _convert_="partial")
            if not isinstance(dataset_downloaded, Dataset):
                raise ValueError(f"Invalid dataset object: {dataset_name}")

    def setup(self, stage: str):
        # load downloaded datasets
        self._datasets = {
            dataset_name: hydra.utils.instantiate(dataset_dict, _convert_="partial")
            for dataset_name, dataset_dict in self.dataset_dict.items()
        }

    def _get_dataloaders(self, split, shuffle=False):
        dataloaders = OrderedDict()
        for dataset_name in self.dataset_dict.keys():
            if split in ("train", "val", "test"):
                
                split_idx = self._datasets[dataset_name].get_idx_split(
                    self.split_dict_names[dataset_name], split_val=self.split_vals[dataset_name]
                )[split]

                dataset = self._datasets[dataset_name][split_idx]
            else:
                # split == 'predict'
                dataset = self._datasets[dataset_name]
            dataloaders[dataset_name] = DataLoader(
                dataset,
                batch_size=self.batch_sizes[dataset_name],
                shuffle=shuffle,
                num_workers=self.num_workers,
                collate_fn=self.collate_fns[dataset_name],
                pin_memory=True,
            )
        return CombinedLoader(dataloaders, mode=self.mode if split == "train" else "sequential")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._get_dataloaders("train", True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloaders("val", False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloaders("test", False)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloaders("predict", False)
