import logging
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import List, Optional
import os
import errno

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning_fabric.utilities.cloud_io import _load as load

from qip.typing import PATH

log = logging.getLogger(__name__)


class ModelCheckpointWithSubModules(ModelCheckpoint):
    def __init__(
        self,
        dirpath: Optional[PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        linkpath: str = None,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        submodule_dirpath: Optional[PATH] = None,
        submodule_names: Optional[List[str]] = None,
    ):
        super().__init__(
            dirpath,
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            train_time_interval,
            every_n_epochs,
            save_on_train_epoch_end,
        )
        self.submodule_dirpath = Path(self.dirpath) if submodule_dirpath is None else Path(submodule_dirpath)
        self.submodule_names = submodule_names
        self.best_model_paths = OrderedDict()
        self.linkpath = linkpath
        
    @staticmethod
    def symlink_force(src, dst, overwrite=True):
        """
        Create a symbolic link at the specified destination pointing to the source.

        Parameters:
        - src: Source path of the target file or directory.
        - dst: Destination path where the symlink will be created.
        - overwrite: If True, overwrite the existing symlink or file at the destination.

        Raises:
        - FileExistsError: If the destination already exists and overwrite is False.
        - OSError: If the symlink creation fails for other reasons.
        """
        if os.path.exists(dst):
            if overwrite:
                os.remove(dst)
            else:
                raise FileExistsError(f"Destination '{dst}' already exists.")
        
        os.symlink(src, dst)

    def on_fit_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """When pretrain routine starts we build the submodule dir on the fly."""
        super().on_fit_start(trainer, pl_module)

        if not trainer.fast_dev_run and trainer.is_global_zero:
            self._fs.makedirs(self.submodule_dirpath, exist_ok=True)

    def _save_checkpoint(self, trainer: "L.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            self._save_best_submodules(trainer)

    def _save_best_submodules(self, trainer: "L.Trainer"):
        best_model_path = Path(self.best_model_path)
        
        self.symlink_force(best_model_path, self.linkpath)

        if best_model_path not in self.best_model_paths.keys():
            submodule_subdirpath = Path(self.submodule_dirpath) / best_model_path.stem
            self.best_model_paths[best_model_path] = submodule_subdirpath

            best_model_state_dict = load(best_model_path, map_location="cpu")["state_dict"]

            for submodule_name in self.submodule_names:
                submodule_file_path = submodule_subdirpath / f"{submodule_name}.pt"
                # state_dict
                best_submodule_state_dict = OrderedDict()
                for k in best_model_state_dict.keys():
                    module_name_splitted = k.split(".")
                    if module_name_splitted[0] == submodule_name:
                        best_submodule_state_dict[".".join(module_name_splitted[1:])] = best_model_state_dict[k]

                if len(best_submodule_state_dict) > 0:
                    trainer.strategy.save_checkpoint(best_submodule_state_dict, submodule_file_path)

        if len(self.best_model_paths) > self.save_top_k:
            _, least_best_submodel_dir = self.best_model_paths.popitem(last=False)
            trainer.strategy.remove_checkpoint(str(least_best_submodel_dir))
    
    