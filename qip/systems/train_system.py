from __future__ import annotations

import lightning as L
import copy
import hydra
import torch
import torch.nn as nn
import os

from abc import abstractmethod
from omegaconf import OmegaConf, DictConfig
from collections import defaultdict

from typing import Optional, Any, Mapping, Dict, Callable, Tuple, List, Sequence, Union, Iterable
from torchmetrics import Metric

from qip.utils.train import freeze_module, unfreeze_module
from qip.utils.misc import get_logger

from qip.typing import PATH
from lightning.pytorch.core.optimizer import do_nothing_closure

log = get_logger(__name__)


def _default_list():
    return []


def _default_none():
    return None


class TrainSystem(L.LightningModule):
    """
    Base class for training systems using PyTorch Lightning.

    This class extends PyTorch Lightning's `LightningModule` class and adds some utility methods.

    """

    VERSION = "unknown"

    @property
    def metrics(self) -> Dict[str, List[Metric]]:
        """Returns a dictionary of metrics used to track training progress."""
        if getattr(self, "_metrics", None) is None:
            self._metrics = defaultdict(_default_list)  # pickable
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Dict[str, Union[Metric, List[Metric]]]) -> None:
        if isinstance(metrics, Mapping):
            for metric_name, metric in metrics.items():
                if isinstance(metric, Metric):
                    self._metrics[metric_name] = [metric]
                elif isinstance(metric, Iterable):
                    self._metrics[metric_name] = list(metric)

    @property
    def post_processes(self) -> Dict[str, Optional[Callable]]:
        if getattr(self, "_post_processes", None) is None:
            self._post_processes = defaultdict(_default_none)  # pickable
        return self._post_processes

    @post_processes.setter
    def post_processes(self, post_processes: Dict[str, Optional[Callable]]):
        if isinstance(post_processes, Mapping):
            for task_name, post_process in post_processes.items():
                if isinstance(post_process, Callable):
                    self._post_processes[task_name] = post_process

    @property
    def optimizer_configs(self):
        return getattr(
            self,
            "_optimizer_configs",
            [
                {
                    "optimizer": {
                        "_target_": "torch.optim.AdamW",
                        "lr": 5e-5,
                        "modules": None,
                        "params": None,
                    },
                    "lr_scheduler": {
                        "scheduler": {
                            "_target_": "qip.modules.lr_scheduler.CosineAnnealingWarmUpRestartsWithDecay",
                            "T_0": self.trainer.estimated_stepping_batches,
                            "T_mult": None,
                            "eta_max": 5e-4,
                            "T_up": max(int(self.trainer.estimated_stepping_batches * 0.002), 1),
                            "gamma": 1.0,
                            "warmup_base_lr": 0,
                        },
                        "interval": "step",
                        # 'interval': 'epoch',
                        "frequency": 1,
                    },
                }
            ],
        )

    @optimizer_configs.setter
    def optimizer_configs(self, config: Optional[Mapping]) -> List[Mapping]:
        # default configs
        if config is None:
            log.info("Default optimizer_configs will be used.")
            if hasattr(self, "_optimizer_configs"):
                del self._optimizer_configs
        # single optimizer
        elif isinstance(config, Mapping):
            self._optimizer_configs = [config]

        # multiple optimizer
        elif isinstance(config, Iterable):
            config_list = list(config)
            self._optimizer_configs = []
            for config in config_list:
                if isinstance(config, Mapping):
                    self._optimizer_configs.append(config)

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers for the system.

        Returns:
            Union[Dict, Tuple[List[Optimizer]], Tuple[List[Optimizer], List[LRScheduler]]]:
                If a single optimizer is defined in optimizer_configs, returns the optimizer as a dictionary with the key
                'optimizer'. If multiple optimizers are defined, returns a tuple with a list of optimizers and optionally a
                list of learning rate schedulers if defined. If no parameters are included in the optimizers, raises a
                ValueError. If any excluded parameters have requires_grad == True, raises a warning.
        """
        if self.optimizer_configs is None:
            raise ValueError("optimizer_configs should be set before calling configure_optimizers")
        _included_params = set()
        _excluded_params = set([param_name for param_name, _ in self.named_parameters()])

        # single optimizers, lr_schedulers
        def _parse_single_optim_config(optim_config: Mapping, included_params: set, excluded_params: set):
            optim_config = dict(copy.deepcopy(optim_config))
            if "optimizer" not in optim_config:
                raise ValueError("'optimizer' key is not in optimizer_configs")
            optim_params: Optional[Union[str, List[str]]] = optim_config["optimizer"].pop("params", None)
            optim_modules: Optional[Union[str, List[str]]] = optim_config["optimizer"].pop("modules", None)

            if optim_params is None and optim_modules is None:
                # use all params in the current module
                params_to_use = list(self.parameters())
                included_params |= set([param_name for param_name, _ in self.named_parameters()])
                excluded_params -= set([param_name for param_name, _ in self.named_parameters()])
            else:
                params_to_use = list()
                if optim_modules is not None:
                    if isinstance(optim_modules, str):
                        optim_modules = [optim_modules]
                    _optim_module_params = [
                        (".".join([module_name, param_name]), params)
                        for module_name in optim_modules
                        for param_name, params in self.get_submodule(module_name).named_parameters()
                    ]

                    params_to_use.extend([param for param_name, param in _optim_module_params])
                    param_names_to_use = set([param_name for param_name, param in _optim_module_params])

                    if len(included_params.intersection(param_names_to_use)) > 0:
                        log.warn(
                            f"Parameters already included, it will be updated several times.: {sorted(list(param_names_to_use))}"
                        )
                    included_params |= param_names_to_use
                    excluded_params -= param_names_to_use

                if optim_params is not None:
                    if isinstance(optim_params, str):
                        optim_params = [optim_params]
                    _optim_module_params = [(param_name, self.get_parameter(param_name)) for param_name in optim_params]

                    params_to_use.extend([param for param_name, param in _optim_module_params])
                    param_names_to_use = set([param_name for param_name, param in _optim_module_params])
                    if len(included_params.intersection(param_names_to_use)) > 0:
                        log.warn(
                            f"Parameters already included, it will be updated several times.: {sorted(list(param_names_to_use))}"
                        )
                    included_params |= param_names_to_use
                    excluded_params -= param_names_to_use

                if len(params_to_use) == 0:
                    raise ValueError("Invalid params and modules for optimizers")

            single_optim_config = dict()
            optimizer_config: DictConfig = OmegaConf.create(optim_config)
            optimizer = hydra.utils.instantiate(optimizer_config.optimizer, params=params_to_use)
            single_optim_config.update({"optimizer": optimizer})

            # update lr_scheduler if it exists
            if getattr(optimizer_config, "lr_scheduler", None) is not None:
                lr_scheduler = hydra.utils.instantiate(optimizer_config.lr_scheduler.scheduler, optimizer=optimizer)
                lr_scheduler_config = hydra.utils.instantiate(
                    optimizer_config.lr_scheduler, scheduler=lr_scheduler, _convert_="all"
                )
                single_optim_config.update({"lr_scheduler": lr_scheduler_config})
            return single_optim_config

        return_configs = []
        for optim_config in self.optimizer_configs:
            return_configs.append(_parse_single_optim_config(optim_config, _included_params, _excluded_params))

        # check parameter setting
        if not (len(_included_params) > 0):
            raise ValueError("No parameters are included in optimizers")
        # warn excluded_params aren't set as requires_grad == False
        if len(_excluded_params) > 0:
            _warn_params = [
                param_name for param_name in _excluded_params if self.get_parameter(param_name).requires_grad == True
            ]
            if len(_warn_params) > 0:
                log.warn(
                    "parameters [{}] are set requires_grad == True but not included in optimizers: ".format(
                        ", ".join(_warn_params)
                    )
                )
        # return return_configs
        if len(return_configs) == 1:
            # single optimizer. use automatic_optimization
            return return_configs[0]
        else:
            # multiple optimizer. use manual_optimization
            self.automatic_optimization = False
            if "lr_scheduler" in return_configs[0]:
                return [v["optimizer"] for v in return_configs], [v["lr_scheduler"] for v in return_configs]
            else:
                return [v["optimizer"] for v in return_configs]

    def manual_optimization_step(self, train_loss, batch_idx) -> torch.Tensor:
        """
        Performs manual optimization steps a given batch during training
        for multiple optimizers and lr_schedulers.

        Args:
            train_loss (torch.Tensor): The train_loss for the current batch.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The train_loss for the current batch.
        """

        self.manual_backward(train_loss / float(self.trainer.accumulate_grad_batches))

        # optimization
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            optimizers = (
                self.optimizers(use_pl_optimizer=True)
                if isinstance(self.optimizers(), Sequence)
                else [self.optimizers()]
            )
            # clip gradient
            for optimizer in optimizers:
                self.configure_gradient_clipping(
                    optimizer, self.trainer.gradient_clip_val, self.trainer.gradient_clip_algorithm
                )

            # optimizer step
            for opt_idx, optimizer in enumerate(optimizers):
                if opt_idx == len(optimizers) - 1:
                    # count global_step on last optimizer_step
                    self.optimizer_step(self.current_epoch, batch_idx, optimizer=optimizer)
                else:
                    # call optimizer without global_step counting
                    self.trainer.strategy.optimizer_step(optimizer.optimizer, closure=do_nothing_closure)

            # zero_grad
            for optimizer in optimizers:
                self.optimizer_zero_grad(self.current_epoch, batch_idx, optimizer=optimizer)

            for lr_scheduler_config in self.trainer.lr_scheduler_configs:
                lr_scheduler = lr_scheduler_config.scheduler
                metric = (
                    self.trainer.callback_metrics.get("loss")
                    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                    else None
                )

                # learning_rate scheduling interval == 'step'
                if lr_scheduler_config.interval == "step":
                    if self.global_step % lr_scheduler_config.frequency == 0:
                        self.lr_scheduler_step(lr_scheduler, metric=metric)

                # learning_rate scheduling interval == 'epoch'
                elif lr_scheduler_config.interval == "epoch":
                    if (
                        self.trainer.is_last_batch
                        and (self.trainer.current_epoch + 1) % lr_scheduler_config.frequency == 0
                    ):
                        self.lr_scheduler_step(lr_scheduler, metric=metric)

        return train_loss

    @staticmethod
    def format_log_title(split, metric, task_name=None, dataset_name=None, subtask_idx=None):
        if task_name is None:
            return f"{split}/{metric}"
        if subtask_idx is not None:
            task_name = task_name + "_" + str(subtask_idx)
        if dataset_name is None:
            return f"{split}/{metric}/{task_name}"
        else:
            return f"{split}/{metric}/{task_name}.{dataset_name}"

    @staticmethod
    def get_masked_preds_and_target(
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple(List[torch.Tensor], List[torch.Tensor]):
        # preds: (*, 1), (*, T), targets: (*, T) or (*, ), mask: (*,) or (*, T)

        if mask is not None and isinstance(mask, torch.Tensor):
            if not isinstance(mask, torch.BoolTensor):
                if mask.max() > 1 or mask.min() < 0:
                    raise ValueError(
                        f"Invalid mask: the maqsk({mask}) is not a torch.BoolTensor and element has a value greater than 1."
                    )
                mask = mask.bool()

            # if valid mask
            if not mask.all():
                # MAE, binaryAUROC,BinaryAccuracy(sample_wise)...
                if preds.shape == target.shape == mask.shape:
                    # flatten or single target case
                    if mask.dim() == 1 or mask.dim() > 1 and mask.shape[-1] == 1:
                        return ([torch.masked_select(preds, mask)], [torch.masked_select(target, mask)])

                    # multi target
                    else:
                        preds_list = []
                        targets_list = []

                        for subtask_idx in range(mask.shape[-1]):
                            preds_list.append(torch.masked_select(preds[..., subtask_idx], mask[..., subtask_idx]))
                            targets_list.append(torch.masked_select(target[..., subtask_idx], mask[..., subtask_idx]))
                        return (preds_list, targets_list)
                # Acc, ...
                elif preds.shape != target.shape == mask.shape:
                    if preds[..., -1].numel() == mask.numel():
                        return (preds[mask.flatten(), ...], target[mask.flatten(), ...])
                else:
                    raise ValueError("Unexpected route")

            else:
                return ([preds], [target])
        else:
            return ([preds], [target])

    @staticmethod
    def compute_loss(loss_fn: Callable, pred: torch.Tensor, target: torch.Tensor):
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            # pred: (B, C), target: (B, )
            # TODO: Error when batch_size == 1 ?
            loss = loss_fn(pred, target.long().reshape(pred.shape[0]))
        elif isinstance(loss_fn, nn.BCEWithLogitsLoss):
            loss = loss_fn(pred, target.float())
        else:
            loss = loss_fn(pred, target)
        return loss

    def set_trainable_modules(
        self,
        trainable_modules: Optional[Union[str, List[str]]] = None,
        frozen_modules: Optional[Union[str, List[str]]] = None,
    ) -> bool:
        """
        Set trainable or frozen modules.

        This method allows you to set individual modules within the training system as trainable or frozen
        (not trainable). You can specify either the names of the modules to be made trainable, or the names
        of the modules to be frozen. If you specify both, a warning will be issued and the method will return
        False.

        Args:
            trainable_modules (str or list of str, optional): The names of the modules to be made trainable.
            frozen_modules (str or list of str, optional): The names of the modules to be frozen.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        if trainable_modules is not None and frozen_modules is not None:
            log.warn("Cannot set frozen_modules and trainable_module simultaneously")
            return False

        elif frozen_modules is not None:
            unfreeze_module(self)
            for module_name, module in self.named_modules():
                if module_name in frozen_modules:
                    log.info(f"frozen module: {module_name}")
                    freeze_module(module)

        elif trainable_modules is not None:
            freeze_module(self)
            for module_name, module in self.named_modules():
                if module_name in trainable_modules:
                    log.info(f"trainable module: {module_name}")
                    unfreeze_module(module)
        return True

    # @abstractmethod
    # def to_torch(self, file_path: Optional[PATH] = None, **kwargs: Any) -> Any:
    #     """
    #     Saves the model to a PyTorch checkpoint file or loads the model from a
    #     checkpoint file, depending on whether `file_path` is provided.

    #     Args:
    #         file_path: A string indicating the path to a PyTorch checkpoint file
    #             to save the model to or load the model from. If `None`, the method
    #             simply returns the current state of the model.
    #         **kwargs: Any. Additional arguments to pass.

    #     Returns:
    #         returns the saved object. Otherwise, returns None.
    #     """
    #     pass

    # def deploy_to_mlflow(
    #     self,
    #     version: Optional[str] = None,
    #     tracking_url: Optional[str] = None,
    #     eval_outputs: Optional[Dict[str, Any]] = None,
    # ):
    #     import mlflow
    #     from mlflow.utils.file_utils import TempDir
    #     import pandas as pd

    #     def _process_result_df(each_split_outputs) -> pd.DataFrame:
    #         result_dfs = []
    #         for outputs in each_split_outputs:
    #             split = list(outputs.keys())[0].split("/")[0]
    #             outputs = {"/".join(k.split("/")[1:]): v for k, v in outputs.items()}
    #             result_dfs.append(pd.DataFrame([(k, v) for k, v in outputs.items()], columns=["metric", split]))

    #         result_df = result_dfs[0]
    #         for to_merge_df in result_dfs[1:]:
    #             result_df: pd.DataFrame = pd.merge(result_df, to_merge_df, on="metric")

    #         result_df = result_df.sort_values("metric")
    #         result_df.to_csv("result.csv", index=False)
    #         return result_df

    #     # dict to save
    #     version = "unknown" if version is None else version
    #     # dict to save
    #     dict_to_upload = dict()
    #     dict_to_upload["version"] = version  # model version
    #     dict_to_upload["system_version"] = self.VERSION
    #     dict_to_upload["data_pipeline"] = next(iter(self.trainer.datamodule.data_pipelines.values()))
    #     for task_name in self.post_processes.keys():
    #         if task_name not in ["nabladft", "pubchemqc"]:
    #             self.post_processes[task_name] = torch.nn.Sigmoid()
    #     dict_to_upload["post_processes"] = self.post_processes
    #     dict_to_upload["available_tasks"] = list(self.task_heads.keys())

    #     # save to mlflow server
    #     mlflow.set_experiment("ADMET_MODEL")
    #     if tracking_url is None:
    #         tracking_url = os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.standigm.com")
    #     mlflow.set_tracking_uri(tracking_url)
    #     # run_id = mlflow.search_runs(filter_string="run_name='Production'").get('run_id', None)
    #     # print(run_id)

    #     with mlflow.start_run(tags={"version": version, "desc": "ADMET model"}) as run:
    #         # log model
    #         mlflow.pytorch.log_model(self, version)

    #         with TempDir() as tmp:
    #             local_dir = tmp.path()
    #             os.makedirs(local_dir, exist_ok=True)
    #             state_dict_path = os.path.join(local_dir, "infos.pt")
    #             torch.save(dict_to_upload, state_dict_path)
    #             if eval_outputs is not None:
    #                 eval_df = _process_result_df([eval_outputs])
    #                 eval_df.to_csv(os.path.join(local_dir, "evaluation.csv"), index=False)
    #             mlflow.log_artifacts(local_dir, version)

    #             # mlflow.pyfunc.log_model(artifact_path, data_path=dict_to_upload, python_model=DummyModel())
    #         for task_name in dict_to_upload["available_tasks"]:
    #             model_uri = "runs:/{run_id}/{version}".format(run_id=run.info.run_id, version=version)
    #             mlflow.register_model(model_uri, f"admet.{version}.{task_name}")
    #         # mlflow.pytorch.log_state_dict(dict_to_upload, artifact_path)
    #         artifact_uri = mlflow.get_artifact_uri(version)
    #         # mlflow.get_tracking_uri()

    #     return artifact_uri, run.info.run_id
