from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List, Union
import numpy as np
import hydra
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import random # for sequantial update

from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.fabric.utilities.cloud_io import _load as load
from lightning.fabric.utilities.cloud_io import _atomic_save as save
from omegaconf import DictConfig
from torch import ScriptModule
from torch.nn.modules.module import _IncompatibleKeys

from qip.datamodules.multi import MultiDataModule
from qip.systems.train_system import TrainSystem
from qip.typing import (
    PATH,
    Any,
    Callable,
    Dict,
    EncoderTaskOutput,
    Mapping,
    Optional,
    Tuple,
    Iterable,
    LONGTARGETMETRICS,
)
from qip.utils.misc import get_logger, warn_once, get_func_signature
from torch.distributed.fsdp.wrap import wrap

from lightning.pytorch.strategies import FSDPStrategy

log = get_logger(__name__)

class EncoderTrainSystem(TrainSystem):
    """
    A TrainSystem for training an encoder with multiple task_heads.

    Args:
        encoder_config (DictConfig): Configuration for the encoder module.
        task_head_configs (DictConfig): A dictionary containing the configuration for each task_head.
        checkpoint_path (Optional[str]): Path to a checkpoint to resume training from. Defaults to None.
        trainable_modules (Optional[List[str]]): A list of module names to make trainable. Defaults to None.
        frozen_modules (Optional[List[str]]): A list of module names to freeze during training. Defaults to None.
        optimizer_configs (Optional[Union[DictConfig, List[DictConfig]]]): Configuration(s) for the optimizer(s).
            If not provided, default configuration will be used.

    Raises:
        ValueError: If encoder_config does not contain 'module' or '_target_', or if any task_head
            configuration does not contain '_target_'.
        ValueError: If checkpoint_path is not a string or None.
    """

    VERSION = "v2.0.0"

    def __init__(
        self,
        encoder_config: DictConfig,
        task_head_configs: DictConfig,
        checkpoint_path: Optional[str] = None,
        trainable_modules: Optional[List[str]] = None,
        frozen_modules: Optional[List[str]] = None,
        optimizer_configs: Optional[Union[DictConfig, List[DictConfig]]] = None,
    ):
        super().__init__()

        # check args validity
        if not "module" in encoder_config or not "_target_" in encoder_config.module:
            raise ValueError("encoder_config doesn't contain module, module._target_")
        for task_name, head_config in task_head_configs.items():
            if "_target_" in head_config:
                raise ValueError(f"task_head.{task_name} doesn't contain _target_")

        if checkpoint_path is not None and not isinstance(checkpoint_path, PATH):
            raise ValueError(f"Invalid checkpoint_path: {checkpoint_path}")

        # set optimizer config
        self.optimizer_configs = optimizer_configs

        # save hyperparameters
        self.save_hyperparameters(logger=False)

        # load modules
        self.encoder: L.LightningModule = hydra.utils.instantiate(encoder_config.module)
        if encoder_config.get("state_path", None) is not None:
            log.info(f"encoder initialized from {encoder_config.get('state_path', None)}")
            encoder_state_dict = load(encoder_config.get("state_path"))
            self.encoder.load_state_dict(encoder_state_dict)

        # self.encoder_input_keys = self.encoder.example_input_array.keys()
        self.encoder_input_signature = get_func_signature(self.encoder.forward)

        # task_heads and their losses and metrics
        self.task_heads: nn.ModuleDict[str, L.LightningModule] = nn.ModuleDict()
        self.task_losses = nn.ModuleDict()
        self.task_loss_weights = dict()

        for task_name, head_config in task_head_configs.items():
            # task_head module
            # if module not exists skip this head
            if task_head := head_config.get("module", None):
                self.task_heads[task_name] = hydra.utils.instantiate(task_head)

                # task_head metrics
                if task_metric := head_config.get("metrics", None):
                    self.metrics[task_name] = hydra.utils.instantiate(task_metric)

                # task_head loss
                if task_loss := head_config.get("loss", None):
                    self.task_losses[task_name] = hydra.utils.instantiate(task_loss)

                # task_head loss weight
                task_weight = head_config.get("weight", None)

                if task_weight is not None:
                    self.task_loss_weights[task_name] = float(task_weight)
                else:
                    if task_name in self.task_losses:
                        self.task_loss_weights[task_name] = 1.0

                log.info(f"task_head: {task_name} created")

                # load state
                if head_config.get("state_path", None) is not None:
                    log.info(f"task_head: {task_name} initialized from {head_config.get('state_path')}")
                    task_head_state_dict = load(head_config.get("state_path"))
                    self.task_heads[task_name].load_state_dict(task_head_state_dict)

                # post_process
                if post_process := head_config.get("post_process", None):
                    self.post_processes[task_name] = hydra.utils.instantiate(post_process)
            else:
                warnings.warn(f"task_head ({task_name}) module is not defined")

        # resume checkpoint if provided
        if checkpoint_path is not None and isinstance(checkpoint_path, PATH):
            #prev_params = [(n,v.detach().clone()) for n,v in self.named_parameters()]
            checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(checkpoint_dict["state_dict"], strict=False)
            #new_params = [(n,v.detach().clone()) for n,v in self.named_parameters()]
            #load_params = all([(a[1] != b[1]).any() for a,b in zip(prev_params, new_params)])
            #log.info(f"load state dict: {load_params}")

        # set module
        self.set_trainable_modules(trainable_modules=trainable_modules, frozen_modules=frozen_modules)
        
        self.automatic_optimization = True # Turn on for sequential update

    @property
    def encoder_input_signature(self) -> Tuple[str]:
        return getattr(self, "_encoder_input_signature", tuple())

    @encoder_input_signature.setter
    def encoder_input_signature(self, value):
        if isinstance(value, Iterable):
            value = tuple(value)
            if not isinstance(value[0], str):
                raise ValueError(f"Ivalid signature type: {type(value[0])}")
            self._encoder_input_signature = value
        else:
            raise ValueError(f"Invalud encoder_input_signature: {value}")

    @property
    def task_names(self) -> Tuple[str]:
        """Return the names of the tasks that the system is training on.

        If a MultiDataModule is used to train the system, the task names are obtained
        from the datamodule. If a task name is found in the datamodule that does not
        correspond to any task head in the system, a ValueError is raised.

        Returns:
            A tuple of strings representing the names of the tasks.
        """
        if getattr(self, "trainer", None):
            if isinstance(getattr(self.trainer, "datamodule", None), MultiDataModule):
                # get task names from datamodule
                unexpected_task_names = set(self.trainer.datamodule.task_names) - set(self.task_heads.keys())
                if len(unexpected_task_names) != 0:
                    raise ValueError(f"unexpected_task_names: {list(unexpected_task_names)}")
                return tuple(self.trainer.datamodule.task_names)
        else:
            # task names
            return tuple(self.task_heads.keys())

    @property
    def dataset_names(self) -> Tuple[str]:
        """Return the names of the datasets used to train the system.

        If a MultiDataModule is used to train the system, the dataset names are obtained
        from the datamodule. If a MultiDataModule is not used, this property returns None.

        Returns:
            A tuple of strings representing the names of the datasets, or None if the
            system is not using a MultiDataModule.
        """
        if getattr(self, "trainer", None):
            if isinstance(getattr(self.trainer, "datamodule", None), MultiDataModule):
                # get task names from datamodule
                return tuple(self.trainer.datamodule.dataset_names)
        else:
            # MultiDataModule is not connected. so dataset_names are not defined
            return tuple()

    def configure_sharded_model(self):
        """
        Configures the sharded model by wrapping the encoder and each task head with a sharded model wrapper.

        This function first attempts to call the `configure_sharded_model` method on the encoder, if it exists, to configure
        the encoder for sharded training. Then, it wraps the encoder with a sharded model wrapper using the `wrap` function.

        Next, for each task head in the `task_heads` ModuleDict, it attempts to call the `configure_sharded_model` method
        on the task head, if it exists, to configure the task head for sharded training. Then, it wraps the task head with
        a sharded model wrapper using the `wrap` function. The wrapped task head is stored back in the `task_heads`
        ModuleDict with the same task name.

        Note: The `wrap` function is imported from torch.distributed.fsdp.wrap .
        """
        # wrap encoder
        getattr(self.encoder, "configure_sharded_model", lambda: None)()
        self.encoder = wrap(self.encoder)

        # wrap each task_head
        for task_name in self.task_heads.keys():
            getattr(self.task_heads[task_name], "configure_sharded_model", lambda: None)()
            self.task_heads[task_name] = wrap(self.task_heads[task_name])

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ):
        """
        Configures gradient clipping for the model's optimizer.

        Args:
            optimizer (Optimizer): The optimizer to configure gradient clipping for.
            gradient_clip_val (Optional[Union[int, float]]): The maximum value for gradient clipping.
                If None, no gradient clipping will be applied. Default is None.
            gradient_clip_algorithm (Optional[str]): The algorithm to use for gradient clipping.
                Only applicable if gradient_clip_val is not None. Must be one of ['norm', 'value', None].
                If None, the default PyTorch behavior will be used. Default is None.

        Returns:
            None

        Raises:
            TypeError: If optimizer is not an instance of torch.optim.Optimizer.
            ValueError: If gradient_clip_algorithm is not None and not one of ['norm', None].

        Note:
            - If the model's trainer strategy is an instance of FSDPStrategy, the gradient clipping
            will be applied using FullyShardedDataParallel.clip_grad_norm_() function.
            - If the model's trainer strategy is not an instance of FSDPStrategy, the gradient clipping
            will be applied using the superclass's configure_gradient_clipping() function.
        """
        if gradient_clip_algorithm is not None and gradient_clip_algorithm not in ["norm", "value", None]:
            raise ValueError("gradient_clip_algorithm must be one of ['norm', 'value', None]")

        # if FSDP use FullyShardedDataParallel.clip_grad_norm_
        if isinstance(self.trainer.strategy, FSDPStrategy):
            if gradient_clip_val is not None and gradient_clip_algorithm != "value":
                self.encoder.clip_grad_norm_(gradient_clip_val)
                # self.task_heads.clip_grad_norm_(gradient_clip_val)
                for task_name in self.task_heads.keys():
                    self.task_heads[task_name].clip_grad_norm_(gradient_clip_val)
        else:
            super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this system and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        If there are unknown task names in the state dictionary (i.e., task names not present in `self.task_names`),
        a warning will be issued and those task heads will not be updated.
        If there are task names in `self.task_names` that are not present in the state dictionary,
        the current parameter values for those task heads will be used instead.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        Note:
            - If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
            - The `state_dict` parameter is expected to have keys that follow the naming convention
            "task_heads.<task_name>.<parameter_name>", where <task_name> is the name of a specific
            task and <parameter_name> is the name of a parameter in the model. For example,
            "task_heads.task1.fc.weight" would correspond to the weight parameter of the fully
            connected layer in the "task1" task head.
        """

        """
        Loads the state_dict for the system. The state dictionary contains the parameter values
        of the system.
        Args:
            state_dict (Mapping[str, Any]): The state dictionary to load. The keys in the dictionary
                should correspond to the names of the parameters in the model, and the values should
                be the parameter values.
            strict (bool, optional): If True, raises an error if the keys in the state dictionary do
                not match the names of the parameters in the model exactly. If False, allows for
                partial matches. Default is True.

        Returns:
            None


        Example usage:
        ```python
        system = EncoderTrainSystem(...)
        state_dict = torch.load('encoder_train_system.pt')
        system.load_state_dict(state_dict)
        ```
        """
        missing_keys = []
        unexpected_keys = []

        # split encoder state_dict and leftovers
        encoder_state_dict = {
            ".".join(param_name.split(".")[1:]): param
            for param_name, param in state_dict.items()
            if param_name.startswith("encoder")
        }
        left_state_dict = {
            param_name: param for param_name, param in state_dict.items() if not param_name.startswith("encoder")
        }
        # update encoder state_dict by calling encoder.load_state_dict
        encoder_missing_keys, encoder_unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=strict)

        missing_keys.extend(["encoder." + param_name for param_name in encoder_missing_keys])
        unexpected_keys.extend(["encoder." + param_name for param_name in encoder_unexpected_keys])

        updated_encoder_state_dict = {
            "encoder." + param_name: param for param_name, param in self.encoder.state_dict().items()
        }

        state_dict = {}
        state_dict.update(updated_encoder_state_dict)
        state_dict.update(left_state_dict)

        # update task_heads
        state_dict_task_names = set([v.split(".")[1] for v in state_dict.keys() if v.startswith("task_heads")])
        unknown_task_names = set(state_dict_task_names) - set(self.task_heads.keys())
        not_updated_task_names = set(self.task_heads.keys()) - set(state_dict_task_names)
        current_state_dict = self.state_dict()

        if len(unknown_task_names) > 0:
            for unknown_task_name in list(unknown_task_names):
                warn_once(log, f"state(task_heads.{unknown_task_name}) will not be updated")

        if len(not_updated_task_names) > 0:
            not_updated_param_names = [
                param_name
                for param_name in current_state_dict.keys()
                if param_name.startswith("task_heads") and param_name.split(".")[1] in not_updated_task_names
            ]
            for param_name in not_updated_param_names:
                state_dict[param_name] = current_state_dict[param_name]  # use currently initialized params

        # here encoder updated twice with the same params..
        all_missing_keys, all_unexpected_keys = super().load_state_dict(state_dict, False)

        missing_keys.extend(all_missing_keys)
        unexpected_keys.extend(all_unexpected_keys)

        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def on_fit_start(self) -> None:
        # check datamodule is MultiDataModule for train_step
        if isinstance(getattr(self.trainer, "datamodule", None), MultiDataModule):
            # check task_names
            invalid_task_names = [
                task_name for task_name in self.trainer.datamodule.task_names if task_name not in self.task_names
            ]
            if len(invalid_task_names) > 0:
                raise ValueError(f"Invalid task_names: {invalid_task_names}")
        else:
            raise ValueError("self.trainer.datamodule is not MultiDataModule")

        return super().on_fit_start()

    def forward(self, data: Mapping, dataloader_idx: Optional[int] = None) -> OrderedDict[str, Any]:
        # encoder_outputs = {"dataset_name1": encoder_output}
        # task_outputs = {"dataset_name1": {"task_name1": task1_output, "task_name2": task2_output, ...},
        #                 "dataset_name2": {"task_name1": task1_output, "task_name3": task3_output, ...},}
        task_outputs = OrderedDict()
        encoder_outputs = OrderedDict()
  
        # get parameter names
        if dataloader_idx is None:
            if self.dataset_names is not None:
                # multi-task
                # combined loader
                for task_name, dataset_name in zip(self.task_names, self.dataset_names):
                    each_data_dict = data[dataset_name]
                    # add task_name as input to use
                    each_data_dict["task_name"] = task_name
                    inputs_dict = {
                        key: each_data_dict[key] for key in self.encoder_input_signature if key in each_data_dict
                    }

                    encoder_outputs[dataset_name] = self.encoder(**inputs_dict)
                    head_inputs = (
                        encoder_outputs[dataset_name]
                        if isinstance(encoder_outputs[dataset_name], tuple)
                        else (encoder_outputs[dataset_name],)
                    )
                    task_output_dict = task_outputs.get(dataset_name, dict())
                    task_output_dict.update({task_name: self.task_heads[task_name](*head_inputs)})
                    task_outputs[dataset_name] = task_output_dict
            else:
                raise RuntimeError("dataset_names not defined. MultiDataModule is not connected to the system.")

            return task_outputs, encoder_outputs
        else:
            # single-task
            # specific task loader with dataloader_idx
            # task_names are infered from datamodule
            if self.dataset_names is None:
                raise RuntimeError("dataset_names not defined. MultiDataModule is not connected to the system.")
            dataset_name = self.dataset_names[dataloader_idx]
            task_name = self.task_names[dataloader_idx]

            data["task_name"] = task_name
            encoder_input_dict = {key: data[key] for key in self.encoder_input_signature if key in data}

            encoder_outputs[dataset_name] = self.encoder(**encoder_input_dict)
            head_inputs = (
                encoder_outputs[dataset_name]
                if isinstance(encoder_outputs[dataset_name], tuple)
                else (encoder_outputs[dataset_name],)
            )
            task_output_dict = {task_name: self.task_heads[task_name](*head_inputs)}
            task_outputs[dataset_name] = task_output_dict
            return task_outputs, encoder_outputs

    def training_step(self, batch, batch_idx, *args, **kwargs) -> Any:
        task_outputs, encoder_outputs = self(batch)

        total_train_loss = 0.0
        for dataset_name, task_name in zip(self.dataset_names, self.task_names):
            loss_fn = self.task_losses[task_name]
            y = batch[dataset_name].y  # batch from CombinedLoader
            mask = getattr(batch[dataset_name], "y_mask", ~torch.isnan(y))
            preds_list, target_list = self.get_masked_preds_and_target(task_outputs[dataset_name][task_name], y, mask)

            each_train_loss = 0.0
            for preds, target in zip(*(preds_list, target_list)):
                # average over subtasks
                each_train_loss += self.compute_loss(loss_fn, preds, target) / float(len(preds_list))

            # log each train_loss without loss weight
            self.log(
                self.format_log_title("train", "loss", task_name, dataset_name),
                each_train_loss,
                prog_bar=False,
                sync_dist=True,
            )
            # add each_train_loss with corresponding task_weight
            total_train_loss += self.task_loss_weights[task_name] * each_train_loss
        # weighted total loss over multiple tasks(datasets)
        self.log(
            self.format_log_title("train", "loss"),
            total_train_loss,
            prog_bar=True,
            sync_dist=True,
        )

        # manual optimization
        if not self.automatic_optimization:
            self.manual_optimization_step(total_train_loss, batch_idx)

        return total_train_loss

    def _shared_evaluation_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        task_name = self.task_names[dataloader_idx]
        dataset_name = self.dataset_names[dataloader_idx]
        task_outputs, encoder_outputs = self(batch, dataloader_idx)
        loss_fn = self.task_losses[task_name]
        y = batch.y  # batch from Data
        try:
            mask = getattr(batch, "y_mask", ~torch.isnan(y))
        except:
            mask = None

        each_eval_loss = 0.0

        def _adapt_metric_input(metric: torchmetrics.Metric, preds: torch.Tensor, target: torch.Tensor):
            batch_size = preds.shape[0]
            preds, target = preds.to(metric.device).squeeze(), target.to(metric.device).squeeze()
            if isinstance(metric, LONGTARGETMETRICS):
                target = target.long()

            # if batch_size == 1, squeeze function remove batch dimension. restore batch dimension.
            if batch_size == 1:
                preds = preds.unsqueeze(0)
                target = target.unsqueeze(0)
            return preds, target

        preds_list, target_list = self.get_masked_preds_and_target(task_outputs[dataset_name][task_name], y, mask)
        for preds, target in zip(*(preds_list, target_list)):
            each_eval_loss += self.compute_loss(loss_fn, preds, target) / float(len(preds_list))
            for metric in self._metrics_per_dataset[dataset_name]:
                try:
                    _preds, _target = _adapt_metric_input(metric, preds, target)
                    metric.update(_preds, _target)
                except Exception as e:
                    warn_once(log, f"Skip logging {metric} {task_name}-{dataset_name}:\n{e}")

        self._loss_metric_per_dataset[dataset_name].update(each_eval_loss)

        return each_eval_loss

    def _on_shared_compute_evaluation_end(self, split: str):
        total_eval_loss = 0.0
        eval_outputs = dict()

        for dataset_name, task_name in zip(self.dataset_names, self.task_names):
            each_dataset_loss = self._loss_metric_per_dataset[dataset_name].compute()
            eval_outputs[self.format_log_title(split, "loss", dataset_name, task_name)] = each_dataset_loss
            total_eval_loss += self.task_loss_weights[task_name] * each_dataset_loss
            metrics = self._metrics_per_dataset[dataset_name]

            for metric in metrics:
                metric_name = metric.__class__.__name__
                try:
                    metric_val = metric.compute().cpu()
                    if metric_val.numel() == 1:
                        eval_outputs[self.format_log_title(split, metric_name, task_name, dataset_name)] = metric_val
                    elif metric_val.dim() == 1 and metric_val.numel() > 1:
                        for subtask_idx, val in enumerate(metric_val):
                            eval_outputs[
                                self.format_log_title(split, metric_name, task_name, dataset_name, subtask_idx)
                            ] = val
                    # TODO: add other metrics such as confusion matrix.
                except Exception as e:
                    warn_once(log, f"Skip logging {metric} {task_name}-{dataset_name}:\n{e}")

        # log evaluation results
        self.log_dict(
            eval_outputs,
            prog_bar=False,
            sync_dist=True,
        )

        keys = np.array([key.split('/')[1] for key in list(eval_outputs.keys())[1::2]])

        values = [val.detach().cpu().numpy() for val in list(eval_outputs.values())[1::2]]

        o = np.ones(len(keys))
        o[np.where(keys=='MeanAbsoluteError')[0]] = -1
        o[np.where(keys=='MeanSquaredError')[0]] = -1

        score = np.sum(o*values)

        self.log(
            self.format_log_title(split, "score"),
            score,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            self.format_log_title(split, "loss"),
            total_eval_loss,
            prog_bar=True,
            sync_dist=True,
        )

    def _on_shared_evaluation_start(self):
        self._metrics_per_dataset = {
            dataset_name: [metric.clone().to(self.device) for metric in self.metrics[task_name]]
            for dataset_name, task_name in zip(self.dataset_names, self.task_names)
        }
        self._loss_metric_per_dataset = {
            dataset_name: torchmetrics.MeanMetric().to(self.device)
            for dataset_name, task_name in zip(self.dataset_names, self.task_names)
        }

    def _on_shared_evaluation_end(self):
        # clear metrics
        for _, metrics in self._metrics_per_dataset.items():
            for metric in metrics:
                metric.reset()
        self._metrics_per_dataset.clear()

        for _, metric in self._loss_metric_per_dataset.items():
            metric.reset()
        self._loss_metric_per_dataset.clear()

    # validation
    def on_validation_start(self):
        return self._on_shared_evaluation_start()

    def on_validation_end(self):
        return self._on_shared_evaluation_end()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = 0) -> Optional[STEP_OUTPUT]:
        outputs = self._shared_evaluation_step(batch, batch_idx, dataloader_idx)
        return outputs

    def on_validation_epoch_end(self):
        self._on_shared_compute_evaluation_end("val")

    # test steps
    def on_test_start(self):
        return self._on_shared_evaluation_start()

    def on_test_end(self):
        return self._on_shared_evaluation_end()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = 0) -> Optional[STEP_OUTPUT]:
        outputs = self._shared_evaluation_step(batch, batch_idx, dataloader_idx)
        return outputs

    def on_test_epoch_end(self):
        self._on_shared_compute_evaluation_end("test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        # predict over all task
        encoder_outputs = dict()
        task_outputs = dict()
        for dataset_name in self.dataset_names:
            each_data_dict = batch
            # add task_name as input to use
            for task_name in self.task_heads.keys():
                each_data_dict["task_name"] = task_name
                inputs_dict = {
                    key: each_data_dict[key] for key in self.encoder_input_signature if key in each_data_dict
                }

                encoder_outputs[f"{task_name}.{dataset_name}"] = self.encoder(**inputs_dict)
                head_inputs = (
                    encoder_outputs[f"{task_name}.{dataset_name}"]
                    if isinstance(encoder_outputs[f"{task_name}.{dataset_name}"], tuple)
                    else (encoder_outputs[f"{task_name}.{dataset_name}"],)
                )
                task_outputs[f"{task_name}.{dataset_name}"] = self.task_heads[task_name](*head_inputs)
                if self.post_processes.get(task_name, None) is not None:
                    task_outputs[f"{task_name}.{dataset_name}"] = self.post_processes[task_name](
                        task_outputs[f"{task_name}.{dataset_name}"]
                    )

        return EncoderTaskOutput(encoder_outputs=encoder_outputs, task_outputs=task_outputs, batch_idx=batch_idx)

    # def to_torch(self, file_path: Optional[PATH] = None, **kwargs: Any) -> Dict:
    #     file_path = Path(file_path) if file_path else None
    #     outputs = dict()

    #     # encoder
    #     outputs["encoder"] = dict(state_dict=self.encoder.state_dict(), encoder_configs=self.hparams.encoder_config)

    #     # task_heads
    #     outputs["task_heads"] = dict()
    #     for task_name in self.task_names:
    #         outputs["task_heads"][task_name] = dict(
    #             state_dict=self.task_heads[task_name].state_dict(),
    #             task_head_config=self.hparams.task_head_configs[task_name],
    #         )

    #     if file_path is not None:
    #         file_path = Path(file_path)
    #         os.makedirs(file_path.parent, exist_ok=True)
    #         save(outputs, file_path)
    #     return outputs

    # def to_torchscript(
    #     self, file_path: Optional[PATH] = None, method: Optional[str] = "script", **kwargs: Any
    # ) -> Union[ScriptModule, Dict[str, ScriptModule]]:
    #     outputs = dict()

    #     # encoder
    #     scripted_encoder = self.encoder.to_torchscript(file_path=file_path, **kwargs)
    #     outputs["encoder"] = scripted_encoder

    #     # task_heads
    #     outputs["task_heads"] = dict()
    #     for task_name in self.task_names:
    #         scripted_task_head = self.task_heads[task_name].to_torchscript(None, method, None, **kwargs)
    #         outputs["task_heads"][task_name] = scripted_task_head

    #     if file_path is not None:
    #         file_path = Path(file_path)
    #         os.makedirs(file_path.parent, exist_ok=True)
    #         save(outputs, file_path)
    #     return outputs

    # def to_onnx(self, file_dir: PATH, **kwargs: Any) -> None:
    #     file_dir = Path(file_dir)
    #     encoder_file_path = file_dir / "onnx" / "encoder.pt"
    #     os.makedirs(encoder_file_path.parent, exist_ok=True)
    #     os.makedirs(encoder_file_path.parent / "task_heads", exist_ok=True)
    #     self.encoder.to_onnx(encoder_file_path, None, **kwargs)
    #     for task_name, task_head_path in [
    #         (task_name, file_dir / "onnx" / "task_heads" / f"{task_name}.pt") for task_name in self.task_names
    #     ]:
    #         self.task_heads[task_name].to_onnx(task_head_path, None, **kwargs)
