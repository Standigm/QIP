import os.path as osp
import warnings
from typing import List, Optional, Sequence

import hydra
import pandas as pd
import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import DictConfig, OmegaConf

from qip.utils.misc import empty, finish, get_logger

log = get_logger(__name__)


def inference(config: DictConfig):
    """inference model.
    batch_size should be set to 1.

    Args:
        config (DictConfig): system configuration

    Raises:
        ValueError: checkpoint_path should be provided and valid.
    """
    # check checkpoint_path
    if config.checkpoint_path is None:
        if config.system.checkpoint_path is not None:
            warnings.warn(
                "config.checkpoint_path is not provided but config.system.checkpoint_path exists. Use config.system.checkpoint_path."
            )
            config.checkpoint_path = config.system.checkpoint_path

        if not osp.isfile(config.checkpoint_path):
            raise ValueError("config.checkpoint_paht is invalid: {config.checkpoint_path}")

    # set batch_size to 1
    if config.datamodule.batch_size != 1:
        warnings.warn("batch_size should be 1. Set config.datamodule.batch_size = 1")
        config.datamodule.batch_size = 1

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, _recursive_=False)

    # Init lightning model
    log.info(f"Instantiating system <{config.system._target_}>")
    system: LightningModule = hydra.utils.instantiate(config.system, _recursive_=False)

    # log model summary
    log.info("Model Summary:\n" + str(ModelSummary(system)))

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if cb_conf is not None and "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if lg_conf is not None and "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    # log hyperparams
    config_dict = OmegaConf.to_container(config, resolve=True)
    for logger_obj in trainer.loggers:
        logger_obj.log_hyperparams(config_dict)
        # disable logging any more hyperparameters for all loggers
        # this is just a trick to prevent trainer from logging hparams of model
        logger_obj.log_hyperparams = empty  # disable log_hyperparams

    # inference model
    # TODO: save output with different names
    predict_outputs = trainer.predict(model=system, datamodule=datamodule, ckpt_path=config.checkpoint_path)
    torch.save(predict_outputs, osp.join(config.work_dir, "predict_outputs.pt"))

    # Make sure everything closed properly
    finish(config, system, datamodule, trainer, callbacks, logger)

    # Print output path
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Results: {osp.join(config.work_dir, 'predict_outputs.pt')}")
