from typing import List, Optional

import hydra
from lightning.pytorch import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import DictConfig, OmegaConf

from qip.utils.misc import empty, finish, get_logger

log = get_logger(__name__)


def train(config: DictConfig):
    """Train system

    Args:
        config (DictConfig): configuration for training
    """
    # seed random
    seed_everything(config.get("seed", None), workers=True)

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

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=system, datamodule=datamodule, ckpt_path=getattr(system, "checkpoint_path", None))

    # test best model
    trainer.test(model=system, datamodule=datamodule, ckpt_path="best")

    # Make sure everything closed properly
    finish(config, system, datamodule, trainer, callbacks, logger)

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt: {trainer.checkpoint_callback.best_model_path}")
