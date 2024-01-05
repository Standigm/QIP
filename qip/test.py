import os.path as osp
import warnings
from typing import List, Optional, Sequence

import hydra
import pandas as pd
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import DictConfig, OmegaConf

from qip.utils.misc import empty, finish, get_logger

log = get_logger(__name__)


@rank_zero_only
def _process_result_df(each_split_outputs) -> pd.DataFrame:
    result_dfs = []
    for outputs in each_split_outputs:
        split = list(outputs.keys())[0].split("/")[0]
        outputs = {"/".join(k.split("/")[1:]): v for k, v in outputs.items()}
        result_dfs.append(pd.DataFrame([(k, v) for k, v in outputs.items()], columns=["metric", split]))
    result_df = result_dfs[0]
    for to_merge_df in result_dfs[1:]:
        result_df: pd.DataFrame = pd.merge(result_df, to_merge_df, on="metric")

    result_df = result_df.sort_values("metric")
    result_df.to_csv("result.csv", index=False)
    return result_df


def test(config: DictConfig):
    """Test trained system

    Args:
        config (DictConfig): system configuration

    Raises:
        ValueError: checkpoint_path should be provided and valid
    """
    # check checkpoint_path
    if config.checkpoint_path is None:
        if config.system.checkpoint_path is not None:
            warnings.warn(
                "config.checkpoint_path is not provided but config.system.checkpoint_path exists. Use config.system.checkpoint_path."
            )
            config.checkpoint_path = config.system.checkpoint_path

    elif not osp.isfile(config.checkpoint_path):
        raise ValueError("config.checkpoint_paht is invalid: {config.checkpoint_path}")

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

    # Test model
    outputs = []
    if config.get("validate", False):
        # Validate model
        valid_outputs = trainer.validate(model=system, datamodule=datamodule, ckpt_path=config.checkpoint_path)
        # outputs are redundent for multi-dataloaders, take first results
        valid_outputs = valid_outputs[0] if isinstance(valid_outputs, Sequence) else valid_outputs
        outputs.append(valid_outputs)

    eval_outputs = trainer.test(model=system, datamodule=datamodule, ckpt_path=config.checkpoint_path)
    # outputs are redundent for multi-dataloaders, take first results
    eval_outputs = eval_outputs[0] if isinstance(eval_outputs, Sequence) else eval_outputs
    outputs.append(eval_outputs)

    # save outputs
    result_df = _process_result_df(outputs)

    # Make sure everything closed properly
    finish(config, system, datamodule, trainer, callbacks, logger)

    # Print path of result
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Results: {osp.join(config.work_dir, 'result.csv')}")
