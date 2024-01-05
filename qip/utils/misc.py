import logging
import warnings
from typing import Callable, List, Sequence, Any

import lightning as L
import rich
import rich.syntax
import rich.tree
import inspect
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from functools import lru_cache


def _type_error_print_format(var: Any, var_name: str, type: str):
    return f"Expected '{var_name}' to be a '{type}' instance, but got {type(var).__name__}."


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


@lru_cache(256)
def warn_once(logger: logging.Logger, msg: str):
    logger.warning(msg)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "system",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)

def finish(
    config: DictConfig,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    trainer: L.Trainer,
    callbacks: List[L.Callback],
    logger: List[Logger],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, WandbLogger):
            import wandb

            wandb.finish()


def get_func_signature(func):
    return tuple(inspect.signature(func).parameters.keys())
