from typing import Iterable

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import (
    CSVLogger,
    TensorBoardLogger,
    WandbLogger,
)


def get_csv_logger(trainer: Trainer) -> CSVLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use csvlogger callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, CSVLogger):
        return trainer.logger

    if isinstance(trainer.logger, Iterable):
        for logger in trainer.logger:
            if isinstance(logger, CSVLogger):
                return logger

    raise Exception("You are using wandb related callback, but WandbLogger was not found for some reason...")


def get_tensorboard_logger(trainer: Trainer) -> TensorBoardLogger:
    """Safely get TensorBoardLogger logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use tensorboard callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, TensorBoardLogger):
        return trainer.logger

    if isinstance(trainer.logger, Iterable):
        for logger in trainer.logger:
            if isinstance(logger, TensorBoardLogger):
                return logger

    raise Exception(
        "You are using tensorboard related callback, but TensorBoardLogger was not found for some reason..."
    )


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, Iterable):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception("You are using wandb related callback, but WandbLogger was not found for some reason...")
