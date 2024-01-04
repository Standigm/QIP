from typing import Any, Optional

import lightning as L
from lightning.pytorch.callbacks import Callback
from admet_prediction.typing import Data


class GraphSizeTracker(Callback):
    def on_train_batch_start(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if isinstance(batch, Data):
            max_num_nodes = batch.x.shape[1]
            pl_module.log("train/max_nodes", float(max_num_nodes), on_step=True, sync_dist=True, sync_dist_op="max")

    def on_validation_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if isinstance(batch, Data):
            max_num_nodes = batch.x.shape[1]
            pl_module.log("val/max_nodes", float(max_num_nodes), on_step=True, sync_dist=True, sync_dist_op="max")

    def on_test_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if isinstance(batch, Data):
            max_num_nodes = batch.x.shape[1]
            pl_module.log("test/max_nodes", float(max_num_nodes), on_step=True, sync_dist=True, sync_dist_op="max")
