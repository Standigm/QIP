from typing import Dict, Union

import lightning as L
from lightning.pytorch.callbacks import RichProgressBar


class PrecisedRichProgressBar(RichProgressBar):
    def get_metrics(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> Dict[str, Union[int, str]]:
        items = super().get_metrics(trainer, pl_module)
        for key in items.keys():
            if "lr" in key or "learning_rate" in key:
                items[key] = f"{float(items[key]):.3e}"
        return items
