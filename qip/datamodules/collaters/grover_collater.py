import torch
import math
import numpy as np
from qip.typing import Data, Sequence
from qip.datamodules.collaters.default import DefaultCollater
from qip.datamodules.featurizers.ogb import OGBFeaturizer


class GroverCollater(DefaultCollater):
    def __init__(self, follow_batch = None, exclude_keys = None, max_num_nodes=None) -> None:
        super().__init__(
            follow_batch=[],
            exclude_keys=[
                "y",
                "y_mask",
            ],
        )
        self.max_num_nodes = max_num_nodes
        self.featurizer = OGBFeaturizer()
        self.masked_features = torch.LongTensor(self.featurizer.get_atom_feature_dims())
        self.masked_edge_attr = torch.LongTensor(self.featurizer.get_bond_feature_dims())

    def collate_data(self, batch: Sequence[Data]):
        percent = 0.15
        labels = []
        labels_mask = []
        for data in batch:
            x = data["x"]
            n_mask = math.ceil(x.shape[0]*percent)
            mask_index = np.random.permutation(x.shape[0])[:n_mask]
            mask = torch.zeros(x.shape[0]).type(torch.bool)
            mask[mask_index] = 1
            data["x"][mask] = self.masked_features

            y = data["y"].squeeze(0)
            y[~mask] = -100
            y_mask = data["y_mask"].squeeze(0)
            labels.append(y)
            labels_mask.append(y_mask)

            edge_mask = torch.cat([(data.edge_index[0] == i).nonzero() for i in mask_index]).reshape(-1)
            data["edge_attr"][edge_mask] = self.masked_edge_attr
        
        batch = super().collate_data(batch)
        batch["y"] = torch.cat(labels).type(torch.long)
        batch["y_mask"] = torch.cat(labels_mask)

        
        return batch
