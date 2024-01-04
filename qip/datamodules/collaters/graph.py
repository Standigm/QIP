import torch

from admet_prediction.typing import Data, Sequence
from admet_prediction.datamodules.collaters.default import DefaultCollater


class GraphDenseCollater(DefaultCollater):
    def __init__(self, max_num_nodes=None, offset=128) -> None:
        super().__init__(
            follow_batch=[],
            exclude_keys=[
                "x",
                "mask",
                "hop",
                "edge_index",
                "edge_attr",
                "rwse",
            ],
        )

        self.offset = offset
        self.max_num_nodes = max_num_nodes

    def convert_to_single_emb(self, x):
        feature_num = x.size(1) if len(x.size()) > 1 else 1
        feature_offset = 1 + torch.arange(0, feature_num * self.offset, self.offset, dtype=torch.long)
        x = x + feature_offset
        return x

    def collate_data(self, batch: Sequence[Data]):
        # filterout if num_nodes larger than max_num_nodes
        if self.max_num_nodes is not None:
            batch = [b for b in batch if b["x"].shape[0] <= self.max_num_nodes]

        node = [self.convert_to_single_emb(b["x"]) for b in batch]
        hop = [b["hop"] for b in batch]

        if len(hop[0].shape) == 1:
            hop = [d.view(n.shape[0], n.shape[0]) for d, n in zip(hop, node)]

        edge_index = [b["edge_index"] for b in batch]
        edge_attr = [b["edge_attr"] for b in batch]
        max_num_node = max(d.shape[0] for d in hop)

        gathered_node = []
        gathered_hop = []
        gathered_edge_matrix = []
        mask = []

        for n, d, ei, ea in zip(node, hop, edge_index, edge_attr):
            m = torch.zeros(max_num_node, dtype=torch.bool)
            m[n.shape[0] :] = 1

            new_n = -torch.ones((max_num_node, n.shape[1]), dtype=torch.long)
            new_n[: n.shape[0]] = n

            new_d = -torch.ones((max_num_node, max_num_node), dtype=torch.long)
            new_d[: d.shape[0], : d.shape[1]] = d
            new_d[new_d < 0] = -1

            new_ea = -torch.ones((max_num_node, max_num_node), dtype=torch.long)
            new_ea[ei[0], ei[1]] = ea

            mask.append(m)
            gathered_node.append(new_n)
            gathered_hop.append(new_d)
            gathered_edge_matrix.append(new_ea)

        mask = torch.stack(mask, dim=0)
        gathered_node = torch.stack(gathered_node, dim=0)
        gathered_hop = torch.stack(gathered_hop, dim=0)
        gathered_edge_matrix = torch.stack(gathered_edge_matrix, dim=0)

        batch = super().collate_data(batch)
        batch["x"] = gathered_node
        batch["mask"] = mask
        batch["hop"] = gathered_hop
        batch["edge_matrix"] = gathered_edge_matrix
        return batch
