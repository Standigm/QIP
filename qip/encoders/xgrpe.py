# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
from typing import Any, Dict, List, Mapping, Optional

import lightning as L
import torch
import torch.nn as nn

from torch.distributed.fsdp.wrap import wrap
from admet_prediction.encoders.grpe import GRPEEncoder
from admet_prediction.modules.xai import xnn


class XGRPEEncoder(GRPEEncoder):
    def __init__(
        self,
        num_layer: int = 12,
        d_model: int = 768,
        nhead: int = 32,
        dim_feedforward: int = 768,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_hop: int = 5,
        num_node_type: int = 11,  # number of node_types
        node_offset: int = 128,
        num_edge_type: int = 3,
        edge_offset: int = 8,
        perturb_noise: float = 0.0,
        norm_mode: str = "pre",
        use_independent_token: bool = False,
    ):
        super().__init__(
            num_layer,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            attention_dropout,
            max_hop,
            num_node_type,  # number of node_types
            node_offset,
            num_edge_type,
            edge_offset,
            perturb_noise,
            norm_mode,
            use_independent_token,
        )

    def _setup_task_token(self) -> None:
        self.task_token = nn.Embedding(1, self.d_model)

    def _setup_modules(self) -> None:
        if self.hparams.num_node_type < 0:
            # num_feat * offset
            self.node_emb = xnn.Linear(-self.num_node_type * self.hparams.node_offset, self.hparams.d_model)
        else:
            # num_feat * offset
            self.node_emb = nn.Embedding(
                self.num_node_type * self.hparams.node_offset + 1, self.hparams.d_model, padding_idx=-1
            )

        all_edge_type = self.hparams.num_edge_type * self.hparams.edge_offset  # <= num_feat * offset

        self.TASK_DISTANCE = self.hparams.max_hop + 1
        self.UNKNOWN_DISTANCE = self.hparams.max_hop + 2  # padding idx

        self.TASK_EDGE = all_edge_type + 1
        self.SELF_EDGE = all_edge_type + 2
        self.NO_EDGE = all_edge_type + 3
        self.UNKNOWN_EDGE = all_edge_type + 4  # padding idx

        # query_hop_emb: Query Structure Embedding
        # query_edge_emb: Query Edge Embedding
        # key_hop_emb: Key Structure Embedding
        # key_edge_emb: Key Edge Embedding
        # value_hop_emb: Value Structure Embedding
        # value_edge_emb: Value Edge Embedding

        # repeat same module -> jitable
        self.hop_emb_size = self.hparams.max_hop + 3
        self.edge_emb_size = all_edge_type + 5

        self.query_hop_emb = nn.ModuleList()
        self.query_edge_emb = nn.ModuleList()
        self.key_hop_emb = nn.ModuleList()
        self.key_edge_emb = nn.ModuleList()
        self.value_hop_emb = nn.ModuleList()
        self.value_edge_emb = nn.ModuleList()

        for _ in range(self.num_layer if self.use_independent_token else 1):
            self.query_hop_emb.append(nn.Embedding(self.hop_emb_size, self.hparams.d_model, padding_idx=-1))
            self.query_edge_emb.append(nn.Embedding(self.edge_emb_size, self.hparams.d_model, padding_idx=-1))
            self.key_hop_emb.append(nn.Embedding(self.hop_emb_size, self.hparams.d_model, padding_idx=-1))
            self.key_edge_emb.append(nn.Embedding(self.edge_emb_size, self.hparams.d_model, padding_idx=-1))
            self.value_hop_emb.append(nn.Embedding(self.hop_emb_size, self.hparams.d_model, padding_idx=-1))
            self.value_edge_emb.append(nn.Embedding(self.edge_emb_size, self.hparams.d_model, padding_idx=-1))

        self.layers = nn.ModuleList(
            [
                XEncoderLayer(
                    hidden_size=self.hparams.d_model,
                    ffn_size=self.hparams.dim_feedforward,
                    dropout_rate=self.hparams.dropout,
                    attention_dropout_rate=self.hparams.attention_dropout,
                    num_heads=self.hparams.nhead,
                    norm_mode=self.norm_mode,
                )
                for _ in range(self.hparams.num_layer)
            ]
        )

    def _setup_example_input_array(self) -> None:
        # set example input array
        MAX_NUM_NODES = 256
        BATCH_SIZE = 8
        self.example_input_array = {
            "x": torch.zeros(
                [BATCH_SIZE, MAX_NUM_NODES, abs(self.num_node_type)], dtype=torch.long, device=self.device
            ),
            "edge_matrix": torch.zeros(
                [BATCH_SIZE, MAX_NUM_NODES, MAX_NUM_NODES], dtype=torch.long, device=self.device
            ),
            "hop": torch.zeros([BATCH_SIZE, MAX_NUM_NODES, MAX_NUM_NODES], dtype=torch.long, device=self.device),
            "mask": torch.zeros([BATCH_SIZE, MAX_NUM_NODES], dtype=torch.bool, device=self.device),
        }

    def _init_weights(self, module, std=0.02):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_sharded_model(self) -> None:
        self.task_token = wrap(self.task_token)
        self.node_emb = wrap(self.node_emb)

        for emb_idx in range(self.num_layer if self.use_independent_token else 1):
            self.query_hop_emb[emb_idx] = wrap(self.query_hop_emb[emb_idx])
            self.query_edge_emb[emb_idx] = wrap(self.query_edge_emb[emb_idx])
            self.key_hop_emb[emb_idx] = wrap(self.key_hop_emb[emb_idx])
            self.key_edge_emb[emb_idx] = wrap(self.key_edge_emb[emb_idx])
            self.value_hop_emb[emb_idx] = wrap(self.value_hop_emb[emb_idx])
            self.value_edge_emb[emb_idx] = wrap(self.value_edge_emb[emb_idx])

        for layer_idx, layer in enumerate(self.layers):
            self.layers[layer_idx] = wrap(layer)

        return super().configure_sharded_model()

    def _encode_node(self, node_feat: torch.Tensor):
        if isinstance(self.node_emb, nn.Linear):
            return self.node_emb(node_feat)
        else:
            node_feat[node_feat == -1] = self.node_emb.padding_idx
            return self.node_emb(node_feat).sum(dim=2)

    def _prepare_graph_input(
        self,
        x: torch.Tensor,
        edge_matrix: torch.Tensor,
        hop: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        task_name: Optional[str] = None,
    ):
        x = self._encode_node(x)

        if self.training and self.perturb_noise != 0.0:
            perturb = torch.empty_like(x).uniform_(-self.perturb_noise, self.perturb_noise)
            x = x + perturb

        # Append Task Token
        x_with_task = torch.zeros((x.shape[0], x.shape[1] + 1, x.shape[2]), dtype=x.dtype, device=x.device)
        task_token_idx = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)

        x_with_task[:, 1:, :] = x
        if task_name is not None and isinstance(self.task_token, nn.ModuleDict):
            x_with_task[:, 0, :] = self.task_token[task_name](task_token_idx)
        else:
            x_with_task[:, 0, :] = self.task_token(task_token_idx)

        # Mask with task
        if mask is None:
            mask_with_task = None
        else:
            mask_with_task = torch.zeros(
                (mask.shape[0], mask.shape[1] + 1),
                dtype=mask.dtype,
                device=x.device,
            )
            mask_with_task[:, 1:] = mask

        hop_with_task = torch.zeros(
            (
                hop.shape[0],
                hop.shape[1] + 1,
                hop.shape[2] + 1,
            ),
            dtype=hop.dtype,
            device=hop.device,
        )
        # distance with task
        # max_hop is $\mathcal{P}_\text{far}$
        hop_clamped = hop.clamp(max=self.max_hop)
        hop_with_task[:, 1:, 1:] = hop_clamped
        # extend hop for unreachable
        hop_with_task[:, 0, 1:] = hop_clamped[:, 0, :]
        hop_with_task[:, 1:, 0] = hop_clamped[:, :, 0]
        unreachable_mask = hop_with_task == -1
        # set task_distance
        hop_with_task[:, 0, 1:] = self.TASK_DISTANCE
        hop_with_task[:, 1:, 0] = self.TASK_DISTANCE
        # set unreachable_distance
        hop_with_task[unreachable_mask] = self.UNKNOWN_DISTANCE

        # edge matrix with task
        edge_matrix_with_task = torch.zeros(
            (
                edge_matrix.shape[0],
                edge_matrix.shape[1] + 1,
                edge_matrix.shape[2] + 1,
            ),
            dtype=edge_matrix.dtype,
            device=edge_matrix.device,
        )
        edge_matrix_with_task[:, 1:, 1:] = edge_matrix
        edge_matrix_with_task[hop_with_task != 1] = self.NO_EDGE

        # self edge
        edge_matrix_with_task[
            :, list(range(edge_matrix_with_task.shape[1])), list(range(edge_matrix_with_task.shape[2]))
        ] = self.SELF_EDGE
        edge_matrix_with_task[hop_with_task == self.TASK_DISTANCE] = self.TASK_EDGE
        edge_matrix_with_task[unreachable_mask] = self.UNKNOWN_EDGE

        return x_with_task, edge_matrix_with_task, hop_with_task, mask_with_task

    def forward(
        self, x: torch.Tensor, edge_matrix: torch.Tensor, hop: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        x_with_task, edge_matrix_with_task, hop_with_task, mask_with_task = self._prepare_graph_input(
            x, edge_matrix, hop, mask, task_name=None
        )
        hop_arange_list = torch.arange(0, self.hop_emb_size, dtype=torch.long, device=hop_with_task.device)
        edge_arange_list = torch.arange(0, self.edge_emb_size, dtype=torch.long, device=edge_matrix_with_task.device)

        for layer_idx, enc_layer in enumerate(self.layers):
            emb_idx = layer_idx if self.use_independent_token else 0
            x_with_task = enc_layer(
                x_with_task,
                self.query_hop_emb[emb_idx](hop_arange_list),
                self.query_edge_emb[emb_idx](edge_arange_list),
                self.key_hop_emb[emb_idx](hop_arange_list),
                self.key_edge_emb[emb_idx](edge_arange_list),
                self.value_hop_emb[emb_idx](hop_arange_list),
                self.value_edge_emb[emb_idx](edge_arange_list),
                hop_with_task,
                edge_matrix_with_task,
                mask=mask_with_task,
            )
        return x_with_task

    def relprop(self, R, **kwargs):
        for layer_idx, enc_layer in reversed(list(enumerate(self.layers))):
            emb_idx = layer_idx if self.use_independent_token else 0
            R = enc_layer.relprop(R, **kwargs)

        return R


class MultitaskXGRPEEncoder(XGRPEEncoder):
    def __init__(
        self,
        num_layer: int = 12,
        d_model: int = 768,
        nhead: int = 32,
        dim_feedforward: int = 768,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_hop: int = 5,
        num_node_type: int = 11,  # number of node_types
        node_offset: int = 128,
        num_edge_type: int = 3,
        edge_offset: int = 8,
        perturb_noise: float = 0.0,
        norm_mode: str = "pre",
        task_names: List[str] = [],
        use_independent_token: bool = False,  # deprecated
    ):
        for task_name in task_names:
            if not isinstance(task_name, str):
                raise ValueError(f"Invalid task_name: {task_name}")
        self.task_names = task_names
        super().__init__(
            num_layer,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            attention_dropout,
            max_hop,
            num_node_type,  # number of node_types
            node_offset,
            num_edge_type,
            edge_offset,
            perturb_noise,
            norm_mode,
            use_independent_token,
        )

    def _setup_task_token(self) -> None:
        self.task_token = nn.ModuleDict({task_name: nn.Embedding(1, self.d_model) for task_name in self.task_names})

    def _setup_example_input_array(self) -> None:
        # set example input array
        MAX_NUM_NODES = 256
        BATCH_SIZE = 8
        self.example_input_array = {
            "x": torch.zeros(
                [BATCH_SIZE, MAX_NUM_NODES, abs(self.num_node_type)], dtype=torch.long, device=self.device
            ),
            "edge_matrix": torch.zeros(
                [BATCH_SIZE, MAX_NUM_NODES, MAX_NUM_NODES], dtype=torch.long, device=self.device
            ),
            "hop": torch.zeros([BATCH_SIZE, MAX_NUM_NODES, MAX_NUM_NODES], dtype=torch.long, device=self.device),
            "task_name": "example_task_name",
            "mask": torch.zeros([BATCH_SIZE, MAX_NUM_NODES], dtype=torch.bool, device=self.device),
        }

    def configure_sharded_model(self) -> None:
        for task_name in self.task_token.keys():
            self.task_token[task_name] = wrap(self.task_token[task_name])
        self.node_emb = wrap(self.node_emb)

        for emb_idx in range(self.num_layer if self.use_independent_token else 1):
            self.query_hop_emb[emb_idx] = wrap(self.query_hop_emb[emb_idx])
            self.query_edge_emb[emb_idx] = wrap(self.query_edge_emb[emb_idx])
            self.key_hop_emb[emb_idx] = wrap(self.key_hop_emb[emb_idx])
            self.key_edge_emb[emb_idx] = wrap(self.key_edge_emb[emb_idx])
            self.value_hop_emb[emb_idx] = wrap(self.value_hop_emb[emb_idx])
            self.value_edge_emb[emb_idx] = wrap(self.value_edge_emb[emb_idx])

        for layer_idx, layer in enumerate(self.layers):
            self.layers[layer_idx] = wrap(layer)
        return super().configure_sharded_model()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if "task_token.weight" in state_dict.keys():
            print(
                "state_dicts originated from GRRPEEncoder. Copy task_token.weight to "
                + ", ".join([f"task_token.{task_name}.weight" for task_name in self.task_names])
                + ".",
                file=sys.stderr,
            )
            # load from GRPEEncoder, copy task_token to all other task_tokens
            single_task_token_state_dict = state_dict.pop("task_token.weight")
            for task_name in self.task_names:
                state_dict[f"task_token.{task_name}.weight"] = single_task_token_state_dict.clone()
        else:
            state_dict_task_names = [
                v.split(".")[1] for v in state_dict.keys() if v.startswith("task_token") and v.split(".")[1] != "weight"
            ]
            unknown_task_names = list(set(state_dict_task_names) - set(self.task_names))
            not_updated_task_names = list(set(self.task_names) - set(state_dict_task_names))
            current_state_dict = self.state_dict()

            if len(unknown_task_names) > 0:
                for unknown_task_name in unknown_task_names:
                    print(f"state(task_token.{unknown_task_name}.weight) is not used.", file=sys.stderr)
                    state_dict.pop(f"task_token.{unknown_task_name}.weight")

            if len(not_updated_task_names) > 0:
                for not_updated_task_name in not_updated_task_names:
                    state_dict[not_updated_task_name] = current_state_dict[not_updated_task_name]

        return super().load_state_dict(state_dict, strict)

    def forward(
        self,
        x: torch.Tensor,
        edge_matrix: torch.Tensor,
        hop: torch.Tensor,
        task_name: str,
        mask: Optional[torch.Tensor] = None,
    ):
        x_with_task, edge_matrix_with_task, hop_with_task, mask_with_task = self._prepare_graph_input(
            x, edge_matrix, hop, mask, task_name=task_name
        )
        hop_arange_list = torch.arange(0, self.hop_emb_size, dtype=torch.long, device=hop_with_task.device)
        edge_arange_list = torch.arange(0, self.edge_emb_size, dtype=torch.long, device=edge_matrix_with_task.device)

        for layer_idx, enc_layer in enumerate(self.layers):
            emb_idx = layer_idx if self.use_independent_token else 0
            x_with_task = enc_layer(
                x_with_task,
                self.query_hop_emb[emb_idx](hop_arange_list),
                self.query_edge_emb[emb_idx](edge_arange_list),
                self.key_hop_emb[emb_idx](hop_arange_list),
                self.key_edge_emb[emb_idx](edge_arange_list),
                self.value_hop_emb[emb_idx](hop_arange_list),
                self.value_edge_emb[emb_idx](edge_arange_list),
                hop_with_task,
                edge_matrix_with_task,
                mask=mask_with_task,
            )
        return x_with_task


class XFeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(XFeedForwardNetwork, self).__init__()

        self.dense1 = xnn.Linear(hidden_size, ffn_size)
        self.gelu = xnn.GELU()
        self.dense2 = xnn.Linear(ffn_size, hidden_size)

    def forward(self, x: torch.Tensor):
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        return x

    def relprop(self, R, **kwargs):
        R = self.dense2.relprop(R, **kwargs)
        R = self.gelu.relprop(R, **kwargs)
        R = self.dense1.relprop(R, **kwargs)
        return R


class XMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(XMultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size**-0.5

        self.linear_q = xnn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = xnn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = xnn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = xnn.Dropout(attention_dropout_rate)

        self.output_layer = xnn.Linear(num_heads * att_size, hidden_size)

        # xai modules
        self.input_clone = xnn.Clone()
        self.query_clone = xnn.Clone()
        self.query_hop_matmul = xnn.MatMul()
        self.query_hop_gather = xnn.Gather()
        self.query_edge_matmul = xnn.MatMul()
        self.query_edge_gather = xnn.Gather()

        self.key_clone = xnn.Clone()
        self.key_hop_matmul = xnn.MatMul()
        self.key_hop_gather = xnn.Gather()
        self.key_edge_matmul = xnn.MatMul()
        self.key_edge_gather = xnn.Gather()

        self.spatial_add = xnn.Add()
        self.edge_add = xnn.Add()
        self.att_matmul = xnn.MatMul()
        self.att_masked_fill = xnn.MaskedFill()
        self.att_add1 = xnn.Add()
        self.att_add2 = xnn.Add()
        self.att_softmax = xnn.Softmax(dim=3)
        self.att_clone = xnn.Clone()
        self.att_mul = xnn.Mul()
        self.val_add1 = xnn.Add()
        self.val_add2 = xnn.Add()

        self.att_val_matmul = xnn.MatMul()
        self.hop_val_matmul = xnn.MatMul()
        self.edge_val_matmul = xnn.MatMul()

        self.hop_scatter_add = xnn.ScatterAdd()
        self.edge_scatter_add = xnn.ScatterAdd()

    def forward(
        self,
        x: torch.Tensor,
        query_hop_emb: torch.Tensor,
        query_edge_emb: torch.Tensor,
        key_hop_emb: torch.Tensor,
        key_edge_emb: torch.Tensor,
        value_hop_emb: torch.Tensor,
        value_edge_emb: torch.Tensor,
        hop: torch.Tensor,
        edge_matrix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        q, k, v = self.input_clone(x, 3)
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2)  # [b, h, d_k, k_len]

        num_tokens = v.shape[2]
        num_hop_types = query_hop_emb.shape[0]
        num_edge_types = query_edge_emb.shape[0]

        query_hop_emb = query_hop_emb.view(1, num_hop_types, self.num_heads, self.att_size).transpose(1, 2)
        query_edge_emb = query_edge_emb.view(1, -1, self.num_heads, self.att_size).transpose(1, 2)
        key_hop_emb = key_hop_emb.view(1, num_hop_types, self.num_heads, self.att_size).transpose(1, 2)
        key_edge_emb = key_edge_emb.view(1, num_edge_types, self.num_heads, self.att_size).transpose(1, 2)

        q, q_hop, q_edge = self.query_clone(q, 3)

        # query_hop = torch.matmul(q, query_hop_emb.transpose(2, 3))
        # query_hop = torch.gather(query_hop, 3, hop.unsqueeze(1).repeat(1, self.num_heads, 1, 1))
        # query_edge = torch.matmul(q, query_edge_emb.transpose(2, 3))
        # query_edge = torch.gather(query_edge, 3, edge_matrix.unsqueeze(1).repeat(1, self.num_heads, 1, 1))

        query_hop = self.query_hop_matmul([q_hop, query_hop_emb.transpose(2, 3)])
        query_hop = self.query_hop_gather(query_hop, 3, hop.unsqueeze(1).repeat(1, self.num_heads, 1, 1))
        query_edge = self.query_edge_matmul([q_edge, query_edge_emb.transpose(2, 3)])
        query_edge = self.query_edge_gather(query_edge, 3, edge_matrix.unsqueeze(1).repeat(1, self.num_heads, 1, 1))

        k, k_hop, k_edge = self.key_clone(k, 3)

        key_hop = self.key_hop_matmul([k_hop, key_hop_emb.transpose(2, 3)])
        key_hop = self.key_hop_gather(key_hop, 3, hop.unsqueeze(1).repeat(1, self.num_heads, 1, 1))
        key_edge = self.key_edge_matmul([k_edge, key_edge_emb.transpose(2, 3)])
        key_edge = self.key_edge_gather(key_edge, 3, edge_matrix.unsqueeze(1).repeat(1, self.num_heads, 1, 1))

        spatial_bias = self.spatial_add([query_hop, key_hop])
        edge_bais = self.edge_add([query_edge, key_edge])
        att = self.att_matmul([q, k.transpose(2, 3)])

        x = self.att_add2([att, self.att_add1([spatial_bias, edge_bais])])

        x = self.att_mul([x, torch.tensor(self.scale, device=x.device, dtype=x.dtype)])

        if mask is not None:
            # x = x.masked_fill(mask.view(mask.shape[0], 1, 1, mask.shape[1]), float("-inf"))
            x = self.att_masked_fill(x, mask.view(mask.shape[0], 1, 1, mask.shape[1]), float("-inf"))

        x = self.att_softmax(x)

        # save attention and register att_gradients
        self.save_att(x)
        x.register_hook(self.save_att_gradients)

        x = self.att_dropout(x)

        x, x_hop_att, x_edge_att = self.att_clone(x, 3)
        value_hop_emb = value_hop_emb.view(1, num_hop_types, self.num_heads, self.att_size).transpose(1, 2)
        value_edge_emb = value_edge_emb.view(1, num_edge_types, self.num_heads, self.att_size).transpose(1, 2)

        value_hop_att = torch.zeros(
            (batch_size, self.num_heads, num_tokens, num_hop_types),
            dtype=x.dtype,
            device=value_hop_emb.device,
        )

        # value_hop_att = torch.scatter_add(value_hop_att, 3, hop.unsqueeze(1).repeat(1, self.num_heads, 1, 1), x)
        value_hop_att = self.hop_scatter_add(
            [value_hop_att, x_hop_att], 3, hop.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )

        value_edge_att = torch.zeros(
            (batch_size, self.num_heads, num_tokens, num_edge_types),
            dtype=x.dtype,
            device=value_edge_emb.device,
        )
        # value_edge_att = torch.scatter_add(value_edge_att, 3, edge_matrix.unsqueeze(1).repeat(1, self.num_heads, 1, 1), x)
        value_edge_att = self.edge_scatter_add(
            [value_edge_att, x_edge_att], 3, edge_matrix.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )

        x1 = self.att_val_matmul([x, v])
        x2 = self.hop_val_matmul([value_hop_att, value_hop_emb])
        x3 = self.edge_val_matmul([value_edge_att, value_edge_emb])
        x = self.val_add2([x1, self.val_add1([x2, x3])])

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)
        # assert x.size() == orig_q_size
        return x

    def get_att(self):
        return getattr(self, "att", None)

    def save_att(self, att):
        # self.att = att
        self.register_buffer("att", att.detach(), persistent=False)

    def save_att_cam(self, cam):
        # self.att_cam = cam
        self.register_buffer("att_cam", cam.detach(), persistent=False)

    def get_att_cam(self):
        return getattr(self, "att_cam", None)

    def save_att_gradients(self, att_gradients):
        # self.att_gradients = att_gradients
        self.register_buffer("att_gradients", att_gradients.detach(), persistent=False)

    def get_att_gradients(self):
        return getattr(self, "att_gradients", None)

    def relprop(self, R, **kwargs):
        batch_size = R.shape[0]
        num_tokens = R.shape[1]
        R = self.output_layer.relprop(R, **kwargs)

        R = R.view(batch_size, -1, self.num_heads, self.att_size)
        R = R.transpose(1, 2).contiguous()
        R1, R2 = self.val_add2.relprop(R, **kwargs)
        R2, R3 = self.val_add1.relprop(R2, **kwargs)

        Rx, Rv = self.att_val_matmul.relprop(R1, **kwargs)
        R_value_hop_att, R_value_hop_emb = self.hop_val_matmul.relprop(R2, **kwargs)
        R_value_edge_att, R_value_edge_emb = self.edge_val_matmul.relprop(R3, **kwargs)

        R_value_edge_att, Rx_edge_att = self.edge_scatter_add.relprop(R_value_edge_att, **kwargs)
        R_value_hop_att, Rx_hop_att = self.hop_scatter_add.relprop(R_value_hop_att, **kwargs)

        R = self.att_clone.relprop([Rx, Rx_hop_att, Rx_edge_att], **kwargs)

        self.save_att_cam(R)

        R = self.att_dropout.relprop(R, **kwargs)
        R = self.att_softmax.relprop(R, **kwargs)

        if getattr(self.att_masked_fill, "mask", None) is not None:
            R = self.att_masked_fill.relprop(R, **kwargs)

        R, scale = self.att_mul.relprop(R, **kwargs)

        Ratt, Rbias = self.att_add2.relprop(R, **kwargs)
        Rsbias, Rebias = self.att_add1.relprop(Rbias, **kwargs)

        Rq, Rk = self.att_matmul.relprop(Ratt, **kwargs)
        Rq_edge, Rk_edge = self.edge_add.relprop(Rebias, **kwargs)
        Rq_hop, Rk_hop = self.spatial_add.relprop(Rsbias, **kwargs)

        ####
        Rk_edge = self.key_edge_gather.relprop(Rk_edge, **kwargs)
        Rk_edge, Rk_edge_emb = self.key_edge_matmul.relprop(Rk_edge, **kwargs)
        Rk_hop = self.key_hop_gather.relprop(Rk_hop, **kwargs)
        Rk_hop, Rk_edge_emb = self.key_hop_matmul.relprop(Rk_hop, **kwargs)

        Rk = self.key_clone.relprop([Rk.transpose(2, 3), Rk_hop, Rk_edge], **kwargs)

        Rq_edge = self.query_edge_gather.relprop(Rq_edge, **kwargs)
        Rq_edge, Rq_edge_emb = self.query_edge_matmul.relprop(Rq_edge, **kwargs)
        Rq_hop = self.query_hop_gather.relprop(Rq_hop, **kwargs)
        Rq_hop, Rq_edge_emb = self.key_hop_matmul.relprop(Rq_hop, **kwargs)

        Rq = self.query_clone.relprop([Rq, Rq_hop, Rq_edge], **kwargs)

        Rq = Rq.transpose(1, 2).view(batch_size, num_tokens, -1)
        Rv = Rv.transpose(1, 2).view(batch_size, num_tokens, -1)
        Rk = Rk.transpose(1, 2).view(batch_size, num_tokens, -1)

        Rq = self.linear_q.relprop(Rq, **kwargs)
        Rk = self.linear_k.relprop(Rk, **kwargs)
        Rv = self.linear_v.relprop(Rv, **kwargs)

        R = self.input_clone.relprop([Rq, Rk, Rv], **kwargs)
        return R


class XEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        ffn_size,
        dropout_rate,
        attention_dropout_rate,
        num_heads,
        norm_mode="pre",
    ):
        super(XEncoderLayer, self).__init__()

        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.num_heads = num_heads
        self.norm_mode = norm_mode

        self.self_attention = XMultiHeadAttention(
            hidden_size,
            attention_dropout_rate,
            num_heads,
        )
        self.self_dropout = xnn.Dropout(dropout_rate)
        self.self_norm = xnn.LayerNorm(hidden_size)

        self.ffn = XFeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_norm = xnn.LayerNorm(hidden_size)
        self.ffn_dropout = xnn.Dropout(dropout_rate)

        self.att_clone = xnn.Clone()
        self.att_add = xnn.SkipAdd()
        self.ffn_clone = xnn.Clone()
        self.ffn_add = xnn.SkipAdd()

    def forward(
        self,
        x: torch.Tensor,
        query_hop_emb: torch.Tensor,
        query_edge_emb: torch.Tensor,
        key_hop_emb: torch.Tensor,
        key_edge_emb: torch.Tensor,
        value_hop_emb: torch.Tensor,
        value_edge_emb: torch.Tensor,
        hop: torch.Tensor,
        edge_matrix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        if self.norm_mode == "pre":
            # attention
            y, x = self.att_clone(x, 2)
            y = self.self_norm(y)
            y = self.self_attention(
                y,
                query_hop_emb,
                query_edge_emb,
                key_hop_emb,
                key_edge_emb,
                value_hop_emb,
                value_edge_emb,
                hop,
                edge_matrix,
                mask=mask,
            )
            y = self.self_dropout(y)
            x = self.att_add([y, x])

            # ffn
            y, x = self.ffn_clone(x, 2)
            y = self.ffn_norm(y)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            x = self.ffn_add([y, x])
        else:
            # original transformer
            # attention
            y, x = self.att_clone(x, 2)
            y = self.self_attention(
                y,
                query_hop_emb,
                query_edge_emb,
                key_hop_emb,
                key_edge_emb,
                value_hop_emb,
                value_edge_emb,
                hop,
                edge_matrix,
                mask=mask,
            )
            y = self.self_dropout(y)
            x = self.att_add([y, x])
            x = self.self_norm(x)

            # ffn
            y, x = self.ffn_clone(x, 2)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            x = self.ffn_add([y, x])
            x = self.ffn_norm(x)
        return x

    def relprop(self, R, **kwargs):
        if self.norm_mode == "pre":
            # ffn relprop
            Ry, Rx = self.ffn_add.relprop(R, **kwargs)
            Ry = self.ffn_dropout.relprop(Ry, **kwargs)
            Ry = self.ffn.relprop(Ry, **kwargs)
            Ry = self.ffn_norm.relprop(Ry, **kwargs)
            R = self.ffn_clone.relprop((Ry, Rx), **kwargs)

            # attention relprop
            Ry, Rx = self.att_add.relprop(R, **kwargs)
            Ry = self.self_dropout.relprop(Ry, **kwargs)
            Ry = self.self_attention.relprop(Ry, **kwargs)
            Ry = self.self_norm.relprop(Ry, **kwargs)
            R = self.att_clone.relprop((Ry, Rx), **kwargs)

        else:
            # original transformer
            # ffn relprop
            R = self.ffn_norm.relprop(R, **kwargs)
            Ry, Rx = self.ffn_add.relprop(R, **kwargs)
            Ry = self.ffn_dropout.relprop(Ry, **kwargs)
            Ry = self.ffn.relprop(Ry, **kwargs)
            R = self.ffn_clone.relprop((Ry, Rx), **kwargs)

            # attention relprop
            R = self.self_norm.relprop(R, **kwargs)
            Ry, Rx = self.att_add.relprop(R, **kwargs)
            Ry = self.self_dropout.relprop(Ry, **kwargs)
            Ry = self.self_attention.relprop(Ry, **kwargs)
            R = self.att_clone.relprop((Ry, Rx), **kwargs)
        return R
