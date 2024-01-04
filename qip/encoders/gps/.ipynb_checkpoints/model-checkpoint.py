import torch
from admet_prediction.encoders.gps.layers import GPSLayer
from admet_prediction.encoders.gps.encoders import FeatureEncoder
import torch.nn as nn
from omegaconf import DictConfig
import lightning as L
from torch_geometric.utils import to_dense_batch


class GPSEncoder(L.LightningModule):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    """

    def __init__(
        self,
        d_model,
        nhead,
        dropout,
        attention_dropout,
        layer_norm,
        batch_norm,
        log_attention_weights,
        num_layer,
        encoder_config: DictConfig,
        ):
        super(GPSEncoder, self).__init__()
        
        
        self.encoder_config = encoder_config
        self.node_config = self.encoder_config.node_config
        self.edge_config = self.encoder_config.edge_config

        assert d_model == self.node_config.dim_emb
        assert d_model == self.edge_config.dim_emb

        self.encoder = FeatureEncoder(
                node_config = self.node_config,
                edge_config = self.edge_config,
        )

        self.gps = nn.ModuleList(
            [GPSLayer(
                d_model,
                nhead,
                dropout,
                attention_dropout,
                layer_norm,
                batch_norm,
                log_attention_weights,
            ) for _ in range(num_layer-1)
            ]+[GPSLayer(
                d_model,
                nhead,
                dropout,
                attention_dropout,
                layer_norm,
                batch_norm,
                log_attention_weights,
                last=True)
                ]
        )
        

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        rwse: torch.Tensor,
        ):
        
        x, edge_attr = self.encoder(x, edge_attr, rwse)
        
        for layer_idx, enc_layer in enumerate(self.gps):
            last = True if layer_idx == len(self.gps) -1 else False
            x, edge_attr = enc_layer(x, edge_index, edge_attr, batch, last=last)
        
        return x, batch
        # return x, batch # (B, Nmax, dim).mean(1)
