import numpy as np
import torch
import torch.nn as nn
import copy

from typing import Mapping

from admet_prediction.encoders.grpe import GRPEEncoder, MultitaskGRPEEncoder
from admet_prediction.taskheads.grpe import GRPETaskTokenHead
from admet_prediction.taskheads.xgrpe import XGRPETaskTokenHead
from admet_prediction.encoders.xgrpe import XGRPEEncoder, MultitaskXGRPEEncoder

from admet_prediction.utils.misc import get_func_signature
from admet_prediction.modules.xai.explanations import BaseExplanation, compute_rollout_attention


XCLASSMAPPER = {
    GRPEEncoder: XGRPEEncoder,
    GRPETaskTokenHead: XGRPETaskTokenHead,
    MultitaskGRPEEncoder: MultitaskXGRPEEncoder,
}


class GRPEExplanation(BaseExplanation):
    def __init__(
        self,
        grpe_encoder: nn.Module,
        task_head: nn.Module,
        xai_method: str,
    ):
        super().__init__()
        # convert grpe to xgrpe
        # encoder
        encoder_state_dict = grpe_encoder.state_dict()
        self.xencoder = XCLASSMAPPER[grpe_encoder.__class__](**grpe_encoder.hparams)
        self.xencoder.load_state_dict(encoder_state_dict)
        # input signature
        self.encoder_input_signature = get_func_signature(self.xencoder.forward)
        # task_head
        task_head_state_dict = task_head.state_dict()
        self.xtask_head = XCLASSMAPPER[task_head.__class__](**task_head.hparams)
        self.xtask_head.load_state_dict(task_head_state_dict)

        # convert to eval mode
        self.xencoder.eval()
        self.xtask_head.eval()
        self.xai_method = xai_method

    def forward(self, data: Mapping):
        each_data_dict = data
        inputs_dict = {key: each_data_dict[key] for key in self.encoder_input_signature if key in each_data_dict}
        encoder_output = self.xencoder(**inputs_dict)
        task_output = self.xtask_head(encoder_output)
        return task_output

    def generate_lrp(self, data, label=None, alpha=1.0, **kwargs):
        self.zero_grad()

        start_layer = 0
        output = self.forward(data)
        kwargs = {"alpha": alpha}

        if label == None:
            label = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, label] = 1
        score = torch.softmax(output, dim=1)[0, label].item()
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(output.device) * output)
        one_hot = one_hot.sum()

        self.zero_grad()
        one_hot.backward(retain_graph=True)
        # relprop
        R_encoder = self.xtask_head.relprop(one_hot, **kwargs)
        self.xencoder.relprop(R_encoder, **kwargs)

        cams = []
        blocks = self.xencoder.layers
        for blk in blocks:
            grad = blk.self_attention.get_att_gradients()
            cam = blk.self_attention.get_att_cam()
            cam = cam.squeeze(0)
            grad = grad.squeeze(0)
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        rollout = compute_rollout_attention(cams, start_layer=start_layer)
        rollout[:, 0, 0] = rollout[:, 0].min()
        return rollout[:, 0], score

    def generate_lrp_last_layer(self, data, label=None, alpha=1.0, **kwargs):
        self.zero_grad()

        output = self.forward(data)
        score = torch.softmax(output, dim=1)[0, 1].item()
        kwargs = {"alpha": alpha}
        if label == None:
            label = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, label] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(output.device) * output)

        self.zero_grad()
        one_hot.backward(retain_graph=True)

        R_encoder = self.xtask_head.relprop(one_hot, **kwargs)
        self.xencoder.relprop(R_encoder, **kwargs)

        cam = self.xencoder.layers[-1].self_attention.get_att_cam()[0]
        cam = cam.clamp(min=0).mean(dim=0).unsqueeze(0)
        cam[:, 0, 0] = 0
        return cam[:, 0], score

    def generate_full_lrp(self, data, label=None, alpha=1.0, **kwargs):
        self.zero_grad()

        output = self.forward(data)
        score = torch.softmax(output, dim=1)[0, 1].item()
        kwargs = {"alpha": 1}

        if label == None:
            label = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, label] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(output.device) * output)

        self.zero_grad()
        one_hot.backward(retain_graph=True)

        R_encoder = self.xtask_head.relprop(one_hot, **kwargs)
        cam = self.xencoder.relprop(R_encoder, **kwargs)
        cam = cam.sum(dim=2)
        cam[:, 0] = 0
        return cam, score

    def generate_att_last_layer(self, data, label=None, alpha=1.0, **kwargs):
        self.zero_grad()

        output = self.forward(data)
        score = torch.softmax(output, dim=1)[0, 1].item()
        cam = self.xencoder.layers[-1].self_attention.get_att()[0]
        cam = cam.mean(dim=0).unsqueeze(0)
        cam[:, 0, 0] = 0
        return cam[:, 0], score

    def generate_rollout(self, data, label=None, alpha=1.0, **kwargs):
        self.zero_grad()

        start_layer = 0
        output = self.forward(data)
        score = torch.softmax(output, dim=1)[0, 1].item()

        blocks = self.xencoder.layers
        all_layer_attentions = []
        for blk in blocks:
            att_heads = blk.self_attention.get_att()
            avg_heads = (att_heads.sum(dim=1) / att_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        rollout[:, 0, 0] = 0
        return rollout[:, 0], score

    def generate_att_gradcam(self, data, label=None, alpha=1.0, **kwargs):
        self.zero_grad()

        output = self.forward(data)
        score = torch.softmax(output, dim=1)[0, 1].item()
        kwargs = {"alpha": 1}

        if label == None:
            label = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, label] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(output.device) * output)

        one_hot.backward(retain_graph=True)

        R_encoder = self.xtask_head.relprop(one_hot, **kwargs)
        self.xencoder.relprop(R_encoder, **kwargs)

        cam = self.xencoder.layers[-1].self_attention.get_att()
        grad = self.xencoder.layers[-1].self_attention.get_att_gradients()

        cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0).unsqueeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam[:, 0, 0] = 0
        return cam[:, 0], score
