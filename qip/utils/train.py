import torch
import torch.nn as nn


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad_(False)


def unfreeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad_(True)
