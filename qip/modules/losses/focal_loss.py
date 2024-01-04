from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss


class SigmoidFocalLoss(nn.Module):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """

    def __init__(self, gamma=1.5, alpha=0.5):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, reduction: Optional[str] = "mean") -> torch.Tensor:
        return sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, reduction=reduction)


class SoftmaxFocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=0.25):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: Optional[str] = "mean"
        ) -> torch.Tensor:

        targets = targets.view(-1, 1).long()
        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = -log_probs.gather(1, targets).view(-1)
        p_t = torch.exp(-ce_loss)

        alpha_weight = torch.tensor([self.alpha, 1-self.alpha], device=targets.device)
        alpha_weight = alpha_weight.gather(0,targets.data.view(-1))
        focal_loss = alpha_weight * ce_loss * ((1 - p_t) ** self.gamma)

        if reduction == "mean":
            return focal_loss.mean()
        elif reduction == "sum":
            return focal_loss.sum()
