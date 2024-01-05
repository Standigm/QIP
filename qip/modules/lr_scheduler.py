# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
from typing import Optional

from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestartsWithDecay(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_max (float, optional): Maximum learning rate. Default: 0.1
        T_up (int, optional): steps for maximum learning rate. Default: 0
        gamma (float, optional): decay factor for eta_max. Default: 1.0
        last_epoch (int, optional): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: Optional[int] = 1,
        eta_max: float = 0.1,
        T_up: int = 0,
        gamma: float = 1.0,
        warmup_base_lr: Optional[float] = None,
        last_epoch: int = -1,
    ):
        if not isinstance(T_0, int) or T_0 <= 0:
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult is not None and (isinstance(T_mult, int) and T_mult < 1):
            raise ValueError("Expected None or integer T_mult >= 1, but got {}".format(T_mult))
        if not isinstance(T_up, int) or T_up < 0:
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        self.warmup_base_lr = warmup_base_lr
        super(CosineAnnealingWarmUpRestartsWithDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1 or self.T_cur >= self.T_i:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            base_lrs = [
                self.warmup_base_lr if self.warmup_base_lr is not None and self.cycle == 0 else base_lr
                for base_lr in self.base_lrs
            ]
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in base_lrs]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)))
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_mult is not None and self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult is not None:
                    if self.T_mult == 1:
                        self.T_cur = epoch % self.T_0
                        self.cycle = epoch // self.T_0
                    else:
                        n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                        self.cycle = n
                        self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
                        self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
