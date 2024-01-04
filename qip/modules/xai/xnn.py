# from https://github.com/hila-chefer/Transformer-Explainability

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "safe_divide",
    "forward_hook",
    "backward_hook",
    "Clone",
    "Cat",
    "Add",
    "SkipAdd",
    "ReLU",
    "GELU",
    "SiLU",
    "Dropout",
    "BatchNorm2d",
    "Linear",
    "MaxPool2d",
    "AdaptiveAvgPool2d",
    "AvgPool2d",
    "Conv2d",
    "Sequential",
    "Einsum",
    "Softmax",
    "IndexSelect",
    "LayerNorm",
    "AddEye",
    "Tanh",
    "MatMul",
    "Mul",
    "MaskedFill" "Gather",
    "ScatterAdd",
]


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = [t.detach().requires_grad_(True) for t in input[0]]
    else:
        self.X = input[0].detach().requires_grad_(True)
    self.Y = output


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelPropModule(nn.Module):
    def __init__(self):
        super(RelPropModule, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha):
        return R


class RelPropSimple(RelPropModule):
    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = [x * c for x, c in zip(self.X, C)]
        else:
            outputs = self.X * (C[0])
        return outputs


class AddEye(RelPropSimple):
    # input of shape B, C, seq_len, seq_len
    def forward(self, input):
        return input + torch.eye(input.shape[2]).expand_as(input).to(input.device)


class MaskedFill(RelPropSimple):
    def forward(self, inputs, mask, value):
        self.__setattr__("mask", mask)
        self.__setattr__("value", value)

        return torch.masked_fill(inputs, mask, value)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.mask, self.value)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = [x * c for x, c in zip(self.X, C)]
        else:
            outputs = self.X * (C[0])
        return outputs


class ReLU(nn.ReLU, RelPropModule):
    pass


class GELU(nn.GELU, RelPropModule):
    pass


class SiLU(nn.SiLU, RelPropModule):
    pass


class Softmax(nn.Softmax, RelPropModule):
    pass


class Mul(RelPropSimple):
    def forward(self, inputs):
        return torch.mul(*inputs)


class Tanh(nn.Tanh, RelPropModule):
    pass


class LayerNorm(nn.LayerNorm, RelPropModule):
    pass


class Dropout(nn.Dropout, RelPropModule):
    pass


class MatMul(RelPropSimple):
    def forward(self, inputs):
        return torch.matmul(*inputs)


class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelPropSimple):
    pass


class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)


class ScatterAdd(RelPropModule):
    def forward(self, inputs, dim, indices):
        self.__setattr__("dim", dim)
        self.__setattr__("indices", indices)

        return torch.scatter_add(inputs[0], dim, indices, inputs[1])

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = [x * c for x, c in zip(self.X, C)]
        else:
            outputs = self.X * (C[0])
        return outputs


class SkipAdd(RelPropSimple):
    # addition for skip connection
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs


class Einsum(RelPropSimple):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)


class IndexSelect(RelPropModule):
    def forward(self, inputs, dim, indices):
        self.__setattr__("dim", dim)
        self.__setattr__("indices", indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = [x * c for x, c in zip(self.X, C)]
        else:
            outputs = self.X * (C[0])
        return outputs


class Gather(RelPropModule):
    def forward(self, inputs, dim, indices):
        self.__setattr__("dim", dim)
        self.__setattr__("indices", indices)

        return torch.gather(inputs, dim, indices)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = [x * c for x, c in zip(self.X, C)]
        else:
            outputs = self.X * (C[0])
        return outputs


class Clone(RelPropModule):
    def forward(self, input, num):
        self.__setattr__("num", num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]

        R = self.X * C

        return R


class Cat(RelPropModule):
    def forward(self, inputs, dim):
        self.__setattr__("dim", dim)
        return torch.cat(inputs, dim)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        outputs = []
        for x, c in zip(self.X, C):
            outputs.append(x * c)

        return outputs


class Sequential(nn.Sequential):
    def relprop(self, R, alpha):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R


class BatchNorm2d(nn.BatchNorm2d, RelPropModule):
    def relprop(self, R, alpha):
        X = self.X
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5)
        )
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R


class Linear(nn.Linear, RelPropModule):
    def relprop(self, R, alpha):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1 + Z2)
            S2 = safe_divide(R, Z1 + Z2)
            C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
            C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]

            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances

        return R


class Conv2d(nn.Conv2d, RelPropModule):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
            (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        )

        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, alpha):
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = (
                self.X * 0
                + torch.min(
                    torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3, keepdim=True
                )[0]
            )
            H = (
                self.X * 0
                + torch.max(
                    torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3, keepdim=True
                )[0]
            )
            Za = (
                torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding)
                - torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding)
                - torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding)
                + 1e-9
            )

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
                Z2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
                S1 = safe_divide(R, Z1)
                S2 = safe_divide(R, Z2)
                C1 = x1 * self.gradprop(Z1, x1, S1)[0]
                C2 = x2 * self.gradprop(Z2, x2, S2)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R
