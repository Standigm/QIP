import torch
import torch.nn as nn


# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    if start_layer < 0:
        start_layer = len(all_layer_matrices) + start_layer
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [
        all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True) for i in range(len(all_layer_matrices))
    ]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


class BaseExplanation(nn.Module):
    METHODS = ["lrp", "lrp_last_layer", "full_lrp", "att_last_layer", "rollout", "att_gradcam"]

    @property
    def xai_method(self):
        return self._xai_method

    @xai_method.setter
    def xai_method(self, value):
        if value in BaseExplanation.METHODS:
            self._xai_method = value
        else:
            raise ValueError(f"Unknown method: {value}. method should be in {BaseExplanation.METHODS}")

    def generate(self, *args, **kwargs):
        generate_func = getattr(self, f"generate_{self.xai_method}", None)
        if generate_func is None:
            raise NotImplementedError(f"generate_{self.xai_method} is not implemented.")
        else:
            return generate_func(*args, **kwargs)
