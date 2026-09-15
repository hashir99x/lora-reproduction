"""
LoRA (Low-Rank Adaptation) implemented from scratch in PyTorch, following
Hu et al., 2021, Section 4.1:

    h = W0 x + (alpha / r) * B A x

W0 (the original pretrained weight) is frozen. A is initialized from a
random Gaussian (so the low-rank update starts as a random-but-small
direction) and B is initialized to all zeros, which forces the update
BA to be exactly zero at the start of training - i.e. training begins
from the unmodified pretrained model's behavior, exactly as the paper
specifies (Section 4.1, "We use a random Gaussian initialization for A
and zero for B, so delta-W = BA is zero at the beginning of training").
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer, freezes it, and adds a trainable
    low-rank adapter alongside it. Forward pass:
        y = frozen_linear(x) + dropout(x) @ A^T @ B^T * (alpha / r)
    """

    def __init__(self, base_linear: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # A: (r, in_features), random Gaussian init (paper, Section 4.1)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: (out_features, r), zero init (paper, Section 4.1) -> delta_W = B @ A = 0 at start
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base_linear(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def merged_weight(self):
        """Return W0 + (alpha/r) * B @ A - the single merged weight matrix the
        paper notes can be used at inference time for zero extra latency
        (Section 4.1: 'when deployed in production, we can explicitly compute
        and store W = W0 + BA ... introducing no inference latency')."""
        with torch.no_grad():
            return self.base_linear.weight + self.scaling * (self.lora_B @ self.lora_A)


def inject_lora(model: nn.Module, target_module_names, r: int, alpha: float, dropout: float = 0.0):
    """
    Walk the model and replace every nn.Linear submodule whose attribute
    name matches one of target_module_names (e.g. 'q_lin', 'v_lin' for
    DistilBERT's attention projections) with a LoRALinear wrapper.
    Returns the number of layers replaced.
    """
    replaced = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if name in target_module_names and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    return replaced


def freeze_base_model_except_lora_and_head(model: nn.Module, head_param_names_substr=("classifier", "pre_classifier")):
    """
    Freeze everything except: (a) LoRA A/B parameters, (b) the task
    classification head. This matches standard LoRA practice (and the
    paper's own setup): the pretrained backbone is frozen, LoRA adapters
    learn the task-specific weight update, and the small task head is
    always trained fully since it doesn't exist in the pretrained model
    at all.
    """
    trainable, total = 0, 0
    for name, param in model.named_parameters():
        total += param.numel()
        is_lora = "lora_A" in name or "lora_B" in name
        is_head = any(h in name for h in head_param_names_substr)
        param.requires_grad = is_lora or is_head
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def count_trainable_params(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
