"""Small, inspectable LoRA implementation for native nn.Linear layers."""
import math
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (self.dropout(x) @ self.A.T @ self.B.T) * self.scale

def inject_lora(model, config):
    targets = config.get("targets", ["q_proj", "v_proj"])
    found = []
    for name, module in list(model.decoder.named_modules()):
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            parent_name, leaf = name.rsplit(".", 1)
            parent = model.decoder.get_submodule(parent_name)
            setattr(parent, leaf, LoRALinear(module, config["rank"], config["alpha"], config.get("dropout", 0.0)))
            found.append(name)
    if not found:
        raise ValueError(f"No LoRA targets matched {targets}")
    return found
