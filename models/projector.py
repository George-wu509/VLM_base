"""v0.1 pixel shuffle: reduce patch tokens, then a single learned linear layer."""
import math
from torch import nn

class Projector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.factor = cfg.mp_pixel_shuffle_factor
        self.proj = nn.Linear(cfg.vit_hidden_dim * self.factor**2, cfg.lm_hidden_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x):
        b, n, d = x.shape
        s, f = math.isqrt(n), self.factor
        if s*s != n or s % f:
            raise ValueError("Patch tokens must form a square grid divisible by shuffle factor")
        x = x.reshape(b, s//f, f, s//f, f, d).permute(0, 1, 3, 2, 4, 5)
        return self.proj(x.reshape(b, n//(f*f), d*f*f))
