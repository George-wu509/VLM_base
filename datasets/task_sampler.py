"""Deterministic epoch sampling; task weights are optional, uniformly shuffled by default."""
import torch
from torch.utils.data import Sampler

class TaskSampler(Sampler):
    def __init__(self, rows, seed=42, weights=None):
        self.rows, self.seed, self.weights, self.epoch = rows, seed, weights, 0
        if weights and any(weights.get(row["task"], 1) <= 0 for row in rows):
            raise ValueError("All task weights must be positive")

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if not self.weights:
            return iter(torch.randperm(len(self.rows), generator=generator).tolist())
        w = torch.tensor([self.weights.get(r["task"], 1) for r in self.rows], dtype=torch.float)
        return iter(torch.multinomial(w, len(w), replacement=True, generator=generator).tolist())
