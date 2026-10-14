import math
from torch.optim.lr_scheduler import LambdaLR

def make_scheduler(optimizer, total_steps, warmup_steps):
    def scale(step):
        if step < warmup_steps:
            return (step+1) / max(1, warmup_steps)
        progress = min(1.0, (step-warmup_steps) / max(1, total_steps-warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, scale)
