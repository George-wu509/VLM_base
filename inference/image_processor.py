from pathlib import Path
import numpy as np
import torch
from PIL import Image

def load_image(path, size):
    with Image.open(Path(path)) as source:
        return process_image(source, size)

def process_image(image, size):
    # SigLIP: full-image square resize, RGB, [0,1] then mean/std = 0.5.
    # No crop: normalized grounding coordinates remain relative to the full image.
    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).sub(0.5).div(0.5)
