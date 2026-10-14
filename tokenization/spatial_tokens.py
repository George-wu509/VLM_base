"""PaliGemma-style 1024 coordinate bins, in y0, x0, y1, x1 order.

The architecture remains nanoVLM's causal prefix model. Segmentation codebook
tokens are reserved only; there is no segmentation mask decoder in this repo.
"""
import math
import re

LOC_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
SEG_TOKENS = [f"<seg{i:03d}>" for i in range(128)]
SPATIAL_TOKENS = LOC_TOKENS + SEG_TOKENS
TOKEN_PATTERN = re.compile(r"<loc\d{4}>|<seg\d{3}>")

def encode_box(box):
    if len(box) != 4 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in box):
        raise ValueError("bbox must be normalized xyxy in [0,1]")
    x0, y0, x1, y1 = box
    if x0 >= x1 or y0 >= y1:
        raise ValueError("bbox must have positive width and height")
    return "".join(LOC_TOKENS[min(1023, int(v * 1024))] for v in (y0, x0, y1, x1))

def decode_box(values):
    if len(values) != 4 or any(not 0 <= v < 1024 for v in values):
        raise ValueError("Exactly four location bins in [0,1023] are required")
    y0, x0, y1, x1 = [v / 1024 for v in values]
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Decoded box has nonpositive extent")
    return [x0, y0, x1, y1]
