"""nanoVLM v0.1 SigLIP-compatible ViT; tiny configs use the same implementation."""
from torch.utils.checkpoint import checkpoint
from .vision_transformer import ViT

class VisionEncoder(ViT):
    def forward(self, pixels):
        x = self.dropout(self.patch_embedding(pixels))
        for block in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.layer_norm(x)
