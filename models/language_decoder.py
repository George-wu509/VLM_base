"""GQA + RoPE + SwiGLU decoder from nanoVLM v0.1, with checkpointing."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .language_model import LanguageModel

class LanguageDecoder(LanguageModel):
    def forward(self, embeddings, attention_mask=None):
        b, t, _ = embeddings.shape
        positions = torch.arange(t, device=embeddings.device)[None].expand(b, -1)
        cos, sin = self.rotary_embd(positions)
        x = embeddings
        for block in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cos, sin, attention_mask, use_reentrant=False)
            else:
                x = block(x, cos, sin, attention_mask)
        return self.norm(x)

    def resize_token_embeddings(self, size):
        old = self.token_embedding
        if size == old.num_embeddings:
            return
        new = nn.Embedding(size, old.embedding_dim, device=old.weight.device, dtype=old.weight.dtype)
        nn.init.normal_(new.weight, std=0.02)
        with torch.no_grad():
            n = min(size, old.num_embeddings)
            new.weight[:n].copy_(old.weight[:n])
        self.token_embedding = new
        head = nn.Linear(old.embedding_dim, size, bias=False, device=old.weight.device, dtype=old.weight.dtype)
        if self.cfg.lm_tie_weights:
            head.weight = new.weight
        else:
            nn.init.normal_(head.weight, std=0.02)
            with torch.no_grad():
                head.weight[:n].copy_(self.head.weight[:n])
        self.head = head
        self.cfg.lm_vocab_size = size
