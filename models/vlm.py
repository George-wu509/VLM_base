"""Shared model: vision prefix + text embeddings + one causal decoder.

Labels are UNshifted. The model shifts exactly once, including the image prefix,
and ignores prompt, image and padding targets. Text-only uses the same decoder.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .vision_encoder import VisionEncoder
from .language_decoder import LanguageDecoder
from .projector import Projector

class VLM(nn.Module):
    def __init__(self, cfg, load_backbones=None):
        super().__init__()
        self.cfg = cfg
        load = cfg.vlm_load_backbone_weights if load_backbones is None else load_backbones
        if cfg.vit_cls_flag or cfg.lm_use_tokens:
            raise ValueError("VLM requires patch features and an embedding-input decoder")
        self.vision_encoder = VisionEncoder.from_pretrained(cfg) if load else VisionEncoder(cfg)
        self.decoder = LanguageDecoder.from_pretrained(cfg) if load else LanguageDecoder(cfg)
        self.projector = Projector(cfg)  # Always random until loading a VLM checkpoint.
        self.vision_head = nn.Linear(cfg.vit_hidden_dim, cfg.vision_num_classes)

    def image_features(self, pixels):
        # No graph needed for a frozen vision backbone; decoder must retain the
        # graph to pass gradients through its frozen weights into the projector.
        with torch.set_grad_enabled(torch.is_grad_enabled() and any(
            p.requires_grad for p in self.vision_encoder.parameters()
        )):
            return self.vision_encoder(pixels)

    def classify(self, pixels):
        return self.vision_head(self.image_features(pixels).mean(1))

    def forward(self, input_ids, pixel_values=None, attention_mask=None, labels=None):
        text = self.decoder.token_embedding(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        n = 0
        if pixel_values is not None:
            image = self.projector(self.image_features(pixel_values))
            n = image.shape[1]
            text = torch.cat([image, text], 1)
            attention_mask = torch.cat([attention_mask.new_ones((text.size(0), n)), attention_mask], 1)
        if text.size(1) > self.cfg.lm_max_position_embeddings:
            raise ValueError("Image + text exceeds configured context length")
        logits = self.decoder.head(self.decoder(text, attention_mask))
        loss = None
        full_labels = None
        if labels is not None:
            full_labels = torch.cat([labels.new_full((labels.size(0), n), -100), labels], 1)
            targets = full_labels[:, 1:]
            if not (targets != -100).any():
                raise ValueError("Batch has no supervised completion tokens after shifting")
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss, "labels": full_labels, "image_tokens": n}
