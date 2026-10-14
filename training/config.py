import json
import random
import os
from pathlib import Path
import torch
import yaml
from models.config import VLMConfig
from models.vlm import VLM
from tokenization.tokenizer import build_tokenizer

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))

def read_yaml(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    return data

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu or install CUDA torch")
    return device

def new_model(model_config):
    cfg = VLMConfig(**model_config)
    if cfg.vlm_load_backbone_weights:
        if cfg.tokenizer_backend != "hf":
            raise ValueError("Pretrained backbones require the matching HF tokenizer")
        from huggingface_hub import HfApi
        api = HfApi()
        for repo_key, revision_key in (("vit_model_type", "vit_revision"), ("lm_model_type", "lm_revision"),
                                       ("lm_tokenizer", "tokenizer_revision")):
            revision = getattr(cfg, revision_key)
            if len(revision) != 40:
                setattr(cfg, revision_key, api.model_info(getattr(cfg, repo_key), revision=revision).sha)
    tokenizer = build_tokenizer(cfg)
    model = VLM(cfg)
    model.decoder.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def move_batch(batch, device):
    return {key: move_batch(value, device) if isinstance(value, dict) else
            value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
