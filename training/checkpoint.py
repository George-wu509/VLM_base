"""Atomic trusted-local checkpoints; weights_only avoids arbitrary pickle execution."""
import os
import random
from dataclasses import asdict
from pathlib import Path
import torch
from models.config import VLMConfig
from models.vlm import VLM
from models.lora import inject_lora
from tokenization.tokenizer import build_tokenizer, fingerprint

FORMAT_VERSION = 1

def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])

def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)

def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format")
    return payload

def model_from_checkpoint(path):
    payload = read_checkpoint(path)
    cfg = VLMConfig(**payload["model_config"])
    model = VLM(cfg, load_backbones=False)
    if payload.get("lora"):
        inject_lora(model, payload["lora"])
    model.load_state_dict(payload["model"], strict=True)
    tokenizer_path = Path(path).resolve().parent / "tokenizer"
    if not tokenizer_path.is_dir():
        raise ValueError(f"Missing tokenizer bundle {tokenizer_path}")
    tokenizer = build_tokenizer(cfg, tokenizer_path)
    if len(tokenizer) != cfg.lm_vocab_size:
        raise ValueError("Tokenizer size does not match checkpoint vocabulary")
    if payload.get("tokenizer_sha256") and fingerprint(tokenizer) != payload["tokenizer_sha256"]:
        raise ValueError("Tokenizer content does not match checkpoint")
    return model, tokenizer, payload

def save_checkpoint(path, model, tokenizer, **state):
    path = Path(path)
    tokenizer.save_pretrained(path.parent / "tokenizer")
    payload = dict(format_version=FORMAT_VERSION, model_config=asdict(model.cfg), model=model.state_dict(),
                   tokenizer_sha256=fingerprint(tokenizer), **state)
    atomic_save(payload, path)

def merge_lora(model):
    from models.lora import LoRALinear
    for name, module in list(model.decoder.named_modules()):
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.base.weight.add_((module.B @ module.A) * module.scale)
            parent, leaf = name.rsplit(".", 1)
            setattr(model.decoder.get_submodule(parent), leaf, module.base)
    return model
