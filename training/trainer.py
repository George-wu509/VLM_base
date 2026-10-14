"""One optimizer-step loop for vision, language, alignment, SFT, LoRA and DPO."""
import copy
import json
import math
import os
import shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets.collator import Collator
from datasets.task_sampler import TaskSampler
from models.lora import inject_lora
from .stages import get_stage
from .config import move_batch
from .optimizer import make_optimizer
from .scheduler import make_scheduler
from .losses import compute_loss
from .checkpoint import save_checkpoint, restore_rng, rng_state
from .experiment import sha256

class Trainer:
    def __init__(self, model, tokenizer, dataset, config, output, device, resume=None):
        self.model, self.tokenizer, self.dataset, self.config = model, tokenizer, dataset, config
        self.output, self.device = Path(output), device
        self.policy = get_stage(config["stage"])
        self.output.mkdir(parents=True, exist_ok=True)
        if resume and (resume.get("lora") != config.get("lora") or resume["training_config"]["stage"] != config["stage"]):
            raise ValueError("Resume requires identical stage and adapter configuration")
        if config["batch_size"] < 1 or config["gradient_accumulation"] < 1 or config["max_steps"] < 1:
            raise ValueError("batch_size, gradient_accumulation and max_steps must be positive")
        model.requires_grad_(False)
        for name in self.policy["train"]:
            getattr(model, name).requires_grad_(True)
        self.lora = config.get("lora") if self.policy.get("lora") else None
        if self.lora and not resume:
            inject_lora(model, self.lora)
        elif self.lora:
            for name, p in model.named_parameters():
                if name.endswith((".A", ".B")):
                    p.requires_grad_(True)
        if self.policy.get("lora") and not self.lora:
            raise ValueError("This stage requires a lora config")
        if config.get("train_embeddings"):
            model.decoder.token_embedding.requires_grad_(True)
            model.decoder.head.requires_grad_(True)
        # Optional gradual unfreezing: last N vision/decoder blocks.
        for component, field in (("vision_encoder", "vision_unfreeze_last"), ("decoder", "decoder_unfreeze_last")):
            if field in config:
                backbone = getattr(model, component)
                n = config[field]
                if not 0 <= n <= len(backbone.blocks):
                    raise ValueError(f"Invalid {field}")
                backbone.requires_grad_(False)
                if n:
                    for block in backbone.blocks[-n:]:
                        block.requires_grad_(True)
        if config["stage"] == "alignment" and any(p.requires_grad for p in model.decoder.parameters()):
            raise ValueError("Alignment must train only the projector; enable spatial embeddings in SFT")
        model.to(device)
        self.reference = None
        if self.policy["objective"] == "dpo":
            if config["lora"].get("dropout", 0) != 0:
                raise ValueError("Use zero LoRA dropout for DPO")
            if model.cfg.lm_dropout or model.cfg.vit_dropout:
                raise ValueError("DPO requires zero backbone dropout for consistent policy/reference scores")
            self.reference = copy.deepcopy(model).requires_grad_(False).eval()
            if resume:
                self.reference.load_state_dict(resume["reference"])
        self.optimizer = make_optimizer(model, config)
        self.scheduler = make_scheduler(self.optimizer, config["max_steps"], config.get("warmup_steps", 0))
        precision = config.get("precision", "fp32")
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16 or bf16")
        if device.type == "cpu" and precision == "fp16":
            raise ValueError("fp16 is supported only on CUDA; use fp32 on CPU")
        if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support BF16; use fp16")
        self.amp = precision != "fp32"
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
        self.step, self.epoch, self.next_batch = 0, 0, 0
        self.data_hash = sha256(dataset.manifest)
        self.content_hash = dataset.content_sha256
        if resume:
            if resume["training_config"] != config or resume["manifest_sha256"] != self.data_hash:
                raise ValueError("Resume requires identical training config and manifest; use --init-from for a new stage/run")
            if resume["image_content_sha256"] != self.content_hash:
                raise ValueError("Resume image content changed")
            if resume.get("device_type") != device.type:
                raise ValueError("Exact resume requires the same device type; use --init-from to migrate")
            self.optimizer.load_state_dict(resume["optimizer"])
            self.scheduler.load_state_dict(resume["scheduler"])
            self.scaler.load_state_dict(resume["scaler"])
            self.step, self.epoch, self.next_batch = resume["step"], resume["epoch"], resume["next_batch"]
            restore_rng(resume["rng"])

    def set_modes(self):
        self.model.train()
        for name in ("vision_encoder", "decoder", "projector"):
            component = getattr(self.model, name)
            if not any(p.requires_grad for p in component.parameters()):
                component.eval()

    def save(self):
        save_checkpoint(self.output / "last.pt", self.model, self.tokenizer,
            training_config=self.config, lora=self.lora, optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(), scaler=self.scaler.state_dict(), rng=rng_state(),
            reference=self.reference.state_dict() if self.reference is not None else None,
            step=self.step, epoch=self.epoch, next_batch=self.next_batch,
            manifest_sha256=self.data_hash, image_content_sha256=self.content_hash, device_type=self.device.type)
        # Retain immutable optimizer-step versions. Hardlinks save disk space;
        # last.pt is atomically replaced, so older step files keep their bytes.
        snapshot = self.output / f"step-{self.step:06d}.pt"
        if not snapshot.exists():
            try:
                os.link(self.output/"last.pt", snapshot)
            except OSError:
                shutil.copyfile(self.output/"last.pt", snapshot)

    def train(self, stop_after=None):
        cfg, accumulation = self.config, self.config["gradient_accumulation"]
        sampler = TaskSampler(self.dataset.rows, cfg["seed"], cfg.get("task_weights"))
        collator = Collator(self.tokenizer, cfg["max_length"], self.policy["objective"])
        limit = min(cfg["max_steps"], stop_after) if stop_after is not None else cfg["max_steps"]
        self.optimizer.zero_grad(set_to_none=True)
        while self.step < limit:
            sampler.epoch = self.epoch
            loader = DataLoader(self.dataset, batch_size=cfg["batch_size"], sampler=sampler,
                collate_fn=collator, num_workers=0, generator=torch.Generator().manual_seed(cfg["seed"]+self.epoch))
            self.set_modes()
            loss_sum = 0.0
            for index, batch in enumerate(loader):
                if index < self.next_batch:
                    continue
                group_start = index // accumulation * accumulation
                group_size = min(accumulation, len(loader)-group_start)
                batch = move_batch(batch, self.device)
                with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.amp):
                    loss, extra = compute_loss(self.model, batch, self.policy["objective"], self.reference, cfg.get("dpo_beta", 0.1))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss_sum += float(loss.detach()) / group_size
                self.scaler.scale(loss / group_size).backward()
                if (index+1) % accumulation == 0 or index+1 == len(loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], cfg.get("max_grad_norm", 1.0))
                    old_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler.get_scale() < old_scale:
                        raise FloatingPointError("FP16 overflow skipped a step; retry with BF16 or lower learning rate")
                    self.scheduler.step()
                    self.step += 1
                    self.next_batch = index+1
                    if self.next_batch == len(loader):
                        self.epoch += 1
                        self.next_batch = 0
                    metrics = dict(step=self.step, loss=loss_sum, lr=self.optimizer.param_groups[0]["lr"], **extra)
                    if self.device.type == "cuda":
                        metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
                    with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(metrics) + "\n")
                    print(json.dumps(metrics), flush=True)
                    loss_sum = 0.0
                    if self.step % cfg.get("save_every", 100) == 0 or self.step >= limit:
                        self.save()
                    if self.step >= limit:
                        return self.step
        return self.step
