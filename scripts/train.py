import argparse
import json
from dataclasses import asdict
from pathlib import Path
from datasets.multimodal_dataset import MultimodalDataset
from datasets.image_dataset import ImageDataset
from datasets.text_dataset import TextDataset
from models.config import VLMConfig
from tokenization.tokenizer import build_tokenizer, ByteTokenizer, fingerprint
from training.config import ROOT, read_yaml, seed_everything, device_for, new_model
from training.checkpoint import model_from_checkpoint, merge_lora
from training.experiment import write_run_metadata
from training.trainer import Trainer
from training.stages import get_stage

def main():
    parser = argparse.ArgumentParser(description="One trainer for every VLM stage")
    parser.add_argument("--model", default="configs/model/pretrained_vlm.yaml")
    parser.add_argument("--training", required=True)
    parser.add_argument("--data")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", help="Same config/data; restores full optimizer/RNG/cursor state")
    mode.add_argument("--init-from", help="New stage; restores model/tokenizer and merges any old LoRA")
    parser.add_argument("--vision-from", help="Import only vision weights from a vision-pretrain checkpoint")
    parser.add_argument("--decoder-from", help="Import only decoder weights from a llm-pretrain checkpoint")
    parser.add_argument("--enable-spatial-tokens", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Testing interruption without changing max_steps")
    args = parser.parse_args()
    config = read_yaml(args.training)
    if args.data:
        config["data"] = args.data
    seed_everything(config["seed"])
    device = device_for(args.device)
    resume = None
    if args.resume or args.init_from:
        model, tokenizer, payload = model_from_checkpoint(args.resume or args.init_from)
        if args.resume:
            resume = payload
            if args.enable_spatial_tokens or args.vision_from or args.decoder_from:
                raise ValueError("Resume cannot change model/tokenizer/backbones")
        else:
            merge_lora(model)
    else:
        model, tokenizer = new_model(read_yaml(args.model))
    if args.enable_spatial_tokens and not model.cfg.spatial_tokens:
        if config["stage"] == "alignment" or not config.get("train_embeddings"):
            raise ValueError("Adding spatial tokens requires an SFT/LoRA stage with train_embeddings: true")
        model.cfg.spatial_tokens = True
        tokenizer = (ByteTokenizer(True) if model.cfg.tokenizer_backend == "byte" else
                     build_tokenizer(model.cfg, Path(args.init_from).resolve().parent/"tokenizer" if args.init_from else None))
        model.decoder.resize_token_embeddings(len(tokenizer))
    for source, name in ((args.vision_from, "vision_encoder"), (args.decoder_from, "decoder")):
        if source:
            other, other_tok, _ = model_from_checkpoint(source)
            if name == "decoder" and fingerprint(other_tok) != fingerprint(tokenizer):
                raise ValueError("Decoder tokenizer must match target tokenizer")
            getattr(model, name).load_state_dict(getattr(other, name).state_dict(), strict=True)
    model.cfg.gradient_checkpointing = config.get("gradient_checkpointing", False)
    stage = config["stage"]
    dataset_cls = ImageDataset if stage == "vision_pretrain" else TextDataset if stage == "llm_pretrain" else MultimodalDataset
    kw = {"num_classes": model.cfg.vision_num_classes} if stage == "vision_pretrain" else {}
    dataset = dataset_cls(config["data"], args.root, model.cfg.vit_img_size, **kw)
    if stage not in ("vision_pretrain", "llm_pretrain"):
        allowed = {"preference"} if stage == "dpo" else {"caption", "image_text", "vqa", "grounding"}
        if any(r["task"] not in allowed for r in dataset.rows):
            raise ValueError(f"Stage {stage} supports only {sorted(allowed)}")
        if any(r["task"] == "grounding" for r in dataset.rows) and not model.cfg.spatial_tokens:
            raise ValueError("Grounding requires --enable-spatial-tokens and train_embeddings: true")
    output = Path(args.output)
    if not args.resume and (output/"last.pt").exists():
        raise ValueError("Output already contains a checkpoint; choose a new output or --resume")
    trainer = Trainer(model, tokenizer, dataset, config, output, device, resume)
    write_run_metadata(output, model, config, dataset, args.resume, args.init_from)
    trainable = {name: sum(p.numel() for p in getattr(model, name).parameters() if p.requires_grad)
                 for name in ("vision_encoder", "projector", "decoder", "vision_head")}
    print(json.dumps({"stage": stage, "device": str(device), "trainable": trainable}))
    trainer.train(args.stop_after)

if __name__ == "__main__":
    main()
