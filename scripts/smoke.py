"""Run ALL seven stage policies through the SAME Trainer on tiny synthetic data."""
import argparse
import json
from pathlib import Path
import torch
from datasets.multimodal_dataset import MultimodalDataset
from scripts.make_demo_data import make_demo
from scripts.evaluate import evaluate
from scripts.export import export
from training.config import ROOT, read_yaml, new_model, seed_everything, device_for
from training.trainer import Trainer
from training.checkpoint import model_from_checkpoint, merge_lora
from training.experiment import write_run_metadata

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--output", default="artifacts/runs/smoke")
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = Path(args.output)
    if (output/"smoke_report.json").exists():
        raise ValueError("Choose a new --output for a new smoke run")
    make_demo(ROOT)
    device = device_for(args.device)
    checkpoints = {}
    for stage in ("vision_pretrain", "llm_pretrain", "alignment", "multimodal_pretrain", "sft", "lora", "dpo"):
        config = read_yaml(ROOT/"configs/training/smoke.yaml")
        config["stage"], config["precision"] = stage, args.precision
        data = "image" if stage == "vision_pretrain" else "text" if stage == "llm_pretrain" else "preference" if stage == "dpo" else "multitask" if stage in ("sft", "lora") else "alignment"
        config["data"] = str(ROOT/f"data/manifests/demo/{data}/train.jsonl")
        if stage in ("lora", "dpo"):
            config["lora"] = {"rank": 2, "alpha": 4, "dropout": 0.0, "targets": ["q_proj", "v_proj"]}
        if stage == "sft":
            config["train_embeddings"] = True
        seed_everything(config["seed"])
        previous = {"multimodal_pretrain": "alignment", "sft": "multimodal_pretrain", "lora": "sft", "dpo": "lora"}.get(stage)
        if previous:
            model, tok, _ = model_from_checkpoint(checkpoints[previous])
            merge_lora(model)
        else:
            model, tok = new_model(read_yaml(ROOT/"configs/model/tiny_vlm.yaml"))
        if stage == "alignment":
            for name, source in (("vision_encoder", "vision_pretrain"), ("decoder", "llm_pretrain")):
                backbone, _, _ = model_from_checkpoint(checkpoints[source])
                getattr(model, name).load_state_dict(getattr(backbone, name).state_dict())
        dataset = MultimodalDataset(config["data"], ROOT, model.cfg.vit_img_size)
        stage_out = output/stage
        trainer = Trainer(model, tok, dataset, config, stage_out, device)
        write_run_metadata(stage_out, model, config, dataset)
        trainer.train()
        checkpoints[stage] = str(stage_out/"last.pt")
        del trainer, model
    report = evaluate(checkpoints["dpo"], ROOT/"data/manifests/demo/multitask/val.jsonl", ROOT,
                      device, output/"evaluation", max_new_tokens=8)
    exported = export(checkpoints["dpo"], output/"export")
    summary = {"device": str(device), "precision": args.precision, "stages": checkpoints,
               "evaluation": report["metrics"], "export": exported,
               "note": "Synthetic pipeline verification only; not a model quality benchmark"}
    (output/"smoke_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
