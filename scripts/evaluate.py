import argparse
import json
import math
from pathlib import Path
import torch
from datasets.multimodal_dataset import MultimodalDataset
from datasets.collator import Collator
from inference.generate import generate_text
from training.config import ROOT, device_for, move_batch
from training.checkpoint import model_from_checkpoint
from training.experiment import sha256
from evaluation.caption import rouge_l
from evaluation.vqa import exact_match
from evaluation.grounding import grounding_metrics
from evaluation.hallucination import score_yes_no
from evaluation.compare_checkpoints import compare

def evaluate(checkpoint, data, root, device, output, max_new_tokens=64, presence_probe=False):
    model, tokenizer, payload = model_from_checkpoint(checkpoint)
    model.to(device).eval()
    dataset = MultimodalDataset(data, root, model.cfg.vit_img_size)
    collator = Collator(tokenizer, model.cfg.lm_max_position_embeddings - 49)
    scores, predictions, presence_pred, presence_truth = {}, [], [], []
    nll, token_count = 0.0, 0
    for index in range(len(dataset)):
        row = dataset[index]
        if row["task"] not in ("caption", "image_text", "vqa", "grounding", "text"):
            raise ValueError("Evaluation expects caption/VQA/grounding/text records")
        batch = move_batch(collator([row]), device)
        with torch.inference_mode():
            result = model(**batch)
        count = int((result["labels"][:, 1:] != -100).sum())
        nll += float(result["loss"]) * count
        token_count += count
        text = generate_text(model, tokenizer, row["prompt"], row.get("pixel_values"), max_new_tokens=max_new_tokens)
        local = {"exact_match": exact_match(text, row["response"])}
        if row["task"] in ("caption", "image_text"):
            local["caption_rouge_l"] = rouge_l(text, row["response"])
        if row["task"] == "grounding":
            local.update(grounding_metrics(text, row.get("objects", row["response"])))
        if presence_probe and row["task"] == "vqa":
            presence_pred.append(text)
            presence_truth.append(row["response"])
        for key, value in local.items():
            scores.setdefault(f"{row['task']}/{key}", []).append(value)
        predictions.append({"id": row.get("id", str(index)), "task": row["task"],
                            "prediction": text, "reference": row["response"], "scores": local})
    metrics = {k: sum(v)/len(v) for k, v in scores.items()}
    metrics.update(loss=nll/token_count, perplexity=math.exp(min(100, nll/token_count)), examples=len(dataset))
    if presence_probe:
        metrics.update(score_yes_no(presence_pred, presence_truth))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256(checkpoint),
                  manifest_sha256=sha256(data), image_content_sha256=dataset.content_sha256, metrics=metrics,
                  generation={"temperature": 0.0, "max_new_tokens": max_new_tokens})
    (output/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output/"predictions.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False)+"\n" for p in predictions), encoding="utf-8")
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--data")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="artifacts/metrics/eval")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--presence-probe", action="store_true")
    parser.add_argument("--compare", nargs="+")
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(compare(args.compare), indent=2))
    else:
        if not args.checkpoint or not args.data:
            parser.error("--checkpoint and --data required unless --compare is used")
        print(json.dumps(evaluate(args.checkpoint, args.data, args.root, device_for(args.device),
              args.output, args.max_new_tokens, args.presence_probe), indent=2))

if __name__ == "__main__":
    main()
