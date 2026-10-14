import argparse
import json
from pathlib import Path
from training.checkpoint import model_from_checkpoint, save_checkpoint, merge_lora
from training.experiment import sha256

def export(checkpoint, output):
    output = Path(output)
    if (output/"model.pt").exists():
        raise ValueError("Export destination already contains model.pt; choose a versioned directory")
    model, tokenizer, payload = model_from_checkpoint(checkpoint)
    merge_lora(model).eval()
    save_checkpoint(output/"model.pt", model, tokenizer, lora=None,
                    source_checkpoint_sha256=sha256(checkpoint), training_stage=payload.get("training_config", {}).get("stage"))
    files = {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob("*") if p.is_file()}
    (output/"artifact.json").write_text(json.dumps({"source": str(Path(checkpoint).resolve()), "files": files}, indent=2), encoding="utf-8")
    return str(output/"model.pt")

def main():
    parser = argparse.ArgumentParser(description="Export inference weights, merged LoRA and exact tokenizer")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export(args.checkpoint, args.output))

if __name__ == "__main__":
    main()
