"""Optional integration check: native v0.1 outputs vs Transformers reference.

Requires cached pretrained models (or network for first download). This is not
part of the fast offline unit suite. Run with HF_HUB_OFFLINE=1 after alignment.
"""
import argparse
import json
from pathlib import Path
import torch
from training.config import ROOT, new_model, read_yaml, device_for, seed_everything

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="configs/model/pretrained_vlm.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/metrics/backbone-parity.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed_everything(42)
    model, tokenizer = new_model(read_yaml(args.model))
    device = device_for(args.device)
    model.to(device).eval()
    from transformers import SiglipVisionModel, AutoModelForCausalLM
    cfg = model.cfg
    reference_vision = SiglipVisionModel.from_pretrained(cfg.vit_model_type, revision=cfg.vit_revision,
                                                       local_files_only=True).to(device).eval()
    pixels = torch.randn(1, 3, cfg.vit_img_size, cfg.vit_img_size, device=device)
    with torch.inference_mode():
        vision = model.vision_encoder(pixels)
        expected = reference_vision(pixels).last_hidden_state
    vision_delta = float((vision-expected).abs().max())
    del reference_vision
    reference_lm = AutoModelForCausalLM.from_pretrained(cfg.lm_model_type, revision=cfg.lm_revision,
                                                      local_files_only=True).to(device).eval()
    ids = torch.tensor([[tokenizer.bos_token_id]+tokenizer.encode("A red rectangle is visible.")], device=device)
    with torch.inference_mode():
        actual = model(ids)["logits"]
        expected = reference_lm(ids).logits
    lm_delta = float((actual[..., :expected.size(-1)]-expected).abs().max())
    report = {"vision_max_abs_delta": vision_delta, "lm_max_abs_delta": lm_delta,
              "device": str(device), "dtype": "fp32", "vision_revision": cfg.vit_revision,
              "lm_revision": cfg.lm_revision, "tolerance": 0.001,
              "passed": vision_delta < 0.001 and lm_delta < 0.001}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Backbone output parity failed")

if __name__ == "__main__":
    main()
