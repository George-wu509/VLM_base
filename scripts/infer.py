import argparse
import json
from training.config import read_yaml, device_for
from training.checkpoint import model_from_checkpoint
from inference.image_processor import load_image
from inference.generate import generate_text
from inference.grounding_parser import parse_grounding
from inference.visualization import draw_grounding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image")
    parser.add_argument("--prompt", default="caption en")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--device")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--draw", help="Output grounding visualization PNG")
    args = parser.parse_args()
    settings = read_yaml(args.config)
    device = device_for(args.device or settings.pop("device"))
    settings.pop("device", None)
    if args.max_new_tokens:
        settings["max_new_tokens"] = args.max_new_tokens
    model, tok, _ = model_from_checkpoint(args.checkpoint)
    model.to(device)
    image = load_image(args.image, model.cfg.vit_img_size) if args.image else None
    text = generate_text(model, tok, args.prompt, image, **settings)
    objects = parse_grounding(text)
    print(json.dumps({"text": text, "objects": objects}, ensure_ascii=True))
    if args.draw:
        if not args.image:
            raise ValueError("--draw requires --image")
        draw_grounding(args.image, objects, args.draw)

if __name__ == "__main__":
    main()
