"""Small synthetic fixtures for pipeline checks; no claim of learned VLM capability."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from scripts.prepare_data import prepare
from training.config import ROOT

def make_demo(root):
    root = Path(root)
    images = root / "data/raw/images/demo"
    annotations = root / "data/annotations/demo"
    images.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(parents=True, exist_ok=True)
    records = {k: [] for k in ("alignment", "multitask", "preference", "image", "text")}
    for i in range(16):
        color = "red" if i % 2 else "blue"
        image = Image.new("RGB", (64, 64), "white")
        x = 8 + i
        ImageDraw.Draw(image).rectangle((x, 16, x+20, 44), fill=color)
        relative = f"data/raw/images/demo/{i:02d}.png"
        image.save(root / relative)
        base = {"image": relative, "group_id": str(i)}
        caption = dict(base, id=f"caption-{i}", task="caption", response=f"A {color} rectangle.")
        records["alignment"].append(caption)
        records["multitask"].extend([caption,
            dict(base, id=f"vqa-{i}", task="vqa", question="What color?", answer=color),
            dict(base, id=f"ground-{i}", task="grounding", query="rectangle",
                 objects=[{"label": color + " rectangle", "bbox": [x/64, 16/64, (x+20)/64, 44/64]}])])
        records["preference"].append(dict(base, id=f"pref-{i}", task="preference", prompt="answer What color?",
             chosen=color, rejected="green"))
        records["image"].append(dict(base, id=f"image-{i}", task="image", label=i % 2))
        records["text"].append(dict(id=f"text-{i}", task="text", text=f"Example {i}: a {color} rectangle is visible."))
    for name, rows in records.items():
        file = annotations / f"{name}.jsonl"
        file.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
        prepare([file], root, root/f"data/manifests/demo/{name}", 0.25)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    make_demo(args.root)
    print("Synthetic manifests created under data/manifests/demo")

if __name__ == "__main__":
    main()
