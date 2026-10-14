import json
import hashlib
from pathlib import Path
from torch.utils.data import Dataset
from .adapters import adapt
from inference.image_processor import load_image

class MultimodalDataset(Dataset):
    def __init__(self, manifest, root, image_size):
        self.manifest = Path(manifest).resolve()
        self.root, self.image_size = Path(root).resolve(), image_size
        self.rows = []
        for line_no, line in enumerate(self.manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            if line.strip():
                try:
                    self.rows.append(adapt(json.loads(line)))
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{manifest}:{line_no}: {exc}") from exc
        if not self.rows:
            raise ValueError("Dataset is empty")
        # Hash each unique image once: resume/evaluation must see identical bytes.
        hashes = {}
        for row in self.rows:
            if row.get("image"):
                path = (self.root / row["image"]).resolve()
                try:
                    path.relative_to(self.root)
                except ValueError as exc:
                    raise ValueError("Image paths must stay inside dataset root") from exc
                if path not in hashes:
                    digest = hashlib.sha256()
                    with path.open("rb") as f:
                        for block in iter(lambda: f.read(1024*1024), b""):
                            digest.update(block)
                    hashes[path] = digest.hexdigest()
                if row.get("image_sha256") and row["image_sha256"] != hashes[path]:
                    raise ValueError(f"Image content changed: {path}")
        self.content_sha256 = hashlib.sha256(json.dumps(sorted(hashes.items(), key=lambda kv: str(kv[0])),
            default=str).encode()).hexdigest()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        if row.get("image"):
            row["pixel_values"] = load_image(self.root / row["image"], self.image_size)
        return row
