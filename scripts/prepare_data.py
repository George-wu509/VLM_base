"""Validate, deduplicate and split canonical JSONL by IMAGE content, not by row."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from PIL import Image
from datasets.adapters import adapt
from training.config import ROOT
from training.experiment import sha256

def prepare(inputs, root, output, val_ratio=0.1, seed=42):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0,1)")
    root, output = Path(root).resolve(), Path(output).resolve()
    groups, seen, ids, duplicate_count = defaultdict(list), set(), set(), 0
    for path in inputs:
        for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = adapt(json.loads(line))
                if row.get("image"):
                    image = (root / row["image"]).resolve()
                    image.relative_to(root)
                    with Image.open(image) as source:
                        source.verify()
                    row["image"] = image.relative_to(root).as_posix()
                    row["image_sha256"] = sha256(image)
                    group = row["image_sha256"]
                else:
                    group = hashlib.sha256(row["response"].encode()).hexdigest()
                canonical = json.dumps({k: v for k, v in row.items() if k != "id"}, sort_keys=True, ensure_ascii=False)
                if canonical in seen:
                    duplicate_count += 1
                    continue
                seen.add(canonical)
                row.setdefault("id", hashlib.sha256(canonical.encode()).hexdigest()[:20])
                if row["id"] in ids:
                    raise ValueError("Duplicate record ID with different content")
                ids.add(row["id"])
                # Optional group_id keeps documents, subjects or source clusters together.
                groups[group].append(row)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    # Connected components enforce BOTH image disjointness and user group_id.
    parent = {k: k for k in groups}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    clusters = {}
    for key, rows in groups.items():
        for row in rows:
            if row.get("group_id") is not None:
                gid = str(row["group_id"])
                if gid in clusters:
                    parent[find(key)] = find(clusters[gid])
                clusters[gid] = key
    merged = defaultdict(list)
    for key, rows in groups.items():
        merged[find(key)].extend(rows)
    keys = sorted(merged)
    if len(keys) < 2:
        raise ValueError("At least two independent image/text groups required for a disjoint split")
    random.Random(seed).shuffle(keys)
    n_val = max(1, min(len(keys)-1, round(len(keys)*val_ratio)))
    splits = {"val": keys[:n_val], "train": keys[n_val:]}
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, split_keys in splits.items():
        rows = [r for key in split_keys for r in merged[key]]
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
        counts[split] = len(rows)
    meta = dict(seed=seed, val_ratio=val_ratio, root=str(root), records=counts, groups=len(keys),
                duplicates_removed=duplicate_count, inputs={str(p): sha256(p) for p in inputs},
                splits={s: sha256(output/f"{s}.jsonl") for s in splits})
    (output/"manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output", default="data/manifests/main")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.root, args.output, args.val_ratio, args.seed), indent=2))

if __name__ == "__main__":
    main()
