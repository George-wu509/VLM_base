import json
from pathlib import Path

def compare(paths):
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if len({r["manifest_sha256"] for r in reports}) != 1:
        raise ValueError("Compare checkpoints on the same evaluation manifest")
    if len({r["image_content_sha256"] for r in reports}) != 1:
        raise ValueError("Evaluation image content must match")
    if len({json.dumps(r["generation"], sort_keys=True) for r in reports}) != 1:
        raise ValueError("Generation settings must match for a fair comparison")
    return [{"checkpoint": r["checkpoint"], **r["metrics"]} for r in reports]
