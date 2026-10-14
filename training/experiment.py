import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
import torch

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_run_metadata(out, model, config, dataset, resume=None, initialize=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    source_files = []
    root = Path(__file__).resolve().parents[1]
    for component in ("models", "datasets", "tokenization", "training", "evaluation", "inference", "serving", "scripts", "configs"):
        for file in sorted((root/component).rglob("*")):
            if file.is_file() and file.suffix in (".py", ".yaml"):
                source_files.append((file.relative_to(root).as_posix(), sha256(file)))
    metadata = {"model": asdict(model.cfg), "training": config, "manifest_sha256": sha256(dataset.manifest),
                "image_content_sha256": dataset.content_sha256,
                "manifest": str(dataset.manifest), "git_commit": commit, "python": platform.python_version(),
                "packages": packages, "cuda": torch.version.cuda, "resume": resume, "initialize": initialize,
                "source_sha256": dict(source_files),
                "upstream_commit": "6ba9082e16f1fc8c21a1f8d0c54b26c9233c8771"}
    filename = "resume.json" if resume else "run.json"
    (out / filename).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
