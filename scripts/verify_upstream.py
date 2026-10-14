"""Verify vendored reference bytes against the pinned GitHub tree's Git blob IDs."""
import argparse
import hashlib
import json
from pathlib import Path
from training.config import ROOT

COMMIT = "6ba9082e16f1fc8c21a1f8d0c54b26c9233c8771"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    root = ROOT/"vendor/nanovlm_v01"
    tree = json.loads((root/"tree.json").read_text(encoding="utf-8-sig"))
    if tree["sha"] != COMMIT:
        raise ValueError("Upstream tree has incorrect revision")
    manifest = {"repository": "https://github.com/huggingface/nanoVLM", "tag": "v0.1", "commit": COMMIT, "files": {}}
    for entry in tree["tree"]:
        path = root/entry["path"]
        if entry["type"] != "blob" or not path.is_file():
            continue
        content = path.read_bytes()
        blob = hashlib.sha1(b"blob "+str(len(content)).encode()+b"\x00"+content).hexdigest()
        if blob != entry["sha"]:
            raise ValueError(f"Vendored bytes changed: {entry['path']}")
        manifest["files"][entry["path"]] = {"git_blob_sha1": blob, "sha256": hashlib.sha256(content).hexdigest()}
    if len(manifest["files"]) != 8:
        raise ValueError("Expected eight upstream reference files")
    if args.write_manifest:
        (root/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        expected = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
        if manifest != expected:
            raise ValueError("Upstream manifest mismatch")
    print(f"Verified {len(manifest['files'])} original files at {COMMIT}")

if __name__ == "__main__":
    main()
