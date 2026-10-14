import json
import hashlib
from pathlib import Path
from .spatial_tokens import SPATIAL_TOKENS, TOKEN_PATTERN

class ByteTokenizer:
    """Deterministic UTF-8 tokenizer for OFFLINE pipeline tests, not a pretrained LM."""
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    def __init__(self, spatial=False):
        self.spatial = spatial
        self.extra = {s: 259+i for i, s in enumerate(SPATIAL_TOKENS)} if spatial else {}
        self.inverse = {v: k for k, v in self.extra.items()}

    def __len__(self):
        return 259 + len(self.extra)

    def encode(self, text):
        out, start = [], 0
        for match in TOKEN_PATTERN.finditer(text) if self.spatial else []:
            out.extend(3 + b for b in text[start:match.start()].encode("utf-8"))
            if match.group() not in self.extra:
                raise ValueError(f"Invalid spatial token {match.group()}")
            out.append(self.extra[match.group()])
            start = match.end()
        out.extend(3+b for b in text[start:].encode("utf-8"))
        return out

    def decode(self, ids):
        result, pending = [], bytearray()
        for value in ids:
            value = int(value)
            if 3 <= value < 259:
                pending.append(value-3)
            else:
                result.append(pending.decode("utf-8", errors="replace"))
                pending.clear()
                if value in self.inverse:
                    result.append(self.inverse[value])
        result.append(pending.decode("utf-8", errors="replace"))
        return "".join(result)

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "byte_tokenizer.json").write_text(json.dumps({"spatial": self.spatial}), encoding="utf-8")

class HFTokenizer:
    def __init__(self, tokenizer):
        self.raw = tokenizer
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires an EOS token")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.raw)

    def encode(self, text):
        return self.raw.encode(text, add_special_tokens=False)

    def decode(self, ids):
        # Keep loc/seg tokens for grounding; remove only boundary/padding tokens.
        ids = [int(i) for i in ids if int(i) not in (self.bos_token_id, self.eos_token_id, self.pad_token_id)]
        return self.raw.decode(ids, skip_special_tokens=False)

    def save_pretrained(self, path):
        self.raw.save_pretrained(path)

def build_tokenizer(cfg, saved_path=None):
    path = Path(saved_path) if saved_path else None
    if path and (path / "byte_tokenizer.json").exists():
        return ByteTokenizer(**json.loads((path / "byte_tokenizer.json").read_text(encoding="utf-8")))
    if cfg.tokenizer_backend == "byte":
        return ByteTokenizer(cfg.spatial_tokens)
    from transformers import AutoTokenizer
    raw = AutoTokenizer.from_pretrained(str(path) if path else cfg.lm_tokenizer,
                                        revision=None if path else cfg.tokenizer_revision)
    if cfg.spatial_tokens:
        raw.add_special_tokens({"additional_special_tokens": SPATIAL_TOKENS})
    return HFTokenizer(raw)

def fingerprint(tokenizer):
    data = ({"backend": "byte", "spatial": tokenizer.spatial} if isinstance(tokenizer, ByteTokenizer)
            else json.loads(tokenizer.raw.backend_tokenizer.to_str()))
    data["boundary_ids"] = [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
