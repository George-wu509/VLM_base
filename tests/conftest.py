import json
import pytest
import torch
from PIL import Image
from models.config import VLMConfig
from models.vlm import VLM
from tokenization.tokenizer import ByteTokenizer
from datasets.multimodal_dataset import MultimodalDataset
from training.config import ROOT, read_yaml, seed_everything

@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    seed_everything(123)

@pytest.fixture
def cfg():
    return VLMConfig(**read_yaml(ROOT/"configs/model/tiny_vlm.yaml"))

@pytest.fixture
def model(cfg):
    return VLM(cfg)

@pytest.fixture
def tokenizer():
    return ByteTokenizer(True)

@pytest.fixture
def make_dataset(tmp_path):
    def make(task="caption", count=5):
        rows = []
        for i in range(count):
            Image.new("RGB", (32, 32), (i*20, 40, 100)).save(tmp_path/f"{i}.png")
            row = dict(id=str(i), task=task, image=f"{i}.png", prompt="Color?", response="blue"+"!"*i)
            if task == "preference":
                row.update(chosen="blue", rejected="red")
            rows.append(row)
        file = tmp_path / f"{task}.jsonl"
        file.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
        return MultimodalDataset(file, tmp_path, 32)
    return make

@pytest.fixture
def train_config():
    return dict(stage="alignment", seed=42, batch_size=1, gradient_accumulation=2, max_steps=5,
                max_length=128, learning_rate=0.001, warmup_steps=0, precision="fp32", save_every=2,
                weight_decay=0, max_grad_norm=1.0)
