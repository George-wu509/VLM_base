import json
import pytest
import torch
from PIL import Image
from datasets.collator import Collator
from datasets.adapters import adapt
from datasets.multimodal_dataset import MultimodalDataset
from datasets.task_sampler import TaskSampler
from scripts.prepare_data import prepare
from tokenization.spatial_tokens import encode_box, decode_box

def test_completion_mask_and_eos_padding(tokenizer, make_dataset):
    data = make_dataset()
    batch = Collator(tokenizer)([data[0], data[4]])
    n_prompt = 1+len(tokenizer.encode("Color?\n"))
    assert (batch["labels"][:, :n_prompt] == -100).all()
    assert batch["labels"][0, n_prompt] == tokenizer.encode("b")[0]
    assert batch["labels"][0, n_prompt+4] == tokenizer.eos_token_id
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    with pytest.raises(ValueError, match="exceeds max_length"):
        Collator(tokenizer, 3)([data[0]])

def test_spatial_and_unicode_roundtrip(tokenizer):
    box = [0.125, 0.25, 0.75, 0.875]
    text = encode_box(box)+" 紅色方框"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert len(tokenizer.encode(encode_box(box))) == 4
    assert decode_box([256, 128, 896, 768]) == box
    with pytest.raises(ValueError):
        encode_box([0.5, 0.5, 0.2, 0.7])

def test_data_split_groups_and_duplicate_removal(tmp_path):
    for i in range(4):
        Image.new("RGB", (10, 10), (i*50, 0, 0)).save(tmp_path/f"{i}.png")
    rows = [dict(id=f"{i}-{task}", task="caption", image=f"{i}.png", caption=f"{i} {task}")
            for i in range(4) for task in ("first", "second")]
    rows.append(rows[0])
    # A different filename with the same bytes must follow the same split.
    (tmp_path/"duplicate.png").write_bytes((tmp_path/"0.png").read_bytes())
    rows.append(dict(id="same-image", task="vqa", image="duplicate.png", question="Color?", answer="red"))
    raw = tmp_path/"raw.jsonl"
    raw.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    report = prepare([raw], tmp_path, tmp_path/"prepared", 0.5)
    assert report["duplicates_removed"] == 1
    train = MultimodalDataset(tmp_path/"prepared/train.jsonl", tmp_path, 32)
    val = MultimodalDataset(tmp_path/"prepared/val.jsonl", tmp_path, 32)
    assert {r["image_sha256"] for r in train.rows}.isdisjoint({r["image_sha256"] for r in val.rows})
    row = train.rows[0]
    Image.new("RGB", (10, 10), "green").save(tmp_path/row["image"])
    with pytest.raises(ValueError, match="content changed"):
        MultimodalDataset(tmp_path/"prepared/train.jsonl", tmp_path, 32)

def test_group_id_connects_multiple_images(tmp_path):
    rows = []
    for i in range(6):
        Image.new("RGB", (10, 10), (i*30, 0, 0)).save(tmp_path/f"{i}.png")
        rows.append(dict(task="caption", image=f"{i}.png", caption="example", group_id=str(i//2)))
    raw = tmp_path/"raw.jsonl"
    raw.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    prepare([raw], tmp_path, tmp_path/"prepared", 0.4)
    split_groups = []
    for split in ("train", "val"):
        data = MultimodalDataset(tmp_path/f"prepared/{split}.jsonl", tmp_path, 32)
        split_groups.append({r["group_id"] for r in data.rows})
    assert split_groups[0].isdisjoint(split_groups[1])

def test_sampler_resume_determinism(make_dataset):
    data = make_dataset(count=12)
    sampler = TaskSampler(data.rows, seed=42)
    first = list(sampler)
    assert first == list(sampler)
    sampler.epoch = 1
    assert first != list(sampler)

@pytest.mark.parametrize("record", [dict(task="vqa", image="x", answer=["blue"]),
    dict(task="image", image="x", label=-1), dict(task="preference", image="x", prompt="a", chosen="b", rejected="b")])
def test_malformed_rows_rejected(record):
    with pytest.raises(ValueError):
        adapt(record)
