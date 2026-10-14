import io
import torch
from PIL import Image
from fastapi.testclient import TestClient
from training.checkpoint import save_checkpoint, model_from_checkpoint
from scripts.export import export
from inference.generate import generate_ids, generate_text
from inference.grounding_parser import parse_grounding
from tokenization.spatial_tokens import encode_box
from tokenization.tokenizer import ByteTokenizer
import pytest
from serving.api import create_app

def test_greedy_generation_is_reproducible(model, tokenizer):
    image = torch.zeros(3, 32, 32)
    first = generate_ids(model, tokenizer, "caption en", image, max_new_tokens=4)
    assert first == generate_ids(model, tokenizer, "caption en", image, max_new_tokens=4)
    assert len(first) <= 4

def test_eos_stops_generation(model, tokenizer):
    with torch.no_grad():
        model.decoder.head.weight.zero_()
    # Tied zero head returns all-zero logits; argmax chooses 0.
    tokenizer.eos_token_id = 0
    assert generate_ids(model, tokenizer, "", max_new_tokens=8) == []

def test_grounding_parser_keeps_valid_boxes_and_rejects_invalid():
    box = [0.125, 0.25, 0.75, 0.875]
    text = encode_box(box)+" rectangle ; <loc9999><loc0000><loc0100><loc0100> invalid"
    assert parse_grounding(text) == [{"bbox": box, "label": "rectangle"}]
    assert parse_grounding("nothing") == []

def test_export_bundle_is_offline_and_preserves_predictions(model, tokenizer, tmp_path):
    save_checkpoint(tmp_path/"training/last.pt", model, tokenizer, lora=None)
    model_file = export(tmp_path/"training/last.pt", tmp_path/"export")
    restored, tok, _ = model_from_checkpoint(model_file)
    assert generate_ids(model, tokenizer, "caption en", max_new_tokens=3) == generate_ids(restored, tok, "caption en", max_new_tokens=3)
    assert (tmp_path/"export/artifact.json").exists()

def test_checkpoint_rejects_replaced_tokenizer(model, tokenizer, tmp_path):
    file = tmp_path/"model.pt"
    save_checkpoint(file, model, tokenizer, lora=None)
    ByteTokenizer(False).save_pretrained(tmp_path/"tokenizer")
    with pytest.raises(ValueError, match="vocabulary"):
        model_from_checkpoint(file)

def test_api_upload_inference_and_bad_image(model, tokenizer, tmp_path):
    file = tmp_path/"api/model.pt"
    save_checkpoint(file, model, tokenizer, lora=None)
    with TestClient(create_app(file, device="cpu")) as client:
        assert client.get("/health").json()["status"] == "ready"
        buffer = io.BytesIO()
        Image.new("RGB", (20, 20), "red").save(buffer, format="PNG")
        result = client.post("/generate", files={"image": ("example.png", buffer.getvalue(), "image/png")}, data={"max_new_tokens": "2"})
        assert result.status_code == 200
        assert set(result.json()) == {"text", "objects"}
        bad = client.post("/generate", files={"image": ("bad.png", b"bad", "image/png")})
        assert bad.status_code == 400
