import pytest
import torch
import torch.nn.functional as F
from datasets.collator import Collator

def test_prefix_shift_and_padding_loss(model, tokenizer, make_dataset):
    dataset = make_dataset()
    batch = Collator(tokenizer)([dataset[0], dataset[4]])
    result = model(**batch)
    assert result["logits"].shape == (2, batch["input_ids"].shape[1]+4, len(tokenizer))
    assert (result["labels"][:, :4] == -100).all()
    logits, labels = result["logits"], result["labels"]
    terms = []
    for b in range(2):
        for t in range(1, labels.shape[1]):
            if labels[b, t] != -100:
                terms.append(-logits[b, t-1].float().log_softmax(-1)[labels[b, t]])
    assert torch.allclose(result["loss"], torch.stack(terms).mean())

def test_causality_and_right_padding(model):
    model.eval()
    ids = torch.tensor([[1, 10, 20, 30]])
    later = ids.clone()
    later[0, -1] = 80
    a = model(ids)["logits"]
    b = model(later)["logits"]
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)
    padded = torch.cat([ids, torch.zeros((1, 3), dtype=torch.long)], 1)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
    c = model(padded, attention_mask=mask)["logits"]
    assert torch.allclose(a, c[:, :4], atol=1e-6)

def test_projector_grid_validation(model):
    with pytest.raises(ValueError, match="square grid"):
        model.projector(torch.randn(1, 15, 32))

def test_empty_targets_rejected(model):
    ids = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="no supervised"):
        model(ids, labels=torch.full_like(ids, -100))
