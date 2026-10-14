import copy
import pytest
import torch
from datasets.collator import Collator
from training.trainer import Trainer
from training.losses import compute_loss, sequence_log_probs
from training.checkpoint import merge_lora
from models.lora import inject_lora

def test_alignment_updates_only_projector_and_reaches_frozen_decoder(model, tokenizer, make_dataset, train_config, tmp_path):
    data = make_dataset()
    trainer = Trainer(model, tokenizer, data, train_config, tmp_path/"alignment", torch.device("cpu"))
    batch = Collator(tokenizer)([data[0]])
    model(**batch)["loss"].backward()
    assert model.projector.proj.weight.grad is not None
    assert model.projector.proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.decoder.parameters())
    assert all(p.grad is None for p in model.vision_encoder.parameters())
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trainer.train(stop_after=1)
    for k, v in model.state_dict().items():
        if not k.startswith("projector"):
            assert torch.equal(v, before[k]), k
    assert not torch.equal(model.projector.proj.weight, before["projector.proj.weight"])

def test_lora_merges_without_changing_predictions(model):
    model.requires_grad_(False)
    inject_lora(model, dict(rank=2, alpha=4, targets=["q_proj", "v_proj"]))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".B"):
                p.normal_(std=0.01)
    model.eval()
    ids = torch.tensor([[1, 10, 20]])
    expected = model(ids)["logits"].detach()
    merge_lora(model)
    assert torch.allclose(model(ids)["logits"], expected, atol=1e-6)
    assert not any(name.endswith((".A", ".B")) for name, _ in model.named_parameters())

def test_dpo_reference_is_fixed_and_loss_starts_at_log_two(model, tokenizer, make_dataset, train_config, tmp_path):
    data = make_dataset("preference")
    train_config.update(stage="dpo", lora=dict(rank=2, alpha=4, dropout=0, targets=["q_proj", "v_proj"]))
    trainer = Trainer(model, tokenizer, data, train_config, tmp_path/"dpo", torch.device("cpu"))
    reference = {k: v.clone() for k, v in trainer.reference.state_dict().items()}
    batch = Collator(tokenizer, objective="dpo")([data[0]])
    loss, _ = compute_loss(model, batch, "dpo", trainer.reference)
    assert abs(loss.item() - 0.69314718) < 1e-6
    trainer.train(stop_after=2)
    assert all(torch.equal(v, reference[k]) for k, v in trainer.reference.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in trainer.reference.parameters())
    after, _ = compute_loss(model, batch, "dpo", trainer.reference)
    assert after < loss

def test_log_probs_ignore_prompt_and_sum_completion():
    logits = torch.tensor([[[0., 1.], [1., 0.], [0., 2.], [3., 0.]]])
    labels = torch.tensor([[-100, -100, 1, 1]])
    expected = logits[0, 1].log_softmax(0)[1]+logits[0, 2].log_softmax(0)[1]
    result = sequence_log_probs({"logits": logits, "labels": labels})
    assert torch.allclose(result[0], expected)

def test_checkpointing_preserves_alignment_gradients(model, tokenizer, make_dataset):
    data = make_dataset()
    model.requires_grad_(False)
    model.projector.requires_grad_(True)
    model.train()
    batch = Collator(tokenizer)([data[0]])
    model(**batch)["loss"].backward()
    expected = model.projector.proj.weight.grad.clone()
    model.zero_grad()
    model.cfg.gradient_checkpointing = True
    model(**batch)["loss"].backward()
    assert torch.allclose(model.projector.proj.weight.grad, expected, atol=1e-7)
