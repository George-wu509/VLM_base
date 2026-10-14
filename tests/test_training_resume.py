import copy
import pytest
import torch
from models.vlm import VLM
from training.config import seed_everything
from training.trainer import Trainer
from training.checkpoint import model_from_checkpoint

@pytest.mark.parametrize("stage", ["alignment", "dpo", "sft"])
def test_resume_matches_uninterrupted_even_last_partial_accumulation(cfg, tokenizer, make_dataset, train_config, tmp_path, stage):
    cfg.lm_dropout = 0.1 if stage == "sft" else 0.0
    data = make_dataset("preference" if stage == "dpo" else "caption", count=5)
    train_config["stage"] = stage
    if stage == "dpo":
        train_config["lora"] = dict(rank=2, alpha=4, dropout=0, targets=["q_proj", "v_proj"])
    seed_everything(777)
    full = Trainer(VLM(cfg), tokenizer, data, train_config, tmp_path/"full", torch.device("cpu"))
    full.train()
    expected = {k: v.clone() for k, v in full.model.state_dict().items()}
    seed_everything(777)
    partial = Trainer(VLM(cfg), tokenizer, data, train_config, tmp_path/"partial", torch.device("cpu"))
    partial.train(stop_after=2)
    # Disturb RNG and construct new model to make sure the checkpoint restores it.
    seed_everything(999)
    model, saved_tok, payload = model_from_checkpoint(tmp_path/"partial/last.pt")
    continued = Trainer(model, saved_tok, data, train_config, tmp_path/"partial", torch.device("cpu"), payload)
    continued.train()
    for key, value in continued.model.state_dict().items():
        assert torch.equal(value, expected[key]), key
    assert continued.scheduler.state_dict() == full.scheduler.state_dict()
    assert continued.epoch == full.epoch and continued.next_batch == full.next_batch

def test_resume_rejects_changed_config(model, tokenizer, make_dataset, train_config, tmp_path):
    data = make_dataset()
    trainer = Trainer(model, tokenizer, data, train_config, tmp_path/"run", torch.device("cpu"))
    trainer.train(stop_after=1)
    model, tok, payload = model_from_checkpoint(tmp_path/"run/last.pt")
    changed = dict(train_config, learning_rate=0.5)
    with pytest.raises(ValueError, match="identical training config"):
        Trainer(model, tok, data, changed, tmp_path/"run", torch.device("cpu"), payload)
