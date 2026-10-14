from importlib import import_module

STAGES = ("vision_pretrain", "llm_pretrain", "alignment", "multimodal_pretrain", "sft", "lora", "dpo")

def get_stage(name):
    if name not in STAGES:
        raise ValueError(f"Unknown stage {name!r}; choose from {STAGES}")
    return import_module(f"training.stages.{name}").POLICY
