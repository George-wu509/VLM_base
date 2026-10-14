import torch

def make_optimizer(model, config):
    groups = []
    for component in ("projector", "decoder", "vision_encoder", "vision_head"):
        params = [p for p in getattr(model, component).parameters() if p.requires_grad]
        if params:
            lr = config.get("learning_rates", {}).get(component, config["learning_rate"])
            groups.append({"params": params, "lr": lr, "name": component})
    if not groups:
        raise ValueError("Stage has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=config.get("weight_decay", 0.01))
