import torch
import torch.nn.functional as F

def sequence_log_probs(output):
    logits, labels = output["logits"][:, :-1].float(), output["labels"][:, 1:]
    valid = labels != -100
    if not valid.any(dim=1).all():
        raise ValueError("Every DPO sequence needs completion tokens")
    log_probs = logits.log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (log_probs * valid).sum(-1)  # DPO uses sums, not length-normalized CE.

def compute_loss(model, batch, objective, reference=None, beta=0.1):
    if objective == "vision":
        return F.cross_entropy(model.classify(batch["pixel_values"]).float(), batch["class_labels"]), {}
    if objective == "causal":
        return model(**batch)["loss"], {}
    chosen = sequence_log_probs(model(**batch["chosen"]))
    rejected = sequence_log_probs(model(**batch["rejected"]))
    with torch.no_grad():
        rc = sequence_log_probs(reference(**batch["chosen"]))
        rr = sequence_log_probs(reference(**batch["rejected"]))
    margin = beta * ((chosen-rejected) - (rc-rr))
    return -F.logsigmoid(margin).mean(), {"preference_accuracy": float((margin > 0).float().mean().detach())}
