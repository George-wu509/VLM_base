"""Translate local annotations to one explicit task schema."""
from tokenization.spatial_tokens import encode_box

TASKS = ("caption", "image_text", "vqa", "grounding", "preference", "text", "image")

def adapt(record):
    row = dict(record)
    task = row.get("task", "caption")
    if task not in TASKS:
        raise ValueError(f"Unknown task {task}")
    row["task"] = task
    if task in ("caption", "image_text"):
        row.setdefault("prompt", "caption en")
        row.setdefault("response", row.get("caption", row.get("text", "")))
    elif task == "vqa":
        row.setdefault("prompt", "answer " + row.get("question", ""))
        row.setdefault("response", row.get("answer", ""))
    elif task == "grounding":
        row.setdefault("prompt", "detect " + row.get("query", "objects"))
        if "objects" in row:
            row["response"] = " ; ".join(encode_box(obj["bbox"]) + " " + obj["label"] for obj in row["objects"])
    elif task == "text":
        row.setdefault("prompt", "")
        row.setdefault("response", row.get("text", ""))
    elif task == "preference":
        if not row.get("prompt") or not row.get("chosen") or not row.get("rejected"):
            raise ValueError("Preference requires prompt, chosen and rejected")
        if row["chosen"] == row["rejected"]:
            raise ValueError("Chosen and rejected completions must differ")
    elif task == "image":
        if type(row.get("label")) is not int or row["label"] < 0:
            raise ValueError("Image pretraining requires a nonnegative integer label")
    if task not in ("image", "preference") and not row.get("response"):
        raise ValueError(f"Task {task} needs a nonempty response")
    if task != "text" and not row.get("image"):
        raise ValueError(f"Task {task} requires an image")
    for field in ("prompt", "response", "image", "chosen", "rejected"):
        if field in row and not isinstance(row[field], str):
            raise ValueError(f"{field} must be a string")
    if task == "grounding" and not row.get("response"):
        raise ValueError("Grounding requires at least one labeled box")
    return row
