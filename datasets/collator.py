import torch

class Collator:
    def __init__(self, tokenizer, max_length=128, objective="causal"):
        self.tokenizer, self.max_length, self.objective = tokenizer, max_length, objective

    def sequence(self, row, response):
        tok = self.tokenizer
        # Encode separately to make the exact completion boundary explicit.
        prompt = [tok.bos_token_id] + tok.encode(row.get("prompt", "") + "\n")
        answer = tok.encode(response) + [tok.eos_token_id]
        if len(prompt) + len(answer) > self.max_length:
            raise ValueError(f"Record {row.get('id')} exceeds max_length={self.max_length}; increase it or split the record")
        return prompt + answer, [-100]*len(prompt) + answer

    def pad(self, rows, field="response"):
        sequences = [self.sequence(row, row[field]) for row in rows]
        width = max(len(ids) for ids, _ in sequences)
        ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for i, (tokens, targets) in enumerate(sequences):
            ids[i, :len(tokens)] = torch.tensor(tokens)
            labels[i, :len(tokens)] = torch.tensor(targets)
            mask[i, :len(tokens)] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    def __call__(self, rows):
        image_flags = ["pixel_values" in r for r in rows]
        if any(image_flags) and not all(image_flags):
            raise ValueError("Mix text-only and multimodal in separate runs; mixed modality batches are unsupported")
        pixels = torch.stack([r["pixel_values"] for r in rows]) if all(image_flags) else None
        if self.objective == "vision":
            return {"pixel_values": pixels, "class_labels": torch.tensor([r["label"] for r in rows])}
        if self.objective == "dpo":
            return {key: dict(self.pad(rows, key), pixel_values=pixels) for key in ("chosen", "rejected")}
        return dict(self.pad(rows), pixel_values=pixels)
