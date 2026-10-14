"""Readable single-example autoregressive decoding; cache vision prefix, no KV cache."""
import torch

@torch.inference_mode()
def generate_ids(model, tokenizer, prompt, pixels=None, max_new_tokens=64, temperature=0.0, top_k=0):
    if max_new_tokens < 1 or temperature < 0 or top_k < 0:
        raise ValueError("Invalid decoding settings")
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([[tokenizer.bos_token_id] + tokenizer.encode(prompt+"\n")], device=device)
    embeddings = model.decoder.token_embedding(ids)
    if pixels is not None:
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
        if pixels.shape[0] != 1:
            raise ValueError("Generation supports one example at a time")
        image = model.projector(model.image_features(pixels.to(device)))
        embeddings = torch.cat([image, embeddings], 1)
    if embeddings.size(1) + max_new_tokens > model.cfg.lm_max_position_embeddings:
        raise ValueError("Prompt + image + generation budget exceeds model context")
    output = []
    for _ in range(max_new_tokens):
        logits = model.decoder.head(model.decoder(embeddings)[:, -1]).float()
        if temperature == 0:
            next_token = logits.argmax(-1, keepdim=True)
        else:
            logits /= temperature
            if top_k:
                cutoff = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, float('-inf'))
            next_token = torch.multinomial(logits.softmax(-1), 1)
        token = next_token.item()
        if token == tokenizer.eos_token_id:
            break
        output.append(token)
        embeddings = torch.cat([embeddings, model.decoder.token_embedding(next_token)], 1)
    return output

def generate_text(model, tokenizer, prompt, pixels=None, **kwargs):
    return tokenizer.decode(generate_ids(model, tokenizer, prompt, pixels, **kwargs))
