import torch


def generate_text_simple(model, idx, max_text_tokens, context_size):
    for _ in range(max_text_tokens):
        with torch.no_grad():
            logits = model(idx[:, -context_size:])[:, -1, :]
        idx = torch.cat((idx, logits.argmax(dim=-1, keepdim=True)), dim=1)
    return idx


def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(idx[:, -context_size:])[:, -1, :] / temperature
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -torch.inf
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        idx = torch.cat((idx, next_id), dim=1)
        if eos_id is not None and (next_id == eos_id).all():
            break
    return idx


def classify_text(text, model, tokenizer, device, max_length, pad_token_id=50256):
    ids = tokenizer.encode(text)[:max_length]
    ids += [pad_token_id] * (max_length - len(ids))
    with torch.no_grad():
        label = model(torch.tensor(ids, device=device).unsqueeze(0))[:, -1, :].argmax(dim=-1).item()
    return "spam" if label == 1 else "not spam"
