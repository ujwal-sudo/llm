import torch


def get_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def text_to_token_ids(text, tokenizer):
    return torch.tensor(tokenizer.encode(text, allowed_special={"<|endoftext|>"})).unsqueeze(0)


def token_ids_to_text(encoded, tokenizer):
    return tokenizer.decode(encoded.squeeze(0).tolist())
