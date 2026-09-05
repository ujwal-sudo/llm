import argparse

import torch

from custom_llm.config import GPT2_124M
from custom_llm.inference import generate_text_simple
from custom_llm.model import GPTModel
from custom_llm.tokenizer import get_tokenizer, text_to_token_ids, token_ids_to_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokens", type=int, default=50)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(GPT2_124M.as_dict())
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.to(device).eval()
    tokenizer = get_tokenizer()
    ids = generate_text_simple(model, text_to_token_ids(args.prompt, tokenizer).to(device), args.tokens, GPT2_124M.context_length)
    print(token_ids_to_text(ids.cpu(), tokenizer))


if __name__ == "__main__":
    main()
