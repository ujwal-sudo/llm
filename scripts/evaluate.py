import argparse

import torch
from torch.utils.data import DataLoader

from custom_llm.config import GPT2_124M
from custom_llm.data.spam import SpamDataset
from custom_llm.evaluation import accuracy
from custom_llm.model import GPTModel
from custom_llm.tokenizer import get_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/spam_classifier.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    model = GPTModel(GPT2_124M.as_dict())
    model.out_head = torch.nn.Linear(GPT2_124M.emb_dim, 2)
    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.to(device)
    for name in ("train", "validation", "test"):
        loader = DataLoader(SpamDataset(f"{args.data_dir}/{name}.csv", tokenizer, 120), batch_size=8)
        print(f"{name}: {accuracy(loader, model, device) * 100:.2f}%")


if __name__ == "__main__":
    main()
