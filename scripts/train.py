import argparse

import torch
from torch.utils.data import DataLoader

from custom_llm.config import GPT2_124M, SEED
from custom_llm.data.spam import SpamDataset
from custom_llm.model import GPTModel
from custom_llm.pretrained import load_model_weights_into_gpt
from custom_llm.tokenizer import get_tokenizer
from custom_llm.training import train_classifier


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the GPT-2-compatible model on SMS spam.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=120)
    parser.add_argument("--gpt2-model-dir", default=None, help="Directory containing GPT-2 files; requires TensorFlow conversion")
    parser.add_argument("--hf-checkpoint", default=None, help="Local Hugging Face GPT-2 checkpoint directory")
    parser.add_argument("--checkpoint", default="checkpoints/spam_classifier.pt")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    datasets = {name: SpamDataset(f"{args.data_dir}/{name}.csv", tokenizer, args.max_length) for name in ("train", "validation")}
    train_loader = DataLoader(datasets["train"], batch_size=8, shuffle=True)
    val_loader = DataLoader(datasets["validation"], batch_size=8)
    model = GPTModel(GPT2_124M.as_dict())
    if args.hf_checkpoint:
        from transformers import GPT2LMHeadModel
        from custom_llm.convert_hf_gpt2 import load_hf_gpt2_into_custom_model

        hf_model = GPT2LMHeadModel.from_pretrained(args.hf_checkpoint, local_files_only=True)
        load_hf_gpt2_into_custom_model(model, hf_model.state_dict())
    if args.gpt2_model_dir:
        from custom_llm.pretrained_download import download_and_load_gpt2
        _, params = download_and_load_gpt2("124M", args.gpt2_model_dir)
        if params is None:
            raise RuntimeError("GPT-2 parameters were not loaded; install TensorFlow or omit --gpt2-model-dir")
        load_model_weights_into_gpt(model, params)
    model.out_head = torch.nn.Linear(GPT2_124M.emb_dim, 2)
    for param in model.parameters():
        param.requires_grad = False
    for param in list(model.out_head.parameters()) + list(model.final_norm.parameters()) + list(model.trf_blocks[-1].parameters()):
        param.requires_grad = True
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    history = train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs=args.epochs)
    torch.save({"model": model.state_dict(), "config": GPT2_124M.as_dict(), "history": history}, args.checkpoint)
    print(f"Saved {args.checkpoint}")


if __name__ == "__main__":
    main()
