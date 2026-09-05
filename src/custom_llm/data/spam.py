import ssl
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


SPAM_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["text"]]
        self.max_length = max_length or max(map(len, self.encoded_texts))
        self.encoded_texts = [tokens[:self.max_length] + [pad_token] * (self.max_length - len(tokens[:self.max_length])) for tokens in self.encoded_texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.encoded_texts[index]), torch.tensor(int(self.data.iloc[index]["Label"]))


def download_spam_data(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tsv = data_dir / "SMSSpamCollection.tsv"
    if not tsv.exists():
        archive = data_dir / "sms_spam_collection.zip"
        with urllib.request.urlopen(SPAM_URL, context=ssl._create_unverified_context()) as response:
            archive.write_bytes(response.read())
        with zipfile.ZipFile(archive) as files:
            files.extractall(data_dir)
        (data_dir / "SMSSpamCollection").rename(tsv)
    return tsv


def prepare_spam_splits(tsv_path, output_dir, seed=123):
    output_dir = Path(output_dir)
    df = pd.read_csv(tsv_path, sep="\t", names=["Label", "text"])
    num_spam = (df["Label"] == "spam").sum()
    balanced = pd.concat([df[df.Label == "ham"].sample(num_spam, random_state=seed), df[df.Label == "spam"]])
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    balanced["Label"] = balanced["Label"].map({"ham": 0, "spam": 1})
    train_end, val_end = int(len(balanced) * 0.7), int(len(balanced) * 0.8)
    splits = {"train": balanced[:train_end], "validation": balanced[train_end:val_end], "test": balanced[val_end:]}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(output_dir / f"{name}.csv", index=False)
    return {name: output_dir / f"{name}.csv" for name in splits}
