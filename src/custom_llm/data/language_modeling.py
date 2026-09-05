import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        starts = range(0, len(token_ids) - max_length, stride)
        self.input_ids = [torch.tensor(token_ids[i:i + max_length]) for i in starts]
        self.target_ids = [torch.tensor(token_ids[i + 1:i + max_length + 1]) for i in starts]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_lm_dataloader(text, tokenizer, max_length=256, stride=128, batch_size=4, shuffle=True, drop_last=True, num_workers=0):
    return DataLoader(GPTDataset(text, tokenizer, max_length, stride), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
