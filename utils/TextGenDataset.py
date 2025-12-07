import torch
from torch.utils.data import Dataset

class TextGenDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_len):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tok = tokenizer
        self.max_len = max_len

    def pad(self, ids):
        if len(ids) > self.max_len:
            return ids[:self.max_len]
        return ids + [self.tok.pad] * (self.max_len - len(ids))

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = self.tok.encode(self.src_texts[idx])
        tgt = self.tok.encode(self.tgt_texts[idx])

        src_ids = self.pad(src)
        tgt_input = self.pad(tgt[:-1])
        tgt_label = self.pad(tgt[1:])

        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt_input': torch.tensor(tgt_input, dtype=torch.long),
            'tgt_label': torch.tensor(tgt_label, dtype=torch.long)
        }

