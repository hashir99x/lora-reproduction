"""
Loads the GLUE MRPC (Microsoft Research Paraphrase Corpus) dataset and
tokenizes it for DistilBERT. MRPC is a sentence-pair binary classification
task: given two sentences, predict whether they're paraphrases of each
other. Chosen because the LoRA paper's own GLUE table reports both
accuracy AND F1 for MRPC, matching this project's required deliverables.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

import config


class MRPCDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=config.MAX_LENGTH):
        self.data = load_dataset("nyu-mll/glue", "mrpc", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["sentence1"],
            item["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def get_dataloaders(batch_size=config.BATCH_SIZE):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_ds = MRPCDataset("train", tokenizer)
    val_ds = MRPCDataset("validation", tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    train_loader, val_loader, tok = get_dataloaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    batch = next(iter(train_loader))
    print({k: v.shape for k, v in batch.items()})
