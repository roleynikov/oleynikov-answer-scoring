import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer


class CommonLitDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float),
        }


class CommonLitDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.encoder_name)

    def setup(self, stage=None):
        df = pd.read_csv(self.cfg.data.train_path)

        texts = df["text"].tolist()
        targets = df[["content", "wording"]].values

        split = int(len(df) * 0.8)
        self.train_dataset = CommonLitDataset(
            texts[:split],
            targets[:split],
            self.tokenizer,
            self.cfg.data.max_length,
        )

        self.val_dataset = CommonLitDataset(
            texts[split:],
            targets[split:],
            self.tokenizer,
            self.cfg.data.max_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=True,
            num_workers=self.cfg.trainer.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=False,
            num_workers=self.cfg.trainer.num_workers,
        )
