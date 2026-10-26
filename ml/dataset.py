from __future__ import annotations

import torch
from torch.utils.data import Dataset


def encode_labels(taxonomy, labels) -> list[float]:
    return taxonomy.multi_hot(labels)


class MultiLabelDataset(Dataset):
    def __init__(self, rows, tokenizer, taxonomy, max_length: int = 128):
        self._rows = list(rows)
        self._tokenizer = tokenizer
        self._taxonomy = taxonomy
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int):
        row = self._rows[index]
        encoded = self._tokenizer(
            row["text"], max_length=self._max_length, truncation=True, padding="max_length",
        )
        item = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self._taxonomy.multi_hot(row["labels"]), dtype=torch.float),
        }
        return item
