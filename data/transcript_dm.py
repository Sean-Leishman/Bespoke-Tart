import os

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from data.maptask import MapTaskDataset
from data.edacc import EdAccDataset


def get_abs_path(filepath):
    return os.path.join(os.path.abspath(__file__), filepath)


DATASETS = [MapTaskDataset, EdAccDataset]


class TranscriptDataset(Dataset):
    def __init__(self, split="train", tokenizer=None):
        self.tokenizer = tokenizer
        self.datasets = []

        for dataset in DATASETS:
            dataset = dataset(split=split, tokenizer=tokenizer)
            self.datasets.append(dataset)
            self.tokenizer = dataset.get_tokenizer()

        self.dataset = ConcatDataset(
            [x(split=split, tokenizer=tokenizer) for x in DATASETS])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
