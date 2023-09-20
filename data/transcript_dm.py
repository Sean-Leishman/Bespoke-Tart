import os

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .maptask import MapTaskDataset


def get_abs_path(filepath):
    return os.path.join(os.path.abspath(__file__), filepath)


DATASETS = [MapTaskDataset]


class TranscriptDataset(Dataset):
    def __init__(self, split="train"):
        self.dataset = ConcatDataset([x(split) for x in DATASETS])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
