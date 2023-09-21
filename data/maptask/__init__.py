import os
import re
import numpy as np
import warnings

from typing import List

import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer

from sklearn.model_selection import train_test_split

from .utils import get_abs_path, transform_files, SPLITS_DIR, OUTPUT_MAP_TASK_DIR


class MapTaskDataset(Dataset):
    def __init__(self, split="train", tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")):
        self.split = split
        self.dialogs, self.labels = self.read_dialog()
        self.dialog_tokens = None

        self.tokenizer = tokenizer
        self.tokens = self.tokenize()

    def __len__(self):
        return len(self.dialog_tokens) if self.dialog_tokens is not None else len(self.dialogs)

    def __getitem__(self, idx):
        return {'input_ids': self.tokens['input_ids'][idx], 'token_type_ids': self.tokens['token_type_ids'][idx], 'attention_mask': self.tokens['attention_mask'][idx], 'labels': self.labels[idx]}

    def read_dialog(self):
        filename = get_abs_path(os.path.join(SPLITS_DIR, self.split + ".txt"))
        dialog_files = os.listdir(get_abs_path(OUTPUT_MAP_TASK_DIR))
        
        # Parse transcripts and save in another directory if not done yet
        if len(dialog_files) == 0:
            transform_files() 
            dialog_files = os.listdir(get_abs_path(OUTPUT_MAP_TASK_DIR))

        dialog = []
        labels = []
        with open(filename, "r") as indexes:
            for index in indexes:
                dialog_file = get_abs_path(os.path.join(
                    OUTPUT_MAP_TASK_DIR, dialog_files[int(index)]))
                curr_dialog = []

                with open(dialog_file, "r") as d:
                    for line in d.readlines():
                        curr_dialog.append(line)
                        labels.append(1)

                dialog.extend(curr_dialog)
        return dialog, labels

    def tokenize(self):
        return self.tokenizer(self.dialogs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")


if __name__ == "__main__":
    ds = MapTaskDataset()
    ds.tokenize()
