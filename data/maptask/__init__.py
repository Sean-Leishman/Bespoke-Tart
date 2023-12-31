import os
import logging

from torch.utils.data import Dataset
from transformers import BertTokenizer
from data.maptask.utils import get_abs_path, transform_files, weighted_random, SPLITS_DIR, OUTPUT_MAP_TASK_DIR


class MapTaskDataset(Dataset):
    def __init__(self, split="train", tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"), config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config

        self.split = split
        self.dialogs, self.labels = self.read_dialog()
        self.dialog_tokens = None

        self.tokenizer = tokenizer
        self.tokens = self.tokenize()

    def __len__(self):
        return len(self.dialog_tokens) if self.dialog_tokens is not None else len(self.dialogs)

    """
        Of the form as below
        {'input_ids': self.tokens['input_ids'][idx],
                'token_type_ids': self.tokens['token_type_ids'][idx],
                'attention_mask': self.tokens['attention_mask'][idx],
                'labels': self.labels[idx]}
        But require robustness due to choice of tokenizer
    """

    def __getitem__(self, idx):
        dict = {}
        for key, value in self.tokens.items():
            dict[key] = value[idx]
        dict['labels'] = self.labels[idx]
        return dict

    def read_dialog(self):
        self.logger.info(f'data ({self.split}): loading MapTask Data')

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
                        curr_dialog.append(line.strip())
                        labels.append(1)

                        for word in range(1, len(line.split(" "))-1):
                            dialog.append(
                                " ".join(line.split(" ")[0:word]).strip())
                            labels.append(0)

                dialog.extend(curr_dialog)

        self.logger.info(
            f"data ({self.split}): done loading {len(curr_dialog)} sentences of maptask data")
        return dialog, labels

    """
    Generate a sentence from the ipu that is considered a non-trp
    Exclude the final word
    Randomly generate start index of the word (weighted to start of the sequence)
    Randonly generate length of sentence (weighted to length - 1 of the sequence)
    """

    def add_non_trp(self, ipu):
        words = ipu.split(" ")

        if len(words) < 2:
            return ""

        start_index = weighted_random(range(0, len(words) - 2), first=10)
        length = weighted_random(
            range(1, len(words) - start_index), first=0,  last=10)
        return " ".join(words[start_index: start_index+length])

    def tokenize(self):
        self.logger.info(f"data ({self.split}): tokenizing maptask data")
        tokens = self.tokenizer(self.dialogs,
                                padding="max_length",
                                truncation=True,
                                max_length=512,
                                return_tensors="pt")
        self.logger.info(f"data ({self.split}): done tokenizing maptask data")
        return tokens

    def get_tokenizer(self):
        return self.tokenizer


if __name__ == "__main__":
    ds = MapTaskDataset()
    ds.tokenize()
