import logging
import os

from torch.utils.data import Dataset
from transformers import AutoTokenizer


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


TRANSCRIPT_DIRECTORIES = [
    # get_abs_path("switchboard/cellular/transcripts/data"),
    get_abs_path(
        "switchboard/switchboard1/transcripts/swb_ms98_transcriptions")
]


class SwitchboardDataset(Dataset):
    def __init__(self, split="traiN", tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")):
        self.logger = logging.getLogger(__name__)

        self.tokenizer = tokenizer
        self.split = split

        self.filenames = self.read_file_splits()
        self.dialogs, self.labels = self.read_dialog()
        self.tokens = self.tokenize()

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return None

    def read_file_splits(self):
        self.logger.info(f"data: reading {self.split} data")

        filename = get_abs_path(os.path.join("splits", self.split))
        if not os.path.isfile(filename):
            self.logger.error(f"data: no splits {self.split} files")

            self.generate_file_splits()

        split_filenames = []
        with open(filename) as f:
            lines = f.readlines()

        split_filenames = [get_abs_path(line.strip()) for line in lines]
        return split_filenames

    def generate_file_splits():
        if not os.path.exists(get_abs_path("splits")):
            os.mkdir(get_abs_path("splits"))

        train_filename = get_abs_path("splits", "train.txt")
        test_filename = get_abs_path("splits", "test.txt")

    def read_dialog(self):
        self.logger.info(f"data ({self.split}): loading switchboard data")
