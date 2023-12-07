import logging
import os
import tqdm

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split

def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


TRANSCRIPT_DIRECTORIES = [
    # get_abs_path("switchboard/cellular/transcripts/data"),
    get_abs_path(
        "fisher/data/trans")
]

class FisherDataset(Dataset):
    def __init__(self, split="train"):
        self.logger = logging.getLogger(__name__)
        self.split = split

        self.filenames = self.read_file_splits()
        #self.dialogs = self.read_dialog()

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx]


    def read_dialog(self):
        self.logger.info(f"fisher ({self.split}): loading data")

        dialogs = []
        for key in tqdm.tqdm(self.filenames):
            dialog = extract_dialog(self.filenames[key])

            dialogs.append(dialog)

        return dialogs

    def read_file_splits(self):
        self.logger.info(f"fisher: reading {self.split} data")

        filename = get_abs_path(os.path.join("splits", f"{self.split}.txt"))
        if not os.path.isfile(filename):
            self.logger.error(f"fisher: no split for {self.split} files")
            self.generate_file_splits()

    def generate_file_splits(self):
        if not os.path.exists(get_abs_path("splits")):
            os.mkdir(get_abs_path("splits"))

        splits_dir = get_abs_path("splits")
        files = []
        for transcript_dir in TRANSCRIPT_DIRECTORIES:
            for dir in os.listdir(transcript_dir):
                for file in os.listdir(os.path.join(transcript_dir, dir)):
                    files.append(os.path.join(transcript_dir, dir, file))
        train, test = train_test_split(files, test_size=.2)


        train_filename = get_abs_path("splits/train.txt")
        test_filename = get_abs_path("splits/test.txt")

        with open(train_filename, "w") as f:
            for key in train:
                f.write(
                    f"{key}\n")

        with open(test_filename, "w") as f:
            for key in test:
                f.write(
                    f"{key}\n")

if __name__=="__main__":
    ds = FisherDataset()