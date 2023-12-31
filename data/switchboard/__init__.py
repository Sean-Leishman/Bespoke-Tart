import logging
import os
import tqdm

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split
from .utils import extract_dialog, extract_speaker_timings, \
    combine_dialogue_with_timings, remove_backchannels, combine_consecutive_trps, remove_overlaps, \
    combine_dialogue_without_timings


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


TRANSCRIPT_DIRECTORIES = [
    # get_abs_path("switchboard/cellular/transcripts/data"),
    get_abs_path(
        "switchboard/switchboard1/transcriptions/swb_ms98_transcriptions")
]

FILENAMES_FILE = get_abs_path(
    "switchboard/switchboard1/transcriptions/swb_ms98_transcriptions/AAREADME.text")


class SwitchboardDataset(Dataset):
    def __init__(self, split="train"):
        self.logger = logging.getLogger(__name__)

        self.split = split



    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx]

    def __str__(self):
        return "Switchboard"

    def __call__(self):
        self.filenames = self.read_file_splits()
        self.dialogs = self.read_dialog()
        # self.tokens = self.tokenize()

    def read_file_splits(self):
        self.logger.info(f"data: reading {self.split} data")

        filename = get_abs_path(os.path.join("splits", f"{self.split}.txt"))
        if not os.path.isfile(filename):
            self.logger.error(f"data: no splits {self.split} files")
            self.generate_file_splits()

        split_filenames = {}
        with open(filename) as f:
            line = f.readline()
            while line:
                values = line.split("\t")
                conv_id = values[0].strip()

                prefix_dict = os.path.join(
                    TRANSCRIPT_DIRECTORIES[0], conv_id[:2], conv_id)

                split_filenames[conv_id] = []
                filenames = os.listdir(prefix_dict)
                filenames.sort()

                for file in filenames:
                    split_filenames[conv_id].append(os.path.join(
                        prefix_dict, file.strip()))

                line = f.readline()

        return split_filenames

    def generate_file_splits(self):
        if not os.path.exists(get_abs_path("splits")):
            os.mkdir(get_abs_path("splits"))

        line_idx = 0
        files = {}
        lines = []
        with open(FILENAMES_FILE) as f:
            line = f.readline()
            while line:
                if line_idx >= 17:
                    if (line_idx - 17) % 5 == 0:
                        if len(lines) >= 5:
                            files[lines[0]] = [
                                lines[1], lines[2], lines[3], lines[4]]
                            lines = []
                    lines.append(line.strip())
                line_idx += 1
                line = f.readline()

        train, test = train_test_split(list(files.keys()), test_size=.2)

        train_filename = get_abs_path("splits/train.txt")
        test_filename = get_abs_path("splits/test.txt")

        with open(train_filename, "w") as f:
            for key in train:
                value = files[key]
                f.write(
                    f"{key}\t{value[0]}\t{value[1]}\t{value[2]}\t{value[3]}\n")

        with open(test_filename, "w") as f:
            for key in test:
                value = files[key]
                f.write(
                    f"{key}\t{value[0]}\t{value[1]}\t{value[2]}\t{value[3]}\n")

    def read_dialog(self):
        self.logger.info(f"data ({self.split}): loading switchboard data")

        dialogs = []
        for key in tqdm.tqdm(self.filenames):
            dialog = extract_dialog(self.filenames[key])
            vad = extract_speaker_timings(dialog)
            # dialog = remove_words_from_dialog(dialog)

            # dialog = combine_dialogue_with_timings(dialog, vad)
            dialog = combine_dialogue_without_timings(dialog)
            # dialog, speaker = remove_backchannels(dialog)
            dialog = remove_overlaps(dialog)

            dialog = combine_consecutive_trps(dialog)

            dialogs.append(dialog)

        # print(dialogs[0])
        return dialogs

    def save_dialogs(self, prefix_dir):
        # Assume self.filenames correspond with self.dialogs
        for idx, key in enumerate(self.filenames):
            filename = key
            filename = os.path.join(prefix_dir, filename)

            if idx >= len(self.dialogs):
                return

            with open(filename, "w") as f:
                f.writelines(self.dialogs[idx])


if __name__ == "__main__":
    sd = SwitchboardDataset()
    sd.save_dialogs(get_abs_path(os.path.join("splits", "data")))
