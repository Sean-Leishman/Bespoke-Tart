import logging
import os

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split


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
    def __init__(self, split="train", tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")):
        self.logger = logging.getLogger(__name__)

        self.tokenizer = tokenizer
        self.split = split

        self.filenames = self.read_file_splits()
        self.dialogs, self.labels = self.read_dialog()
        # self.tokens = self.tokenize()

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return None

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
                conv_id = values[0]

                prefix_dict = os.path.join(
                    TRANSCRIPT_DIRECTORIES[0], conv_id[:2], conv_id)

                split_filenames[conv_id] = []

                for file in values[1:]:
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
        for key in list(self.filenames.keys())[:2]:
            dialogs.append("")

            filenameA, _, filenameB, _ = self.filenames[key]
            if not os.path.exists(filenameA) or not os.path.exists(filenameB):
                self.logger.info(
                    f"data ({self.split}): {filenameA} does not exist")
                continue

            with open(filenameA) as fA, open(filenameB) as fB:
                lines_fA = fA.readlines()
                lines_fB = fB.readlines()

            i, j = 0, 0
            prev_speaker = None
            while i < len(lines_fA) and j < len(lines_fB):
                lineA, lineB = lines_fA[i], lines_fB[j]

                textA, startA, endA = self._read_transcript_line(lineA)
                textB, startB, endB = self._read_transcript_line(lineB)

                if textA == '[silence]':
                    i += 1
                    continue
                if textB == '[silence]':
                    j += 1
                    continue

                """
                At each iteration add either textA or textB when both are non-silence
                Cases 
                1. TextA starts prior to TextB and TextA ends prior to TextB 
                2. Vice versa as above 
                3. TextA starts prior to TextB and TextB end prior to TextA 
                    3.1 TextB is an overlap of TextA 
                    3.2 Check if the overlap is surrounded by 1s of silence from TextB
                        3.2.1 If so move to the next text of textA 
                        3.2.2 Else move the overlap to the closest side of the overlapped instance 
                4. Vice Versa 
                """

                # Check if text is overlapped within other speaker's text
                overlap, speaker = self._return_overlap(textA, textB, startA, startB,
                                                        endA, endB)
                # Check overlap and time threshold
                if overlap is not None:
                    if speaker == 'A':
                        # If there is enough silence prior, post curr than move on
                        # Else add the overlap to the closest part of speech
                        if self._check_overlap_silence(lines_fA[i-1], lines_fA[i+1]):
                            i += 1
                            continue
                        else:
                            pass
                    elif speaker == 'B':
                        if self._check_overlap_silence(lines_fB[j-1], lines_fB[j+1]):
                            j += 1
                            continue
                        else:
                            pass
                """
                At this point complete portions of overlap are covered
                Where a portion of overlapped speech is surrounded by the other's 
                speaker's speech 
                Now determine which speaker's speech begins and ends first  
                and handle it.
                Ordinarily partial overlap occurs:
                    A: 0->10 B:9->15 
                So here augment such that the resulting transcript reads as:
                    A: 0->10 B:10->16 
                where the timestamps are just the words resulting 
                """

                # print(startA, startB, endA, endB, textA, textB)

                # Add token

                # Perhaps whichever starts first
                if startA < startB:
                    if prev_speaker == 'B':
                        dialogs[-1] += "[SEP]"
                    dialogs[-1] += textA
                    prev_speaker = 'A'
                    i += 1
                elif startB > startA:
                    if prev_speaker == 'A':
                        dialogs[-1] += "[SEP]"

                    dialogs[-1] += textB
                    prev_speaker = 'B'
                    j += 1
                else:
                    if endA < endB:
                        if prev_speaker == 'B':
                            dialogs[-1] += "[SEP]"

                        dialogs[-1] += textA
                        prev_speaker = 'A'
                        i += 1
                    else:
                        if prev_speaker == 'A':
                            dialogs[-1] += "[SEP]"

                        dialogs[-1] += textB
                        prev_speaker = 'B'
                        j += 1

        print(dialogs)
        print(len(dialogs))
        return dialogs, None

    def _read_transcript_line(self, line):
        sepLine = line.split(" ")

        text = " ".join(sepLine[3:]).strip()
        start = float(sepLine[1])
        end = float(sepLine[2])

        return text, start, end

    def _return_overlap(self, textA, textB, startA, startB, endA, endB):
        if startA > startB and endA < endB:
            return textA, 'A'
        elif startB > startA and endB < endA:
            return textB, 'B'

        return None, None

    def _check_overlap_silence(self, past_line, next_line, thresh=1):
        past_text, past_start, past_end = self._read_transcript_line(past_line)
        next_text, next_start, next_end = self._read_transcript_line(next_line)

        if past_text != '[silence]' or next_text != '[silence]':
            return False

        if past_end - past_start < 1:
            return False

        if next_end - next_start < 1:
            return False

        return True


if __name__ == "__main__":
    sd = SwitchboardDataset()
    sd.generate_file_splits()
