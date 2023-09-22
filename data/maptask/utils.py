import os
import re
import random

from typing import List
from sklearn.model_selection import train_test_split

MAP_TASK_DIR = "transcripts/"
OUTPUT_MAP_TASK_DIR = "parsed_transcripts/"

SPLITS_DIR = "splits/"


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


def transform_files():
    for file in os.listdir(get_abs_path(MAP_TASK_DIR)):
        output = ""
        speaker = ""
        with open(get_abs_path(os.path.join(MAP_TASK_DIR, file)), "r") as stream:
            lines = stream.readlines()[3:]
            for line in lines:
                line = line.replace("\n", "")
                new_speaker = line[0]

                if new_speaker == speaker or speaker == "":
                    output += line[2:]
                    if speaker == "":
                        speaker = new_speaker
                else:
                    output += "\n"
                    output += line[2:]
                    speaker = new_speaker

        output = re.sub('[^\S\r\n]+', ' ', output)

        write_txt(output, get_abs_path(
            os.path.join(OUTPUT_MAP_TASK_DIR, file)))


def write_txt(content, file):
    with open(file, "w") as f:
        if type(content) is list:
            for text in content:
                f.write(str(text))
                f.write("\n")
        elif type(content) is str:
            f.write(str(content))


def read_txt(file):
    content = []
    with open(file, "r") as f:
        for line in f.readlines():
            content.append(int(line))
    return content


def generate_train_test_split():
    file_count = len(os.listdir(get_abs_path(MAP_TASK_DIR)))

    if not os.path.exists(get_abs_path(SPLITS_DIR)):
        os.makedirs(get_abs_path(SPLITS_DIR))

    train_file = get_abs_path(os.path.join(SPLITS_DIR, "train.txt"))
    test_file = get_abs_path(os.path.join(SPLITS_DIR, "test.txt"))

    train_idxs, test_idxs = train_test_split(range(file_count), test_size=0.2)

    write_txt(train_idxs, train_file)
    write_txt(test_idxs, test_file)


def weighted_random(choices, first=1, last=1):
    choices = [0] * first + \
        [x for x in range(1, len(choices)-2)] + [len(choices)-1] * last
    return random.choice(choices)


if __name__ == "__main__":
    transform_files()
