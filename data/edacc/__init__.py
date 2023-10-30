from torch.utils.data import DataLoader
import logging
import os
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


def collate_fn(batch):
    batched_data = {'input': None, 'output': None}
    for key in ['input', 'output']:
        input_ids = [item[key]['input_ids'] for item in batch]
        attention_masks = [item[key]['attention_mask'] for item in batch]

        max_len = max(len(item[key]) for item in batch)

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0)
        attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0)

        if key == 'input':
            batched_data = {'input_ids': input_ids_padded,
                            'attention_mask': attention_masks_padded}
        else:
            batched_data[key] = {'input_ids': input_ids_padded,
                                 'attention_mask': attention_masks_padded}
    return batched_data


INPUT_TRANSCRIPT = {
    "train": "data/edacc_v1.0/dev/text",
    "test": "data/edacc_v1.0/test/text"
}


class EdAccDataset(Dataset):
    def __init__(self, split="train",
                 prior_window_size=20,
                 post_window_size=5,
                 max_length=20):
        self.logger = logging.getLogger(__name__)

        self.split = split

        self.prior_window_size = prior_window_size
        self.post_window_size = post_window_size
        self.max_length = max_length

        self.dialogs = self.read_dialog()
        self.tokens = None

        # self.tokens = [None for _ in range(len(self.dialogs))]

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx]

    def read_dialog(self):
        self.logger.info(f"data ({self.split}): loading edAcc data")

        input_transcript = get_abs_path(INPUT_TRANSCRIPT[self.split])

        if not os.path.isfile(input_transcript):
            self.logger.error(
                f"data ({self.split}): transcript file does not exist at {input_transcript}")
            return []

        dialogs = []

        lines = []
        with open(input_transcript) as f:
            lines = f.readlines()

        current_conv_id = None
        curr_speaker = 'A'
        for line in lines:
            conv_id = line[:9]
            if current_conv_id is None or conv_id != current_conv_id:
                current_conv_id = conv_id
                dialogs.append([])
                curr_speaker = 'A'

            # index_of_utterance = line[10:19]
            text = line[20:]

            dialogs[-1].append({"text": text, "start": 0, "end": 0, "speaker":curr_speaker})
            curr_speaker = 'A' if curr_speaker == 'B' else 'B'

        self.logger.info(
            f" data({self.split}): done loading {len(dialogs)} conversations of edacc data")
        return dialogs


if __name__ == "__main__":
    ed = EdAccDataset()
    dl = DataLoader(ed, batch_size=16, collate_fn=collate_fn)
    it = iter(dl)

    for i in range(250):
        n = next(it)
    print(n)
