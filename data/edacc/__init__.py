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
    "train": "data/dev/text",
    "test": "data/test/text"
}


class EdAccDataset(Dataset):
    def __init__(self, split="train",
                 tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
                 prior_window_size=20,
                 post_window_size=5,
                 max_length=20):
        self.logger = logging.getLogger(__name__)

        self.split = split
        self.tokenizer = self.init_tokenizer(tokenizer)

        self.prior_window_size = prior_window_size
        self.post_window_size = post_window_size
        self.max_length = max_length

        self.dialogs = self.read_dialog()
        self.tokens = self.tokenize(padding=False)
        self.prefix_sum = self.generate_cumsum()

        # self.tokens = [None for _ in range(len(self.dialogs))]

    def __len__(self):
        return sum(len(i['input_ids'][0]) for i in self.tokens)

    def __getitem__(self, idx):
        dialog_idx, token_idx = self.get_dialog_idx(idx)
        dialog = self.dialogs[dialog_idx]

        if self.tokens[dialog_idx] is None:
            self.tokens[dialog_idx] = self.tokenize_sentence(dialog)

        tokens = self.tokens[dialog_idx]

        start_idx = max(0, token_idx - self.prior_window_size)
        end_idx = min(len(self), token_idx + self.post_window_size)

        dict = {'input': {}, 'output': {}}
        for key, value in tokens.items():
            dict['input'][key] = value[0][start_idx:token_idx]
            dict['output'][key] = value[0][token_idx:end_idx]

        return dict

    def init_tokenizer(self, tokenizer):
        return tokenizer

    def get_dialog_idx(self, token_idx):
        dialog_idx = next(i for i, v in enumerate(
            self.prefix_sum) if v > token_idx) - 1
        token_of_dialog_idx = token_idx - self.prefix_sum[dialog_idx]

        return dialog_idx, token_of_dialog_idx

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
        for line in lines:
            conv_id = line[:9]
            if current_conv_id is None or conv_id != current_conv_id:
                current_conv_id = conv_id
                dialogs.append("")

            # index_of_utterance = line[10:19]
            text = line[20:]

            dialogs[-1] += text
            # dialogs[-1] += "[EOT]"
            dialogs[-1] += "[SEP]"

        self.logger.info(
            f" data({self.split}): done loading {len(dialogs)} sentences of edacc data")

        return dialogs

    def generate_cumsum(self):
        cumsum = [0]
        for tokenized in self.tokens:
            cumsum.append(len(tokenized['input_ids'][0]) + cumsum[-1])

        print(cumsum)
        return cumsum

    def tokenize(self, padding=False):
        self.logger.info(f"data ({self.split}): tokenizing edacc data")

        if padding:
            self.logger.info(f"data ({self.split}): padding edacc data")
            tokens = self.tokenizer(self.dialogs,
                                    padding="max length",
                                    truncation=True,
                                    max_length=1024,
                                    return_tensors="pt")

        else:
            tokens = [self.tokenize_sentence(dialog)
                      for dialog in self.dialogs]
        self.logger.info(
            f"data ({self.split}): done tokenizing edacc data with keys: {tokens[0].keys}")
        return tokens

    def tokenize_sentence(self, dialog):
        tokens = self.tokenizer(
            dialog,
            padding="do_not_pad",
            truncation=True,
            max_length=12400,
            return_tensors="pt")
        return tokens

    def get_tokenizer(self):
        return self.tokenizer


if __name__ == "__main__":
    ed = EdAccDataset()
    dl = DataLoader(ed, batch_size=16, collate_fn=collate_fn)
    it = iter(dl)

    tokenizer = ed.get_tokenizer()
    for i in range(250):
        n = next(it)
    print(n)
    print([tokenizer.decode(x) for x in n['input_ids']])
