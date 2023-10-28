import os
import logging

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from transformers import AutoTokenizer

from data.edacc import EdAccDataset
from data.switchboard import SwitchboardDataset


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


def collate_fn(batch):
    batched_data = {'input': None, 'output': None}
    for key in ['input', 'output']:
        input_ids = [item[key]['input_ids'] for item in batch]
        attention_masks = [item[key]['attention_mask'] for item in batch]

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


DATASETS = [SwitchboardDataset, EdAccDataset]
CACHE_PATH = get_abs_path(os.path.join(".cache", "dataset"))


class TranscriptDataset(Dataset):
    def __init__(self, split="train",
                 tokenizer=None,
                 savepath=None,
                 overwrite=False,
                 load_from_cache_file=False,
                 max_prior_window_size=50,
                 context_window=2,
                 post_window_size=10):
        self.logger = logging.getLogger(__name__)

        self.tokenizer = tokenizer
        self.datasets = []

        self.split = split

        if savepath is None:
            savepath = CACHE_PATH
        self.savepath = savepath

        self.overwrite = overwrite
        self.load_from_cache_file = load_from_cache_file

        self.max_prior_window_size = max_prior_window_size
        self.context_window = context_window
        self.post_window_size = post_window_size

        self.prepare_data()

    def __len__(self):
        return sum([len(i['input_ids'][0]) for i in self.data])

    def __getitem__(self, idx):
        dialog_idx, token_idx = self.get_dialog_idx(idx)

        conv = self.data[dialog_idx]
        if 'input_ids' not in conv.keys():
            self.data = self.tokenize()

        start_idx = max(0, token_idx-self.max_prior_window_size)
        end_idx = min(len(self), token_idx + self.post_window_size)

        dict = {'input': {}, 'output': {}}
        for k, v in conv.items():
            if k == "dialog":
                continue

            dict["input"][k] = v[0][start_idx:token_idx]
            dict["output"][k] = v[0][token_idx:end_idx]

        return dict

    def get_save_load_path(self):
        save_load_dir = get_abs_path(os.path.dirname(self.savepath))

        if not os.path.exists(save_load_dir):
            os.mkdir(save_load_dir)

        return os.path.join(save_load_dir, self.split)

    def prepare_data(self):
        if self.load_from_cache_file or not self.overwrite:
            self.setup()
            return

        for load_dataset in DATASETS:
            dataset = load_dataset(split=self.split)
            self.datasets.append(dataset)

        self.dataset = ConcatDataset(self.datasets)
        self.data = self.tokenize()

        self.prefix_sum = self.generate_indexes()

        self.save_to_disk()

    def setup(self):
        saved_ds = torch.load(self.get_save_load_path())
        self.dataset = saved_ds.dataset
        self.data = saved_ds.data
        self.prefix_sum = saved_ds.prefix_sum

    def save_to_disk(self):
        self.logger.info(f"data {self.split}: saving combined transcript")
        torch.save(self, self.get_save_load_path())

    def get_dialog_idx(self, token_idx):
        dialog_idx = next(i for i, v in enumerate(
            self.prefix_sum) if v > token_idx) - 1
        token_idx = token_idx - self.prefix_sum[dialog_idx]

        return dialog_idx, token_idx

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
            result = []
            for dataset in self.dataset:
                output = {}
                dialog = [dialog['text'] for dialog in dataset]

                output['dialog'] = "[SEP]".join(dialog)
                tokens = self.tokenize_sentence(output['dialog'])

                output['input_ids'] = tokens['input_ids']
                output['attention_mask'] = tokens['attention_mask']

                current_speaker = dataset[0]['speaker']
                token_type_ids = [[]]
                for token in output['input_ids'][0]:
                    # Is [SEP] token self.tokenizer.encode('[SEP]') -> [101, 102, 102]
                    if token == 102:
                        current_speaker = 'A' if current_speaker == 'B' else 'A'

                    token_type_ids[0].append(0 if current_speaker == 'A' else 1)

                output['token_type_ids'] = torch.tensor(token_type_ids)
                result.append(output)

        self.logger.info(
            f"data ({self.split}): done tokenizing edacc data with keys: {result[0].keys}")
        return result

    def tokenize_sentence(self, dialog):
        tokens = self.tokenizer(
            dialog,
            padding="do_not_pad",
            truncation=True,
            max_length=12400,
            return_tensors="pt")
        return tokens

    def generate_indexes(self):
        indexes = [0]
        for conv in self.data:
            indexes.append(
                len(conv['input_ids'][0]) + indexes[-1])

        return indexes


if __name__ == "__main__":
    ts = TranscriptDataset(
        tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"))
    ts.setup()
    dl = DataLoader(ts, batch_size=320, collate_fn=collate_fn)

    batch = next(iter(dl))

    for i in range(100):
        print(ts.tokenizer.decode(batch['input_ids'][i]))
