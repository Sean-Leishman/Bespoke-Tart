import copy
import os
import logging
import re

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from transformers import AutoTokenizer

from data.edacc import EdAccDataset
from data.switchboard import SwitchboardDataset
from data.fisher import FisherDataset
from gptonly.tokenizer import SpokenDialogTokenizer

def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)

    output = {}
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0)
    token_type_ids_padded = torch.nn.utils.rnn.pad_sequence(
        token_type_ids, batch_first=True, padding_value=0)

    output['input_ids'] = input_ids_padded
    output['attention_mask'] = attention_masks_padded
    output['token_type_ids'] = token_type_ids_padded
    return output


DATASETS = [SwitchboardDataset]
CACHE_PATH = get_abs_path(".cache")


class GenerationDM(Dataset):
    """
    Implementation of Pytorch Dataset to load text data for transcript tasks.

    Attributes
    -----------
    tokenizer: Transformers.TokenizerFast
        used to tokenize text when loading data
    max_length: int
        maximum number of tokens within a sequence
    keep_length: int
        only keep sequences with this number of tokens
    overlap_length: int
        number of tokens that overlap between consecutive sequences if the total sequence exceeds maximum length
    """
    def __init__(self, split="train",
                 tokenizer=None,
                 savepath=None,
                 overwrite=False,
                 load_from_cache_file=False,
                 max_length=256,
                 keep_length=64,
                 overlap_length=10,
                 dev_mode=False,
                 datasets=["switchboard", "fisher"],
                 ):
        self.logger = logging.getLogger(__name__)

        self.tokenizer = tokenizer
        self.datasets = []

        self.data = []

        self.split = split

        if savepath is None:
            dirname = self.tokenizer.__str__()[:self.tokenizer.__str__().index("(")]
            savepath = os.path.join(CACHE_PATH, dirname)
        self.savepath = savepath

        self.overwrite = overwrite
        self.load_from_cache_file = load_from_cache_file


        self.max_length = max_length
        self.keep_length = keep_length
        self.overlap_length = overlap_length

        self.dev_mode = dev_mode

        for ds in datasets:
            if ds == "switchboard":
                self.datasets.append(SwitchboardDataset(split=self.split))
            elif ds == "fisher":
                self.datasets.append(FisherDataset(split=self.split))

    def __len__(self):
        if self.dev_mode:
            return 1000

        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self,batch):
        ret = self.tokenizer.pad(
            {"input_ids": [b["input_ids"][: self.max_length] for b in batch]}
        )
        ret["token_type_ids"] = self.tokenizer.pad(
            {"input_ids": [b["token_type_ids"][: self.max_length] for b in batch]}
        )["input_ids"]
        for k, v in ret.items():
            ret[k] = v.clone().detach()
        return ret

    def get_save_load_path(self):
        save_load_dir = get_abs_path(self.savepath)

        if not os.path.exists(save_load_dir):
            os.mkdir(save_load_dir)

        return os.path.join(save_load_dir, "".join(str(x) for x in self.datasets) + self.split)

    def prepare_data(self):
        if self.load_from_cache_file or not self.overwrite:
            # Returns false if setup fails for whatever reason
            if self.setup():
                return

        for ds in self.datasets:
            # Extracts Dialog
            ds()

        self.dataset = ConcatDataset(self.datasets)
        self.data = self.tokenize()

        self.data = self.split_to_length()
        # self.data = self.pad_to_length()

        self.save_to_disk()

    def setup(self):
        filename = self.get_save_load_path()
        try:
            saved_ds = torch.load(filename)
            self.dataset = saved_ds.dataset
            self.data = saved_ds.data
            return True
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {filename}")

        return False

    def save_to_disk(self):
        self.logger.info(f"data {self.split}: saving combined transcript")
        torch.save(self, self.get_save_load_path())

    def get_dialog_idx(self, token_idx):
        dialog_idx = next(i for i, v in enumerate(
            self.prefix_sum) if v > token_idx) - 1
        token_idx = (
                            token_idx - self.prefix_sum[dialog_idx]) * self.window_step

        return dialog_idx, token_idx

    def filter_empty_turns(self, dataset):
        """
        return only dialogs with no empty turns
        """
        for idx,dataset in enumerate(dataset):
            dialogs = []
            for dialog in dataset:
                if not (dialog['text'] == "" or not re.search(r"\w", dialog['text'])):  # utt is empty
                    dialogs.append(dialog)
            dataset[idx] = dialogs
        return dataset

    """
    Splits each dialog into sequences of `max_length` tokens or whatever is left with overlap length in token overlap
    between sequences.
    Also pads to max_length
    """
    def split_to_length(self):
        # Augment entirety of datasets
        result = []
        start_idx = 0
        end_idx = self.overlap_length
        # self.dataset iterates data of each dataset

        for dialog in self.data:
            output = {}
            tokens = dialog['input_ids'][0]
            types = dialog['token_type_ids'][0]

            while end_idx <= len(tokens):
                # Split to appropriate lengths
                start_idx = end_idx - self.overlap_length
                # End either with a maximum length sequence or whatever is left
                end_idx = min(self.max_length + start_idx, len(tokens))

                if end_idx - start_idx < self.keep_length:
                    end_idx = self.overlap_length
                    break
                output['input_ids'] = tokens[start_idx:end_idx].clone().detach()
                output['token_type_ids'] = types[start_idx:end_idx].clone().detach()
                output['attention_mask'] = torch.ones(end_idx - start_idx)

                result.append(copy.deepcopy(output))

        return result


    """
    Uses self.tokenizer to tokenize each dialog within a dataset and then also initialise attention mask 
    and token_type_ids as speaker_ids
    """
    def tokenize(self):
        self.logger.info(f"data ({self.split}): tokenizing data")

        result = []
        for dataset in self.dataset:
            output = {}
            dialog = [dialog['text'] for dialog in dataset]

            # Add space after turn shift to produce G-hat word
            output['dialog'] = "<ts> ".join(dialog)
            tokens = self.tokenize_sentence(output['dialog'])

            output['input_ids'] = tokens['input_ids']
            output['attention_mask'] = tokens['attention_mask']

            current_speaker = dataset[0]['speaker']
            token_type_ids = [[]]
            for token in output['input_ids'][0]:
                # Is [SEP] token self.tokenizer.encode('[SEP]') -> [101, 102, 102]
                if token.item() == self.tokenizer.eos_token_id:
                    current_speaker = 'A' if current_speaker == 'B' else 'B'

                token_type_ids[0].append(
                    0 if current_speaker == 'A' else 1)

            output['token_type_ids'] = torch.tensor(token_type_ids)
            result.append(output)

        self.logger.info(
            f"data ({self.split}): done tokenizing dataset with keys: {result[0].keys}")
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
                (len(conv['input_ids'][0])//self.window_step + indexes[-1]))

        return indexes


def init_tokenizer(tokens=['!', '?', '.']):
    tokenizer = SpokenDialogTokenizer()
    return tokenizer


if __name__ == "__main__":
    tokenizer = init_tokenizer()
    ts = GenerationDM(
        tokenizer=tokenizer, overwrite=True, split="test")
    ts.prepare_data()
    dl = DataLoader(ts, batch_size=4, collate_fn=ts.collate_fn)

    batch = next(iter(dl))

    for i in range(10):
        print(batch['input_ids'].shape)
        print(ts.tokenizer.decode(batch['input_ids'][i]))
        print("\n")
    print("YES")