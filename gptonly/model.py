import torch
import logging
import random

from argparse import ArgumentParser

from transformers import BertModel, DistilBertModel, AutoTokenizer, AutoModel
from transformers import GPT2LMHeadModel, GPT2Config

class GPT(torch.nn.Module):
    def __init__(self, pretrained_model_name="gpt2", bert_finetuning=True, num_labels=1, config=None):
        super(GPT, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        """
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.bert.to(config.device)
        """

        self.dropout = torch.nn.Dropout(p=0.1)

        config = GPT2Config.from_pretrained(pretrained_model_name)
        self.gpt = GPT2LMHeadModel.from_pretrained(pretrained_model_name, config=config)
        self.init_tokenizer()

        if not bert_finetuning:
            self.logger.info('model: bert parameters frozen')
            for param in self.gpt.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                output_ids=None, output_attention=None, output_token_type_ids=None):

        out = self.gpt(
            input_ids,
            labels=output_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return out

    def generate(self, input_ids=None, speaker_ids=None, mask=None, output_scores=False, n_sequences=1):
        if input_ids is None:
            sample_output = self.gpt.generate(
                bos_token_id=random.randint(1,30000),
                token_type_ids=speaker_ids,
                do_sample=True,
                top_k=50,
                max_length=100,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id
            )
        else:
            sample_output = self.gpt.generate(
                input_ids=input_ids,
                token_type_ids=speaker_ids,
                do_sample=True,
                top_k=50,
                max_length=300,
                top_p=0.95,
                num_return_sequences=n_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        return sample_output

    def init_tokenizer(self):
        num_added_token = self.tokenizer.add_special_tokens(
            {
                "eos_token": "[SEP]",
                "pad_token": "<|endoftext|>"
            }
        )
        self.gpt.resize_token_embeddings(len(self.tokenizer))
        # self.tokenizer.sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')

        self.logger.info(
            f"model: add {self.tokenizer.all_special_tokens} token/s")

    def get_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        #self.tra May be useful to separate params

    def from_string(self, string):
        output = {}
        output['dialog'] = "[SEP]".join(string)
        tokens = self.tokenizer(output['dialog'], return_tensors="pt", truncation=True)

        output['input_ids'] = tokens['input_ids'].to(self.config.device)

        current_speaker = 'A'
        token_type_ids = [[]]

        SEP_token = self.tokenizer.convert_tokens_to_ids("[SEP]")
        for token in output['input_ids'][0]:
            # Is [SEP] token self.tokenizer.encode('[SEP]') -> [101, 102, 102]
            if token.item() == SEP_token:
                current_speaker = 'A' if current_speaker == 'B' else 'B'

            token_type_ids[0].append(
                0 if current_speaker == 'A' else 1)

        output['token_type_ids'] = torch.tensor(token_type_ids).to(self.config.device)
        return output
