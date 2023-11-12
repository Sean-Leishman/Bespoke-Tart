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

    def generate(self, input_ids=None, mask=None):
        if input_ids is None:
            sample_output = self.gpt.generate(
                bos_token_id=random.randint(1,30000),
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
                do_sample=True,
                top_k=50,
                max_length=300,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return sample_output

    def init_tokenizer(self):
        num_added_token = self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[SEP]']}
        )
        self.gpt.resize_token_embeddings(len(self.tokenizer))

        self.logger.info(
            f"model: add {self.tokenizer.all_special_tokens} token/s")

    def get_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        #self.tra May be useful to separate params
