import torch
import logging

from argparse import ArgumentParser

from transformers import BertModel, DistilBertModel, AutoTokenizer, AutoModel
from transformers import BertLMHeadModel, BertForMaskedLM

class MaskedBert(torch.nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", bert_finetuning=True, num_labels=1, config=None):
        super(MaskedBert, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        """
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.bert.to(config.device)
        """

        self.dropout = torch.nn.Dropout(p=0.1)

        self.bertmasklm = BertForMaskedLM.from_pretrained(pretrained_model_name)
        self.bertclm = BertLMHeadModel.from_pretrained(pretrained_model_name)

        if not bert_finetuning:
            self.logger.info('model: bert parameters frozen')
            for param in self.bertclm.parameters():
                param.requires_grad = False
            for param in self.bertmasklm.parameters():
                param.requires_grad = False

        self.softmax = torch.nn.Softmax(dim=2)
        self.relu = torch.nn.ReLU()

        self.output = torch.nn.Linear(
           768 , self.tokenizer.vocab_size)

        self.output_activation = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                output_ids=None, output_attention=None, output_token_type_ids=None):

        out = self.bertmasklm(input_ids=input_ids, attention_mask=attention_mask, labels=output_ids)
        return out

    def get_probability(self, logits):
        return self.output_activation(logits)

    # out: Seq2SeqLMOutput
    def output_word(self, out):
        return self.tokenizer.decode(torch.argmax(self.softmax(out.logits))[1])
    def init_tokenizer(self, tokenizer):
        num_added_token = tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[EOT]']}
        )

        self.logger.info(
            f"model: add {tokenizer.all_special_tokens} token/s")

        return tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        #self.tra May be useful to separate params

