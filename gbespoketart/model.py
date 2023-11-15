import torch
import logging

from argparse import ArgumentParser

from transformers import BertModel, DistilBertModel, AutoTokenizer, AutoModel, BertLMHeadModel, BertConfig, GenerationConfig
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel

class GenerationBert(torch.nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", bert_finetuning=True, num_labels=1, config=None):
        super(GenerationBert, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, padding_side='left')
        """
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.bert.to(config.device)
        """

        config = BertConfig.from_pretrained("bert-base-uncased")
        config.update({'is_decoder':True})
        self.bertlmhead = BertLMHeadModel.from_pretrained(pretrained_model_name_or_path="bert-base-uncased",config=config)

        self.generation_config=GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.cls_token_id,
            max_length= 200,
            num_return_sequences= 1,
            do_sample=True,
        )

        if not bert_finetuning:
            self.logger.info('model: bert parameters frozen')
            for param in self.bertlmhead.parameters():
                param.requires_grad = False


    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                output_ids=None, output_attention=None, output_token_type_ids=None):

        out = self.bertlmhead(input_ids=input_ids,
                                  labels=output_ids,
                                  attention_mask=attention_mask,
        )

        return out

    def generate(self, input_ids, mask=None):
        out = self.bertlmhead.generate(
            inputs=input_ids,
            attention_mask=mask,
            generation_config=self.generation_config,
        )
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

