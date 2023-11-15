import torch
import logging

from argparse import ArgumentParser

from transformers import BertModel, DistilBertModel, AutoTokenizer, AutoModel, BertLMHeadModel, BertConfig, GenerationConfig
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel

class EncoderDecoderBert(torch.nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", bert_finetuning=True, num_labels=1, config=None):
        super(EncoderDecoderBert, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, padding_side='left')

        self.encoder = BertGenerationEncoder.from_pretrained(pretrained_model_name, bos_token_id=101, eos_token_id=102)
        self.decoder = BertLMHeadModel.from_pretrained(pretrained_model_name,
                                                             add_cross_attention=True, is_decoder=True,
                                                             bos_token_id=101, eos_token_id=102)
        self.bertgeneration = EncoderDecoderModel(encoder=self.encoder, decoder=self.decoder)
        self.bertgeneration.config.pad_token_id = self.tokenizer.pad_token_id
        self.bertgeneration.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.bertgeneration.config.eos_token_id = self.tokenizer.sep_token_id
        self.bertgeneration.config.vocab_size = self.bertgeneration.config.encoder.vocab_size

        if not bert_finetuning:
            self.logger.info('model: bert parameters frozen')
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.bertgeneration.parameters():
                param.requires_grad = False


    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                output_ids=None, output_attention=None, output_token_type_ids=None):

        out = self.bertgeneration(input_ids=input_ids,
                                  decoder_input_ids=output_ids,
                                  attention_mask=attention_mask)

        if output_ids is not None:
            # Next token prediction; shift prediction and input ids by one;
            # https://github.com/huggingface/transformers/blob/v4.35.1/src/transformers/models/bert/modeling_bert.py#L1154
            shifted_prediction_scores = out.logits[:, :-1, :].contiguous()
            labels = output_ids[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.decoder.config.vocab_size), labels.view(-1))
            out.loss = lm_loss

        return out

    def generate(self, input_ids, mask=None):
        out = self.bertgeneration.generate(
            inputs=input_ids,
            attention_mask=mask,
            do_sample=True,
            top_k=50,
            max_length=100,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id
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

