import torch
import logging

from transformers import BertModel, DistilBertModel, AutoTokenizer


class DistilledBert(torch.nn.Module):
    def __init__(self, pretrained_model_name="distilbert-base-uncased",
                 bert_finetuning=True, num_labels=1, config=None):
        super(DistilledBert, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            'distilbert-base-uncased')

        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        self.bert.to(config.device)
        self.logger = logging.getLogger(__name__)

        if not bert_finetuning:
            self.logger.info('model: bert parameters frozen')
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_labels)

        self.output_activation = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        output = self.bert(
            input_ids, attention_mask=attention_mask)
        hidden_state = output[0]
        pooled_output = hidden_state[:, 0]

        final_output = self.classifier(pooled_output)
        return self.output_activation(final_output)

    def get_tokenizer(self):
        return self.tokenizer


class Bert(torch.nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", bert_finetuning=True, num_labels=1, config=None):
        super(Bert, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.bert.to(config.device)
        self.logger = logging.getLogger(__name__)

        if not bert_finetuning:
            self.logger.info('model: bert parameters frozen')
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_labels)
        self.output_activation = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        output = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=None)
        hidden_state = output[0]
        pooled_output = hidden_state[:, 0]

        return self.classifier(pooled_output)

    def get_tokenizer(self):
        return self.tokenizer
