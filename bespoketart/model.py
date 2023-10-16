import torch
import logging

from transformers import BertModel, DistilBertModel, AutoTokenizer


class DistilledBert(torch.nn.Module):
    def __init__(self,
                 pretrained_model_name="distilbert-base-uncased",
                 bert_finetuning=True, num_labels=1, config=None):
        super(DistilledBert, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.tokenizer = self.init_tokenizer(
            AutoTokenizer.from_pretrained(pretrained_model_name))

        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        self.bert.to(config.device)

        if not bert_finetuning:
            self.logger.info('model: bert parameters frozen')
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = torch.nn.Dropout(p=0.1)
        self.fc1 = torch.nn.Linear(self.bert.config.hidden_size, 20)
        self.relu = torch.nn.ReLU()

        self.output = torch.nn.Linear(
            20, num_labels)

        self.output_activation = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        bert_output = self.bert(
            input_ids, attention_mask=attention_mask)

        # mean_pooled = torch.mean(bert_output.last_hidden_state, dim=1)
        x = self.dropout(0.2)
        x = self.fc1(x)
        x = self.relu(x)

        final_output = self.output(x)
        return final_output

    def get_probability(self, logits):
        return self.output_activation(logits)

    def init_tokenizer(self, tokenizer):
        num_added_token = tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[EOT]']}
        )

        self.logger.info(
            f"model: add {tokenizer.all_special_tokens} token/s")

        return tokenizer

    def get_tokenizer(self):
        return self.tokenizer


class Bert(torch.nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", bert_finetuning=True, num_labels=1, config=None):
        super(Bert, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

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
