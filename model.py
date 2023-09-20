import torch

from transformers import BertConfig, BertModel


class Bert(torch.nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", freeze_bert=False, num_labels=1, config=None):
        super(Bert, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.bert.to(config.device)

        if freeze_bert:
            for param in self.bert.parameters():

                param.requires_grad = False

        self.classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        output = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=None)
        hidden_state = output[0]
        pooled_output = hidden_state[:, 0]
        return self.classifier(pooled_output)
