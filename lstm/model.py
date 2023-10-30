import torch
import logging

from argparse import ArgumentParser
from transformers import AutoTokenizer

class LSTMBase(torch.nn.Module):
    def __init__(self, tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'), config=None):
        super(LSTMBase, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config

        self.tokenizer = tokenizer

        self.embedding_dim = 200
        self.hidden_size = 768
        self.num_layers = 3

        self.embedding = torch.nn.Embedding(len(tokenizer), embedding_dim=self.embedding_dim)
        self.lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(self.hidden_size, self.embedding_dim)

        self.output_activation = torch.nn.Sigmoid()

    def get_probability(self, logits):
        return self.output_activation(logits)

    def get_word(self, vector):
        return self.embedding(vector)
    def forward(self, input_ids, hidden):
        embedding = self.dropout(self.embedding(input_ids))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output[:,-1,:])
        return prediction, hidden

    def get_tokenizer(self):
        return self.tokenizer

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden,cell

