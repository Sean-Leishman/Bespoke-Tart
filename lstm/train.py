import torch
from torch.utils.data import DataLoader

import argparse
import logging
import json
import os
import warnings

from datetime import datetime

from data import TranscriptDataset
from types import SimpleNamespace
from lstm.model import LSTMBase
from trainer import Trainer



def build_logger():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    logging.info("Logger built")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='true',
                        help="true/false if cuda should be enabled")
    parser.add_argument('--epoch-size', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--early-stop', type=int, default=5,
                        help='number of iterations without improvement for early stop')
    parser.add_argument('--weight-decay', type=float, default=0.01)

    parser.add_argument('--save-path', type=str, default='trained_model/',
                        help="model weights and config options save directory")

    parser.add_argument('--overwrite', type=str, default="false",
                        help="overwrite and regenerate dataset")

    parser.add_argument('--loss-weight', type=float, default=1.355,
                        help="for binary classification and weighting EOTs")
    parser.add_argument('--output-window', type=int, default=5,
                        help="number of tokens to determine label in binary classification")
    parser.add_argument('--context-window', type=int, default=2,
                        help="number of full turns to be used in the prior context")
    parser.add_argument('--max-prior-window', type=int, default=300,
                        help="number of tokens to be used as a max in the prior context")
    return parser.parse_args()

def collate_fn(batch):
    batched_data = {'input': None, 'output': None}
    for key in ['input', 'output']:
        input_ids = [item[key]['input_ids'] for item in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0)

        if key == 'input':
            batched_data = {'input_ids': input_ids_padded}
        else:
            batched_data[key] = {'input_ids': input_ids_padded}
    return batched_data


def main(config):
    # model = Bert(
    #    pretrained_model_name='bert-base-uncased',
    #    bert_finetuning=True if config.bert_finetuning == 'true' else False,
    #    config=config,
    # )
    logging.getLogger(__name__).info("model: initialising model")

    model = LSTMBase()
    model.to(config.device)

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.FloatTensor([config.loss_weight]).to(config.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    trainer = Trainer(model=model, criterion=criterion,
                      optimizer=optimizer, config=config)

    train_dl = DataLoader(TranscriptDataset(
            split="train",
            tokenizer=model.get_tokenizer(),
            overwrite=True if config.overwrite == 'true' else False,
            max_prior_window_size=config.max_prior_window,
            context_window=config.context_window
        ),
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    test_dl = DataLoader(TranscriptDataset(
        split="test",
        tokenizer=model.get_tokenizer(),
        overwrite=True if config.overwrite == 'true' else False,
        max_prior_window_size=config.max_prior_window,
        context_window=config.context_window
    ),
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )

    logging.getLogger(__name__).info("model: train model")
    history = trainer.train(train_dl, test_dl)

    return history


if __name__ == "__main__":
    warnings.filterwarnings('always')

    config = build_parser()
    config.device = "cuda" if torch.cuda.is_available() and (
        config.cuda == "true") else "cpu"

    build_logger()

    logging.getLogger(__name__).info(f"{config}")
    main(config)
