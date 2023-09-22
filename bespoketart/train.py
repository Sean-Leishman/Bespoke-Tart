import torch
from torch.utils.data import DataLoader

import argparse
import logging

from data import TranscriptDataset
from model import Bert, DistilledBert
from trainer import Trainer


def build_logger():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    logging.info("Logger built")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Bespoke Tart model used to predict turn-taking from linguistic features")
    parser.add_argument('--cuda', type=str, default='true')
    parser.add_argument('--epoch-size', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.0002)
    parser.add_argument('--bert-finetuning', type=str, default='false')
    parser.add_argument('--save-path', type=str, default='model/')

    return parser.parse_args()


def collate_loader(batch):
    input_ids = [item['input_ids'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]


def main(config):
    # model = Bert(
    #    pretrained_model_name='bert-base-uncased',
    #    bert_finetuning=True if config.bert_finetuning == 'true' else False,
    #    config=config,
    # )
    logging.getLogger(__name__).info("model: initialising model")
    model = DistilledBert(
        bert_finetuning=True if config.bert_finetuning == 'true' else False,
        config=config,
    )
    model.to(config.device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    trainer = Trainer(model=model, criterion=criterion,
                      optimizer=optimizer, config=config)

    train_dl = DataLoader(TranscriptDataset(
        "train", model.get_tokenizer()), batch_size=config.batch_size)
    test_dl = DataLoader(TranscriptDataset(
        "test", model.get_tokenizer()), batch_size=config.batch_size)

    logging.getLogger(__name__).info("model: train model")
    history = trainer.train(train_dl)


if __name__ == "__main__":
    config = build_parser()
    config.device = "cuda" if torch.cuda.is_available() and (
        config.cuda == "true") else "cpu"

    build_logger()

    logging.getLogger(__name__).info(f"{config}, {__name__}")
    main(config)
