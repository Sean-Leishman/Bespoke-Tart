import torch
from torch.utils.data import DataLoader

import argparse

from data import TranscriptDataset
from model import Bert
from trainer import Trainer


def build_parser():
    parser = argparse.ArgumentParser(description="Diss")
    parser.add_argument('--cuda', type=str, default='true')
    parser.add_argument('--epoch_size', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.0002)

    return parser.parse_args()


def collate_loader(batch):
    input_ids = [item['input_ids'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]


def main(config):
    print(config)

    model = Bert(
        pretrained_model_name='bert-base-uncased',
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    trainer = Trainer(model=model, criterion=criterion,
                      optimizer=optimizer, config=config)

    train_dl = DataLoader(TranscriptDataset(
        "train"), batch_size=config.batch_size)
    test_dl = DataLoader(TranscriptDataset(
        "test"), batch_size=config.batch_size)

    print(next(iter(train_dl)))
    history = trainer.train(train_dl)


if __name__ == "__main__":
    config = build_parser()
    config.device = torch.device("cuda" if torch.cuda.is_available() and (
        config.device == "true") else "cpu")
    main(config)
