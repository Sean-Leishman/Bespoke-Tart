import torch
from torch.utils.data import DataLoader

import argparse
import logging
import json
import os
import warnings

from datetime import datetime

from data import TranscriptDataset, GenerationDM
from types import SimpleNamespace

from gptonly import GPT
from trainer import Trainer, get_abs_path

from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup


import wandb


def build_logger():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    logging.info("Logger built")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Bespoke Tart model used to predict turn-taking from linguistic features")
    parser.add_argument('--cuda', action="store_true",
                        help="true/false if cuda should be enabled")
    parser.add_argument('--load-model', action="store_true",
                        help="true/false if model should be loaded")
    parser.add_argument('--load-path', type=str, default='trained_model/',
                        help="load model config and weights from this file and ignore input configurations")
    parser.add_argument('--save-path', type=str, default='trained_model/',
                        help="model weights and config options save directory")

    parser.add_argument('--finetune', action="store_true",
                        help='true/false if BERT should be finetuned')
    parser.add_argument('--pretrained', type=str,
                        default='bert-base-uncased',
                        help="name of pretrained BERT model")

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--early-stop', type=int, default=5,
                        help='number of iterations without improvement for early stop')

    parser.add_argument('--evaluate', action='store_true',
                        help='model should only be evaluated. load-model and load-path should be set')

    parser.add_argument('--description', type=str, default='',
                        help="description of model")

    # Wandb
    parser.add_argument('--log_interval', type=int, default=100,
                        help="frequency with which to report logs to wandb")

    # Dataset
    parser.add_argument('--overwrite', action='store_true',
                        help="overwrite and regenerate dataset")
    parser.add_argument('--dev-mode', action='store_true',
                        help="decrease dataset size to test post-processing steps/disable writing to wandb")
    parser.add_argument('--datasets', nargs="+", help="Datasets to use",
                        default=["switchboard", "fisher"])
    parser.add_argument('--max-length', type=int, default=256,
                        help="max length of a sequence")
    parser.add_argument('--keep-length', type=int, default=64,
                        help="minimum length of a sequence")
    parser.add_argument('--overlap-length', type=int, default=10,
                        help="number of tokens to overlap between sequences")

    parser.add_argument('--speaker-tokens', action="store_true",
                        help="add speaker tokens as token type ids")
    parser.add_argument('--projection-labels', action='store_true',
                        help="add projection labels and convert to multitask learning")

    return parser.parse_args()


def get_latest_model(path):
    list_dir = os.listdir(path)
    latest_model = None
    latest_index = -1
    for item in list_dir:
        if item[:5] == 'model':
            index = int(''.join(x for x in item if x.isdigit()))
            if latest_model is None or index > latest_index:
                latest_index = index
                latest_model = item

    if latest_model is None:
        raise RuntimeError("model file not found")
    return os.path.join(path, latest_model)


def main(config):
    # model = Bert(
    #    pretrained_model_name='bert-base-uncased',
    #    bert_finetuning=True if config.bert_finetuning == 'true' else False,
    #    config=config,
    # )
    logging.getLogger(__name__).info("model: initialising model")

    if not config.dev_mode:
        name = input("Enter new change(s) for wandb run: ")
        if name == "":
            name = None

        wandb.init(
            config=config,
            name=name
        )

    if config.load_model:
        load_path = get_abs_path(config.load_path)
        logging.getLogger(__name__).info(
            f"model: loading model from {load_path}")

        with open(os.path.join(load_path, "config.json")) as f:
            new_config = SimpleNamespace(**json.load(f))
            for arg in vars(new_config):
                config.arg = getattr(new_config, arg)

        logging.getLogger(__name__).info(f"Loaded config: {config}")

    logging.getLogger(__name__).info(
        f"Loaded model: gpt with finetuning: {config.finetune}")
    model = GPT(
        pretrained_model_name=config.pretrained,
        finetune=config.finetune,
        device=config.device,
        speaker_tokens=config.speaker_tokens,
        projection_labels=config.projection_labels,
    )

    model.to(config.device)

    # criterion = torch.nn.BCEWithLogitsLoss(
    #    pos_weight=torch.FloatTensor([config.loss_weight]).to(config.device))
    criterion = torch.nn.CrossEntropyLoss().to(config.device)
    # , weight_decay=config.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if config.load_model:
        trainer = Trainer(model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          config=config,
                          load_from_checkpoint=get_latest_model(load_path),
                          **vars(config)
                          )
    else:
        trainer = Trainer(model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          config=config,
                          **vars(config)
                          )

    if not config.evaluate:
        train_ds = GenerationDM(
            split="train",
            tokenizer=model.get_tokenizer(),
            overwrite=config.overwrite,
            max_length=config.max_length,
            keep_length=config.keep_length,
            overlap_length=config.overlap_length,
            datasets=config.datasets,
        )
        train_ds.prepare_data()
        train_dl = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            collate_fn=train_ds.collate_fn,
            num_workers=8,
            shuffle=False
        )
        test_ds = GenerationDM(
            split="test",
            tokenizer=model.get_tokenizer(),
            overwrite=config.overwrite,
            max_length=config.max_length,
            keep_length=config.keep_length,
            overlap_length=config.overlap_length,
            datasets=config.datasets,
        )
        test_ds.prepare_data()
        test_dl = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            collate_fn=test_ds.collate_fn,
            num_workers=8,
            shuffle=False
        )

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
        scheduler = None

        logging.getLogger(__name__).info("model: train model")
        history = trainer.train(train_dl, test_dl, scheduler=scheduler)
    else:
        test_ds = GenerationDM(
            split="test",
            tokenizer=model.get_tokenizer(),
            overwrite=True if config.overwrite else False,
            max_length=config.max_length,
            keep_length=config.keep_length,
            overlap_length=config.overlap_length,
            datasets=config.datasets,
        )
        test_ds.prepare_data()
        test_dl = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            collate_fn=test_ds.collate_fn,
            num_workers=8,
            shuffle=True
        )

        logging.getLogger(__name__).info("model: evaluate model")
        if not config.load_model:
            logging.getLogger(__name__).error(
                "model: model is not being loaded")
            return None

        history = trainer.evaluate(test_dl)

    return history


if __name__ == "__main__":
    warnings.filterwarnings(
        action="ignore", category=DeprecationWarning, module="transformers")

    config = build_parser()
    config.device = "cuda" if torch.cuda.is_available() and (
        config.cuda) else "cpu"

    build_logger()

    logging.getLogger(__name__).info(f"{config}")
    main(config)
