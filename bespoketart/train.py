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
from model import Bert, DistilledBert
from trainer import Trainer


def build_logger():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    logging.info("Logger built")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Bespoke Tart model used to predict turn-taking from linguistic features")
    parser.add_argument('--cuda', type=str, default='true',
                        help="true/false if cuda should be enabled")
    parser.add_argument('--load-model', type=str, default='false',
                        help="true/false if model should be loaded")
    parser.add_argument('--load-path', type=str, default='trained_model/',
                        help="load model config and weights from this file and ignore input configurations")
    parser.add_argument('--save-path', type=str, default='trained_model/',
                        help="model weights and config options save directory")

    parser.add_argument('--bert-type', type=str, default='distilbert',
                        help="choose which BERT version to use (bert, distilbert)")
    parser.add_argument('--bert-finetuning', type=str, default='false',
                        help='true/false if BERT should be finetuned')
    parser.add_argument('--bert-pretraining', type=str,
                        default='bert-base-uncased',
                        help="name of pretrained BERT model")

    parser.add_argument('--epoch-size', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.0002)
    parser.add_argument('--early-stop', type=int, default=5,
                        help='number of iterations without improvement for early stop')

    parser.add_argument('--evaluate', type=str, default='false',
                        help='model should only be evaluated. load-model and load-path should be set')

    return parser.parse_args()


def get_new_filename(save_dir):
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d:%H-%M-%S")
    return os.path.join(save_dir, current_time_str)


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


def collate_fn(batch):
    batched_data = {'input': None, 'output': None}
    for key in ['input', 'output']:
        input_ids = [item[key]['input_ids'] for item in batch]
        attention_masks = [item[key]['attention_mask'] for item in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0)
        attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0)

        if key == 'input':
            batched_data = {'input_ids': input_ids_padded,
                            'attention_mask': attention_masks_padded}
        else:
            batched_data[key] = {'input_ids': input_ids_padded,
                                 'attention_mask': attention_masks_padded}
    return batched_data


def main(config):
    # model = Bert(
    #    pretrained_model_name='bert-base-uncased',
    #    bert_finetuning=True if config.bert_finetuning == 'true' else False,
    #    config=config,
    # )
    logging.getLogger(__name__).info("model: initialising model")

    if config.load_model == 'true':
        load_path = config.load_path
        logging.getLogger(__name__).info(
            f"model: loading model from {load_path}")

        with open(os.path.join(load_path, "config.json")) as f:
            new_config = SimpleNamespace(**json.load(f))
            for arg in vars(new_config):
                config.arg = getattr(new_config, arg)

        logging.getLogger(__name__).info(f"Loaded config: {config}")

    if config.bert_type == 'distilbert':
        logging.getLogger(__name__).info(f"Loaded model: distilbert")
        model = DistilledBert(
            bert_finetuning=True if config.bert_finetuning == 'true' else False,
            config=config,
        )
    elif config.bert_type == 'bert':
        logging.getLogger(__name__).info(f"Loaded model: bert")
        model = Bert(bert_finetuning=True if config.bert_finetuning == 'true' else False,
                     config=config)
    else:
        logging.getLogger(__name__).info(
            f"Loaded model: invalid bert name {config.bert_type}. loaded distilbert")
        model = DistilledBert(
            bert_finetuning=True if config.bert_finetuning == 'true' else False,
            config=config,
        )

    model.to(config.device)

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.FloatTensor([1.5]).cuda())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if (config.load_model == 'true'):
        trainer = Trainer(model=model, criterion=criterion,
                          optimizer=optimizer, config=config,
                          load_from_checkpoint=get_latest_model(load_path))
    else:
        trainer = Trainer(model=model, criterion=criterion,
                          optimizer=optimizer, config=config)

    if config.evaluate == 'false':
        train_dl = DataLoader(TranscriptDataset(
            "train", model.get_tokenizer()), batch_size=config.batch_size,
            collate_fn=collate_fn,
            shuffle=True)
        test_dl = DataLoader(TranscriptDataset(
            "test", model.get_tokenizer()), batch_size=config.batch_size, collate_fn=collate_fn)

        logging.getLogger(__name__).info("model: train model")
        history = trainer.train(train_dl, test_dl)
    else:
        test_dl = DataLoader(TranscriptDataset(
            "test", model.get_tokenizer()), batch_size=config.batch_size, collate_fn=collate_fn)

        logging.getLogger(__name__).info("model: evaluate model")
        if not config.load_model:
            logging.getLogger(__name__).error(
                "model: model is not being loaded")
            return None

        history = trainer.validate(test_dl)

    return history


if __name__ == "__main__":
    warnings.filterwarnings('always')

    config = build_parser()
    config.device = "cuda" if torch.cuda.is_available() and (
        config.cuda == "true") else "cpu"

    build_logger()

    logging.getLogger(__name__).info(f"{config}")
    main(config)
