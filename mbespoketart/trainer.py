import torch
import os
import json
import numpy as np
from datetime import datetime

import logging
import time

from tqdm import tqdm
from sklearn.metrics import f1_score

from seqeval.metrics import classification_report, f1_score, accuracy_score
from torchmetrics.text import BLEUScore
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryRecall
from torchmetrics.classification import F1Score, Accuracy
from torchmetrics.text import Perplexity

def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)

def get_new_filename(save_dir):
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d:%H-%M-%S")
    return os.path.join(save_dir, current_time_str)



class Trainer:
    def __init__(self, model=None, criterion=None, optimizer=None, config=None,
                 load_from_checkpoint=None):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.epochs = config.epochs
        self.device = torch.device(
            config.device if config is not None else "cpu")

        self.epoch = 0

        self.train_history = {}
        self.test_history = {}

        self.best = {
            'epoch': 0,
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'loss': None,
            'config': config,
        }

        self.save_path = get_abs_path(get_new_filename(config.save_path))
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        with open(os.path.join(self.save_path, "config.json"), "w") as config_file:
            print(config)
            json.dump(vars(config), config_file)

        self.logger = logging.getLogger(__name__)

        self.load_model_file = load_from_checkpoint
        if self.load_model_file is not None:
            self.load_from_checkpoint()

        self.metrics = {
            'acc': Accuracy(task="multiclass", num_classes=self.model.tokenizer.vocab_size).to(self.config.device),
            'f1': F1Score(task="multiclass", num_classes=self.model.tokenizer.vocab_size).to(self.config.device),
            'perplexity': Perplexity(ignore_index=-100).to(self.config.device),
        }


    def load_from_checkpoint(self):
        checkpoint = torch.load(self.load_model_file)
        self.logger.info(
            f"model: loading parameters for model {checkpoint.keys()}")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']

    def train(self, train_dl, test_dl, scheduler=None):
        best_loss = float('Inf')
        self.scheduler = scheduler

        # For early stopping, number of iterations without loss improvement
        not_improving_x = 0
        progress_bar = tqdm(range(self.epoch, self.epoch +
                            self.epochs), desc='Epoch   ')

        for idx in progress_bar:
            train_metrics = self.train_epoch(train_dl)
            test_metrics = self.validate(test_dl)

            for key in train_metrics:
                if key not in self.train_history:
                    self.train_history[key] = []
                self.train_history[key].append(train_metrics[key])
            for key in test_metrics:
                if key not in self.test_history:
                    self.test_history[key] = []
                self.test_history[key].append(test_metrics[key])

            self.save_history(self.save_path)

            avg_valid_loss = (train_metrics['avg_loss'] + test_metrics['avg_loss']) / 2

            if avg_valid_loss < best_loss:
                not_improving_x = 0

                best_loss = avg_valid_loss
                self.best['model_state_dict'] = self.model.state_dict()
                self.best['optimizer_state_dict'] = self.optimizer.state_dict()
                self.best['loss'] = best_loss
                self.best['epoch'] = idx + 1

                model_name = "model_" + str(self.best['epoch']) + ".pt"
                self.save_training(os.path.join(self.save_path, model_name))

                self.logger.info(
                    f"train: saving model at {os.path.join(self.save_path, model_name)}")
            else:
                not_improving_x += 1

                if not_improving_x >= self.config.early_stop and self.config.early_stop > 0:
                    self.logger.info("train: early stop")
                    progress_bar.close()
                    return self.train_history

        progress_bar.close()
        return self.train_history

    def get_postfix_str(self, step, f1, loss, count, tp, fp, fn, tn):
        return (f'loss={loss / (step + 1): .4f}, f1={f1 / (step + 1): .4f}, accuracy={(tp+tn) / count: .4f} '
        f'precision={(tp/(tp+fp)) if tp+fp != 0 else 1: .4f}, recall={tp/(tp+fn) if tp+fn != 0 else 0: .4f}, '
                f'bAcc={0.5 * (tp/(tp+fn) + tn/(fp+tn)) if tp+fn != 0 and fp+tn!=0 else 0: .4f}')

    def metric_output(self, metrics):
        output = ""
        if 'avg_loss' in metrics.keys():
            output += f"loss={metrics['avg_loss'] :.4f} "
        if 'f1' in metrics.keys():
            output += f"f1={metrics['f1'] :.4f} "
        if 'acc' in metrics.keys():
            output += f"acc={metrics['acc'] :.4f} "
        if 'recall' in metrics.keys():
            output += f"loss={metrics['eot_recall'] :.4f} "
        if 'rouge_mean' in metrics.keys():
            output += f"rouge2={metrics['rouge_mean'] :.4f} "

        return output

    def compute_metrics(self, p):
        # SEP Token within output window
        id = self.model.tokenizer.convert_tokens_to_ids('[SEP]')

        results = {
            'f1': torch.mean(torch.tensor([self.metrics['f1'](pred, label) for pred, label in p])),
            'acc': torch.mean(torch.tensor([self.metrics['acc'](pred, label) for pred, label in p])),
        }
        return results

    def train_epoch(self, train_dl):
        self.model.train()
        total_loss, total_count = 0,0
        total_f1 = 0
        tp, fp, fn, tn = 0,0,0,0

        progress_bar = tqdm(train_dl, desc='Training', unit="batch")
        padding = torch.zeros((8,183)).to(self.config.device)
        padding[:,:5] = 1

        pred_label = []

        for step, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)

            labels = self.generate_labels(input_ids, mask=attention_mask)

            mask = torch.rand(input_ids.shape) > 0.85
            input_ids[mask] = self.model.tokenizer.mask_token_id

            out = self.model.forward(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_ids=labels)

            predicted_token_ids = torch.argmax(out.logits[mask], dim=-1)
            labels = torch.where(input_ids==self.model.tokenizer.mask_token_id, labels, -100)
            pred_label.append((predicted_token_ids, labels[mask]))

            loss = out.loss
            if loss is None:
                loss = self.calculate_loss(out.logits, labels)
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            total_count += input_ids.shape[0] * input_ids.shape[1]

            avg_loss = round(total_loss / (step+1),4)
            progress_bar.set_postfix_str(f"loss={avg_loss}")


        avg_loss = total_loss / len(train_dl)
        metrics = self.compute_metrics(pred_label)
        metrics['avg_loss'] = avg_loss

        progress_bar.disable = False
        progress_bar.set_postfix_str(self.metric_output(metrics))
        progress_bar.close()


        return metrics

    def generate_labels(self, input_ids, mask=None, pad_id=-100):
        labels = input_ids.clone()
        labels[torch.logical_not(mask)] = pad_id

        return labels

    def calculate_loss(self, output, labels, padding=None):
        mask = torch.ones(labels.shape).to(self.config.device)
        if padding is not None:
            mask = padding

        self.criterion.reduction = "none"
        loss = self.criterion(output.view(-1, output.size(-1)), labels.view(-1))
        loss = loss.view(labels.shape)

        loss *= mask
        return loss.sum() / mask.sum()

    def validate(self, test_dl):
        total_loss, total_count = 0, 0
        total_f1 = 0
        tp, fp, fn, tn = 0,0,0,0

        padding = torch.zeros((8,183)).to(self.config.device)
        padding[:,:5] = 1

        pred_label = []

        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(test_dl, desc='Validation')

            for step, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)


                labels = self.generate_labels(input_ids, mask=attention_mask)

                mask = torch.rand(input_ids.shape) > 0.85
                input_ids[mask] = self.model.tokenizer.mask_token_id

                out = self.model.forward(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_ids=labels)

                predicted_token_ids = torch.argmax(out.logits[mask], dim=-1)
                labels = torch.where(input_ids==self.model.tokenizer.mask_token_id, labels, -100)
                pred_label.append((predicted_token_ids, labels[mask]))

                loss = out.loss
                if loss is None:
                    loss = self.calculate_loss(out.logits, labels)

                total_loss += loss.item()
                total_count += input_ids.shape[0] * input_ids.shape[1]

                avg_loss = round(total_loss / (step+1),4)
                progress_bar.set_postfix_str(f"loss={avg_loss}")


            avg_loss = total_loss / len(test_dl)
            metrics = self.compute_metrics(pred_label)

        progress_bar.disable = False
        # progress_bar.set_postfix_str(self.metric_output(metrics))

        progress_bar.close()
        self.model.train()

        return metrics

    def save_training(self, path):
        torch.save(self.best, path)

    def save_history(self, path):
        self.logger.info("trainer: save history")
        for key in self.train_history:
            np.save(os.path.join(path, f"train_{key}"), self.train_history[key])
        for key in self.test_history:
            np.save(os.path.join(path, f"test_{key}"), self.test_history[key])

    def print_dialogue(self, input_ids, prediction, output, label):
        output = f"Input: {self.model.tokenizer.decode(input_ids)}\n"
        output += f"Output: {self.model.tokenizer.decode(output['input_ids'])}\n"
        output += f"Prediction: {prediction}, Label: {label}"

        print(output)