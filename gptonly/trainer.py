import torch
import os
import json
import numpy as np
from datetime import datetime

import logging
import time

import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

from seqeval.metrics import classification_report, f1_score, accuracy_score
from torchmetrics.text import BLEUScore
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryRecall
import evaluate

from gptonly.utils import plot_trp


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


def get_new_filename(save_dir):
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d:%H-%M-%S")
    return os.path.join(save_dir, current_time_str)


def get_latest_model(path, before=None):
    list_dir = os.listdir(path)
    latest_model = None
    max_index = 100000

    if before is not None:
        before = before.split("/")[-1][:-3]
        max_index = int(before[6:])

    latest_index = -1
    for item in list_dir:
        if item[:5] == 'model':
            index = int(''.join(x for x in item if x.isdigit()))
            if latest_model is None or index > latest_index and index < max_index:
                latest_index = index
                latest_model = item

    if latest_model is None:
        raise RuntimeError("model file not found")

    return os.path.join(path, latest_model)


class Trainer:
    def __init__(self,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 epochs=None,
                 load_from_checkpoint=None,
                 device=None,
                 log_interval=None,
                 save_path="./",
                 early_stop=5,
                 dev_mode=False,
                 config=None,
                 **kwargs,
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.log_interval = log_interval

        self.epoch = 0
        self.dev_mode = dev_mode

        self.train_history = {}
        self.test_history = {}

        self.best = {
            'epoch': 0,
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'loss': None,
        }

        self.save_path = get_abs_path(get_new_filename(save_path))
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        with open(os.path.join(self.save_path, "config.json"), "w") as config_file:
            json.dump(vars(config), config_file)

        self.logger = logging.getLogger(__name__)

        self.early_stop = early_stop

        self.load_model_file = load_from_checkpoint
        if self.load_model_file is not None:
            self.load_from_checkpoint()

        self.metrics = {
            'rouge': evaluate.load("rouge"),
            'acc': BinaryAccuracy().to(self.device),
            'f1': BinaryF1Score().to(self.device),
            'recall': BinaryRecall().to(self.device),
        }

        self.thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.bacc = [{
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "bacc": 0,
            "threshold": threshold,
        } for threshold in self.thresholds]

        self.global_step = 0

    def load_from_checkpoint(self):
        try:
            checkpoint = torch.load(self.load_model_file)
        except:
            self.load_model_file = get_latest_model(os.path.dirname(
                self.load_model_file), before=self.load_model_file)
            self.load_from_checkpoint()
        else:
            self.logger.info(
                f"model: loading parameters for model {checkpoint.keys()}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.model.tokenizer.from_pretrained(os.path.join(os.path.dirname(self.load_model_file), "tokenizer"))
            # self.model.init_tokenizer()
            self.epoch = checkpoint['epoch']

    def train(self, train_dl, test_dl, scheduler):
        best_loss = float('Inf')

        # For early stopping, number of iterations without loss improvement
        not_improving_x = 0
        progress_bar = tqdm(range(self.epoch, self.epoch +
                                  self.epochs), desc='Epoch   ')

        self.scheduler = scheduler
        test_metrics = self.validate(test_dl)

        for idx in progress_bar:
            train_metrics = self.train_epoch(train_dl)
            test_metrics = self.validate(test_dl)

            self.trp_example_plots()
            self.text_generation_examples()

            for key in train_metrics:
                if key not in self.train_history:
                    self.train_history[key] = []
                self.train_history[key].append(train_metrics[key])
            for key in test_metrics:
                if key not in self.test_history:
                    self.test_history[key] = []
                self.test_history[key].append(test_metrics[key])

            self.save_history(self.save_path)

            avg_valid_loss = (
                train_metrics['avg_loss'] + test_metrics['avg_loss']) / 2

            if avg_valid_loss < best_loss:
                not_improving_x = 0

                self.trp_example_plots()
                self.text_generation_examples()

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

                if not_improving_x >= self.early_stop and self.early_stop > 0:
                    self.logger.info("train: early stop")
                    progress_bar.close()
                    return self.train_history

        progress_bar.close()
        return self.train_history

    def get_postfix_str(self, step, f1, loss, count, tp, fp, fn, tn):
        return (f'loss={loss / (step + 1): .4f}, f1={f1 / (step + 1): .4f}, accuracy={(tp + tn) / count: .4f} '
                f'precision={(tp / (tp + fp)) if tp + fp != 0 else 1: .4f}, recall={tp / (tp + fn) if tp + fn != 0 else 0: .4f}, '
                f'bAcc={0.5 * (tp / (tp + fn) + tn / (fp + tn)) if tp + fn != 0 and fp + tn != 0 else 0: .4f}')

    def metric_output(self, metrics):
        output = ""
        if 'avg_loss' in metrics.keys():
            output += f"loss={metrics['avg_loss'] :.4f}"
        if 'eot_f1' in metrics.keys():
            output += f"f1={metrics['eot_f1'] :.4f}"
        if 'eot_acc' in metrics.keys():
            output += f"acc={metrics['eot_acc'] :.4f}"
        if 'recall' in metrics.keys():
            output += f"loss={metrics['eot_recall'] :.4f}"
        if 'rouge_mean' in metrics.keys():
            output += f"rouge2={metrics['rouge_mean'] :.4f}"

        return output

    def compute_metrics(self, p):
        # SEP Token within output window
        id = self.model.tokenizer.convert_tokens_to_ids('[SEP]')
        true_predictions = [(prediction_batch[:, :self.output_window] == id).any(dim=1) for prediction_batch, _
                            in p]
        true_labels = [(label_batch[:, :self.output_window] ==
                        id).any(dim=1) for _, label_batch in p]

        # Rouge
        pred_strs = [self.model.tokenizer.batch_decode(pred) for pred, _ in p]
        label_strs = [self.model.tokenizer.batch_decode(
            torch.where(label == -100, 0, label)) for _, label in p]

        rouge_out = [
            self.metrics['rouge'].compute(
                predictions=pred_str, references=label_str, rouge_types=["rouge2"])['rouge2']
            for
            pred_str, label_str in zip(pred_strs, label_strs)]
        results = {
            'eot_acc': torch.mean(torch.tensor([self.metrics['acc'].to(self.device)(p, l) for p, l in
                                                zip(true_predictions, true_labels)])).item(),
            'eot_f1': torch.mean(torch.tensor([self.metrics['f1'].to(self.device)(p, l) for p, l in
                                               zip(true_predictions, true_labels)])).item(),
            'eot_recall': torch.mean(torch.tensor([self.metrics['recall'].to(self.device)(p, l) for p, l in
                                                   zip(true_predictions, true_labels)])).item(),
            'rouge_mean': np.mean(rouge_out),
        }
        return results

    def train_epoch(self, train_dl):
        self.model.train()
        total_loss, total_count = 0, 0
        total_f1 = 0
        tp, fp, fn, tn = 0, 0, 0, 0

        progress_bar = tqdm(train_dl, desc='Training', unit="batch")
        padding = torch.zeros((8, 183)).to(self.device)
        padding[:, :5] = 1

        pred_label = []

        for step, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)

            labels = self.generate_labels(input_ids, mask=attention_mask)
            projection_labels = self.generate_projection_labels(labels)

            out = self.model.forward(
                input_ids, labels=labels, projection_labels=projection_labels, attention_mask=attention_mask, token_type_ids=token_type_ids)

            loss = out.loss
            if out.mc_loss is not None:
                loss = out.loss + out.mc_loss

            loss.backward()
            self.optimizer.step()

            if step % self.log_interval == 0:
                wandb.log({"loss": loss,
                           "global_step": self.global_step})

            total_loss += loss.item()
            avg_loss = round(total_loss / (step + 1), 4)
            progress_bar.set_postfix_str(f"loss={avg_loss}")

            self.global_step += 1

        if self.scheduler:
            # self.scheduler.step()
            pass

        avg_loss = total_loss / len(train_dl)
        metrics = self.compute_metrics(pred_label)
        metrics['avg_loss'] = avg_loss

        wandb.log({"train_loss": avg_loss,
                   "global_step": self.global_step})

        progress_bar.disable = False
        progress_bar.set_postfix_str(self.metric_output(metrics))
        progress_bar.close()

        return metrics

    def evaluate(self, test_dl):
        self.model.eval()

        with torch.no_grad():
            for step, batch in enumerate(test_dl):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)

                labels = self.generate_labels(input_ids, mask=attention_mask)
                projection_labels = self.generate_projection_labels(labels)
                out = self.model.forward(
                    input_ids, labels=labels, projection_labels=projection_labels, attention_mask=attention_mask, token_type_ids=token_type_ids)

                if self.is_not_trp_example(out.logits, labels):
                    print(labels)

    def validate(self, test_dl):
        total_loss, total_count = 0, 0
        tp, fp, fn, tn = 0, 0, 0, 0

        pred_label = []

        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(test_dl, desc='Validation')

            for step, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)

                labels = self.generate_labels(input_ids, mask=attention_mask)
                projection_labels = self.generate_projection_labels(labels)
                out = self.model.forward(
                    input_ids, labels=labels, projection_labels=projection_labels, attention_mask=attention_mask, token_type_ids=token_type_ids)

                loss = out.loss
                total_loss += loss.item()

                self.add_to_bacc(out.logits, labels)

                avg_loss = round(total_loss / (step + 1), 4)
                progress_bar.set_postfix_str(f"loss={avg_loss}")

            metrics = self.compute_metrics(pred_label)
            metrics['avg_loss'] = total_loss / len(test_dl)

            metrics['bacc'] = self.compute_bacc()

            if not self.dev_mode:
                self.generate_on_validation_set(
                    input_ids, mask=attention_mask, speaker_ids=token_type_ids)
                wandb.log({"val_loss": round(total_loss / len(test_dl), 4),
                           "global_step": self.global_step,
                           "bacc": metrics['bacc']})

            progress_bar.disable = False
            progress_bar.set_postfix_str(self.metric_output(metrics))

        progress_bar.close()
        self.model.train()

        return metrics

    def generate_labels(self, input_ids, mask=None, pad_id=-100):
        labels = input_ids.clone()
        labels[torch.logical_not(mask)] = pad_id

        return labels

    def generate_projection_labels(self, labels):
        batch_size, num_labels = labels.size()

        mask = (labels == self.model.tokenizer.eos_token_id)
        distances = torch.full((batch_size, num_labels),
                               num_labels, device=labels.device)
        distances[mask] = 0

        for i in range(num_labels - 2, -1, -1):
            distances[:, i] = torch.minimum(
                distances[:, i], distances[:, i+1] + 1)

        return distances

    def calculate_loss(self, output, labels, padding=None):
        mask = torch.ones(labels.shape).to(self.device)
        if padding is not None:
            mask = padding

        output = output[:, :100].contiguous()

        self.criterion.reduction = "none"
        loss = self.criterion(
            output.view(-1, output.size(-1)), labels.view(-1))
        loss = loss.view(labels.shape)

        loss *= mask
        return loss.sum() / mask.sum()

    def add_to_bacc(self, logits, labels):
        probs = logits.softmax(dim=-1)
        trp_prob = probs[..., self.model.tokenizer.eos_token_id]
        trp_prob = trp_prob[..., :-1]

        labels = labels[..., 1:]
        is_trp = labels == self.model.tokenizer.eos_token_id
        not_trp = labels != self.model.tokenizer.eos_token_id
        for bacc in self.bacc:
            thresh = bacc['threshold']
            bacc['tp'] += (is_trp & (trp_prob > thresh)).sum()
            bacc['tn'] += (not_trp & (trp_prob < thresh)).sum()
            bacc['fp'] += (not_trp & (trp_prob > thresh)).sum()
            bacc['fn'] += (is_trp & (trp_prob < thresh)).sum()

            bacc['bacc'] = ((bacc['tp'] / (bacc['tp'] + bacc['fn'])) +
                            (bacc['tn'] / (bacc['fp'] + bacc['tn']))) / 2

    def compute_bacc(self):
        max_bacc = max(self.bacc, key=lambda x: x['bacc'])

        self.bacc = [{
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "bacc": 0,
            "threshold": threshold,
        } for threshold in self.thresholds]

        return max_bacc['bacc'].cpu()

    """
    Returns true/false if logits predict ground truth trp sufficiently wrong
    """

    def is_not_trp_example(self, logits, labels):
        probs = logits.softmax(dim=-1)
        trp_prob = probs[..., self.model.tokenizer.eos_token_id]
        trp_prob = trp_prob[..., :-1]

        labels = labels[..., 1:]
        is_trp = labels == self.model.tokenizer.eos_token_id
        not_trp = labels != self.model.tokenizer.eos_token_id

        return torch.max(trp_prob - is_trp).item() > 0.5

    def save_training(self, path):
        # self.model.tokenizer.save_pretrained(os.path.join(os.path.dirname(path), "tokenizer"))
        torch.save(self.best, path)

    def save_history(self, path):
        self.logger.info("trainer: save history")
        for key in self.train_history:
            np.save(os.path.join(
                path, f"train_{key}"), self.train_history[key])
        for key in self.test_history:
            np.save(os.path.join(path, f"test_{key}"), self.test_history[key])

    def print_dialogue(self, input_ids, prediction, output, label):
        output = f"Input: {self.model.tokenizer.decode(input_ids)}\n"
        output += f"Output: {self.model.tokenizer.decode(output['input_ids'])}\n"
        output += f"Prediction: {prediction}, Label: {label}"

    @torch.no_grad
    def generate_from_string(self, t):
        out = self.model(t["input_ids"], token_type_ids=t["token_type_ids"])
        out["probs"] = out["logits"].softmax(dim=-1)
        out["trp_probs"] = (out["probs"])[...,
                                          self.model.tokenizer.eos_token_id]
        out["tokens"] = self.model.tokenizer.convert_ids_to_tokens(
            t["input_ids"][0])
        return out

    @torch.no_grad
    def generate_on_validation_set(self, validation_text, mask=None, speaker_ids=None, name="Generate/Validation"):
        text_idx = validation_text.shape[1] // 2

        text = validation_text[:, :text_idx]
        speaker_id = speaker_ids[:, :text_idx]
        m = mask[:, :text_idx]

        out = self.model.generate(
            input_ids=text, speaker_ids=speaker_id, mask=m, output_scores=True, n_sequences=10)
        G = {"tokens": self.model.tokenizer.batch_decode(
            out['sequences'][:, len(text):])}

        input = self.model.tokenizer.batch_decode(text)
        ground_truth = self.model.tokenizer.batch_decode(
            validation_text[:, text_idx:])

        table = wandb.Table(
            columns=["context", "truth", "sample"],
            data=[
                [sentence, truth, sample]
                for sentence, truth, sample in zip(input, ground_truth, G["tokens"])
            ]
        )
        wandb.log({
            f"{name}": table,
            "global_step": self.global_step,
        })

    def trp_example_plots(self, name="TRP/example"):
        turn_list = [
            ["yesterday we met in the park",
                "okay when will you meet again", "tomorrow"],
            [
                "Hello there I basically had the worst day of my life",
                "Oh no, what happened?",
                "Do you want the long or the short story?",
            ],
        ]

        figs = []
        global_steps = []
        for b in range(len(turn_list)):
            out = self.model.from_string(turn_list[b])
            out = self.generate_from_string(out)
            fig, _ = plot_trp(
                trp=out["trp_probs"][0].cpu(),
                text=out["tokens"],
                eos_token='[SEP]'
            )
            figs.append(fig)
            global_steps.append(self.global_step)

        wandb.log({"graphs": [wandb.Image(im)
                  for im in figs], "global_step": self.global_step})

    def text_generation_examples(self, name="Generate/example"):
        turn_list = [
            ["yesterday we met in the park",
                "okay when will you meet again", "tomorrow"],
            [
                "Hello there I basically had the worst day of my life",
                "Oh no, what happened?",
                "Do you want the long or the short story?",
            ],
        ]

        inp = self.model.from_string(turn_list[-1])
        out = self.model.generate(
            input_ids=inp['input_ids'], speaker_ids=inp["token_type_ids"], output_scores=True, n_sequences=10)

        G = {"tokens": self.model.tokenizer.batch_decode(
            out['sequences'][:, len(inp['input_ids'][0]):])}
        """
        for i, g in enumerate(out["sequences"][1:]):
            if g not in G["tokens"]:
                G["tokens"].append(g)
                # G["probs"].append(out["probs"][i].cpu())
        """

        table = wandb.Table(
            columns=["context", "sample"],
            data=[
                [turn_list[-1][-1], toks]
                for toks in G["tokens"]
            ]
        )
        wandb.log({
            f"{name}": table,
            "global_step": self.global_step,
        })

