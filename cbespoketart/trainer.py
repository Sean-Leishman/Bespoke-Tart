import torch
import os
import json
import numpy as np
from datetime import datetime

import logging
from tqdm import tqdm
from sklearn.metrics import f1_score

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
        self.epochs = config.epoch_size
        self.device = torch.device(
            config.device if config is not None else "cpu")

        self.epoch = 0

        self.standardise = None

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_correct": [],
            "val_f1":[]
        }

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


    def load_from_checkpoint(self):
        checkpoint = torch.load(self.load_model_file)
        self.logger.info(
            f"model: loading parameters for model {checkpoint.keys()}")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']

    def train(self, train_dl, test_dl):
        best_loss = float('Inf')

        # For early stopping, number of iterations without loss improvement
        not_improving_x = 0
        progress_bar = tqdm(range(self.epoch, self.epoch +
                            self.epochs), desc='Epoch   ')

        self.standardise = self.std(train_dl)

        for idx in progress_bar:
            avg_train_loss = self.train_epoch(train_dl)
            avg_valid_loss, avg_valid_correct, avg_valid_f1 = self.validate(test_dl)

            # self.logger.info(
            #    f'train_loss= {avg_train_loss: .4f}, avg_valid_loss= {avg_valid_loss: .4f}, avg_valid_correct={avg_valid_correct: .4f}')

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_valid_loss)
            self.history["val_correct"].append(avg_valid_correct)
            self.history["val_f1"].append(avg_valid_f1)
            self.save_history(self.save_path)

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
                    return self.history

        progress_bar.close()
        return self.history

    def get_postfix_str(self, step, f1, loss, count, tp, fp, fn, tn):
        return (f'loss={loss / (step + 1): .4f}, f1={f1 / (step + 1): .4f}, accuracy={(tp+tn) / count: .4f} '
        f'precision={(tp/(tp+fp)) if tp+fp != 0 else 1: .4f}, recall={tp/(tp+fn) if tp+fn != 0 else 0: .4f}, '
                f'bAcc={0.5 * (tp/(tp+fn) + tn/(fp+tn)) if tp+fn != 0 and fp+tn!=0 else 0: .4f}')

    def train_epoch(self, train_dl):
        self.model.train()
        total_loss, total_count = 0,0
        total_f1 = 0
        tp, fp, fn, tn = 0,0,0,0

        progress_bar = tqdm(train_dl, desc='Training')
        padding = torch.zeros((8,183)).to(self.config.device)
        padding[:,:5] = 1

        for step, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)

            output_ids = batch["output"]["input_ids"].to(self.device)
            output_attention = batch["output"]["attention_mask"].to(self.device)
            output_token_types = batch["output"]["token_type_ids"].to(self.device)

            if input_ids.shape[1] > output_ids.shape[1]:
                padding_size = input_ids.shape[1] - output_ids.shape[1]
                output_ids = torch.nn.functional.pad(output_ids, (0, padding_size), value=0)
                output_attention = torch.nn.functional.pad(output_attention, (0, padding_size), value=0)
                output_token_types = torch.nn.functional.pad(output_token_types, (0, padding_size), value=0)
            elif input_ids.shape[1] < output_ids.shape[1]:
                padding_size = output_ids.shape[1] - input_ids.shape[1]
                input_ids = torch.nn.functional.pad(input_ids, (padding_size, 0), value=0)
                attention_mask = torch.nn.functional.pad(attention_mask, (padding_size, 0), value=0)
                token_type_ids = torch.nn.functional.pad(token_type_ids, (padding_size, 0), value=0)

            out = self.model.forward(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            )

            dist_to_eot = self.generate_output(batch["output"], 300)
            loss = self.calculate_loss(out, dist_to_eot)
            loss.backward()
            self.optimizer.step()

            labels = self.generate_labels(batch['output']['input_ids'][:,:self.config.output_window])

            total_loss += loss.item()
            total_count += len(labels)


            predicted_trp = dist_to_eot < 5

            tp += ((predicted_trp == 1) & (labels == 1)).any(dim=1).sum().item()
            fp += ((predicted_trp == 1) & (labels == 0)).any(dim=1).sum().item()
            fn += ((predicted_trp == 0) & (labels == 1)).any(dim=1).sum().item()
            tn += ((predicted_trp == 0) & (labels == 0)).any(dim=1).sum().item()

            f1 = f1_score(torch.flatten(labels.cpu()),
                          torch.flatten(predicted_trp.cpu()))
            total_f1 += f1

            progress_bar.set_postfix_str(self.get_postfix_str(step, total_f1, total_loss, total_count, tp, fp, fn , tn))

        progress_bar.close()
        avg_loss = total_loss / len(train_dl)

        return avg_loss

    def generate_output(self, batch_output, number_of_tokens=10):
        eot_id = self.model.tokenizer.convert_tokens_to_ids('[SEP]')

        masked_idx = (batch_output['input_ids'][:,:number_of_tokens] == eot_id).float()
        masked_idx[masked_idx == 0] = masked_idx.size(1)
        index = masked_idx.argmin(dim=1).to(self.config.device)

        if self.standardise is None:
            return index

        return self.standardise(index)

    def generate_labels(self, batch_output):
        eot_id = 102

        labels = (batch_output == eot_id).any(dim=1).float().to(self.config.device)
        return labels.unsqueeze(1)

    def calculate_loss(self, output, labels, padding=None):
        loss = self.criterion(output.view(-1).float(), labels.view(-1).float())
        return loss

    def std(self, dl):
        mean = 0
        std = 0
        for batch in dl:
            first_eot = self.generate_output(batch, number_of_tokens=300).float()
            mean += torch.mean(first_eot)
            std += torch.std(first_eot)
        mean /= len(dl)
        std /= len(dl)
        return lambda x: (x - mean) / std



    def validate(self, test_dl):
        total_loss, total_count = 0, 0
        total_f1 = 0
        tp, fp, fn, tn = 0,0,0,0

        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(test_dl, desc='Validation')

            for step, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)

                output_ids = batch["output"]["input_ids"].to(self.device)
                output_attention = batch["output"]["attention_mask"].to(self.device)
                output_token_types = batch["output"]["token_type_ids"].to(self.device)

                if input_ids.shape[1] > output_ids.shape[1]:
                    padding_size = input_ids.shape[1] - output_ids.shape[1]
                    output_ids = torch.nn.functional.pad(output_ids, (0, padding_size), value=0)
                    output_attention = torch.nn.functional.pad(output_attention, (0, padding_size), value=0)
                    output_token_types = torch.nn.functional.pad(output_token_types, (0, padding_size), value=0)
                elif input_ids.shape[1] < output_ids.shape[1]:
                    padding_size = output_ids.shape[1] - input_ids.shape[1]
                    input_ids = torch.nn.functional.pad(input_ids, (padding_size, 0), value=0)
                    attention_mask = torch.nn.functional.pad(attention_mask, (padding_size, 0), value=0)
                    token_type_ids = torch.nn.functional.pad(token_type_ids, (padding_size, 0), value=0)
                # labels = self.generate_labels(batch['output'], self.config.output_window).to(self.device)

                out = self.model.forward(
                    input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                )

                dist_to_eot = self.generate_output(batch["output"], 300)
                loss = self.calculate_loss(out, dist_to_eot)

                labels = self.generate_labels(batch['output']['input_ids'][:,:self.config.output_window])

                total_loss += loss.item()
                total_count += len(labels)


                predicted_trp = dist_to_eot < 5

                tp += ((predicted_trp == 1) & (labels == 1)).any(dim=1).sum().item()
                fp += ((predicted_trp == 1) & (labels == 0)).any(dim=1).sum().item()
                fn += ((predicted_trp == 0) & (labels == 1)).any(dim=1).sum().item()
                tn += ((predicted_trp == 0) & (labels == 0)).any(dim=1).sum().item()

                f1 = f1_score(torch.flatten(labels.cpu()),
                              torch.flatten(predicted_trp.cpu()))
                total_f1 += f1

                progress_bar.set_postfix_str(self.get_postfix_str(step, total_f1, total_loss, total_count, tp, fp, fn , tn))

        progress_bar.close()
        self.model.train()

        return total_loss / step, (tp+tn) / total_count, total_f1/step

    def save_training(self, path):
        torch.save(self.best, path)

    def save_history(self, path):
        np.save(os.path.join(path, "train_loss"), self.history["train_loss"])
        np.save(os.path.join(path, "val_loss"), self.history["val_loss"])
        np.save(os.path.join(path, "val_f1"), self.history["val_f1"])
        np.save(os.path.join(path, "val_correct"), self.history["val_correct"])

    def print_dialogue(self, input_ids, prediction, output, label):
        output = f"Input: {self.model.tokenizer.decode(input_ids)}\n"
        output += f"Output: {self.model.tokenizer.decode(output['input_ids'])}\n"
        output += f"Prediction: {prediction}, Label: {label}"

        print(output)