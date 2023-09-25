import torch
import os
import json
import numpy as np

import logging
from tqdm import tqdm


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

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_correct": [],
        }

        self.best = {
            'epoch': 0,
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'loss': None,
            'config': config,
        }

        self.save_path = os.path.join(config.save_path)
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        with open(os.path.join(self.save_path, "config.json"), "w") as config_file:
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

        for idx in progress_bar:
            avg_train_loss = self.train_epoch(train_dl)
            avg_valid_loss, avg_valid_correct = self.validate(test_dl)

            progress_bar.set_postfix_str(
                f'train_loss={avg_train_loss: .4f}, avg_valid_loss={avg_valid_loss: .4f}, avg_valid_correct={avg_valid_correct: .4f}')
            # self.logger.info(
            #    f'train_loss= {avg_train_loss: .4f}, avg_valid_loss= {avg_valid_loss: .4f}, avg_valid_correct={avg_valid_correct: .4f}')

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_valid_loss)
            self.history["val_correct"].append(avg_valid_correct)
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

    def train_epoch(self, train_dl):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(train_dl, desc='Training')

        for step, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].type(
                torch.float).unsqueeze(1).to(self.device)

            output = self.model.forward(
                input_ids, attention_mask=attention_mask)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        progress_bar.close()
        avg_loss = total_loss / len(train_dl)

        return avg_loss

    def validate(self, test_dl):
        total_loss, total_count = 0, 0
        total_correct = 0

        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(test_dl, desc='Validation')

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].type(
                    torch.float).unsqueeze(1).to(self.device)

                output = self.model.forward(
                    input_ids, attention_mask=attention_mask)
                loss = self.criterion(output, labels)

                total_loss += float(loss)
                total_count += 1

                predicted_trp = (output > 0.5).float()
                correct_predictions = (
                    predicted_trp == labels).float().sum().item()
                total_correct += correct_predictions

                progress_bar.set_postfix_str(
                    f'avg_valid_loss= {total_loss/total_count: .4f}, avg_valid_correct={total_correct/total_count: .4f}')

            progress_bar.close()
        self.model.train()
        return total_loss / total_count, total_correct / total_count

    def save_training(self, path):
        torch.save(self.best, path)

    def save_history(self, path):
        np.save(os.path.join(path, "train_loss"), self.history["train_loss"])
        np.save(os.path.join(path, "val_loss"), self.history["val_loss"])
        np.save(os.path.join(path, "val_correct"), self.history["val_correct"])
