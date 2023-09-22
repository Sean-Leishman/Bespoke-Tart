import torch
import os
import json
import numpy as np

from tqdm import tqdm


class Trainer:
    def __init__(self, model=None, criterion=None, optimizer=None, config=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.epochs = config.epoch_size
        self.device = torch.device(config.device)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_correct": [],
        }

        self.best = {
            'epoch': 0,
            'config': config,
        }

        self.save_path = os.path.join(config.save_path)
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        with open(os.path.join(self.save_path, "config.json"), "w") as config_file:
            json.dump(vars(config), config_file)

    def train(self, train_dl):
        progress_bar = tqdm(range(0, self.epochs), desc='Epoch   ')

        for idx in progress_bar:
            avg_train_loss = self.train_epoch(train_dl)
            print("step")

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

    def save_training(self, path):
        torch.save(self.best, path)

    def save_history(self, path):
        np.save(os.path.join(path, "train_loss"), self.history["train_loss"])
        np.save(os.path.join(path, "val_loss"), self.history["val_loss"])
        np.save(os.path.join(path, "val_correct"), self.history["val_correct"])
