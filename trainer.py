import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model=None, criterion=None, optimizer=None, config=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.epochs = config.epoch_size
        self.device = config.device

    def train(self, train_dl):
        progress_bar = tqdm(range(0, self.epochs))

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
