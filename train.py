import os
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from IPython.display import clear_output
from torch.utils.data import DataLoader

from models import QuartzNet
from dataset import NumbersDataset
from utils import decode, save_spec, cer


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class Trainer:

    def __init__(self, config, from_checkpoint):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Parameters
        self.batch_size = config["batch_size"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.epochs = config["epochs"] + 1

        # Data
        train_data = os.path.join(config["data_dir"], config["train_data"])
        val_data = os.path.join(config["data_dir"], config["val_data"])
        self.train_set = NumbersDataset(config, train_data)
        self.val_set = NumbersDataset(config, val_data)
        self.train_loader = self.loader(self.train_set)
        self.val_loader = self.loader(self.val_set)

        # Model stuff
        self.model = QuartzNet().to(self.device)
        self.criterion = criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))

        if from_checkpoint:
            self.load_checkpoint(self.checkpoint_dir, map_location=self.device)
            print("=> Loaded checkpoint")

        # Logging
        self.train_writer = SummaryWriter(os.path.join(config["log_dir"], "train"))
        self.val_writer = SummaryWriter(os.path.join(config["log_dir"], "val"))

    def train(self):

        best_loss = None

        # Training
        for epoch in range(1, self.epochs):

            self.train_step(epoch)
            loss = self.val_step(epoch)

            self.save_checkpoint(self.checkpoint_dir, postfix="last")

            if best_loss is None:
                best_loss = loss
            elif loss < best_loss:
                self.save_checkpoint(self.checkpoint_dir, postfix="best")
                print("=> Checkpoint updated")
                best_loss = loss

    def train_step(self, step):

        self.model.train()
        loop = tqdm(self.train_loader)
        losses = 0
        num_batches = 0

        for batch_idx, (specs, labels) in enumerate(loop):

            clear_output(wait=True)
            loop.set_description(f"Epoch {step} (train)")
            self.optimizer.zero_grad()

            specs = specs.to(self.device)
            labels = labels.to(self.device)

            output = self.model(specs)
            loss = self.criterion(output, labels)
            losses += loss
            loss.backward()
            self.optimizer.step()

            loop.set_postfix(loss=loss.item())
            num_batches += 1

            self.train_writer.add_scalar(f"Epoch {step}: loss", loss, global_step=batch_idx)

            for param_group in self.optimizer.param_groups:
                rate = param_group["lr"]

            self.train_writer.add_scalar("Learning Rate", rate, global_step=batch_idx + len(self.train_loader) * (step - 1))

            if batch_idx % 100 == 0:
                rand_idx = random.randint(0, specs.shape[0] - 1)
                self.train_writer.add_image(f"Epoch {step} (train): specs", save_spec(specs[rand_idx].to("cpu").detach()), global_step=batch_idx)

        loss = losses / num_batches
        self.train_writer.add_scalar("loss", loss, global_step=step)

    def val_step(self, step):

        self.model.eval()
        loop = tqdm(self.val_loader)
        losses = 0
        cers = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (specs, labels) in enumerate(loop):

                clear_output(wait=True)
                loop.set_description(f"Epoch {step} (val)")

                specs = specs.to(self.device)
                labels = labels.to(self.device)

                output = self.model(specs)
                loss = self.criterion(output, labels)
                losses += loss

                loop.set_postfix(loss=loss.item())
                num_batches += 1

                decoded_preds, decoded_targets = decode(output, labels)
                error = cer(decoded_targets, decoded_preds)
                cers += error

                # Save training logs to Tensorboard
                rand_idx = random.randint(0, specs.shape[0] - 1)

                self.val_writer.add_text(f"Epoch {step} (val): preds", decoded_preds[rand_idx], global_step=batch_idx)
                self.val_writer.add_text(f"Epoch {step} (val): targets", decoded_targets[rand_idx], global_step=batch_idx)
                self.val_writer.add_scalar(f"Epoch {step}: loss", loss, global_step=batch_idx)

        loss = losses / num_batches
        error = cers / num_batches

        self.val_writer.add_scalar("loss", loss, global_step=step)
        self.val_writer.add_scalar("CER", error, global_step=step)

        return loss

    def loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size)

    def save_checkpoint(self, path, postfix=""):

        if not os.path.exists(path):
            os.mkdir(path)

        torch.save(self.model.state_dict(), os.path.join(path, f"model_{postfix}.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(path, f"optimizer_{postfix}.pt"))

    def load_checkpoint(self, path, map_location):

        self.model.load_state_dict(torch.load(os.path.join(path, "model_last.pt"), map_location=map_location))
        self.optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer_last.pt"), map_location=map_location))


def main():

    parser = ArgumentParser()
    parser.add_argument('--conf', default="config.yml", help='Path to the configuration file')
    parser.add_argument('--from_checkpoint', action="store_true", help='Continue training from the last checkpoint')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.conf))
    from_checkpoint = args.from_checkpoint

    trainer = Trainer(config, from_checkpoint)
    print("=> Initialised trainer")
    print("=> Training...")
    trainer.train()


if __name__ == "__main__":
    main()
