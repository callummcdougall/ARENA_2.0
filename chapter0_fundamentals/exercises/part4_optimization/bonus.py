# %%
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import pandas as pd
import torch as t
from torch import optim
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear, Module
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from typing import Callable, Iterable, Tuple, Optional
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from IPython.display import display, HTML
import wandb
from torchvision import transforms


# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_optimization"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from plotly_utils import bar, imshow
from part3_resnets.solutions import (
    IMAGENET_TRANSFORM,
    get_resnet_for_feature_extraction,
    plot_train_loss_and_test_accuracy_from_metrics,
)
from part4_optimization.utils import plot_fn, plot_fn_with_points
import part4_optimization.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %%

if MAIN:
    MNIST_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )


def get_mnist(subset: int = 1):
    """Returns MNIST training data, sampled by the frequency given in `subset`."""
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=MNIST_TRANSFORM
    )
    mnist_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=MNIST_TRANSFORM
    )

    if subset > 1:
        mnist_trainset = Subset(
            mnist_trainset, indices=range(0, len(mnist_trainset), subset)
        )
        mnist_testset = Subset(
            mnist_testset, indices=range(0, len(mnist_testset), subset)
        )

    return mnist_trainset, mnist_testset

# %%

@dataclass
class ConvNetFinetuningArgs:
    batch_size: int = 64
    max_epochs: int = 1
    max_steps: int = 500
    optimizer: t.optim.Optimizer = t.optim.AdamW
    learning_rate: float = 1e-3
    log_dir: str = os.getcwd() + "/logs"
    log_name: str = "day4-mnist"
    log_every_n_steps: int = 1
    n_classes: int = 10
    subset: int = 10
    trainset: Optional[datasets.MNIST] = None
    testset: Optional[datasets.MNIST] = None

    def __post_init__(self):
        if self.trainset is None or self.testset is None:
            self.trainset, self.testset = get_mnist(self.subset)
        self.trainloader = DataLoader(
            self.trainset, shuffle=True, batch_size=self.batch_size
        )
        self.testloader = DataLoader(
            self.testset, shuffle=False, batch_size=self.batch_size
        )
        self.logger = WandbLogger(
            save_dir=self.log_dir, project=self.log_name, name=self.log_name
        )

class ConvNet(Module):
    def __init__(self, channels=32):
        super().__init__()
        self.model = Sequential(
            Conv2d(in_channels=1, out_channels=channels, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0),
            Flatten(),
            Linear(in_features=channels*14*14, out_features=10)
        )
    def forward(self, x):
        return self.model(x)

class LitConvNet(pl.LightningModule):
    def __init__(self, args: ConvNetFinetuningArgs, channels=32):
        super().__init__()
        self.model = ConvNet(channels)
        self.args = args

    def _shared_train_val_step(
        self, batch: Tuple[t.Tensor, t.Tensor]
    ) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        """
        Convenience function since train/validation steps are similar.
        """
        imgs, labels = batch
        logits = self.model(imgs)
        return logits, labels

    def training_step(
        self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int
    ) -> t.Tensor:
        """
        Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        logits, labels = self._shared_train_val_step(batch)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> None:
        """
        Operates on a single batch of data from the validation set. In this step you might
        generate examples or calculate anything of interest like accuracy.
        """
        logits, labels = self._shared_train_val_step(batch)
        loss = F.cross_entropy(logits, labels).detach().item()
        classifications = logits.argmax(dim=1)
        accuracy = t.sum(classifications == labels) / len(classifications)
        self.log("accuracy", accuracy)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = self.args.optimizer(
            self.model.parameters(), lr=self.args.learning_rate
        )
        return optimizer

# %%

if MAIN:
    args = ConvNetFinetuningArgs()
    model = LitConvNet(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=args.logger,
        log_every_n_steps=args.log_every_n_steps,
    )
    trainer.fit(
        model=model, train_dataloaders=args.trainloader, val_dataloaders=args.testloader
    )
    wandb.finish()

# %%
from math import floor, sqrt

if MAIN:
    sweep_config = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "accuracy"},
        "parameters": {
            "channels": {"distribution": "q_log_uniform_values", "min": 1, "max": 3, "q": 1},
            "subset": {"distribution": "q_log_uniform_values", "min": 1, "max": 6, "q": 2},
            "learning_rate": {
                "distribution": "log_uniform_values",
                "max": 1e-1,
                "min": 1e-4,
            },
        },
    }
# %%

# (2) Define a training function which takes no args, and uses `wandb.config` to get hyperparams


def train():
    wandb.init()
    args = ConvNetFinetuningArgs(
        learning_rate=wandb.config["learning_rate"],
        subset=wandb.config['subset']
    )
    model = LitConvNet(args, wandb.config['channels'])

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=args.logger,
        log_every_n_steps=args.log_every_n_steps,
    )
    trainer.fit(
        model=model, train_dataloaders=args.trainloader, val_dataloaders=args.testloader
    )
    wandb.finish()


if MAIN:
    sweep_id = wandb.sweep(sweep=sweep_config, project="day4-resnet-sweep")
    wandb.agent(sweep_id=sweep_id, function=train, count=5)
   # %%
