# %%
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import pandas as pd
import torch as t
from torch import optim
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


def get_cifar(subset: int = 1):
    cifar_trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=IMAGENET_TRANSFORM
    )
    cifar_testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=IMAGENET_TRANSFORM
    )
    if subset > 1:
        cifar_trainset = Subset(
            cifar_trainset, indices=range(0, len(cifar_trainset), subset)
        )
        cifar_testset = Subset(
            cifar_testset, indices=range(0, len(cifar_testset), subset)
        )
    return cifar_trainset, cifar_testset


# if MAIN:
#     cifar_trainset, cifar_testset = get_cifar()

#     imshow(
#         cifar_trainset.data[:15],
#         facet_col=0,
#         facet_col_wrap=5,
#         facet_labels=[cifar_trainset.classes[i] for i in cifar_trainset.targets[:15]],
#         title="CIFAR-10 images",
#         height=600,
#     )


# %%
@dataclass
class ResNetTrainingArgs:
    batch_size: int = 64 * 10
    max_epochs: int = 3
    max_steps: int = 500
    optimizer: t.optim.Optimizer = t.optim.Adam
    learning_rate: float = 1e-3
    log_dir: str = os.getcwd() + "/logs"
    log_name: str = "day4-resnet"
    log_every_n_steps: int = 1
    n_classes: int = 10
    subset: int = 10


# %%
class LitResNet(pl.LightningModule):
    def __init__(self, args: ResNetTrainingArgs):
        super().__init__()
        self.args = args
        self.resnet = get_resnet_for_feature_extraction(self.args.n_classes)
        self.trainset, self.testset = get_cifar(subset=self.args.subset)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.resnet(x)

    def _shared_train_val_step(
        self, batch: Tuple[t.Tensor, t.Tensor]
    ) -> Tuple[t.Tensor, t.Tensor]:
        imgs, labels = batch
        logits = self(imgs)
        return logits, labels

    def training_step(
        self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int
    ) -> t.Tensor:
        logits, labels = self._shared_train_val_step(batch)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> None:
        logits, labels = self._shared_train_val_step(batch)
        classifications = logits.argmax(dim=1)
        accuracy = t.sum(classifications == labels) / len(classifications)
        self.log("accuracy", accuracy)

    def configure_optimizers(self):
        return self.args.optimizer(
            self.resnet.out_layers.parameters(), lr=self.args.learning_rate
        )

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)


# %%
# if MAIN:
#     args = ResNetTrainingArgs()
#     model = LitResNet(args)
#     logger = CSVLogger(save_dir=args.log_dir, name=args.log_name)

#     trainer = pl.Trainer(
#         max_epochs=args.max_epochs,
#         logger=logger,
#         log_every_n_steps=args.log_every_n_steps,
#     )
#     trainer.fit(model=model)

#     metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

#     plot_train_loss_and_test_accuracy_from_metrics(
#         metrics, "Feature extraction with ResNet34"
#     )


# %%
def test_resnet_on_random_input(n_inputs: int = 3):
    indices = np.random.choice(len(cifar_trainset), n_inputs).tolist()
    classes = [cifar_trainset.classes[cifar_trainset.targets[i]] for i in indices]
    imgs = cifar_trainset.data[indices]
    with t.inference_mode():
        x = t.stack(list(map(IMAGENET_TRANSFORM, imgs)))
        logits: t.Tensor = model.resnet(x)
    probs = logits.softmax(-1)
    if probs.ndim == 1:
        probs = probs.unsqueeze(0)
    for img, label, prob in zip(imgs, classes, probs):
        display(HTML(f"<h2>Classification probabilities (true class = {label})</h2>"))
        imshow(
            img,
            width=200,
            height=200,
            margin=0,
            xaxis_visible=False,
            yaxis_visible=False,
        )
        bar(
            prob,
            x=cifar_trainset.classes,
            template="ggplot2",
            width=600,
            height=400,
            labels={"x": "Classification", "y": "Probability"},
            text_auto=".2f",
            showlegend=False,
        )


# if MAIN:
#     test_resnet_on_random_input()
# %%
import wandb


@dataclass
class ResNetTrainingArgsWandb(ResNetTrainingArgs):
    run_name: Optional[str] = None


# if MAIN:
#     args = ResNetTrainingArgsWandb()
#     model = LitResNet(args)
#     logger = WandbLogger(
#         save_dir=args.log_dir, project=args.log_name, name=args.run_name
#     )

#     trainer = pl.Trainer(
#         max_epochs=args.max_epochs,
#         max_steps=args.max_steps,
#         logger=logger,
#         log_every_n_steps=args.log_every_n_steps,
#     )
#     trainer.fit(model=model)
#     wandb.finish()

# %%

if MAIN:
    sweep_config = dict()

    # sweep_configuration = {
    #     'method': 'random',
    #     'name': 'sweep',
    #     'metric': {
    #         'goal': 'minimize',
    #         'name': 'validation_loss'
    #         },
    #     'parameters': {
    #         'batch_size': {'values': [16, 32, 64]},
    #         'epochs': {'values': [5, 10, 15]},
    #         'lr': {'max': 0.1, 'min': 0.0001}
    #      }
    # }
    sweep_config = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "accuracy"},
        "parameters": {
            "batch_size": {"values": [64, 256, 512, 1024]},
            "max_epochs": {"values": [5, 10, 24]},
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 1e-1,
            },
        },
    }
    # tests.test_sweep_config(sweep_config)
# %%
# (2) Define a training function which takes no args, and uses `wandb.config` to get hyperparams


def train():
    # Define hyperparameters, override some with values from wandb.config
    args = ResNetTrainingArgsWandb()
    logger = WandbLogger(
        save_dir=args.log_dir, project=args.log_name, name=args.run_name
    )

    args.batch_size = wandb.config["batch_size"]
    args.max_epochs = wandb.config["max_epochs"]
    args.learning_rate = wandb.config["learning_rate"]

    model = LitResNet(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
    )
    trainer.fit(model=model)


if MAIN:
    sweep_id = wandb.sweep(sweep=sweep_config, project="day4-resnet-sweep")
    wandb.agent(sweep_id=sweep_id, function=train, count=20)
    wandb.finish()
