# %%
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch as t
import torch.nn.functional as F

from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_optimization"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from part3_resnets.solutions import (
    IMAGENET_TRANSFORM, get_resnet_for_feature_extraction)

IMAGENET_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


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


# %%
if MAIN:
    cifar_trainset, cifar_testset = get_cifar(subset=1)
    cifar_trainset_small, cifar_testset_small = get_cifar(subset=10)


@dataclass
class ResNetFinetuningArgs:
    batch_size: int = 64
    max_epochs: int = 3
    max_steps: int = 500
    optimizer: t.optim.Optimizer = t.optim.Adam
    learning_rate: float = 1e-3
    log_dir: str = os.getcwd() + "/logs"
    log_name: str = "day5-resnet"
    log_every_n_steps: int = 1
    n_classes: int = 10
    subset: int = 10
    trainset: Optional[datasets.CIFAR10] = None
    testset: Optional[datasets.CIFAR10] = None

    def __post_init__(self):
        if self.trainset is None or self.testset is None:
            self.trainset, self.testset = get_cifar(self.subset)
        self.trainloader = DataLoader(
            self.trainset, shuffle=True, batch_size=self.batch_size
        )
        self.testloader = DataLoader(
            self.testset, shuffle=False, batch_size=self.batch_size
        )
        self.logger = CSVLogger(save_dir=self.log_dir, name=self.log_name)


# %%
class LitResNet(pl.LightningModule):
    def __init__(self, args: ResNetFinetuningArgs):
        super().__init__()
        self.resnet = get_resnet_for_feature_extraction(args.n_classes)
        self.args = args

    def _shared_train_val_step(
        self, batch: Tuple[t.Tensor, t.Tensor]
    ) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        """
        Convenience function since train/validation steps are similar.
        """
        imgs, labels = batch
        logits = self.resnet(imgs)
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
        classifications = logits.argmax(dim=1)
        accuracy = t.sum(classifications == labels) / len(classifications)
        self.log("accuracy", accuracy)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = self.args.optimizer(
            self.resnet.out_layers.parameters(), lr=self.args.learning_rate
        )
        return optimizer

import wandb

@dataclass
class ResNetFinetuningArgsWandb(ResNetFinetuningArgs):
    use_wandb: bool = True
    run_name: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.use_wandb:
            self.logger = WandbLogger(
                save_dir=self.log_dir, project=self.log_name, name=self.run_name
            )



if MAIN:
    # YOUR CODE HERE - fill `sweep_config`
    sweep_config = dict(
        method="random",
        metric=dict(name="accuracy", goal="maximize"),
        parameters=dict(
            batch_size=dict(values=[32, 64, 128, 256]),
            max_epochs=dict(min=1, max=4),
            learning_rate=dict(max=0.1, min=0.0001, distribution="log_uniform_values"),
        ),
    )
    # FLAT SOLUTION END

    tests.test_sweep_config(sweep_config)
# %%
def train() -> None:
    args = ResNetFinetuningArgsWandb(
        trainset=cifar_trainset_small,
        testset=cifar_testset_small,
    )
    args.batch_size=wandb.config['batch_size']
    args.max_epochs=wandb.config['max_epochs']
    args.learning_rate=wandb.config['learning_rate']
    model = LitResNet(args)

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
if MAIN:
    sweep_id = wandb.sweep(sweep=sweep_config, project='day4-resnet-sweep')
    wandb.agent(sweep_id=sweep_id, function=train, count=3)
# %%