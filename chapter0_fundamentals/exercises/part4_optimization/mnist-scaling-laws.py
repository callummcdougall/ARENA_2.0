#%%
import os
os.chdir("/root/ARENA_2.0/chapter0_fundamentals/exercises/")

from typing import Tuple

import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

from part2_cnns.solutions import get_mnist

import wandb


MAIN = __name__ == "__main__"

#%%
class ConvNet(nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=k, kernel_size=(3, 3), padding=1, stride=1
        )  # k x 28 x 28
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # k x 14 x 14
        self.conv2 = nn.Conv2d(
            in_channels=k, out_channels=2 * k, kernel_size=(3, 3), padding=1, stride=1
        )  # 2k x 14 x 14
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=2 * k * 7 * 7, out_features=4 * k)
        self.fc2 = nn.Linear(in_features=4 * k, out_features=10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.fc2(self.relu(self.fc1(self.flatten(x))))
        return x

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


#%%
PROJECT_NAME = "mnist-scaling"

class LitConvNet(pl.LightningModule):
    def __init__(self, k):
        super().__init__()
        self.convnet = ConvNet(k=k)

    def _shared_train_val_step(self, batch: Tuple[t.Tensor, t.Tensor]) -> Tuple[t.Tensor, t.Tensor]:
        '''
        Convenience function since train/validation steps are similar.
        '''
        imgs, labels = batch
        logits = self.convnet(imgs)
        return logits, labels

    def training_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> t.Tensor:
        '''
        Here you compute and return the training loss and some additional metrics for e.g. 
        the progress bar or logger.
        '''
        logits, labels = self._shared_train_val_step(batch)
        loss = t.nn.functional.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> None:
        '''
        Operates on a single batch of data from the validation set. In this step you might
        generate examples or calculate anything of interest like accuracy.
        '''
        logits, labels = self._shared_train_val_step(batch)
        loss = t.nn.functional.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("val_loss", loss)
        self.log("accuracy", accuracy)

    def configure_optimizers(self):
        '''
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        '''
        optimizer = t.optim.AdamW(self.convnet.parameters())
        return optimizer

#%%
def main():
    wandb.init(project=PROJECT_NAME)

    k = wandb.config["k"]
    subset = wandb.config["subset"]

    model = LitConvNet(k=k)
    mnist_trainset, mnist_testset = get_mnist(subset=subset)
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

    trainer = pl.Trainer(
        max_epochs=1,
        logger=WandbLogger(project=PROJECT_NAME),
        log_every_n_steps=1,
    )

    trainer.fit(model=model, train_dataloaders=mnist_trainloader, val_dataloaders=mnist_testloader)
    wandb.finish()

#%%
k_values = [int(i) for i in list(8 * np.unique(np.power(np.sqrt(2), np.arange(0, 8)).astype(int)))]
s_values = [16, 8, 4, 2, 1]

sweep_config = {
    'method': 'grid',
    'name': 'mnist-sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'val_loss',
    },
    'parameters': {
        'k': {'values': k_values},
        'subset': {'values': s_values}
     }
}

sweep_id = wandb.sweep(
    sweep=sweep_config,
    entity="jmsdao",
    project=PROJECT_NAME,
)

#%%
wandb.agent(
    sweep_id,
    function=main,
    count=len(k_values) * len(s_values),
)


