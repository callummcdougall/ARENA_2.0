# %%
import os
import einops

import numpy as np

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange
from dataclasses import dataclass
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from typing import List, Tuple, Dict
from PIL import Image
from IPython.display import display
from pathlib import Path
import torchinfo
import json
import pandas as pd
from jaxtyping import Float, Int
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from typing import Any

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part3_resnets"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from part2_cnns.solutions import get_mnist, Linear, Conv2d, Flatten, ReLU, MaxPool2d
from part3_resnets.utils import print_param_count
import part3_resnets.tests as tests
from plotly_utils import line, plot_train_loss_and_test_accuracy_from_metrics

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%


class ConvNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )  # stride=1
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)  # padding=1
        self.conv2 = Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)  # padding=1
        self.flatten = Flatten(start_dim=1, end_dim=-1)
        self.linear1 = Linear(in_features=7 * 7 * 64, out_features=128)  # bias = True
        self.linear2 = Linear(in_features=128, out_features=10)  # bias = True
        self.layers: list[nn.Module] = [
            self.conv1,
            self.relu1,
            self.maxpool1,
            self.conv2,
            self.relu2,
            self.maxpool1,
            self.flatten,
            self.linear1,
            self.linear2
        ]

    def forward(self, x: t.Tensor) -> t.Tensor:
        r = x
        for layer in self.layers:
            r = layer.forward(r)
        return r


if MAIN:
    model = ConvNet()
    print(model)


# %%

if MAIN:
    summary = torchinfo.summary(model, input_size=(1, 1, 28, 28))
    print(summary)

# %%
if MAIN:
    MNIST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_mnist(subset: int = 1):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset



if MAIN:
    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# %%

from tqdm.notebook import tqdm
import time


if MAIN:
    for i in tqdm(range(100)):
        time.sleep(0.01)

# %%

if MAIN:
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
# %%

if MAIN:
    model = ConvNet().to(device)

    batch_size = 64
    epochs = 3

    mnist_trainset, _ = get_mnist(subset = 10)
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)

    optimizer = t.optim.Adam(model.parameters())
    loss_list = []

    for epoch in tqdm(range(epochs)):
        for imgs, labels in mnist_trainloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())   # .item() converts single-elem tensor to scalar

# %%

if MAIN:
    line(
        loss_list,
        yaxis_range=[0, max(loss_list) + 0.1],
        labels={"x": "Num batches seen", "y": "Cross entropy loss"},
        title="ConvNet training on MNIST",
        width=700
    )

# %%

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

# %%

class LitConvNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.convnet = ConvNet()


    def _shared_train_val_step(self, batch: Tuple[t.Tensor, t.Tensor]) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        '''Convenience function since train/validation steps are similar.'''
        pass

    def training_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> t.Tensor:
        '''
        Here you compute and return the training loss and some additional metrics for e.g. the progress bar or logger.
        '''
        imgs, labels = batch
        logits = self.convnet(imgs)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> None:
        print("validation print")
        self.log("validation_is_called", 1)

    def configure_optimizers(self):
        '''
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        '''
        optimizer = t.optim.Adam(self.parameters())
        return optimizer

# Set batch size

if MAIN:
    batch_size = 64
    max_epochs = 3

    # Create the model & training system
    model = LitConvNet()

    # Get dataloaders
    trainset, testset = get_mnist(subset = 10)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    # Get a logger, to record metrics during training
    logger = CSVLogger(save_dir=os.getcwd() + "/logs", name="day4-convenet")

    # Train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(model=model, train_dataloaders=trainloader)

# %%

if MAIN:
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

    metrics.head()

# %%

if MAIN:
    line(
        metrics["train_loss"].values,
        x=metrics["step"].values,
        yaxis_range=[0, metrics["train_loss"].max() + 0.1],
        labels={"x": "Batches seen", "y": "Cross entropy loss"},
        title="ConvNet training on MNIST",
        width=800,
        hovermode="x unified",
        template="ggplot2", # alternative aesthetic for your plots (-:
    )
# %%

@dataclass
class ConvNetTrainingArgs():
    '''
    Defining this class implicitly creates an __init__ method, which sets arguments as
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = ConvNetTrainingArgs(batch_size=128).
    '''
    batch_size: int = 64
    max_epochs: int = 3
    optimizer: t.optim.Optimizer = t.optim.Adam
    learning_rate: float = 1e-3
    log_dir: str = os.getcwd() + "/logs"
    log_name: str = "day4-convenet"
    log_every_n_steps: int = 1
    sample: int = 10

    def __post_init__(self):
        '''
        This code runs after the class is instantiated. It can reference things like
        self.sample, which are defined in the __init__ block.
        '''
        trainset, testset = get_mnist(subset=self.sample)
        self.trainloader = DataLoader(trainset, shuffle=True, batch_size=self.batch_size)
        self.testloader = DataLoader(testset, shuffle=False, batch_size=self.batch_size)
        self.logger = CSVLogger(save_dir=self.log_dir, name=self.log_name)


class LitConvNet(pl.LightningModule):
    def __init__(self, args: ConvNetTrainingArgs):
        super().__init__()
        self.convnet = ConvNet()
        self.args = args

    def training_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> t.Tensor:
        imgs, labels = batch
        logits = self.convnet(imgs)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = self.args.optimizer(self.parameters(), lr=self.args.learning_rate)
        return optimizer

if MAIN:
    args = ConvNetTrainingArgs()
    model = LitConvNet(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=args.logger,
        log_every_n_steps=1
    )
    trainer.fit(model=model, train_dataloaders=args.trainloader)

# %%

class LitConvNetTest(pl.LightningModule):
    def __init__(self, args: ConvNetTrainingArgs):
        super().__init__()
        self.convnet = ConvNet()
        self.args = args

    def _logits_labels(self, batch: Tuple[t.Tensor, t.Tensor]) -> Tuple[t.Tensor]:
        '''Convenience function since train/validation steps are similar.'''
        imgs, labels = batch
        logits = self.convnet(imgs)
        return logits, labels

    def training_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> t.Tensor:
        '''
        Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        '''
        logits, labels = self._logits_labels(batch)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> None:
        '''
        Operates on a single batch of data from the validation set. In this step you might
        generate examples or calculate anything of interest like accuracy.
        '''
        logits, labels = self._logits_labels(batch)
        # Test accuracy is the number of correct classifications on the dataset
        our_labels = logits.argmax(dim=1)
        labels_correct = our_labels == labels
        accuracy = labels_correct.mean(dtype=t.float32)
        loss = F.cross_entropy(logits, labels)
        self.log("validation_loss", loss)
        self.log("accuracy", accuracy)

    def configure_optimizers(self):
        '''
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        '''
        optimizer = self.args.optimizer(self.parameters(), lr=self.args.learning_rate)
        return optimizer



if MAIN:
    args = ConvNetTrainingArgs()
    model = LitConvNetTest(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=args.logger,
        log_every_n_steps=args.log_every_n_steps
    )
    trainer.fit(model=model, train_dataloaders=args.trainloader, val_dataloaders=args.testloader)

# %%
if MAIN:
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

    plot_train_loss_and_test_accuracy_from_metrics(metrics, "Training ConvNet on MNIST data")
# %%

class Sequential(nn.Module):
    _modules: Dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        if index < 0: index += len(self._modules) # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        if index < 0: index += len(self._modules) # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for k, mod in self._modules.items():
            if False:
                k
            x = mod(x)
            assert x is not None
        return x
# %%
class BatchNorm2dUS(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    num_features: int
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""] # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("num_batches_tracked", t.tensor(0))
        self.register_buffer("running_mean", t.zeros(num_features)[None, :, None, None])
        self.register_buffer("running_var", t.ones(num_features)[None, :, None, None])
        self.weight = nn.Parameter(t.ones(num_features)[None, :, None, None])
        self.bias = nn.Parameter(t.zeros(num_features)[None, :, None, None])

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''

        if self.training:
            batch_mean = x.mean(dim=[0, 2, 3], keepdim=True)
            batch_var = x.var(dim=[0, 2, 3], unbiased=False, keepdim=True)

            self.running_mean = self.running_mean * (1 - self.momentum) + batch_mean * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + batch_var * self.momentum

            nbatches, _, _, _ = x.shape
            self.num_batches_tracked += nbatches

        # weight = self.weight[:, None]
        # bias = self.bias[:, None]
        # # weight = einops.rearrange(self.weight, )
        # bias = einops.rearrange(self.bias, )

        result = ((x - self.running_mean) / t.sqrt(self.running_var + self.eps)) * self.weight + self.bias
        return result


    def extra_repr(self) -> str:
        return f"{self.num_features=},{self.momentum=},{self.eps=}"


if MAIN and False:
    tests.test_batchnorm2d_module(BatchNorm2d)
    tests.test_batchnorm2d_forward(BatchNorm2d)
    tests.test_batchnorm2d_running_mean(BatchNorm2d)
# %%
class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""] # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        # SOLUTION
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        # SOLUTION

        # Calculating mean and var over all dims except for the channel dim
        if self.training:
            # Using keepdim=True so we don't have to worry about broadasting them with x at the end
            mean = t.mean(x, dim=(0, 2, 3), keepdim=True)
            var = t.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
            # Updating running mean and variance, in line with PyTorch documentation
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            self.num_batches_tracked += 1
        else:
            mean = rearrange(self.running_mean, "channels -> 1 channels 1 1")
            var = rearrange(self.running_var, "channels -> 1 channels 1 1")

        # Rearranging these so they can be broadcasted (although there are other ways you could do this)
        weight = rearrange(self.weight, "channels -> 1 channels 1 1")
        bias = rearrange(self.bias, "channels -> 1 channels 1 1")

        return ((x - mean) / t.sqrt(var + self.eps)) * weight + bias

    def extra_repr(self) -> str:
        # SOLUTION
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])
# %%
class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return x.mean(dim=[2, 3])

# %%
class ResidualBlock(nn.Module):
    left_branch: nn.Sequential
    right_branch: nn.Module
    relu1: nn.Module

    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()

        self.left_branch = Sequential(*OrderedDict([
            ("conv1", Conv2d(
                in_channels=in_feats,
                out_channels=out_feats,
                stride=first_stride,
                kernel_size=(3, 3),
                padding=1,
            )),
            ("batchnorm1", BatchNorm2d(num_features=out_feats)),
            ("relu1", ReLU()),
            ("conv2", Conv2d(
                in_channels=out_feats,
                out_channels=out_feats,
                kernel_size=(3, 3),
                padding=1,
            )),
        ]).values())

        if first_stride == 1:
            self.right_branch = nn.Identity()
        else:
            self.right_branch = Sequential(*OrderedDict([
                ("conv1", Conv2d(
                    in_channels=in_feats,
                    out_channels=out_feats,
                    stride=first_stride,
                    kernel_size=(1, 1),
                    padding=0,
                )),
                ("batchnorm1", BatchNorm2d(num_features=out_feats))
            ]).values())

        self.relu1 = ReLU()


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        '''
        result = self.left_branch.forward(x)
        residual = self.right_branch.forward(x)
        return self.relu1.forward(result + residual)
# %%

class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()

        blocks = [ResidualBlock(in_feats=in_feats, out_feats=out_feats, first_stride=first_stride)]
        for i in range(n_blocks - 1):
            b = ResidualBlock(in_feats=out_feats, out_feats=out_feats)
            blocks.append(b)
        self.parts = Sequential(*blocks)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        return self.parts.forward(x)

# %%
# resnet = models.resnet34()
# print(torchinfo.summary(resnet, input_size=(1, 3, 64, 64)))

# %%
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        in_feats = 64
        block_groups = []
        for blocks, out_feats, strides in zip(n_blocks_per_group, out_features_per_group, first_strides_per_group):
            group = BlockGroup(n_blocks=blocks, in_feats=in_feats, out_feats=out_feats, first_stride=strides)
            block_groups.append(group)
            in_feats = out_feats
        final_in_features = out_features_per_group[-1]
        self.parts = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
            BatchNorm2d(num_features=64),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            *block_groups,
            AveragePool(),
            Flatten(),
            Linear(in_features=final_in_features, out_features=n_classes)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        return self.parts.forward(x)


if MAIN:
    my_resnet = ResNet34()
    th_resnet = models.resnet34()
    infos = []
    for s in (my_resnet, th_resnet):
        infos.append(str(torchinfo.summary(my_resnet, input_size=(1, 3, 64, 64))))
    if t1 == t2:
        print("YAAAAAY!")
    for t1, t2 in zip(*infos):
        if t1 != t2:
            print(f"{t1} {t2}")

# %%
