import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["WANDB_API_KEY"] = None
import sys
import pandas as pd
import torch as t
from torch import optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from typing import Callable, Iterable, Tuple, Optional, Any, Dict
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from IPython.display import display, HTML
import torchvision.transforms as transforms

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_optimization"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from plotly_utils import bar, imshow
from part3_resnets.solutions import IMAGENET_TRANSFORM, get_resnet_for_feature_extraction, plot_train_loss_and_test_accuracy_from_metrics
from part4_optimization.utils import plot_fn, plot_fn_with_points
import part4_optimization.tests as tests
from part2_cnns.solutions import Conv2d, ReLU, MaxPool2d, Flatten, Linear
import wandb
import yaml
import math


##############################


class ConvNet(t.nn.Module):
	def __init__(self, h):
		super().__init__()
		
		self.conv1 = Conv2d(in_channels=1, out_channels=32*h, kernel_size=3, stride=1, padding=1)
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)
		
		self.conv2 = Conv2d(in_channels=32*h, out_channels=64*h, kernel_size=3, stride=1, padding=1)
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=2, stride=2, padding=0)
		
		self.flatten = Flatten()
		self.fc1 = Linear(in_features=7*7*64*h, out_features=128*h)
		self.fc2 = Linear(in_features=128*h, out_features=10)
		
	def forward(self, x: t.Tensor) -> t.Tensor:
		x = self.maxpool1(self.relu1(self.conv1(x)))
		x = self.maxpool2(self.relu2(self.conv2(x)))
		x = self.fc2(self.fc1(self.flatten(x)))
		return x


@dataclass
class ConvNetTrainingArgs():
	'''
	Defining this class implicitly creates an __init__ method, which sets arguments as 
	given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
	when you create an instance, e.g. args = ConvNetTrainingArgs(batch_size=128).
	'''
	batch_size: int = 64
	max_epochs: int = 1
	optimizer: t.optim.Optimizer = t.optim.Adam
	learning_rate: float = 1e-3
	log_dir: str = os.getcwd() + "/logs"
	log_name: str = "day4-convnet"
	log_every_n_steps: int = 1
	sample: int = 10
	model_width: int = 128

	def __post_init__(self):
		trainset, testset = get_mnist(subset=self.sample)
		self.trainloader = DataLoader(trainset, shuffle=True, batch_size=self.batch_size)
		self.testloader = DataLoader(testset, shuffle=False, batch_size=self.batch_size)
		self.logger = CSVLogger(save_dir=self.log_dir, name=self.log_name)

@dataclass
class ConvNetTrainingArgsWandb(ConvNetTrainingArgs):
    use_wandb: bool = True
    run_name: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.use_wandb:
            self.logger = WandbLogger(save_dir=self.log_dir, project=self.log_name, name=self.run_name)



class LitConvNet(pl.LightningModule):
	def __init__(self, args: ConvNetTrainingArgs):
		super().__init__()
		self.convnet = ConvNet(args.model_width)
		self.param_count = sum(p.numel() for p in self.parameters())
		args.learning_rate = 0.003239 + ((-0.0001395)*math.log(self.param_count))
		self.args = args

	def _shared_train_val_step(self, batch: Tuple[t.Tensor, t.Tensor]) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
		imgs, labels = batch
		logits = self.convnet(imgs)
		return logits, labels

	def training_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> t.Tensor:
		logits, labels = self._shared_train_val_step(batch)
		loss = F.cross_entropy(logits, labels)
		self.log("train_loss", loss)
		return loss
	
	def validation_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> None:
		logits, labels = self._shared_train_val_step(batch)
		classifications = logits.argmax(dim=1)
		accuracy = t.sum(classifications == labels) / len(classifications)
		self.log("test accuracy", accuracy)
		self.log("param count", self.param_count)
		self.log("sample", self.args.sample)

	def configure_optimizers(self):
		optimizer = self.args.optimizer(self.parameters(), lr=self.args.learning_rate)
		return optimizer
	



MNIST_TRANSFORM = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	
def get_mnist(subset: int = 1):
	'''Returns MNIST training data, sampled by the frequency given in `subset`.'''
	MNIST_TRANSFORM = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
	mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)

	if subset > 1:
		mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
		mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

	return mnist_trainset, mnist_testset


def train():
    args = ConvNetTrainingArgsWandb()
    args.model_width = wandb.config["model_width"]
    args.sample = wandb.config["sample"]
    model = LitConvNet(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=args.logger,
        log_every_n_steps=args.log_every_n_steps
    )
    trainer.fit(model=model, train_dataloaders=args.trainloader, val_dataloaders=args.testloader)
    wandb.finish()


def yaml_to_dict(file_path):
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

if __name__ == "__main__":
	sweep_config = yaml_to_dict("/root/ARENA_2.0/chapter0_fundamentals/exercises/part4_optimization/config.YAML")
	sweep_id = wandb.sweep(sweep_config)
	wandb.agent(sweep_id, train)