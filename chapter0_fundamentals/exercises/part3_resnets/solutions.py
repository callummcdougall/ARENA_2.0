# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
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

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part3_resnets"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from part2_cnns.solutions import get_mnist, Linear, Conv2d, Flatten, ReLU, MaxPool2d
from part3_resnets.utils import print_param_count
import part3_resnets.tests as tests
from plotly_utils import line, plot_train_loss_and_test_accuracy_from_metrics

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

MAIN = __name__ == "__main__"

# %% 1️⃣ BUILDING & TRAINING A CNN

class ConvNet(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)
		
		self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=2, stride=2, padding=0)
		
		self.flatten = Flatten()
		self.fc1 = Linear(in_features=7*7*64, out_features=128)
		self.fc2 = Linear(in_features=128, out_features=10)
		self.relu3 = ReLU()
		
	def forward(self, x: t.Tensor) -> t.Tensor:
		x = self.maxpool1(self.relu1(self.conv1(x)))
		x = self.maxpool2(self.relu2(self.conv2(x)))
		x = self.fc2(self.relu3(self.fc1(self.flatten(x))))
		return x



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


class LitConvNet(pl.LightningModule):
	def __init__(self, batch_size: int, max_epochs: int, subset: int = 10):
		super().__init__()
		self.convnet = ConvNet()
		self.batch_size = batch_size
		self.max_epochs = max_epochs
		self.trainset, self.testset = get_mnist(subset = 10)

	def forward(self, x: t.Tensor) -> t.Tensor:
		'''
		Here you should define the forward pass of your model.
		'''
		return self.convnet(x)

	def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> t.Tensor:
		'''
		Here you compute and return the training loss and some additional metrics for e.g. the progress bar or logger.
		'''
		imgs, labels = batch
		logits = self(imgs)
		loss = F.cross_entropy(logits, labels)
		self.log("train_loss", loss)
		return loss

	def configure_optimizers(self):
		'''
		Choose what optimizers and learning-rate schedulers to use in your optimization.
		'''
		optimizer = t.optim.Adam(self.parameters())
		return optimizer
	
	def train_dataloader(self):
		'''
		Return the training dataloader.
		'''
		return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

# %%

# Create the model & training system

if MAIN:
	batch_size = 64
	max_epochs = 3
	model = LitConvNet(batch_size=batch_size, max_epochs=max_epochs)
	
	# Get a logger, to record metrics during training
	logger = CSVLogger(save_dir=os.getcwd() + "/logs", name="day4-convenet")
	
	# Train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
	trainer = pl.Trainer(
		max_epochs=max_epochs,
		logger=logger,
		log_every_n_steps=1,
	)
	trainer.fit(model=model)

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
	log_name: str = "day3-convenet"
	log_every_n_steps: int = 1
	sample: int = 10


class LitConvNet(pl.LightningModule):
	def __init__(self, args: ConvNetTrainingArgs):
		super().__init__()
		self.convnet = ConvNet()
		self.args = args
		self.trainset, self.testset = get_mnist(subset=args.sample)

	def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> t.Tensor:
		imgs, labels = batch
		logits = self.convnet(imgs)
		loss = F.cross_entropy(logits, labels)
		self.log("train_loss", loss)
		return loss

	def configure_optimizers(self):
		return self.args.optimizer(self.parameters(), lr=self.args.learning_rate)
	
	def train_dataloader(self):
		return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
	


if MAIN:
	args = ConvNetTrainingArgs()
	model = LitConvNet(args)
	logger = CSVLogger(save_dir=args.log_dir, name=args.log_name)
	
	trainer = pl.Trainer(
		max_epochs=args.max_epochs,
		logger=logger,
		log_every_n_steps=1
	)
	trainer.fit(model=model)

# %%


if MAIN:
	metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
	
	metrics.head()
	
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

class LitConvNetTest(pl.LightningModule):
	def __init__(self, args: ConvNetTrainingArgs):
		super().__init__()
		self.convnet = ConvNet()
		self.args = args
		self.trainset, self.testset = get_mnist(subset=args.sample)

	def forward(self, x: t.Tensor) -> t.Tensor:
		return self.convnet(x)

	def _shared_train_val_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
		imgs, labels = batch
		logits = self(imgs)
		return logits, labels

	def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
		logits, labels = self._shared_train_val_step(batch)
		loss = F.cross_entropy(logits, labels)
		self.log("train_loss", loss)
		return loss
	
	def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
		logits, labels = self._shared_train_val_step(batch)
		classifications = logits.argmax(dim=1)
		accuracy = t.sum(classifications == labels) / len(classifications)
		self.log("accuracy", accuracy)

	def configure_optimizers(self):
		return self.args.optimizer(self.parameters(), lr=self.args.learning_rate)
	
	def train_dataloader(self):
		return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
	
	def val_dataloader(self):
		return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)
	


if MAIN:
	args = ConvNetTrainingArgs()
	model = LitConvNetTest(args)
	logger = CSVLogger(save_dir=args.log_dir, name=args.log_name)
	
	trainer = pl.Trainer(
		max_epochs=args.max_epochs,
		logger=logger,
		log_every_n_steps=args.log_every_n_steps
	)
	trainer.fit(model=model)

# %%


if MAIN:
	metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
	
	plot_train_loss_and_test_accuracy_from_metrics(metrics, "Training ConvNet on MNIST data")

# %%


if MAIN:
	data_augmentation_transform = transforms.Compose([
	transforms.RandomRotation(degrees=15),
	transforms.RandomResizedCrop(size=28, scale=(0.8, 1.2)),
	transforms.RandomHorizontalFlip(p=0.5),
	transforms.RandomVerticalFlip(p=0.5),
	transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))
	])

# %%

def get_mnist_augmented(subset: int = 1, train_transform=None, test_transform=None):
	if train_transform is None:
		train_transform = MNIST_TRANSFORM
	if test_transform is None:
		test_transform = MNIST_TRANSFORM
	mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
	mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)
	if subset > 1:
		mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
		mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))
	return mnist_trainset, mnist_testset

# %% 2️⃣ ASSEMBLING RESNET

class Sequential(nn.Module):
	_modules: Dict[str, nn.Module]

	def __init__(self, *modules: nn.Module):
		super().__init__()
		for index, mod in enumerate(modules):
			self._modules[str(index)] = mod

	def __getitem__(self, index: int) -> nn.Module:
		index %= len(self._modules) # deal with negative indices
		return self._modules[str(index)]

	def __setitem__(self, index: int, module: nn.Module) -> None:
		index %= len(self._modules) # deal with negative indices
		self._modules[str(index)] = module

	def forward(self, x: t.Tensor) -> t.Tensor:
		'''Chain each module together, with the output from one feeding into the next one.'''
		for mod in self._modules.values():
			x = mod(x)
		return x

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
		return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])



if MAIN:
	tests.test_batchnorm2d_module(BatchNorm2d)
	tests.test_batchnorm2d_forward(BatchNorm2d)
	tests.test_batchnorm2d_running_mean(BatchNorm2d)

# %%

class AveragePool(nn.Module):
	def forward(self, x: t.Tensor) -> t.Tensor:
		'''
		x: shape (batch, channels, height, width)
		Return: shape (batch, channels)
		'''
		return t.mean(x, dim=(2, 3))

# %%

class ResidualBlock(nn.Module):
	def __init__(self, in_feats: int, out_feats: int, first_stride=1):
		'''
		A single residual block with optional downsampling.

		For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

		If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
		'''
		super().__init__()
		
		self.left = Sequential(
			Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
			BatchNorm2d(out_feats),
			ReLU(),
			Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(out_feats)
		)
		
		if first_stride > 1:
			self.right = Sequential(
				Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride),
				BatchNorm2d(out_feats)
			)
		else:
			self.right = nn.Identity()
			
		self.relu = ReLU()

	def forward(self, x: t.Tensor) -> t.Tensor:
		'''
		Compute the forward pass.

		x: shape (batch, in_feats, height, width)

		Return: shape (batch, out_feats, height / stride, width / stride)

		If no downsampling block is present, the addition should just add the left branch's output to the input.
		'''
		x_left = self.left(x)
		x_right = self.right(x)
		return self.relu(x_left + x_right)

# %%

class BlockGroup(nn.Module):
	def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
		'''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
		super().__init__()
		
		blocks = [ResidualBlock(in_feats, out_feats, first_stride)] + [
			ResidualBlock(out_feats, out_feats) for n in range(n_blocks - 1)
		]
		self.blocks = nn.Sequential(*blocks)
		
	def forward(self, x: t.Tensor) -> t.Tensor:
		'''
		Compute the forward pass.
		
		x: shape (batch, in_feats, height, width)

		Return: shape (batch, out_feats, height / first_stride, width / first_stride)
		'''
		return self.blocks(x)

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
		in_feats0 = 64
		self.n_blocks_per_group = n_blocks_per_group
		self.out_features_per_group = out_features_per_group
		self.first_strides_per_group = first_strides_per_group
		self.n_classes = n_classes

		self.in_layers = Sequential(
			Conv2d(3, in_feats0, kernel_size=7, stride=2, padding=3),
			BatchNorm2d(in_feats0),
			ReLU(),
			MaxPool2d(kernel_size=3, stride=2, padding=1),
		)

		all_in_feats = [in_feats0] + out_features_per_group[:-1]
		self.residual_layers = Sequential(
			*(
				BlockGroup(*args)
				for args in zip(
					n_blocks_per_group,
					all_in_feats,
					out_features_per_group,
					first_strides_per_group,
				)
			)
		)

		self.out_layers = Sequential(
			AveragePool(),
			Flatten(),
			Linear(out_features_per_group[-1], n_classes),
		)

	def forward(self, x: t.Tensor) -> t.Tensor:
		'''
		x: shape (batch, channels, height, width)
		Return: shape (batch, n_classes)
		'''
		x = self.in_layers(x)
		x = self.residual_layers(x)
		x = self.out_layers(x)
		return x



if MAIN:
	my_resnet = ResNet34()

# %%

def copy_weights(my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet) -> ResNet34:
	'''Copy over the weights of `pretrained_resnet` to your resnet.'''
	
	# Get the state dictionaries for each model, check they have the same number of parameters & buffers
	mydict = my_resnet.state_dict()
	pretraineddict = pretrained_resnet.state_dict()
	assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."
	
	# Define a dictionary mapping the names of your parameters / buffers to their values in the pretrained model
	state_dict_to_load = {
		mykey: pretrainedvalue
		for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
	}
	
	# Load in this dictionary to your model
	my_resnet.load_state_dict(state_dict_to_load)
	
	return my_resnet



if MAIN:
	pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
	my_resnet = copy_weights(my_resnet, pretrained_resnet)

# %%


if MAIN:
	IMAGE_FILENAMES = [
		"chimpanzee.jpg",
		"golden_retriever.jpg",
		"platypus.jpg",
		"frogs.jpg",
		"fireworks.jpg",
		"astronaut.jpg",
		"iguana.jpg",
		"volcano.jpg",
		"goofy.jpg",
		"dragonfly.jpg",
	]
	
	IMAGE_FOLDER = section_dir / "resnet_inputs"
	
	images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]

# %%


if MAIN:
	images[0]

# %%

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
	transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# %%

def prepare_data(images: List[Image.Image]) -> t.Tensor:
	'''
	Return: shape (batch=len(images), num_channels=3, height=224, width=224)
	'''
	x = t.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0)
	return x



if MAIN:
	prepared_images = prepare_data(images)
	
	assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)

# %%

def predict(model, images):
	logits: t.Tensor = model(images)
	return logits.argmax(dim=1)

# %%


if MAIN:
	with open(section_dir / "imagenet_labels.json") as f:
		imagenet_labels = list(json.load(f).values())

# %%

# Check your predictions match the pretrained model's

if MAIN:
	my_predictions = predict(my_resnet, prepared_images)
	pretrained_predictions = predict(pretrained_resnet, prepared_images)
	assert all(my_predictions == pretrained_predictions)

# %%

# Print out your predictions, next to the corresponding images

if MAIN:
	for img, label in zip(images, my_predictions):
		print(f"Class {label}: {imagenet_labels[label]}")
		display(img)
		print()

# %%

class NanModule(nn.Module):
	'''
	Define a module that always returns NaNs (we will use hooks to identify this error).
	'''
	def forward(self, x):
		return t.full_like(x, float('nan'))


if MAIN:
	model = nn.Sequential(
		nn.Identity(),
		NanModule(),
		nn.Identity()
	)
	
	
def hook_check_for_nan_output(module: nn.Module, input: Tuple[t.Tensor], output: t.Tensor) -> None:
	'''
	Hook function which detects when the output of a layer is NaN.
	'''
	if t.isnan(output).any():
		raise ValueError(f"NaN output from {module}")


def add_hook(module: nn.Module) -> None:
	'''
	Register our hook function in a module.

	Use model.apply(add_hook) to recursively apply the hook to model and all submodules.
	'''
	module.register_forward_hook(hook_check_for_nan_output)


def remove_hooks(module: nn.Module) -> None:
	'''
	Remove all hooks from module.

	Use module.apply(remove_hooks) to do this recursively.
	'''
	module._backward_hooks.clear()
	module._forward_hooks.clear()
	module._forward_pre_hooks.clear()



if MAIN:
	model = model.apply(add_hook)
	input = t.randn(3)
	
	try:
		output = model(input)
	except ValueError as e:
		print(e)
	
	model = model.apply(remove_hooks)

# %% 3️⃣ RESNET FEATURE EXTRACTION


if MAIN:
	layer0, layer1 = nn.Linear(3, 4), nn.Linear(4, 5)
	
	layer0.requires_grad_(False) # generic code to set `param.requires_grad = False` recursively for a module (or entire model)
	
	x = t.randn(3)
	out = layer1(layer0(x)).sum()
	out.backward()
	
	assert layer0.weight.grad is None
	assert layer1.weight.grad is not None

# %%

def get_resnet_for_feature_extraction(n_classes: int) -> ResNet34:
	'''
	Creates a ResNet34 instance, replaces its final linear layer with a classifier
	for `n_classes` classes, and freezes all weights except the ones in this layer.

	Returns the ResNet model.
	'''
	# Create a ResNet34 with the default number of classes
	my_resnet = ResNet34()

	# Load the pretrained weights
	pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

	# Copy the weights over
	my_resnet = copy_weights(my_resnet, pretrained_resnet)

	# Freeze gradients for all layers (note that when we redefine the last layer, it will be unfrozen)
	my_resnet.requires_grad_(False)    

	# Redefine last layer
	my_resnet.out_layers[-1] = Linear(
		my_resnet.out_features_per_group[-1],
		n_classes
	)

	return my_resnet



if MAIN:
	tests.test_get_resnet_for_feature_extraction(get_resnet_for_feature_extraction)

# %%

def get_cifar(subset: int):
	cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=IMAGENET_TRANSFORM)
	cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=IMAGENET_TRANSFORM)

	if subset > 1:
		cifar_trainset = Subset(cifar_trainset, indices=range(0, len(cifar_trainset), subset))
		cifar_testset = Subset(cifar_testset, indices=range(0, len(cifar_testset), subset))
		
	return cifar_trainset, cifar_testset


@dataclass
class ResNetTrainingArgs():
	batch_size: int = 64
	max_epochs: int = 3
	max_steps: int = 500
	optimizer: t.optim.Optimizer = t.optim.Adam
	learning_rate: float = 1e-3
	log_dir: str = os.getcwd() + "/logs"
	log_name: str = "day3-resnet"
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

	def _shared_train_val_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
		imgs, labels = batch
		logits = self(imgs)
		return logits, labels

	def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> t.Tensor:
		logits, labels = self._shared_train_val_step(batch)
		loss = F.cross_entropy(logits, labels)
		self.log("train_loss", loss)
		return loss
	
	def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
		logits, labels = self._shared_train_val_step(batch)
		classifications = logits.argmax(dim=1)
		accuracy = t.sum(classifications == labels) / len(classifications)
		self.log("accuracy", accuracy)

	def configure_optimizers(self):
		return self.args.optimizer(self.resnet.out_layers.parameters(), lr=self.args.learning_rate)
	
	def train_dataloader(self):
		return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
	
	def val_dataloader(self):
		return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)
	


if MAIN:
	args = ResNetTrainingArgs()
	model = LitResNet(args)
	logger = CSVLogger(save_dir=args.log_dir, name=args.log_name)
	
	trainer = pl.Trainer(
		max_epochs=args.max_epochs,
		logger=logger,
		log_every_n_steps=args.log_every_n_steps
	)
	trainer.fit(model=model)
	
	metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
	
	plot_train_loss_and_test_accuracy_from_metrics(metrics, "Feature extraction with ResNet34")

# %%

