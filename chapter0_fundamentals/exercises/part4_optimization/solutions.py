# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
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
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from plotly_utils import bar, imshow
from part3_resnets.solutions import IMAGENET_TRANSFORM, get_resnet_for_feature_extraction, plot_train_loss_and_test_accuracy_from_metrics
from part4_optimization.utils import plot_fn, plot_fn_with_points
import part4_optimization.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %% 1️⃣ OPTIMIZERS

def pathological_curve_loss(x: t.Tensor, y: t.Tensor):
	# Example of a pathological curvature. There are many more possible, feel free to experiment here!
	x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
	y_loss = t.sigmoid(y)
	return x_loss + y_loss


if MAIN:
	plot_fn(pathological_curve_loss)

# %%

def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
	'''
	Optimize the a given function starting from the specified point.

	xy: shape (2,). The (x, y) starting point.
	n_iters: number of steps.
	lr, momentum: parameters passed to the torch.optim.SGD optimizer.

	Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
	'''
	assert xy.requires_grad

	xys = t.zeros((n_iters, 2))

	# YOUR CODE HERE: run optimization, and populate `xys` with the coordinates before each step
	optimizer = optim.SGD([xy], lr=lr, momentum=momentum)

	for i in range(n_iters):
		xys[i] = xy.detach()
		out = fn(xy[0], xy[1])
		out.backward()
		optimizer.step()
		optimizer.zero_grad()
		
	return xys

# %%


if MAIN:
	points = []
	
	optimizer_list = [
		(optim.SGD, {"lr": 0.1, "momentum": 0.0}),
		(optim.SGD, {"lr": 0.02, "momentum": 0.99}),
	]
	
	for optimizer_class, params in optimizer_list:
		xy = t.tensor([2.5, 2.5], requires_grad=True)
		xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params['lr'], momentum=params['momentum'])
	
		points.append((xys, optimizer_class, params))
	
	plot_fn_with_points(pathological_curve_loss, points=points)

# %%

class SGD:
	def __init__(
		self, 
		params: Iterable[t.nn.parameter.Parameter], 
		lr: float, 
		momentum: float = 0.0, 
		weight_decay: float = 0.0
	):
		'''Implements SGD with momentum.

		Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
			https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

		'''
		params = list(params) # turn params into a list (because it might be a generator)
		self.params = params
		self.lr = lr
		self.mu = momentum
		self.lmda = weight_decay
		self.t = 0

		self.gs = [t.zeros_like(p) for p in self.params]

	def zero_grad(self) -> None:
		for param in self.params:
			param.grad = None

	@t.inference_mode()
	def step(self) -> None:
		for i, (g, param) in enumerate(zip(self.gs, self.params)):
			# Implement the algorithm from the pseudocode to get new values of params and g
			new_g = param.grad
			if self.lmda != 0:
				new_g = new_g + (self.lmda * param)
			if self.mu != 0 and self.t > 0:
				new_g = (self.mu * g) + new_g
			# Update params (remember, this must be inplace)
			self.params[i] -= self.lr * new_g
			# Update g
			self.gs[i] = new_g
		self.t += 1

	def __repr__(self) -> str:
		return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"



if MAIN:
	tests.test_sgd(SGD)

# %%

class RMSprop:
	def __init__(
		self,
		params: Iterable[t.nn.parameter.Parameter],
		lr: float = 0.01,
		alpha: float = 0.99,
		eps: float = 1e-08,
		weight_decay: float = 0.0,
		momentum: float = 0.0,
	):
		'''Implements RMSprop.

		Like the PyTorch version, but assumes centered=False
			https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

		'''
		params = list(params) # turn params into a list (because it might be a generator)
		self.params = params
		self.lr = lr
		self.eps = eps
		self.mu = momentum
		self.lmda = weight_decay
		self.alpha = alpha

		self.bs = [t.zeros_like(p) for p in self.params]
		self.vs = [t.zeros_like(p) for p in self.params]

	def zero_grad(self) -> None:
		for p in self.params:
			p.grad = None

	@t.inference_mode()
	def step(self) -> None:
		for i, (p, b, v) in enumerate(zip(self.params, self.bs, self.vs)):
			new_g = p.grad
			if self.lmda != 0:
				new_g = new_g + self.lmda * p
			new_v = self.alpha * v + (1 - self.alpha) * new_g.pow(2)
			self.vs[i] = new_v
			if self.mu > 0:
				new_b = self.mu * b + new_g / (new_v.sqrt() + self.eps)
				p -= self.lr * new_b
				self.bs[i] = new_b
			else:
				p -= self.lr * new_g / (new_v.sqrt() + self.eps)

	def __repr__(self) -> str:
		return f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"



if MAIN:
	tests.test_rmsprop(RMSprop)

# %%

class Adam:
	def __init__(
		self,
		params: Iterable[t.nn.parameter.Parameter],
		lr: float = 0.001,
		betas: Tuple[float, float] = (0.9, 0.999),
		eps: float = 1e-08,
		weight_decay: float = 0.0,
	):
		'''Implements Adam.

		Like the PyTorch version, but assumes amsgrad=False and maximize=False
			https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
		'''
		params = list(params) # turn params into a list (because it might be a generator)
		self.params = params
		self.lr = lr
		self.beta1, self.beta2 = betas
		self.eps = eps
		self.lmda = weight_decay
		self.t = 1

		self.gs = [t.zeros_like(p) for p in self.params]
		self.ms = [t.zeros_like(p) for p in self.params]
		self.vs = [t.zeros_like(p) for p in self.params]

	def zero_grad(self) -> None:
		for p in self.params:
			p.grad = None

	@t.inference_mode()
	def step(self) -> None:
		for i, (p, g, m, v) in enumerate(zip(self.params, self.gs, self.ms, self.vs)):
			new_g = p.grad
			if self.lmda != 0:
				new_g = new_g + self.lmda * p
			self.gs[i] = new_g
			new_m = self.beta1 * m + (1 - self.beta1) * new_g
			new_v = self.beta2 * v + (1 - self.beta2) * new_g.pow(2)
			self.ms[i] = new_m
			self.vs[i] = new_v
			m_hat = new_m / (1 - self.beta1 ** self.t)
			v_hat = new_v / (1 - self.beta2 ** self.t)
			p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
		self.t += 1

	def __repr__(self) -> str:
		return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"



if MAIN:
	tests.test_adam(Adam)

# %%

class AdamW:
	def __init__(
		self,
		params: Iterable[t.nn.parameter.Parameter],
		lr: float = 0.001,
		betas: Tuple[float, float] = (0.9, 0.999),
		eps: float = 1e-08,
		weight_decay: float = 0.0,
	):
		'''Implements Adam.

		Like the PyTorch version, but assumes amsgrad=False and maximize=False
			https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
		'''
		params = list(params) # turn params into a list (because it might be a generator)
		self.params = params
		self.lr = lr
		self.beta1, self.beta2 = betas
		self.eps = eps
		self.lmda = weight_decay
		self.t = 1

		self.gs = [t.zeros_like(p) for p in self.params]
		self.ms = [t.zeros_like(p) for p in self.params]
		self.vs = [t.zeros_like(p) for p in self.params]

	def zero_grad(self) -> None:
		for p in self.params:
			p.grad = None

	@t.inference_mode()
	def step(self) -> None:
		for i, (p, g, m, v) in enumerate(zip(self.params, self.gs, self.ms, self.vs)):
			new_g = p.grad
			if self.lmda != 0:
				# new_g = new_g + self.lmda * p
				p -= p * self.lmda * self.lr
			self.gs[i] = new_g
			new_m = self.beta1 * m + (1 - self.beta1) * new_g
			new_v = self.beta2 * v + (1 - self.beta2) * new_g.pow(2)
			self.ms[i] = new_m
			self.vs[i] = new_v
			m_hat = new_m / (1 - self.beta1 ** self.t)
			v_hat = new_v / (1 - self.beta2 ** self.t)
			p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
		self.t += 1

	def __repr__(self) -> str:
		return f"AdamW(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"



if MAIN:
	tests.test_adamw(AdamW)

# %%

def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_hyperparams: dict, n_iters: int = 100):
	'''Optimize the a given function starting from the specified point.

	optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
	optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
	'''
	assert xy.requires_grad

	xys = t.zeros((n_iters, 2))
	optimizer = optimizer_class([xy], **optimizer_hyperparams)

	for i in range(n_iters):
		xys[i] = xy.detach()
		out = fn(xy[0], xy[1])
		out.backward()
		optimizer.step()
		optimizer.zero_grad()
	
	return xys

# %%


if MAIN:
	points = []
	
	optimizer_list = [
		(SGD, {"lr": 0.03, "momentum": 0.99}),
		(RMSprop, {"lr": 0.02, "alpha": 0.99, "momentum": 0.8}),
		(Adam, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
	]
	
	for optimizer_class, params in optimizer_list:
		xy = t.tensor([2.5, 2.5], requires_grad=True)
		xys = opt_fn(pathological_curve_loss, xy=xy, optimizer_class=optimizer_class, optimizer_hyperparams=params)
		points.append((xys, optimizer_class, params))
	
	plot_fn_with_points(pathological_curve_loss, points=points)

# %%

def bivariate_gaussian(x, y, x_mean=0.0, y_mean=0.0, x_sig=1.0, y_sig=1.0):
	norm = 1 / (2 * np.pi * x_sig * y_sig)
	x_exp = (-1 * (x - x_mean) ** 2) / (2 * x_sig**2)
	y_exp = (-1 * (y - y_mean) ** 2) / (2 * y_sig**2)
	return norm * t.exp(x_exp + y_exp)

def neg_trimodal_func(x, y):
	z = -bivariate_gaussian(x, y, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
	z -= bivariate_gaussian(x, y, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
	z -= bivariate_gaussian(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)
	return z


if MAIN:
	plot_fn(neg_trimodal_func, x_range=(-2, 2), y_range=(-2, 2))

# %%

def rosenbrocks_banana_func(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
	return (a - x) ** 2 + b * (y - x**2) ** 2 + 1


if MAIN:
	plot_fn(rosenbrocks_banana_func, x_range=(-2, 2), y_range=(-1, 3), log_scale=True)

# %%

class SGD:

	def __init__(self, params, **kwargs):
		'''Implements SGD with momentum.

		Accepts parameters in groups, or an iterable.

		Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
			https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
		'''

		if not isinstance(params, (list, tuple)):
			params = [{"params": params}]

		# assuming params is a list of dictionaries, we make self.params also a list of dictionaries (with other kwargs filled in)
		default_param_values = dict(momentum=0.0, weight_decay=0.0)

		# creating a list of param groups, which we'll iterate over during the step function
		self.param_groups = []
		# creating a list of params, which we'll use to check whether a param has been added twice
		params_to_check_for_duplicates = set()

		for param_group in params:
			# update param_group with kwargs passed in init; if this fails then update with the default values
			param_group = {**default_param_values, **kwargs, **param_group}
			# check that "lr" is defined (it should be either in kwargs, or in all of the param groups)
			assert "lr" in param_group, "Error: one of the parameter groups didn't specify a value for required parameter `lr`."
			# set the "params" and "gs" in param groups (note that we're storing 'gs' within each param group, rather than as self.gs)
			param_group["params"] = list(param_group["params"])
			param_group["gs"] = [t.zeros_like(p) for p in param_group["params"]]
			self.param_groups.append(param_group)
			# check that no params have been double counted
			for param in param_group["params"]:
				assert param not in params_to_check_for_duplicates, "Error: some parameters appear in more than one parameter group"
				params_to_check_for_duplicates.add(param)

		self.t = 1

	def zero_grad(self) -> None:
		for param_group in self.param_groups:
			for p in param_group["params"]:
				p.grad = None

	@t.inference_mode()
	def step(self) -> None:
		# loop through each param group
		for i, param_group in enumerate(self.param_groups):
			# get the parameters from the param_group
			lmda = param_group["weight_decay"]
			mu = param_group["momentum"]
			gamma = param_group["lr"]
			# loop through each parameter within each group
			for j, (p, g) in enumerate(zip(param_group["params"], param_group["gs"])):
				# Implement the algorithm in the pseudocode to get new values of params and g
				new_g = p.grad
				if lmda != 0:
					new_g = new_g + (lmda * p)
				if mu > 0 and self.t > 1:
					new_g = (mu * g) + new_g
				# Update params (remember, this must be inplace)
				param_group["params"][j] -= gamma * new_g
				# Update g
				self.param_groups[i]["gs"][j] = new_g
		self.t += 1



if MAIN:
	tests.test_sgd_param_groups(SGD)

# %% 2️⃣ WEIGHTS AND BIASES

def get_cifar(subset: int = 1):
	cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=IMAGENET_TRANSFORM)
	cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=IMAGENET_TRANSFORM)
	if subset > 1:
		cifar_trainset = Subset(cifar_trainset, indices=range(0, len(cifar_trainset), subset))
		cifar_testset = Subset(cifar_testset, indices=range(0, len(cifar_testset), subset))
	return cifar_trainset, cifar_testset


if MAIN:
	cifar_trainset, cifar_testset = get_cifar()
	
	imshow(
		cifar_trainset.data[:15],
		facet_col=0,
		facet_col_wrap=5,
		facet_labels=[cifar_trainset.classes[i] for i in cifar_trainset.targets[:15]],
		title="CIFAR-10 images",
		height=600
	)

# %%

@dataclass
class ResNetTrainingArgs():
	batch_size: int = 64
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

	def _shared_train_val_step(self, batch: Tuple[t.Tensor, t.Tensor]) -> Tuple[t.Tensor, t.Tensor]:
		imgs, labels = batch
		logits = self(imgs)
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
		self.log("accuracy", accuracy)

	def configure_optimizers(self):
		return self.args.optimizer(self.resnet.out_layers.parameters(), lr=self.args.learning_rate)
	
	def train_dataloader(self):
		return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
	
	def val_dataloader(self):
		return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)

# %%


if MAIN:
	args = ResNetTrainingArgs()
	model = LitResNet(args)
	logger = CSVLogger(save_dir=args.log_dir, name=args.log_name)
	
	trainer = pl.Trainer(
		max_epochs=args.max_epochs,
		logger=logger,
		log_every_n_steps=args.log_every_n_steps,
	)
	trainer.fit(model=model)
	
	metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
	
	plot_train_loss_and_test_accuracy_from_metrics(metrics, "Feature extraction with ResNet34")

# %%

def test_resnet_on_random_input(n_inputs: int = 3):
	indices = np.random.choice(len(cifar_trainset), n_inputs).tolist()
	classes = [cifar_trainset.classes[cifar_trainset.targets[i]] for i in indices]
	imgs = cifar_trainset.data[indices]
	with t.inference_mode():
		x = t.stack(list(map(IMAGENET_TRANSFORM, imgs)))
		logits: t.Tensor = model.resnet(x)
	probs = logits.softmax(-1)
	if probs.ndim == 1: probs = probs.unsqueeze(0)
	for img, label, prob in zip(imgs, classes, probs):
		display(HTML(f"<h2>Classification probabilities (true class = {label})</h2>"))
		imshow(
			img, 
			width=200, height=200, margin=0,
			xaxis_visible=False, yaxis_visible=False
		)
		bar(
			prob,
			x=cifar_trainset.classes,
			template="ggplot2", width=600, height=400,
			labels={"x": "Classification", "y": "Probability"}, 
			text_auto='.2f', showlegend=False,
		)


if MAIN:
	test_resnet_on_random_input()

# %%

import wandb

# %%

@dataclass
class ResNetTrainingArgsWandb(ResNetTrainingArgs):
	run_name: Optional[str] = None

# %%


if MAIN:
	args = ResNetTrainingArgsWandb()
	model = LitResNet(args)
	logger = WandbLogger(save_dir=args.log_dir, project=args.log_name, name=args.run_name)
	
	trainer = pl.Trainer(
		max_epochs=args.max_epochs,
		max_steps=args.max_steps,
		logger=logger,
		log_every_n_steps=args.log_every_n_steps,
	)
	trainer.fit(model=model)
	wandb.finish()

# %%


if MAIN:
	sweep_config = dict()
	# FLAT SOLUTION
	# YOUR CODE HERE - fill `sweep_config`
	sweep_config = dict(
		method = 'random',
		metric = dict(name = 'accuracy', goal = 'maximize'),
		parameters = dict(
			batch_size = dict(values = [32, 64, 128, 256]),
			max_epochs = dict(min = 1, max = 4),
			learning_rate = dict(max = 0.1, min = 0.0001, distribution = 'log_uniform_values'),
		)
	)
	# FLAT SOLUTION END
	
	tests.test_sweep_config(sweep_config)

# %%

# (2) Define a training function which takes no args, and uses `wandb.config` to get hyperparams

def train():
	# Define hyperparameters, override some with values from wandb.config
	args = ResNetTrainingArgsWandb()
	logger = WandbLogger(save_dir=args.log_dir, project=args.log_name, name=args.run_name)

	args.batch_size=wandb.config["batch_size"]
	args.max_epochs=wandb.config["max_epochs"]
	args.learning_rate=wandb.config["learning_rate"]

	model = LitResNet(args)

	trainer = pl.Trainer(
		max_epochs=args.max_epochs,
		max_steps=args.max_steps,
		logger=logger,
		log_every_n_steps=args.log_every_n_steps
	)
	trainer.fit(model=model)

# %%


if MAIN:
	sweep_id = wandb.sweep(sweep=sweep_config, project='day4-resnet-sweep')
	wandb.agent(sweep_id=sweep_id, function=train, count=3)
	wandb.finish()

# %% 3️⃣ BONUS

# Load from checkpoint

if MAIN:
	trained_model = LitResNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args=trainer.model.args)
	
	# Check models are identical
	assert all([(p1.to(device) == p2.to(device)).all() for p1, p2 in zip(model.resnet.parameters(), trained_model.resnet.parameters())])

# %%

