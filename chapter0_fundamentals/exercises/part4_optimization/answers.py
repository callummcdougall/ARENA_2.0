# %%
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
# %%
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
    sgd = t.optim.SGD(params=[xy], lr=lr, momentum=momentum)
    
    inputs = t.zeros((n_iters, 2))

    for n in range(n_iters):
        value = fn(xy[0], xy[1])
        inputs[n] = xy.detach()
        value.backward()
        sgd.step()
        sgd.zero_grad()

    return inputs


    
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
        self.params = list(params) # turn params into a list (because it might be a generator)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.buffer = [None]*len(self.params)
        

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for idx, param in enumerate(self.params):
            assert param.grad is not None, "gradients of parameters must be defined!"
            grad = param.grad
            if self.lr != 0:
                grad += self.weight_decay*param
            if self.momentum != 0:
                if self.buffer[idx] is not None:
                    step = self.momentum * self.buffer[idx] + grad # type: ignore
                else:
                    step = grad
                grad = step
                self.buffer[idx] = step
            param -= self.lr*grad


    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"

class SGDParamGroups:

    def __init__(self, 
                 params: Iterable[Dict[str, Any]], **kwargs):
        '''Implements SGD with momentum.

        Accepts parameters in groups, or an iterable.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        '''
        if not "momentum" in kwargs.keys():
            kwargs["momentum"] = 0.0
        if not "weight_decay" in kwargs.keys():
            kwargs["weight_decay"] = 0.0
                
        self.params = list(params) # turn params into a list (because it might be a generator)
        for param_group in params:
            for key in ["lr","momentum","weight_decay"]:
                if not key in param_group.keys():
                    if not key in kwargs.keys():
                        raise ValueError()
                    param_group[key] = kwargs[key]
            param_group["params"] = list(param_group["params"])
            param_group["buffer"] = [None]*len(param_group["params"])
        for idx, param_group1 in enumerate(params):
            for param_group2 in params[:idx] + params[idx+1:]:
                for param in param_group1["params"]:
                    for param2 in param_group2["params"]:
                        if param is param2:
                            raise ValueError()
                
    def zero_grad(self) -> None:
        for param_group in self.params:
            for param in param_group["params"]:
                param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for param_group in self.params:
            for idx, param in enumerate(param_group['params']):
                assert param.grad is not None, "gradients of parameters must be defined!"
                grad = param.grad
                if param_group["lr"] != 0:
                    grad += param_group["weight_decay"]*param
                if param_group["momentum"] != 0:
                    if param_group["buffer"][idx] is not None:
                        step = param_group["momentum"] * param_group["buffer"][idx] + grad # type: ignore
                    else:
                        step = grad
                    grad = step
                    param_group["buffer"][idx] = step
                param -= param_group["lr"]*grad


if MAIN:
    tests.test_sgd(SGD)
    tests.test_sgd_param_groups(SGDParamGroups)
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
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        if self.momentum != 0:
            self.buffer = [t.zeros_like(param) for param in self.params] 
        self.square_average = [t.zeros_like(param) for param in self.params] 

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for idx, param in enumerate(self.params):
            grad = param.grad
            assert grad is not None, "gradients of parameters must be defined!"
            if self.weight_decay != 0:
                grad += self.weight_decay*param
            self.square_average[idx] = self.alpha*self.square_average[idx] + (1 - self.alpha)*grad**2
            if self.momentum > 0:
                self.buffer[idx] = self.momentum*self.buffer[idx] + grad / (t.sqrt(self.square_average[idx]) + self.eps)
                param -= self.lr * self.buffer[idx]
            else:
                param -= self.lr * grad / (t.sqrt(self.square_average[idx]) + self.eps)

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
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [t.zeros_like(param) for param in self.params] 
        self.v = [t.zeros_like(param) for param in self.params] 
        self.step_number = 1


    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for idx, param in enumerate(self.params):
            grad = param.grad
            assert grad is not None, "gradients of parameters must be defined!"
            if self.weight_decay != 0:
                grad += self.weight_decay*param
            self.m[idx] = self.betas[0]*self.m[idx] + (1 - self.betas[0])*grad
            self.v[idx] = self.betas[1]*self.v[idx] + (1 - self.betas[1])*grad.pow(2)
            m_scaled = self.m[idx] / (1 - self.betas[0] ** self.step_number)
            v_scaled = self.v[idx] / (1 - self.betas[1] ** self.step_number)

            param -= self.lr*m_scaled / (t.sqrt(v_scaled)  + self.eps)
        self.step_number += 1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.betas[0]}, beta2={self.betas[1]}, eps={self.eps}, weight_decay={self.lr})"



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
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        '''
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [t.zeros_like(param) for param in self.params] 
        self.v = [t.zeros_like(param) for param in self.params] 
        self.step_number = 1


    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None


    @t.inference_mode()
    def step(self) -> None:
        for idx, param in enumerate(self.params):
            grad = param.grad
            assert grad is not None, "gradients of parameters must be defined!"
            param -= self.lr*self.weight_decay*param
            self.m[idx] = self.betas[0]*self.m[idx] + (1 - self.betas[0])*grad
            self.v[idx] = self.betas[1]*self.v[idx] + (1 - self.betas[1])*grad.pow(2)
            m_scaled = self.m[idx] / (1 - self.betas[0] ** self.step_number)
            v_scaled = self.v[idx] / (1 - self.betas[1] ** self.step_number)

            param -= self.lr*m_scaled / (t.sqrt(v_scaled)  + self.eps)
        self.step_number += 1

    def __repr__(self) -> str:
        return f"AdamW(lr={self.lr}, beta1={self.betas[0]}, beta2={self.betas[1]}, eps={self.eps}, weight_decay={self.lr})"



if MAIN:
    tests.test_adamw(AdamW)
# %%

def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_hyperparams: dict, n_iters: int = 100):
    '''Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    '''
    optimizer = optimizer_class(params=[xy], **optimizer_hyperparams)
    
    inputs = t.zeros((n_iters, 2))

    for n in range(n_iters):
        value = fn(xy[0], xy[1])
        inputs[n] = xy.detach()
        value.backward()
        optimizer.step()
        optimizer.zero_grad()

    return inputs

# %%
def optimize_and_plot(fn):
        points = []

        optimizer_list = [
            (SGD, {"lr": 0.03, "momentum": 0.99}),
            (RMSprop, {"lr": 0.02, "alpha": 0.99, "momentum": 0.8}),
            (Adam, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
        ]

        for optimizer_class, params in optimizer_list:
            xy = t.tensor([2.5, 2.5], requires_grad=True)
            xys = opt_fn(fn, xy=xy, optimizer_class=optimizer_class, optimizer_hyperparams=params)
            points.append((xys, optimizer_class, params))

        plot_fn_with_points(fn, points=points)


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
if MAIN:
    for func in [bivariate_gaussian, neg_trimodal_func, rosenbrocks_banana_func]:
        optimize_and_plot(func)



# %%
##### PART 2: WANDB

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

if MAIN:
    cifar_trainset, cifar_testset = get_cifar(subset=1)
    cifar_trainset_small, cifar_testset_small = get_cifar(subset=10)

@dataclass
class ResNetFinetuningArgs():
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
        self.trainloader = DataLoader(self.trainset, shuffle=True, batch_size=self.batch_size)
        self.testloader = DataLoader(self.testset, shuffle=False, batch_size=self.batch_size)
        self.logger = CSVLogger(save_dir=self.log_dir, name=self.log_name)

# %%

class LitResNet(pl.LightningModule):
    def __init__(self, args: ResNetFinetuningArgs):
        super().__init__()
        self.resnet = get_resnet_for_feature_extraction(args.n_classes)
        self.args = args

    def _shared_train_val_step(self, batch: Tuple[t.Tensor, t.Tensor]) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        '''
        Convenience function since train/validation steps are similar.
        '''
        imgs, labels = batch
        logits = self.resnet(imgs)
        return logits, labels

    def training_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> t.Tensor:
        '''
        Here you compute and return the training loss and some additional metrics for e.g. 
        the progress bar or logger.
        '''
        logits, labels = self._shared_train_val_step(batch)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> None:
        '''
        Operates on a single batch of data from the validation set. In this step you might
        generate examples or calculate anything of interest like accuracy.
        '''
        logits, labels = self._shared_train_val_step(batch)
        classifications = logits.argmax(dim=1)
        accuracy = t.sum(classifications == labels) / len(classifications)
        self.log("accuracy", accuracy)

    def configure_optimizers(self):
        '''
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        '''
        optimizer = self.args.optimizer(self.resnet.out_layers.parameters(), lr=self.args.learning_rate)
        return optimizer
    
# %%

if MAIN:
    args = ResNetFinetuningArgs(trainset=cifar_trainset_small, testset=cifar_testset_small)
    model = LitResNet(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=args.logger,
        log_every_n_steps=args.log_every_n_steps,
    )
    trainer.fit(model=model, train_dataloaders=args.trainloader, val_dataloaders=args.testloader)

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
            template="ggplot2",
            width=600, height=400,
            labels={"x": "Classification", "y": "Probability"}, 
            text_auto='.2f', showlegend=False,
        )


if MAIN:
    test_resnet_on_random_input()

# %%
import wandb
# %%
@dataclass
class ResNetFinetuningArgsWandb(ResNetFinetuningArgs):
    use_wandb: bool = True
    run_name: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.use_wandb:
            self.logger = WandbLogger(save_dir=self.log_dir, project=self.log_name, name=self.run_name)

# %%
if MAIN:
    args = ResNetFinetuningArgsWandb(trainset=cifar_trainset_small, testset=cifar_testset_small)
    model = LitResNet(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=args.logger,
        log_every_n_steps=args.log_every_n_steps
    )
    trainer.fit(model=model, train_dataloaders=args.trainloader, val_dataloaders=args.testloader)
    wandb.finish()

# %%
if MAIN:
    sweep_config = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'maximize', 
            'name': 'accuracy'
            },
        'parameters': {
            'batch_size': {'values': [32,64,128,256]},
            'max_epochs': {'values': [1,2,3]},
            'learning_rate': {'max': 0.1, 'min': 0.0001, 'distribution': 'log_uniform_values'}
        }
    }
    tests.test_sweep_config(sweep_config)

# %%
def train():
    args = ResNetFinetuningArgsWandb(trainset=cifar_trainset_small, testset=cifar_testset_small)
    args.batch_size = wandb.config["batch_size"]
    args.max_epochs = wandb.config["max_epochs"]
    args.learning_rate = wandb.config["learning_rate"]
    model = LitResNet(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=args.logger,
        log_every_n_steps=args.log_every_n_steps
    )
    trainer.fit(model=model, train_dataloaders=args.trainloader, val_dataloaders=args.testloader)
    wandb.finish()

# %%
if MAIN:
    sweep_id = wandb.sweep(sweep=sweep_config, project='day4-resnet-sweep')
    wandb.agent(sweep_id=sweep_id, function=train, count=3)

##### PART 3
