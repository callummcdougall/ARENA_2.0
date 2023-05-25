#%%
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
from tqdm import tqdm

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
#%%

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
    optimizer = t.optim.SGD([xy], lr=lr, momentum = momentum)
    points = []

    for iter in tqdm(range(n_iters)):
        out = fn(xy[0],xy[1])
        points.append(t.tensor(xy.detach()))
        out.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    points = t.stack(points, dim=0)
    assert points.shape[0] == n_iters
    return points


if MAIN:
    points = []

    optimizer_list = [
        (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
        (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
    ]

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params['lr'], momentum=params['momentum'], n_iters=200)

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
        self.v = [t.zeros_like(param) for param in self.params]
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr = lr

        self.step_num = 1
        pass

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None
        pass

    @t.inference_mode()
    def step(self) -> None:
        
        for param_iter in range(len(self.params)):
            param = self.params[param_iter]
        
            # calculate a different version of gradient based on weight decay and momentum        
            altered_grad = param.grad if param.grad is not None else t.zeros_like(param)

            # weight decay first:
            if self.weight_decay:
                altered_grad = altered_grad + self.weight_decay * param

            if self.momentum:
                if self.step_num > 1:
                    self.v[param_iter] = self.momentum * self.v[param_iter] + altered_grad
                else:
                    self.v[param_iter] = altered_grad
            
                altered_grad = self.v[param_iter]
            self.params[param_iter] -= self.lr * altered_grad

        self.step_num += 1


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
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.mu = momentum
        self.lmda = weight_decay
        self.v = [t.zeros_like(p) for p in self.params]
        self.b = [t.zeros_like(p) for p in self.params]
        self.t = 1

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = t.zeros_like(p)

    @t.inference_mode()
    def step(self) -> None:
        for i in range(len(self.params)):
            p = self.params[i]
            new_g = p.grad
            if self.lmda:
                new_g += self.lmda * p
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * new_g ** 2
            if self.mu > 0:
                self.b[i] = self.mu * self.b[i] + new_g / (t.sqrt(self.v[i]) + self.eps)
                self.params[i] -= self.lr *  self.b[i]
            else:
                self.params[i] -= self.lr * new_g / (t.sqrt(self.v[i]) + self.eps)
        self.t +=1


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
        self.betas = betas
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.lmda = weight_decay

        self.m = [t.zeros_like(p) for p in self.params]
        self.v = [t.zeros_like(p) for p in self.params]

        self.t = 1
        pass

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = t.zeros_like(p)


    @t.inference_mode()
    def step(self) -> None:
        for i in range(len(self.params)):
            p = self.params[i]
            new_g = p.grad

            if self.lmda:
                new_g += self.lmda * p

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * new_g
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * new_g ** 2

            if self.t > 0:
                m_corrected = self.m[i] / (1 - self.betas[0] ** self.t)
                v_corrected = self.v[i] / (1 - self.betas[1] ** self.t)
            else:
                m_corrected = self.m[i]
                v_corrected = self.v[i]
            
            self.params[i] -= self.lr * (m_corrected / (t.sqrt(v_corrected) + self.eps))
        
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
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        '''
        self.betas = betas
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.lmda = weight_decay

        self.m = [t.zeros_like(p) for p in self.params]
        self.v = [t.zeros_like(p) for p in self.params]

        self.t = 1
        pass

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = t.zeros_like(p)


    @t.inference_mode()
    def step(self) -> None:
        for i in range(len(self.params)):
            p = self.params[i]
            new_g = p.grad
            
            self.params[i] -= self.params[i] * self.lr * self.lmda

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * new_g
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * new_g ** 2

            if self.t > 0:
                m_corrected = self.m[i] / (1 - self.betas[0] ** self.t)
                v_corrected = self.v[i] / (1 - self.betas[1] ** self.t)
            else:
                m_corrected = self.m[i]
                v_corrected = self.v[i]
            
            self.params[i] -= self.lr * (m_corrected / (t.sqrt(v_corrected) + self.eps))
        
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
    optimizer = optimizer_class(xy, **optimizer_hyperparams)
    points = []

    for iter in tqdm(range(n_iters)):

        optimizer.zero_grad()
        out = fn(xy[0],xy[1])
        points.append(xy.detach().clone())
        out.backward()
        optimizer.step()
        
    points = t.stack(points, dim=0)
    assert points.shape[0] == n_iters
    return points
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
        xys = opt_fn(pathological_curve_loss, xy=xy, optimizer_class=optimizer_class, optimizer_hyperparams=params, n_iters = 300)
        points.append((xys.detach().clone(), optimizer_class, params))

    plot_fn_with_points(pathological_curve_loss, points=points)
# %%
