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


# %%
def pathological_curve_loss(x: t.Tensor, y: t.Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss


if MAIN:
    plot_fn(pathological_curve_loss)

# %%


def opt_fn_with_sgd(
    fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100
):
    """
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum: parameters passed to the torch.optim.SGD optimizer.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    """
    points = []
    opt = optim.SGD([xy], lr=lr, momentum=momentum)
    for _ in range(n_iters):
        points.append(xy.detach().clone())
        z = fn(*xy)
        z.backward()
        opt.step()
        opt.zero_grad()
    return t.stack(points)


# %%
if MAIN:
    points = []

    optimizer_list = [
        (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
        (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
    ]

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn_with_sgd(
            pathological_curve_loss, xy=xy, lr=params["lr"], momentum=params["momentum"]
        )

        points.append((xys, optimizer_class, params))

    plot_fn_with_points(pathological_curve_loss, points=points)

# %%


class SGD:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        """
        self.params = list(
            params
        )  # turn params into a list (because it might be a generator)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.inertia = len(self.params) * [None]

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    @t.inference_mode()
    def step(self) -> None:
        for i, param in enumerate(self.params):
            g = param.grad.clone()

            if abs(self.weight_decay) > 1e-10:
                g += self.weight_decay * param

            if abs(self.momentum) > 1e-10:
                if self.inertia[i] is None:
                    self.inertia[i] = g.clone()
                else:
                    self.inertia[i] = self.momentum * self.inertia[i] + g
                    g = self.inertia[i]

            param -= self.lr * g

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
        """Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

        """
        self.params = list(
            params
        )  # turn params into a list (because it might be a generator)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.inertia = [t.zeros_like(param) for param in self.params]
        self.inertia_var = [t.zeros_like(param) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    @t.inference_mode()
    def step(self) -> None:
        for i, param in enumerate(self.params):
            g = param.grad.clone()

            if abs(self.weight_decay) > 1e-10:
                g += self.weight_decay * param

            self.inertia_var[i] = (
                self.alpha * self.inertia_var[i] + (1 - self.alpha) * g**2
            )
            if self.momentum > 1e-10:
                self.inertia[i] = self.momentum * self.inertia[i] + g / (
                    t.sqrt(self.inertia_var[i]) + self.eps
                )
                param -= self.lr * self.inertia[i]
            else:
                param -= self.lr * g / (t.sqrt(self.inertia_var[i]) + self.eps)

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
        self.params = list(
            params
        )  # turn params into a list (because it might be a generator)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.inertia = [t.zeros_like(param) for param in self.params]
        self.inertia_var = [t.zeros_like(param) for param in self.params]
        self.steps = 1

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    @t.inference_mode()
    def step(self) -> None:
        for i, param in enumerate(self.params):
            g = param.grad.clone()

            if abs(self.weight_decay) > 1e-10:
                g += self.weight_decay * param

            self.inertia[i] = self.betas[0] * self.inertia[i] + (1 - self.betas[0]) * g
            self.inertia_var[i] = self.betas[1] * self.inertia_var[i] + (1 - self.betas[1]) * (g ** 2)
            inertia = self.inertia[i] / ((1 - self.betas[0] ** self.steps))
            inertia_var = self.inertia_var[i] / ((1 - self.betas[1] ** self.steps))
            param -= self.lr * inertia / (inertia_var.sqrt() + self.eps)
        self.steps += 1


    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"



if MAIN:
    tests.test_adam(Adam)
# %%