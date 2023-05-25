# %%
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import pandas as pd
import torch as t
from torch import Tensor, optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from typing import Callable, Iterable, List, Tuple, Optional
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
    results = []
    optim = t.optim.SGD([xy], lr=lr, momentum=momentum)

    for i in range(n_iters):
        optim.param_groups[0]["momentum"] = momentum ** (i / n_iters)
        results.append(xy.tolist())
        val = fn(*xy)
        val.backward()
        optim.step()
        optim.zero_grad()

    return t.tensor(results)


if MAIN:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    out = opt_fn_with_sgd(pathological_curve_loss, xy, lr=0.02, momentum=0.99)
    print(out[-1])

# %%
if MAIN:
    points = []

    optimizer_list = [
        (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
        (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
        (optim.SGD, {"lr": 0.06, "momentum": 0.5}),
        (optim.SGD, {"lr": 0.06, "momentum": 0.5}),
    ]
    optimizer_list += [
        (optim.SGD, {"lr": 0.4, "momentum": 0.9 + x / 1000}) for x in range(0, 101, 10)
    ]
    n_iters = 500

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn_with_sgd(
            pathological_curve_loss,
            xy=xy,
            lr=params["lr"],
            momentum=params["momentum"],
            n_iters=n_iters,
        )

        points.append((xys, optimizer_class, params))

    plot_fn_with_points(pathological_curve_loss, points=points[-9:])


for x in points:
    print(x[0][-1])


# %% Plot the evolution of momentum
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

        self.running_momentums: List[Optional[Tensor]] = [None] * len(self.params)

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, param in enumerate(self.params):
            if self.weight_decay != 0:
                grad = param.grad + self.weight_decay * param
            else:
                grad = param.grad
            assert grad is not None

            if self.momentum != 0:
                if self.running_momentums[i] is None:
                    self.running_momentums[i] = grad  # clone?
                else:
                    self.running_momentums[i].set_(
                        self.momentum * self.running_momentums[i] + grad
                    )
                grad = self.running_momentums[i]
            param.add_(grad, alpha=-self.lr)

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"


if MAIN:
    tests.test_sgd(SGD)


#%%
t.optim.SGD