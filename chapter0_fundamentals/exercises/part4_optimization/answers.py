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
    xy_history = []

    optimizer = t.optim.SGD([xy], lr=lr, momentum=momentum)
    for _ in range(n_iters):
        xy_history.append(xy.detach().clone())
        x, y = xy
        loss = fn(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return t.stack(xy_history, dim=0)


# %%
if MAIN:
    points = []

    optimizer_list = [
        # (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
        # (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
        # (optim.SGD, {"lr": 0.1, "momentum": 0.90}),
        (optim.SGD, {"lr": 0.5, "momentum": 0.90}),
        # (optim.SGD, {"lr": 0.6, "momentum": 0.90}),
        # (optim.SGD, {"lr": 0.8, "momentum": 0.90}),
        (optim.SGD, {"lr": 1.0, "momentum": 0.90}),
        (optim.SGD, {"lr": 0.5, "momentum": 0.95}),
        (optim.SGD, {"lr": 1.0, "momentum": 0.95}),
        (optim.SGD, {"lr": 0.5, "momentum": 0.99}),
        (optim.SGD, {"lr": 1.0, "momentum": 0.99}),
    ]

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn_with_sgd(
            pathological_curve_loss,
            xy=xy,
            lr=params["lr"],
            momentum=params["momentum"],
            n_iters=100,
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
        self.prev_g = [0] * len(self.params)

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = t.zeros_like(param)

    @t.inference_mode()
    def step(self) -> None:
        for i, (param, prev_g) in enumerate(zip(self.params, self.prev_g)):
            g = param.grad + self.weight_decay * param.data + self.momentum * prev_g
            param -= self.lr * g
            self.prev_g[i] = g.detach().clone()

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
        # SOLUTION
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum

        # self.b_prev = [0 for _ in self.params]
        # self.v_prev = [0 for _ in self.params]
        self.b_prev = [t.zeros_like(p) for p in params]
        self.v_prev = [t.zeros_like(p) for p in params]

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self):
        for i, (param, v_prev, b_prev) in enumerate(
            zip(self.params, self.v_prev, self.b_prev)
        ):
            g = param.grad + self.weight_decay * param.data
            v = (self.alpha * v_prev) + ((1 - self.alpha) * g.square())
            b = (self.momentum * b_prev) + (g / (v.sqrt() + self.eps))
            param -= self.lr * b

            self.v_prev[i] = v
            self.b_prev[i] = b

    def __repr__(self) -> str:
        return f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"


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
        """Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        """
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m_prev = [t.zeros_like(p) for p in self.params]
        self.v_prev = [t.zeros_like(p) for p in self.params]
        self.t = 1

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, (param, m_prev, v_prev) in enumerate(
            zip(self.params, self.m_prev, self.v_prev)
        ):
            g = param.grad + self.weight_decay * param.data
            m = self.b1 * m_prev + (1 - self.b1) * g
            v = self.b2 * v_prev + (1 - self.b2) * g.square()

            m_cor = m / (1 - self.b1**self.t)
            v_cor = v / (1 - self.b2**self.t)
            param -= (self.lr * m_cor) / (v_cor.sqrt() + self.eps)

            self.m_prev[i], self.v_prev[i] = m, v
        self.t += 1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.b1}, beta2={self.b2}, eps={self.eps}, weight_decay={self.lmda})"


if MAIN:
    tests.test_adam(Adam)


# %%
def opt_fn(
    fn: Callable,
    xy: t.Tensor,
    optimizer_class,
    optimizer_hyperparams: dict,
    n_iters: int = 100,
):
    """Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    """
    xy_history = []
    optim = optimizer_class([xy], **optimizer_hyperparams)
    for _ in range(n_iters):
        xy_history.append(xy.detach().clone())
        x, y = xy
        loss = fn(x, y)
        loss.backward()
        optim.step()
        optim.zero_grad()

    return t.stack(xy_history, dim=0)


# %%
if MAIN:
    points = []

    optimizer_list = [
        # (SGD, {"lr": 0.03, "momentum": 0.99}),
        # (RMSprop, {"lr": 0.02, "alpha": 0.99, "momentum": 0.8}),
        # (Adam, {"lr": 0.2, "betas": (0.9, 0.99), "weight_decay": 0.005}),
        (Adam, {"lr": 0.2}),
        (Adam, {"lr": 0.2, "betas": (0.95, 0.999)}),
        (Adam, {"lr": 0.3, "betas": (0.95, 0.999)}),
    ]

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn(
            pathological_curve_loss,
            xy=xy,
            optimizer_class=optimizer_class,
            optimizer_hyperparams=params,
        )
        points.append((xys, optimizer_class, params))

    plot_fn_with_points(pathological_curve_loss, points=points)


# %%
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


if MAIN:
    cifar_trainset, cifar_testset = get_cifar()

    imshow(
        cifar_trainset.data[:15],
        facet_col=0,
        facet_col_wrap=5,
        facet_labels=[cifar_trainset.classes[i] for i in cifar_trainset.targets[:15]],
        title="CIFAR-10 images",
        height=600,
    )
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


# %%
if MAIN:
    args = ResNetFinetuningArgs(
        trainset=cifar_trainset_small, testset=cifar_testset_small
    )
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

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

    plot_train_loss_and_test_accuracy_from_metrics(
        metrics, "Feature extraction with ResNet34"
    )


# %%
def test_resnet_on_random_input(n_inputs: int = 3):
    indices = np.random.choice(len(cifar_trainset), n_inputs).tolist()
    classes = [cifar_trainset.classes[cifar_trainset.targets[i]] for i in indices]
    imgs = cifar_trainset.data[indices]
    with t.inference_mode():
        x = t.stack(list(map(IMAGENET_TRANSFORM, imgs)))
        logits: t.Tensor = model.resnet(x)
    probs = logits.softmax(-1)
    if probs.ndim == 1:
        probs = probs.unsqueeze(0)
    for img, label, prob in zip(imgs, classes, probs):
        display(HTML(f"<h2>Classification probabilities (true class = {label})</h2>"))
        imshow(
            img,
            width=200,
            height=200,
            margin=0,
            xaxis_visible=False,
            yaxis_visible=False,
        )
        bar(
            prob,
            x=cifar_trainset.classes,
            template="ggplot2",
            width=600,
            height=400,
            labels={"x": "Classification", "y": "Probability"},
            text_auto=".2f",
            showlegend=False,
        )


if MAIN:
    test_resnet_on_random_input()
# %%
import wandb


@dataclass
class ResNetFinetuningArgsWandb(ResNetFinetuningArgs):
    use_wandb: bool = True
    run_name: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.use_wandb:
            self.logger = WandbLogger(
                save_dir=self.log_dir,
                project=self.log_name,
                name=self.run_name,
                entity="rusheb-shah",
            )


# %%
if MAIN:
    args = ResNetFinetuningArgsWandb(
        trainset=cifar_trainset_small, testset=cifar_testset_small
    )
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
sweep_config = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "max_epochs": {"values": [1, 2, 3]},
        "batch_size": {"values": [32, 64, 128, 256]},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1,
        },
    },
}
tests.test_sweep_config(sweep_config)


# # %%
wandb.init()
def train():
    args = ResNetFinetuningArgsWandb(
        trainset=cifar_trainset_small,
        testset=cifar_testset_small,
    )
    args.max_epochs = wandb.config["max_epochs"],
    args.batch_size = wandb.config["batch_size"],
    args.learning_rate = wandb.config["learning_rate"]
    model = LitResNet(args)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=args.logger,
        log_every_n_steps=args.log_every_n_steps
    )
    trainer.fit(model=model, train_dataloaders=args.trainloader, val_dataloaders=args.testloader)

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    wandb.finish()

if MAIN:
    sweep_id = wandb.sweep(sweep=sweep_config, project="day4-resnet-sweep")
    wandb.agent(sweep_id=sweep_id, function=train, count=3)

# %%
# %%
def train():
    # SOLUTION
    # Define hyperparameters, override some with values from wandb.config
    args = ResNetFinetuningArgsWandb(trainset=cifar_trainset_small, testset=cifar_testset_small)
    args.batch_size=wandb.config["batch_size"]
    args.max_epochs=wandb.config["max_epochs"]
    args.learning_rate=wandb.config["learning_rate"]

    model = LitResNet(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=args.logger,
        log_every_n_steps=args.log_every_n_steps
    )
    trainer.fit(model=model, train_dataloaders=args.trainloader, val_dataloaders=args.testloader)
    wandb.finish()

if MAIN:
    sweep_id = wandb.sweep(sweep=sweep_config, project='day4-resnet-sweep')
    wandb.agent(sweep_id=sweep_id, function=train, count=1)
# %%
wandb.finish()