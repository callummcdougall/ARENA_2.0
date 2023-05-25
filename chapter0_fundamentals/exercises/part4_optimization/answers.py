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
import einops


def pathological_curve_loss(x: t.Tensor, y: t.Tensor, angle=0):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    rotations = t.tensor(
        [
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)],
        ],
        dtype=x.dtype,
    )
    xy = t.stack([x, y], dim=-1)
    x, y = einops.einsum(rotations, xy, "a b, ... b -> a ...")
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
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.eps = eps

        self.v = [t.zeros_like(param) for param in self.params]
        self.b = [t.zeros_like(param) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for param, v, b in zip(self.params, self.v, self.b):
            grad = param.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param

            v.set_(self.alpha * v + (1 - self.alpha) * grad.square())

            if self.momentum > 0:
                b.set_(self.momentum * b + grad / (v.sqrt() + self.eps))
                param.add_(b, alpha=-self.lr)
            else:
                param.add_(grad / (v.sqrt() + self.eps), alpha=-self.lr)

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
        """Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        """
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [t.zeros_like(param) for param in self.params]
        self.v = [t.zeros_like(param) for param in self.params]

        self.t = 0

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        self.t += 1
        for param, m, v in zip(self.params, self.m, self.v):
            grad = param.grad
            assert grad is not None
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param
            m.set_(self.betas[0] * m + (1 - self.betas[0]) * grad)
            v.set_(self.betas[1] * v + (1 - self.betas[1]) * grad.square())

            mhat = m / (1 - self.betas[0] ** self.t)
            vhat = v / (1 - self.betas[1] ** self.t)

            param.add_(mhat / (vhat.sqrt() + self.eps), alpha=-self.lr)

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
        """Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        """
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [t.zeros_like(param) for param in self.params]
        self.v = [t.zeros_like(param) for param in self.params]

        self.t = 0

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        self.t += 1
        for param, m, v in zip(self.params, self.m, self.v):
            grad = param.grad
            assert grad is not None
            if self.weight_decay != 0:
                param *= 1 - self.lr * self.weight_decay

            m.set_(self.betas[0] * m + (1 - self.betas[0]) * grad)
            v.set_(self.betas[1] * v + (1 - self.betas[1]) * grad.square())

            mhat = m / (1 - self.betas[0] ** self.t)
            vhat = v / (1 - self.betas[1] ** self.t)

            param -= self.lr * mhat / (vhat.sqrt() + self.eps)

    def __repr__(self) -> str:
        return f"AdamW(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"


if MAIN:
    tests.test_adamw(AdamW)
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
    results = []
    optim = optimizer_class([xy], **optimizer_hyperparams)

    for i in range(n_iters):
        results.append(xy.tolist())
        val = fn(*xy)
        val.backward()
        optim.step()
        optim.zero_grad()

    return t.tensor(results)


# %%
if MAIN:
    points = []

    optimizer_list = [
        (SGD, {"lr": 0.03, "momentum": 0.99}),
        (RMSprop, {"lr": 0.03, "alpha": 0.99, "momentum": 0.8}),
        (Adam, {"lr": 0.3, "betas": (0.99, 0.99), "weight_decay": 0.001}),
    ]
    optimizer_list += [(AdamW, optimizer_list[-1][1])]

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn(
            pathological_curve_loss,
            xy=xy,
            optimizer_class=optimizer_class,
            optimizer_hyperparams=params,
            n_iters=300,
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


# %%

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
                save_dir=self.log_dir, project=self.log_name, name=self.run_name
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
if MAIN:
    # YOUR CODE HERE - fill `sweep_config`
    sweep_config = dict(
        method="random",
        metric=dict(name="accuracy", goal="maximize"),
        parameters=dict(
            batch_size=dict(values=[32, 64, 128, 256]),
            max_epochs=dict(min=1, max=4),
            learning_rate=dict(max=0.1, min=0.0001, distribution="log_uniform_values"),
        ),
    )
    # FLAT SOLUTION END

    tests.test_sweep_config(sweep_config)
# %%
def train() -> None:
    args = ResNetFinetuningArgsWandb(
        trainset=cifar_trainset_small, testset=cifar_testset_small
    )
    args.batch_size = wandb.config["batch_size"]
    args.max_epochs = wandb.config["max_epochs"]
    args.learning_rate = wandb.config["learning_rate"]

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
if MAIN:
    wandb.init()
    sweep_id = wandb.sweep(sweep=sweep_config, project='day4-resnet-sweep')
    wandb.agent(sweep_id=sweep_id, function=train, count=3)
# %%