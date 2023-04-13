# %%

import torch as t
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from typing import Callable, Iterable, Tuple
from tqdm import tqdm
import plotly.express as px
from dataclasses import dataclass
import time
import wandb
import functools

import part5_optimization_utils as utils
import part5_optimization_tests as tests

from part4_resnets_solutions import ResNet34

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"





# %% SECTION 1: OPTIMIZERS

def rosenbrocks_banana(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1


if MAIN:
    x_range = [-2, 2]
    y_range = [-1, 3]
    fig = utils.plot_fn(rosenbrocks_banana, x_range, y_range, log_scale=True)

# %%

def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
    '''
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum, 

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    '''
    assert xy.requires_grad

    xys = t.zeros((n_iters, 2))
    optimizer = optim.SGD([xy], lr=lr, momentum=momentum)

    for i in range(n_iters):
        xys[i] = xy.detach()
        out = fn(xy[0], xy[1])
        out.backward()
        optimizer.step()
        optimizer.zero_grad()

    return xys


if MAIN:
    xy = t.tensor([-1.5, 2.5], requires_grad=True)
    x_range = [-2, 2]
    y_range = [-1, 3]

    fig = utils.plot_optimization_sgd(opt_fn_with_sgd, rosenbrocks_banana, xy, x_range, y_range, lr=0.001, momentum=0.98, show_min=True)

    fig.show()

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
        self.params = list(params)
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
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop

        '''
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.mu = momentum
        self.lmda = weight_decay
        self.alpha = alpha

        self.gs = [t.zeros_like(p) for p in self.params]
        self.bs = [t.zeros_like(p) for p in self.params]
        self.vs = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, (p, g, b, v) in enumerate(zip(self.params, self.gs, self.bs, self.vs)):
            new_g = p.grad
            if self.lmda != 0:
                new_g = new_g + self.lmda * p
            self.gs[i] = new_g
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
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        '''
        self.params = list(params)
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

def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_kwargs: dict, n_iters: int = 100):
    '''Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    '''
    assert xy.requires_grad

    xys = t.zeros((n_iters, 2))
    optimizer = optimizer_class([xy], **optimizer_kwargs)

    for i in range(n_iters):
        xys[i] = xy.detach()
        out = fn(xy[0], xy[1])
        out.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return xys

# Implementation of SGD which works with parameter groups

# %%

if MAIN:
    xy = t.tensor([-1.5, 2.5], requires_grad=True)
    x_range = [-2, 2]
    y_range = [-1, 3]
    optimizers = [
        (SGD, dict(lr=1e-3, momentum=0.98)),
        (SGD, dict(lr=5e-4, momentum=0.98)),
        (Adam, dict(lr=0.15, betas=(0.85, 0.85))),
    ]

    fig = utils.plot_optimization(opt_fn, rosenbrocks_banana, xy, optimizers, x_range, y_range, show_min=True)

    fig.show()

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











# %% SECTION 2: WEIGHTS AND BIASES

def get_cifar10(subset: int = 1):
    '''Returns CIFAR training data, sampled by the frequency given in `subset`.'''
    cifar_mean = [0.485, 0.456, 0.406]
    cifar_std = [0.229, 0.224, 0.225]

    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)

    if subset > 1:
        cifar_trainset = Subset(cifar_trainset, range(0, len(cifar_trainset), subset))
        cifar_testset = Subset(cifar_testset, range(0, len(cifar_testset), subset))

    return cifar_trainset, cifar_testset

if MAIN:
    cifar_trainset, cifar_testset = get_cifar10(subset=5)
    utils.show_cifar_images(cifar_trainset.dataset, rows=3, cols=5)

# %%

@dataclass
class ResNetTrainingArgs():
    trainset: datasets.VisionDataset
    testset: datasets.VisionDataset
    epochs: int = 3
    batch_size: int = 512
    loss_fn: Callable[..., t.Tensor] = nn.CrossEntropyLoss()
    optimizer: Callable[..., t.optim.Optimizer] = t.optim.Adam
    optimizer_args: Tuple = ()
    device: str = "cuda" if t.cuda.is_available() else "cpu"
    filename_save_model: str = "models/part5_resnet.pt"


def train_resnet(args: ResNetTrainingArgs) -> Tuple[list, list]:
    '''
    Defines and trains a ResNet.

    This is a pretty standard training function, containing a test set for evaluations, plus a progress bar.
    '''

    trainloader = DataLoader(args.trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(args.testset, batch_size=args.batch_size, shuffle=True)

    model = ResNet34().to(args.device).train()
    optimizer = args.optimizer(model.parameters(), *args.optimizer_args)

    loss_list = []
    accuracy_list = []

    for epoch in range(args.epochs):

        progress_bar = tqdm(trainloader)
        for (imgs, labels) in progress_bar:

            imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            probs = model(imgs)
            loss = args.loss_fn(probs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}, Loss = {loss:.3f}")
            
            loss_list.append(loss.item())

        with t.inference_mode():

            accuracy = 0
            total = 0

            for (imgs, labels) in testloader:

                imgs = imgs.to(args.device)
                labels = labels.to(args.device)

                probs = model(imgs)
                predictions = probs.argmax(-1)
                accuracy += (predictions == labels).sum().item()
                total += imgs.size(0)

            accuracy_list.append(accuracy / total)

        print(f"Train loss = {loss:.6f}, Accuracy = {accuracy}/{total}")
    
    print(f"\nSaving model to: {args.filename_save_model}")
    t.save(model, args.filename_save_model)
    return loss_list, accuracy_list


if MAIN:
    args = ResNetTrainingArgs(cifar_trainset, cifar_testset)
    loss_list, accuracy_list = train_resnet(args)

    px.line(
        y=loss_list, x=range(0, len(loss_list)*args.batch_size, args.batch_size),
        title="Training loss for CNN, on MNIST data",
        labels={"x": "Num images seen", "y": "Cross entropy loss"}, template="ggplot2",
        height=400, width=600
    ).show()

    px.line(
        y=accuracy_list, x=range(1, len(accuracy_list)+1),
        title="Training accuracy for CNN, on MNIST data",
        labels={"x": "Epoch", "y": "Accuracy"}, template="seaborn",
        height=400, width=600
    ).show()

# %%

def train_resnet_wandb(args: ResNetTrainingArgs) -> None:
    '''
    Defines and trains a ResNet.

    This is a pretty standard training function, containing weights and biases logging, a test set for evaluations, plus a progress bar.
    '''

    start_time = time.time()
    examples_seen = 0

    config_dict = args.__dict__
    wandb.init(project="part4_model_resnet", config=config_dict)

    trainloader = DataLoader(args.trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(args.testset, batch_size=args.batch_size, shuffle=True)

    model = ResNet34().to(args.device).train()
    optimizer = args.optimizer(model.parameters(), *args.optimizer_args)

    wandb.watch(model, criterion=args.loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(args.epochs):

        progress_bar = tqdm(trainloader)
        for (imgs, labels) in progress_bar:

            imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            probs = model(imgs)
            loss = args.loss_fn(probs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}, Loss = {loss:.3f}")
            
            examples_seen += imgs.size(0)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)

        with t.inference_mode():

            accuracy = 0
            total = 0

            for (imgs, labels) in testloader:

                imgs = imgs.to(args.device)
                labels = labels.to(args.device)

                probs = model(imgs)
                predictions = probs.argmax(-1)
                accuracy += (predictions == labels).sum().item()
                total += imgs.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)
    wandb.finish()


if MAIN:
    args = ResNetTrainingArgs(cifar_trainset, cifar_testset)
    train_resnet_wandb(args)

# %%

def train_resnet_wandb_sweep(args: ResNetTrainingArgs) -> None:
    '''
    Defines and trains a ResNet.

    This is a pretty standard training function, containing weights and biases logging, a test set for evaluations, plus a progress bar.
    '''

    start_time = time.time()
    examples_seen = 0

    # This is the only part of the function that changes
    wandb.init()
    args.epochs = wandb.config.epochs
    args.batch_size = wandb.config.batch_size
    args.optimizer_args = (wandb.config.lr,)

    trainloader = DataLoader(args.trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(args.testset, batch_size=args.batch_size, shuffle=True)

    model = ResNet34().to(args.device).train()
    optimizer = args.optimizer(model.parameters(), *args.optimizer_args)

    wandb.watch(model, criterion=args.loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(args.epochs):

        progress_bar = tqdm(trainloader)
        for (imgs, labels) in progress_bar:

            imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            probs = model(imgs)
            loss = args.loss_fn(probs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}, Loss = {loss:.3f}")
            
            examples_seen += imgs.size(0)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)

        with t.inference_mode():

            accuracy = 0
            total = 0

            for (imgs, labels) in testloader:

                imgs = imgs.to(args.device)
                labels = labels.to(args.device)

                probs = model(imgs)
                predictions = probs.argmax(-1)
                accuracy += (predictions == labels).sum().item()
                total += imgs.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)
    wandb.finish()

sweep_config = dict(
    method = 'random',
    name = 'resnet_sweep',
    metric = dict(name = 'test_accuracy', goal = 'maximize'),
    parameters = dict( 
        batch_size = dict(values = [64, 128, 256, 512]),
        epochs = dict(min = 1, max = 3),
        lr = dict(max = 0.1, min = 0.0001, distribution = 'log_uniform_values'),
    )
)

if MAIN:
    # Define a training function that takes no arguments (this is necessary for doing sweeps)
    train = functools.partial(
        train_resnet_wandb_sweep, 
        args=ResNetTrainingArgs(cifar_trainset, cifar_testset)
    )

    # Run the sweep
    wandb.agent(
        sweep_id=wandb.sweep(sweep=sweep_config, project='resnet_sweep'), 
        function=train, 
        count=5
    )

# %%