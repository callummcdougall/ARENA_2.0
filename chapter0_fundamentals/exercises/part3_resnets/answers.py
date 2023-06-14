# %%
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange, repeat
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
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from part2_cnns.solutions import get_mnist, Linear, Conv2d, Flatten, ReLU, MaxPool2d
from part3_resnets.utils import print_param_count
import part3_resnets.tests as tests
from plotly_utils import line, plot_train_loss_and_test_accuracy_from_metrics

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


# %%
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1
        )
        self.relu1 = ReLU()
        self.mpool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1
        )
        self.relu2 = ReLU()
        self.mpool2 = MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flatten = Flatten()
        self.linear1 = Linear(in_features=64 * 7 * 7, out_features=128)
        self.relu3 = ReLU()
        self.linear2 = Linear(in_features=128, out_features=10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.mpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.mpool2(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)

        return x


if MAIN:
    model = ConvNet()
    print(model)

    summary = torchinfo.summary(model, input_size=(1, 1, 28, 28))
    print(summary)
# %%
if MAIN:
    MNIST_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )


def get_mnist(subset: int = 1):
    """Returns MNIST training data, sampled by the frequency given in `subset`."""
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=MNIST_TRANSFORM
    )
    mnist_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=MNIST_TRANSFORM
    )

    if subset > 1:
        mnist_trainset = Subset(
            mnist_trainset, indices=range(0, len(mnist_trainset), subset)
        )
        mnist_testset = Subset(
            mnist_testset, indices=range(0, len(mnist_testset), subset)
        )

    return mnist_trainset, mnist_testset


if MAIN:
    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# %%
if MAIN:
    model = ConvNet().to(device)

    batch_size = 64
    epochs = 3

    mnist_trainset, _ = get_mnist(subset=10)
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
            loss_list.append(
                loss.item()
            )  # .item() converts single-elem tensor to scalar
if MAIN:
    line(
        loss_list,
        yaxis_range=[0, max(loss_list) + 0.1],
        labels={"x": "Num batches seen", "y": "Cross entropy loss"},
        title="ConvNet training on MNIST",
        width=700,
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
        self.trainset, self.testset = get_mnist(subset=10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Here you should define the forward pass of your model.
        """
        return self.convnet(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> t.Tensor:
        """
        Here you compute and return the training loss and some additional metrics for e.g. the progress bar or logger.
        """
        imgs, labels = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = t.optim.Adam(self.parameters())
        return optimizer

    def train_dataloader(self):
        """
        Return the training dataloader.
        """
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)


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
    line(
        metrics["train_loss"].values,
        x=metrics["step"].values,
        yaxis_range=[0, metrics["train_loss"].max() + 0.1],
        labels={"x": "Batches seen", "y": "Cross entropy loss"},
        title="ConvNet training on MNIST",
        width=800,
        hovermode="x unified",
        template="ggplot2",  # alternative aesthetic for your plots (-:
    )


# %%
@dataclass
class ConvNetTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = ConvNetTrainingArgs(batch_size=128).
    """

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

    trainer = pl.Trainer(max_epochs=args.max_epochs, logger=logger, log_every_n_steps=1)
    trainer.fit(model=model)


# %%
class LitConvNetTest(pl.LightningModule):
    def __init__(self, args: ConvNetTrainingArgs):
        super().__init__()
        self.convnet = ConvNet()
        self.args = args
        self.trainset, self.testset = get_mnist(subset=args.sample)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.convnet(x)

    def _shared_train_val_step(
        self, batch: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        imgs, labels = batch
        logits = self.convnet(imgs)
        return logits, labels

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        logits, labels = self._shared_train_val_step(batch)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        logits, labels = self._shared_train_val_step(batch)
        loss = F.cross_entropy(logits, labels)
        accuracy = ((t.argmax(logits, dim=1) == labels) * 1.0).mean()
        self.log("val_loss", loss)
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
        log_every_n_steps=args.log_every_n_steps,
    )
    trainer.validate(model=model)
    trainer.fit(model=model)


# %%
if MAIN:
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

    plot_train_loss_and_test_accuracy_from_metrics(
        metrics, "Training ConvNet on MNIST data"
    )


# %%
class Sequential(nn.Module):
    _modules: Dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            modules = modules[0].items()
        else:
            modules = enumerate(modules)
        for index, mod in modules:
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules)  # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules)  # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for mod in self._modules.values():
            x = mod(x)
        return x


if MAIN:
    # seq = Sequential(OrderedDict([
    #     nn.Linear(10, 20),
    #     nn.ReLU(),
    #     nn.Linear(20, 30)
    # ]))
    # s = OrderedDict([
    #     ("linear1", nn.Linear(10, 20)),
    #     ("relu", nn.ReLU()),
    #     ("linear2", nn.Linear(20, 30))
    # ])
    # seq = Sequential(s)
    seq = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 30))


# %%
class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""]  # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(t.ones(num_features))
        self.register_buffer("running_mean", t.zeros(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))
        self.register_buffer("running_var", t.zeros(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = t.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var.squeeze()
            self.num_batches_tracked += 1
        else:
            # running_mean = x.mean(dim=(0, 2, 3), keepdim=False)
            # running_var = x.mean(dim=(0, 2, 3), keepdim=False)
            mean = repeat(self.running_mean, "c -> 1 c 1 1")
            var = repeat(self.running_var, "c -> 1 c 1 1")
        # left = ((x - mean) / (var + self.eps).sqrt())
        # b, c, h, w = x.shape
        weight = rearrange(self.weight, "channels -> 1 channels 1 1")
        bias = rearrange(self.bias, "channels -> 1 channels 1 1")
        # left = einsum(left, self.weight, 'b c h w, c -> b c h w')
        # bias = repeat(self.bias, 'c -> b c h w', b=b, h=h, w=w)
        return ((x - mean) / (var + self.eps).sqrt()) * weight + bias

    def extra_repr(self) -> str:
        pass


# %%
class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return x.mean(dim=(2, 3))


if MAIN:
    tests.test_batchnorm2d_module(BatchNorm2d)
    tests.test_batchnorm2d_forward(BatchNorm2d)
    tests.test_batchnorm2d_running_mean(BatchNorm2d)


# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        """
        super().__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride
        self.left_branch = nn.Sequential(
            Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_feats),
        )

        if first_stride == 1:
            self.right_branch = nn.Identity()
        elif first_stride > 1:
            self.right_branch = nn.Sequential(
                Conv2d(
                    in_feats, out_feats, kernel_size=1, stride=first_stride, padding=0
                ),
                BatchNorm2d(out_feats),
            )

        self.relu = ReLU()

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        """
        # if self.first_stride == 1:
        #     left_x = (x, "b inc h w -> b ouc h w", ouc = self.out_feats)
        # else:
        left_x = self.left_branch(x)
        right_x = self.right_branch(x)
        print(left_x.shape)
        print(right_x.shape)
        x = left_x + right_x
        x = self.relu(x)
        return x


# %%
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride."""
        super().__init__()
        self.group = nn.Sequential(
            *(
                [ResidualBlock(in_feats, out_feats, first_stride)]
                + [ResidualBlock(out_feats, out_feats, 1) for i in range(n_blocks - 1)]
            )
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        return self.group(x)


# %%
if MAIN:
    rb = ResidualBlock(4, 3, 1)

    # print(torchinfo.summary(rb, input_size=(1, 3, 64, 64)))
# %%
# class ResidualBlock(nn.Module):
#     def __init__(self, in_feats: int, out_feats: int, first_stride=1):
#         """
#         A single residual block with optional downsampling.

#         For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

#         If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
#         """
#         super().__init__()
#         # SOLUTION

#         self.left = Sequential(
#             Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
#             BatchNorm2d(out_feats),
#             ReLU(),
#             Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(out_feats),
#         )

#         if first_stride > 1:
#             self.right = Sequential(
#                 Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride),
#                 BatchNorm2d(out_feats),
#             )
#         else:
#             self.right = nn.Identity()

#         self.relu = ReLU()

#     def forward(self, x: t.Tensor) -> t.Tensor:
#         """
#         Compute the forward pass.

#         x: shape (batch, in_feats, height, width)

#         Return: shape (batch, out_feats, height / stride, width / stride)

#         If no downsampling block is present, the addition should just add the left branch's output to the input.
#         """
#         # SOLUTION
#         x_left = self.left(x)
#         x_right = self.right(x)
#         return self.relu(x_left + x_right)


# %%
# class BlockGroup(nn.Module):
#     def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
#         """An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride."""
#         super().__init__()
#         # SOLUTION

#         blocks = [ResidualBlock(in_feats, out_feats, first_stride)] + [
#             ResidualBlock(out_feats, out_feats) for n in range(n_blocks - 1)
#         ]
#         self.blocks = nn.Sequential(*blocks)

#     def forward(self, x: t.Tensor) -> t.Tensor:
#         """
#         Compute the forward pass.

#         x: shape (batch, in_feats, height, width)

#         Return: shape (batch, out_feats, height / first_stride, width / first_stride)
#         """
#         # SOLUTION
#         return self.blocks(x)


# %%
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        # (self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1)
        # 64 -> [64, 128, 256, 512]
        super().__init__()
        in_features_per_group = [64] + out_features_per_group[:-1]
        z = zip(
            n_blocks_per_group,
            in_features_per_group,
            out_features_per_group,
            first_strides_per_group,
        )
        self.resnet = nn.Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(num_features=64),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            *[
                BlockGroup(n_blocks, in_feats, out_feats, first_stride)
                for n_blocks, in_feats, out_feats, first_stride in z
            ],
            AveragePool(),
            # Flatten(),
            Linear(out_features_per_group[-1], n_classes),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        """
        return self.resnet(x)


# %%
if MAIN:
    my_resnet = ResNet34()


    # resnet = models.resnet34()
    # print(torchinfo.summary(my_resnet, input_size=(1, 3, 64, 64), depth=4))
# %%
def copy_weights(
    my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet
) -> ResNet34:
    """Copy over the weights of `pretrained_resnet` to your resnet."""

    # Get the state dictionaries for each model, check they have the same number of parameters & buffers
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters / buffers to their values in the pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(
            mydict.items(), pretraineddict.items()
        )
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
    images[0]
# %%
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# %%
def prepare_data(images: List[Image.Image]) -> t.Tensor:
    """
    Return: shape (batch=len(images), num_channels=3,
    height=224, width=224)
    """
    # result = t.zeros((len(images), 3, 224, 224))
    # for i, image in enumerate(images):
    #     result[i] = IMAGENET_TRANSFORM(image)
    # return result
    return t.stack([IMAGENET_TRANSFORM(image) for image in images])


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

# Check your predictions match the pretrained model's
# %%
if MAIN:
    my_predictions = predict(my_resnet, prepared_images)
    pretrained_predictions = predict(pretrained_resnet, prepared_images)
    assert all(my_predictions == pretrained_predictions)

# Print out your predictions, next to the corresponding images
# %%
if MAIN:
    for img, label in zip(images, my_predictions):
        print(f"Class {label}: {imagenet_labels[label]}")
        display(img)
        print()


# %%
def get_resnet_for_feature_extraction(n_classes: int) -> ResNet34:
    """
    Creates a ResNet34 instance,
    replaces its final linear layer with a classifier
    for `n_classes` classes, and
    freezes all weights except the ones in this layer.

    Returns the ResNet model.
    """
    my_resnet = ResNet34()
    pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    my_resnet = copy_weights(my_resnet, pretrained_resnet)
    my_resnet.requires_grad_(False)
    in_features = my_resnet.resnet[-1].in_features
    my_resnet.resnet[-1] = Linear(in_features=in_features, out_features=n_classes)
    return my_resnet


if MAIN:
    tests.test_get_resnet_for_feature_extraction(get_resnet_for_feature_extraction)
# %%


def get_cifar(subset: int):
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


@dataclass
class ResNetTrainingArgs:
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
        self.trainset, self.testset = get_cifar(args.subset)

        self.resnet = get_resnet_for_feature_extraction(n_classes=args.n_classes)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.resnet(x)

    def _shared_train_val_step(
        self, batch: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
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
        accuracy = ((logits.argmax(dim=1) == labels) * 1.0).mean()
        self.log("accuracy", accuracy)

    def configure_optimizers(self):
        return self.args.optimizer(
            self.resnet.resnet[-1].parameters(), lr=self.args.learning_rate
        )

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.testset, batch_size=self.args.batch_size)


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

    plot_train_loss_and_test_accuracy_from_metrics(
        metrics, "Feature extraction with ResNet34"
    )

# %%
