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

# %%
if MAIN:
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

# %%

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.max1 = MaxPool2d(2, stride=2, padding=0)
        self.conv2 = Conv2d(32,64, kernel_size=3, padding=1, stride=1)
        self.relu2 = ReLU()
        self.max2 = MaxPool2d(2, stride=2, padding=0)
        self.flatten = Flatten()
        self.linear1 = Linear(3136, 128)
        self.relu3 = ReLU()
        self.linear2 = Linear(128,10)


    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
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

# %%
class LitConvNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.convnet = ConvNet()

    def training_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> t.Tensor:
        '''
        Here you compute and return the training loss and some additional metrics for e.g. the progress bar or logger.
        '''
        imgs, labels = batch
        logits = self.convnet(imgs)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        '''
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        '''
        optimizer = t.optim.Adam(self.parameters())
        return optimizer
    
# %%

# Set batch size

if MAIN:
    batch_size = 64
    max_epochs = 3

    # Create the model & training system
    model = LitConvNet()

    # Get dataloaders
    trainset, testset = get_mnist(subset = 10)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    # Get a logger, to record metrics during training
    logger = CSVLogger(save_dir=os.getcwd() + "/logs", name="day4-convenet")

    # Train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(model=model, train_dataloaders=trainloader)

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
data_augmentation_transform = transforms.Compose([
       transforms.RandomRotation(degrees=15),
       transforms.RandomResizedCrop(size=28, scale=(0.8, 1.2)),
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomVerticalFlip(p=0.5),
       transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_mnist_augmented(subset: int = 1, train_transform=None, test_transform=None):
    if train_transform is None:
        train_transform = MNIST_TRANSFORM
    if test_transform is None:
        test_transform = MNIST_TRANSFORM
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=data_augmentation_transform)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)
    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))
    return mnist_trainset, mnist_testset
# %%
@dataclass
class ConvNetTrainingArgs():
    '''
    Defining this class implicitly creates an __init__ method, which sets arguments as 
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = ConvNetTrainingArgs(batch_size=128).
    '''
    batch_size: int = 128
    max_epochs: int = 10
    optimizer: t.optim.Optimizer = t.optim.Adam
    learning_rate: float = 1e-3
    log_dir: str = os.getcwd() + "/logs"
    log_name: str = "day4-convenet"
    log_every_n_steps: int = 1
    sample: int = 10
    augmented: bool = True

    def __post_init__(self):
        '''
        This code runs after the class is instantiated. It can reference things like
        self.sample, which are defined in the __init__ block.
        '''

        if self.augmented:
            trainset, testset = get_mnist_augmented(subset=self.sample)
        else:
            trainset, testset = get_mnist(subset=self.sample)
        self.trainloader = DataLoader(trainset, shuffle=True, batch_size=self.batch_size)
        self.testloader = DataLoader(testset, shuffle=False, batch_size=self.batch_size)
        self.logger = CSVLogger(save_dir=self.log_dir, name=self.log_name)


class LitConvNet(pl.LightningModule):
    def __init__(self, args: ConvNetTrainingArgs):
        super().__init__()
        self.convnet = ConvNet()
        self.args = args

    def training_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> t.Tensor:
        imgs, labels = batch
        logits = self.convnet(imgs)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> t.Tensor:
        imgs, labels = batch
        logits = self.convnet(imgs)
        acc = t.mean((labels == t.argmax(logits, dim=-1)), dtype=t.float16)
        self.log("accuracy", acc)

    def configure_optimizers(self):
        optimizer = self.args.optimizer(self.parameters(), lr=self.args.learning_rate)
        return optimizer



if MAIN:
    args = ConvNetTrainingArgs()
    model = LitConvNet(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=args.logger,
        log_every_n_steps=1
    )
    trainer.fit(model=model, train_dataloaders=args.trainloader, val_dataloaders=args.testloader)

# %%
if MAIN:
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    plot_train_loss_and_test_accuracy_from_metrics(metrics, "Training ConvNet on MNIST data")



##### Part 2

# %%

class Sequential(nn.Module):
    _modules: Dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        if index < 0: index += len(self._modules) # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        if index < 0: index += len(self._modules) # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for mod in self._modules.values():
            x = mod(x)
        return x

import einops



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
        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))
        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked",t.tensor(0))
        self.epsilon = eps
        self.momentum = momentum

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        if self.training:
            batch_mean = t.mean(x, dim=(0,2,3), keepdim=True)
            batch_var = t.var(x, dim=(0,2,3), unbiased=False, keepdim=True)
            y = (x - batch_mean)/t.sqrt(batch_var+self.epsilon)
            running_mean = einops.rearrange(self.running_mean, "b -> 1 b 1 1")
            running_var = einops.rearrange(self.running_var, "b -> 1 b 1 1")
            self.running_mean = t.squeeze(running_mean * (1-self.momentum) + self.momentum*batch_mean)
            self.running_var = t.squeeze(running_var * (1-self.momentum) + self.momentum*batch_var)
            self.num_batches_tracked += 1

        else:
            running_mean = einops.rearrange(self.running_mean, "b -> 1 b 1 1")
            running_var = einops.rearrange(self.running_var, "b -> 1 b 1 1")
            y = (x - running_mean)/t.sqrt(running_var+self.epsilon)
        gamma = einops.rearrange(self.weight, "b -> 1 b 1 1")
        beta = einops.rearrange(self.bias, "b -> 1 b 1 1")
        return y*gamma + beta

    def extra_repr(self) -> str:
        pass


if MAIN:
    tests.test_batchnorm2d_module(BatchNorm2d)
    tests.test_batchnorm2d_forward(BatchNorm2d)
    tests.test_batchnorm2d_running_mean(BatchNorm2d)


class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return einops.reduce(x, "b c h w -> b c", "mean")
    


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        # SOLUTION

        if first_stride == 1:
            assert(in_feats == out_feats)

        self.left = Sequential(
                    Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
                    BatchNorm2d(out_feats),
                    ReLU(),
                    Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
                    BatchNorm2d(out_feats)
                )


        if first_stride > 1:
            self.right = Sequential(
                Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride, padding=0),
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
        # SOLUTION
        x_left = self.left(x)
        x_right = self.right(x)
        return self.relu(x_left + x_right)
# class ResidualBlock(nn.Module):
#     def __init__(self, in_feats: int, out_feats: int, first_stride=1):
#         '''
#         A single residual block with optional downsampling.

#         For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

#         If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
#         '''
#         super().__init__()
#         self.relu = ReLU()
#         self.left_branch = Sequential(
#             Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
#             BatchNorm2d(out_feats),
#             ReLU(),
#             Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(out_feats)
#         )
#         if first_stride == 1:
#             self.right_branch = nn.Identity()
#         else:
#             self.right_branch = Sequential(
#                 Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride),
#                 BatchNorm2d(out_feats)
#             )

#     def forward(self, x: t.Tensor) -> t.Tensor:
#         '''
#         Compute the forward pass.

#         x: shape (batch, in_feats, height, width)

#         Return: shape (batch, out_feats, height / stride, width / stride)

#         If no downsampling block is present, the addition should just add the left branch's output to the input.
#         '''
#         x1 = self.left_branch(x)
#         x2 = self.right_branch(x)
#         return self.relu(x1 + x2)
    
    
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()
        self.n_blocks = n_blocks
        self.first_res = ResidualBlock(in_feats, out_feats, first_stride=first_stride)
        self.res_blocks = Sequential(*[ResidualBlock(out_feats, out_feats) for i in range(n_blocks - 1)])
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        x = self.first_res(x)
        x = self.res_blocks(x)
        return x
    
    
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        in_features_per_group = [64] + out_features_per_group[:-1]
        zipped = zip(n_blocks_per_group, in_features_per_group, out_features_per_group, first_strides_per_group)
       

       
        self.resnet34 = Sequential(
            Conv2d(3, 64, kernel_size = 7, stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            Sequential(*[BlockGroup(n_blocks, in_feats, out_feats, first_stride) 
                         for (n_blocks, in_feats, out_feats, first_stride) in zipped]),
            AveragePool(),
            Flatten(),
            Linear(out_features_per_group[-1], n_classes),
        )
       
        
        # self.conv1 = Conv2d(3, 64, kernel_size = 7, stride=2, padding=1)
        # self.bn = BatchNorm2d(64)
        # self.relu = ReLU()
        # self.maxpool = MaxPool2d(kernel_size=3, stride=2)
        # zipped = zip(n_blocks_per_group, in_features_per_group, out_features_per_group, first_strides_per_group)
        # self.blockgroups = Sequential(*[BlockGroup(n_blocks, in_feats, out_feats, first_stride) for (n_blocks, in_feats, out_feats, first_stride) in zipped])
        # self.avgpool = AveragePool()
        # self.flat = Flatten()
        # self.linear = Linear(10,1000)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        x = self.resnet34(x)
        return x


if MAIN:
    my_resnet = ResNet34()
    
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
    print_param_count(my_resnet, pretrained_resnet)


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
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def prepare_data(images: List[Image.Image]) -> t.Tensor:
    '''
    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    '''
    tensors = [IMAGENET_TRANSFORM(img) for img in images]
    return t.stack(tensors, dim=0)


if MAIN:
    prepared_images = prepare_data(images)

    assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)

def predict(model, images):
    logits: t.Tensor = model(images)
    return logits.argmax(dim=1)


# %%
#one = (my_resnet(prepared_images[-1].unsqueeze(dim=0)) - pretrained_resnet(prepared_images[-1].unsqueeze(dim=0)))
their_logits = pretrained_resnet(prepared_images[-1].unsqueeze(dim=0))
my_logits = my_resnet(prepared_images[-1].unsqueeze(dim=0))
# %%
if MAIN:
    my_predictions = predict(my_resnet, prepared_images)
    pretrained_predictions = predict(pretrained_resnet, prepared_images)
    assert all(my_predictions == pretrained_predictions)
# %%
if MAIN:
    with open(section_dir / "imagenet_labels.json") as f:
        imagenet_labels = list(json.load(f).values())
        
if MAIN:
    for img, label in zip(images, my_predictions):
        print(f"Class {label}: {imagenet_labels[label]}")
        display(img)
        print()
# %%


def get_resnet_for_feature_extraction(n_classes: int) -> ResNet34:
    '''
    Creates a ResNet34 instance, replaces its final linear layer with a classifier
    for `n_classes` classes, and freezes all weights except the ones in this layer.

    Returns the ResNet model.
    '''
    my_resnet = ResNet34()
    pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    my_resnet = copy_weights(my_resnet, pretrained_resnet)
    
    my_resnet.requires_grad_(False)
    
    my_resnet.state_dict()['resnet34.7.weight'] = nn.Linear(512, n_classes)


if MAIN:
    tests.test_get_resnet_for_feature_extraction(get_resnet_for_feature_extraction)
    