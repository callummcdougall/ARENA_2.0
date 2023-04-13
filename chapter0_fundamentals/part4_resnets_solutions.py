# %%

import torch as t
from torch import nn
from einops import rearrange
from dataclasses import dataclass
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import List, Callable, Tuple
from PIL import Image
import plotly.express as px
from IPython.display import display
import torchinfo
import json

from part2_cnns_solutions import ReLU, Conv2d, MaxPool2d, Flatten, Linear

import part4_resnets_tests as tests
import part4_resnets_utils as utils

MAIN = __name__ == "__main__"






# %% PART 1: BUILDING & TRAINING A CNN
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.flatten = Flatten()
        self.fc1 = Linear(in_features=3136, out_features=128)
        self.fc2 = Linear(in_features=128, out_features=10)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.fc2(self.fc1(self.flatten(x)))
        return x

if MAIN:
    model = ConvNet()
    print(model)

# %%

if MAIN:
    summary = torchinfo.summary(model, input_size=(1, 1, 28, 28))
    print(summary)

# %%

def get_mnist(subset: int = 1):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=mnist_transform)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset

if MAIN:
    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=True)

# %%

@dataclass
class ConvNetTrainingArgs():
    trainset: datasets.VisionDataset
    testset: datasets.VisionDataset

    epochs: int = 3
    batch_size: int = 512
    loss_fn: Callable[..., t.Tensor] = nn.CrossEntropyLoss()
    optimizer: Callable[..., t.optim.Optimizer] = t.optim.Adam
    optimizer_args: Tuple = ()
    device: str = "cuda" if t.cuda.is_available() else "cpu"
    filename_save_model: str = "models/part4_convnet.pt"

if MAIN:
    args = ConvNetTrainingArgs(mnist_trainset, mnist_testset)
    print(args)

# %%

def train_convnet(args: ConvNetTrainingArgs) -> Tuple[list, list]:
    '''
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.
    
    Returns tuple of (loss_list, accuracy_list), where accuracy_list contains the fraction of accurate classifications on the test set, at the end of each epoch.
    '''

    trainloader = DataLoader(args.trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(args.testset, batch_size=args.batch_size, shuffle=True)

    model = ConvNet().to(args.device).train()
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
            
            loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}, Loss = {loss:.3f}")
        
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

            accuracy_list.append(accuracy/total)
            
        print(f"Train loss = {loss:.6f}, Accuracy = {accuracy}/{total}")
    
    print(f"\nSaving model to: {args.filename_save_model}")
    t.save(model, args.filename_save_model)
    return loss_list, accuracy_list


if MAIN:
    loss_list, accuracy_list = train_convnet(args)

# %%

if MAIN:
    px.line(
        y=loss_list, x=range(0, len(loss_list)*args.batch_size, args.batch_size),
        labels={"y": "Cross entropy loss", "x": "Num images seen"}, title="MNIST training curve (cross entropy loss)", template="ggplot2"
    ).update_layout(
        showlegend=False, yaxis_range=[0, max(loss_list)*1.1], height=400, width=600
    ).show()
    px.line(
        y=accuracy_list, x=range(1, len(accuracy_list)+1),
        title="Training accuracy for CNN, on MNIST data",
        labels={"x": "Epoch", "y": "Accuracy"}, template="seaborn",
        height=400, width=600
    ).show()






# %% SECTION 2: ASSEMBLING RESNET

class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()
        for i, mod in enumerate(modules):
            self.add_module(str(i), mod)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for mod in self._modules.values():
            if mod is not None:
                x = mod(x)
        return x

# %%

class BatchNorm2d(nn.Module):
    running_mean: t.Tensor         # shape: (num_features,)
    running_var: t.Tensor          # shape: (num_features,)
    num_batches_tracked: t.Tensor  # shape: ()

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))
        
        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        
        # Calculating mean and var over all dims except for the channel dim
        if self.training:
            # Using keepdim=True so we don't have to worry about broadasting them with x at the end
            mean = t.mean(x, dim=(0, 2, 3), keepdim=True)
            var = t.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
            # Updating running mean and variance, in line with PyTorch documentation
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            self.num_batches_tracked += 1
        else:
            mean = rearrange(self.running_mean, "channels -> 1 channels 1 1")
            var = rearrange(self.running_var, "channels -> 1 channels 1 1")
        
        # Rearranging these so they can be broadcasted (although there are other ways you could do this)
        weight = rearrange(self.weight, "channels -> 1 channels 1 1")
        bias = rearrange(self.bias, "channels -> 1 channels 1 1")
        
        return ((x - mean) / t.sqrt(var + self.eps)) * weight + bias

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])


if MAIN:
    tests.test_batchnorm2d_module(BatchNorm2d)
    tests.test_batchnorm2d_forward(BatchNorm2d)
    tests.test_batchnorm2d_running_mean(BatchNorm2d)

# %%

class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return t.mean(x, dim=(2, 3))


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        
        self.left = Sequential(
            Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_feats)
        )
        
        if first_stride > 1:
            self.right = Sequential(
                Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride),
                BatchNorm2d(out_feats)
            )
        else:
            self.right = nn.Identity()
            
        self.relu = ReLU()

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        '''
        x_left = self.left(x)
        x_right = self.right(x)
        return self.relu(x_left + x_right)


class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()
        
        blocks = [ResidualBlock(in_feats, out_feats, first_stride)] + [
            ResidualBlock(out_feats, out_feats) for n in range(n_blocks - 1)
        ]
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        return self.blocks(x)

# %%


class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        in_feats0 = 64

        self.in_layers = Sequential(
            Conv2d(3, in_feats0, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(in_feats0),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        all_in_feats = [in_feats0] + out_features_per_group[:-1]
        self.residual_layers = Sequential(
            *(
                BlockGroup(*args)
                for args in zip(
                    n_blocks_per_group,
                    all_in_feats,
                    out_features_per_group,
                    first_strides_per_group,
                )
            )
        )
        # Alternative that uses `add_module`, in a way which makes the layer names line up:
        # for idx, (n_blocks, in_feats, out_feats, first_stride) in enumerate(zip(
        #     n_blocks_per_group, all_in_feats, out_features_per_group, strides_per_group
        # )):
        #     self.add_module(f"layer{idx+1}", BlockGroup(n_blocks, in_feats, out_feats, first_stride))

        self.out_layers = Sequential(
            AveragePool(),
            Flatten(),
            Linear(out_features_per_group[-1], n_classes),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        x = self.in_layers(x)
        x = self.residual_layers(x)
        x = self.out_layers(x)
        return x

    # def __call__(self, x: t.Tensor) -> t.Tensor:
    #     return self.forward(x)

if MAIN:
    my_resnet = ResNet34()

# %%

def copy_weights(my_resnet: ResNet34, pretrained_resnet: torchvision.models.resnet.ResNet) -> ResNet34:
    '''Copy over the weights of `pretrained_resnet` to your resnet.'''
    
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict)
    
    state_dict_to_load = {}
    for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items()):
        state_dict_to_load[mykey] = pretrainedvalue
    
    my_resnet.load_state_dict(state_dict_to_load)
    
    return my_resnet

if MAIN:
    pretrained_resnet = torchvision.models.resnet34(pretrained=True)
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

    IMAGE_FOLDER = "./resnet_inputs"

    images = [Image.open(f"{IMAGE_FOLDER}/{filename}") for filename in IMAGE_FILENAMES]

# %%

if MAIN:
    # ImageNet transforms:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# %%

def prepare_data(images: List[Image.Image]) -> t.Tensor:
    '''
    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    '''
    x = t.stack([transform(img) for img in images], dim=0)  # type: ignore
    return x

if MAIN:
    prepared_images = prepare_data(images)

# %%

def predict(model, images):
    logits = model(images)
    return logits.argmax(dim=1)

# %%

with open("imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())

# %%

if MAIN:
    my_predictions = predict(my_resnet, prepared_images)
    pretrained_predictions = predict(pretrained_resnet, prepared_images)

    assert all(my_predictions == pretrained_predictions)

# %%

if MAIN:
    for img, label in zip(images, my_predictions):
        print(f"Class {label}: {imagenet_labels[label]}")
        display(img)
        print()

# %%

# Example debugging a NaN model:

class NanModule(nn.Module):
    def forward(self, x):
        return t.full_like(x, float('nan'))

model = nn.Sequential(
    nn.Identity(),
    NanModule(),
    nn.Identity()
)

def hook_check_for_nan_output(module: nn.Module, input: Tuple[t.Tensor], output: t.Tensor) -> None:
    if t.isnan(output).any():
        raise ValueError(f"NaN output from {module}")

def add_hook(module: nn.Module) -> None:
    '''
    Register our hook function in a module.

    Use model.apply(add_hook) to recursively apply the hook to model and all submodules.
    '''
    module.register_forward_hook(hook_check_for_nan_output)

def remove_hooks(module: nn.Module) -> None:
    '''
    Remove all hooks from module.

    Use module.apply(remove_hooks) to do this recursively.
    '''
    module._backward_hooks.clear()
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()

if MAIN:
    model.apply(add_hook)

    input = t.randn(3)
    output = model(input)

    model.apply(remove_hooks)

# %%