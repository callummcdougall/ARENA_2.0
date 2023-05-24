#%%
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

MAIN = __name__ == "__main__"\


# 02 - Assembling ResNet
# %% Sequential

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



# %% Batch Norm 2d
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
        self.eps = eps 
        self.momentum = momentum
        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        self.num_batches_tracked += 1

        running_mean = self.running_mean.reshape((1, x.shape[1], 1, 1))
        running_var = self.running_var.reshape((1, x.shape[1], 1, 1))
        weight = self.weight.reshape((1, x.shape[1], 1, 1))
        bias = self.bias.reshape((1, x.shape[1], 1, 1))
        if self.training:
            mean = t.mean(x, dim=(0,2,3), keepdim=True)
            var = t.var(x, unbiased=False, dim=(0,2,3), keepdim=True)
            self.running_mean = ((1-self.momentum) * running_mean + self.momentum * mean).squeeze()
            self.running_var = ((1-self.momentum) * running_var + self.momentum * var).squeeze()
            return (x - mean) / t.sqrt(var + self.eps) * weight + bias

        else:
            return (x - running_mean) / t.sqrt(running_var + self.eps) * weight + bias


    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])



if MAIN:
    tests.test_batchnorm2d_module(BatchNorm2d)
    tests.test_batchnorm2d_forward(BatchNorm2d)
    tests.test_batchnorm2d_running_mean(BatchNorm2d)

# %% Average Pool

class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return x.mean(dim=(2,3))

# %% Implement Residual Block
# Unless otherwise noted, convolutions have a kernel_size of 3x3, a stride of 1, and a padding of 1. None of the convolutions have biases.
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        self.first_stride = first_stride
        self.out_feats = out_feats
        self.in_feats = in_feats
        
        self.left = Sequential(
            Conv2d(in_channels=in_feats,
                   out_channels=out_feats,
                   stride=first_stride,
                   kernel_size=3,
                   padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(in_channels=out_feats,
                   out_channels=out_feats,
                   stride=1,
                   kernel_size=3,
                   padding=1),
            BatchNorm2d(out_feats),
        )
        
        if self.first_stride > 1:
            self.right = Sequential(
                Conv2d(in_channels=in_feats,
                        out_channels=out_feats,
                        stride = first_stride,
                        kernel_size=1),
                BatchNorm2d(out_feats),
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
        left = self.left(x)
        right = self.right(x)
        return self.relu(left + right)


class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()
        self.block1 = ResidualBlock(in_feats, out_feats, first_stride)
        resblocks = [ResidualBlock(out_feats, out_feats, first_stride=1) for _ in range(n_blocks-1)] 
        self.resblocks = nn.Sequential(*resblocks)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        return self.resblocks(self.block1(x))


class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()

        first_out_channels = 64
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes
        
        in_features_per_group = [first_out_channels] + out_features_per_group[:-1]

        self.in_blocks = nn.Sequential(Conv2d(in_channels=3, out_channels=first_out_channels,
                                              kernel_size=7, stride=2, padding=3),
                                              BatchNorm2d(num_features=first_out_channels),
                                              ReLU(),
                                              MaxPool2d(kernel_size=3, stride=2))
        
        self.block_groups = Sequential(
            *(BlockGroup(n_blocks, in_feats, out_feats, first_stride)
            for n_blocks, in_feats, out_feats, first_stride in 
                zip(n_blocks_per_group,
                    in_features_per_group,
                    out_features_per_group,
                    first_strides_per_group)
            )
        )

        self.out_blocks = nn.Sequential(AveragePool(),
                                        Flatten(),
                                        Linear(in_features = out_features_per_group[-1], 
                                               out_features=n_classes))   
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        return self.out_block(self.block_groups(self.in_blocks(x)))
        


if MAIN:
    my_resnet = ResNet34()


# %% Copy weights from OG Resnet
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
    print(torchinfo.summary(my_resnet))
    
    # my_resnet = copy_weights(my_resnet, pretrained_resnet)
# %%
if MAIN:
    print(torchinfo.summary(my_resnet))