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

MAIN = __name__ == "__main__"
# %%
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=32, 
                            kernel_size=(3, 3), padding=1, stride=1) # 32, 28, 28
        self.relu = ReLU()
        self.maxpool = MaxPool2d(kernel_size=2, stride=2) # 32, 14, 14
        self.conv2 = Conv2d(in_channels=32, out_channels=64,
                            kernel_size=(3, 3), padding=1, stride=1) # 64, 14, 14
        self.flatten = Flatten() # 64 * 14 & 14
        self.linear1 = Linear(in_features=(64 * 14 * 14), out_features=128)
        self.linear2 = Linear(in_features=128, out_features=10)


    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.relu(self.conv1(x))


if MAIN:
    model = ConvNet()
    print(model)