# %%
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
import torch.nn.utils.prune as prune
import numpy as np
import utils
import matplotlib.pyplot as plt
from utils import Timer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
model = utils.LeNetv3()
model.load_state_dict(torch.load('LeNetv3_Fashion_9045.pth'))
model.to(device)
utils.compute_acc(model)
# %%
