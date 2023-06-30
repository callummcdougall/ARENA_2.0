# %%
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
from utils import LeNet, train_dataset, test_dataset, compute_acc
import torch.nn.utils.prune as prune
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%

# %%
model = LeNet()
model.load_state_dict(torch.load('mnist_cnn.pth'))

# %%


# Generate logarithmically spaced values
prune_factor = 1-np.logspace(np.log10(1), np.log10(0.01), num=30)

# Define the pruning parameters

pruned_model = LeNet().to(device)
pruned_acc = []

for frac in tqdm(prune_factor):
    pruned_model.load_state_dict(model.state_dict())
    pruning_method = prune.L1Unstructured  # Pruning method: L1 magnitude-based pruning

    # Apply pruning to the model
    for module in pruned_model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.ln_structured(module, name="weight", amount=frac, n=1, dim=0)
            prune.remove(module, "weight")
    
    acc, loss = compute_acc(pruned_model)
    pruned_acc.append(acc)
# %%
    