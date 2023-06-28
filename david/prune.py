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
# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%

model = utils.LeNet()
model.load_state_dict(torch.load('mnist_cnn_max_pool_9922.pth'))
model.to(device)
utils.compute_acc(model)

# %%


# Generate logarithmically spaced values


# Define the pruning parameters

# %%

num_pruned = 200
prune_factor = np.linspace(0, 1, num_pruned)
pruned_acc = [None] * num_pruned


def prune_model(frac, idx):
    pruned_model = utils.LeNet().to(device)
    pruned_model.load_state_dict(model.state_dict())
    pruning_method = prune.L1Unstructured  # Pruning method: L1 magnitude-based pruning

    # Apply pruning to the model
    for module in pruned_model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.ln_structured(module, name="weight", amount=frac, n=1, dim=0)
            prune.remove(module, "weight")

    # Compute the accuracy of the pruned model
    acc, loss = utils.compute_acc(pruned_model)
    pruned_acc[idx] = (acc, loss)

from threading import Thread

threads = []
for i,frac in enumerate(prune_factor):
    t = Thread(target=prune_model, args=(frac,i,))
    threads.append(t)
    t.start()

for t in tqdm(threads):
    t.join()




# %%
plt.plot(prune_factor, pruned_acc, marker='o', markersize=3)
plt.axhline(y=10, color='red', linestyle='--')
plt.xlabel("Fraction of weights pruned")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Fraction of weights pruned")
plt.show()

# %%
