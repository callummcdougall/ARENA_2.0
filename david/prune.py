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
# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%

model = utils.LeNetv2()
model.load_state_dict(torch.load('LeNet_Fashion_9045.pth'))
model.to(device)
utils.compute_acc(model)

# %%


def global_prune_model(frac):
    pruned_model = utils.LeNetv3().to(device)
    pruned_model.load_state_dict(model.state_dict())

    parameters_to_prune = (
        (pruned_model.conv1, 'weight'),
        (pruned_model.conv2, 'weight'),
        (pruned_model.fc1, 'weight'),
        (pruned_model.fc2, 'weight')
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=frac,
    )
    
# %%

# import torch
# import torch.nn.functional as F

# # Define your tensors representing the truth (P) and approximation (Q)
# P = torch.randn(N, 10)  # Replace with your own tensor
# Q = torch.randn(N, 10)  # Replace with your own tensor

# # Apply softmax to normalize the tensors into probability distributions
# P_probs = F.softmax(P, dim=1)
# Q_probs = F.softmax(Q, dim=1)

# # Compute the KL divergence in both directions and take their average
# kl_divergence = F.kl_div(P_probs.log(), Q_probs, reduction='none').sum(1)
# kl_divergence += F.kl_div(Q_probs.log(), P_probs, reduction='none').sum(1)
# average_kl_divergence = kl_divergence.mean()

# # Print the average KL divergence
# print("Average KL Divergence:", average_kl_divergence.item())



# def prune_model(frac):
#     """
#     Prune the model by removing the lowest magnitude weights.
#     Removes frac * 100% of the weights.
#     """
#     pruned_model = utils.LeNet().to(device)
#     pruned_model.load_state_dict(model.state_dict())
#     pruning_method = prune.L1Unstructured  # Pruning method: L1 magnitude-based pruning

#     # Apply pruning to the model
#     for module in pruned_model.modules():
#         if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#             prune.ln_structured(module, name="weight", amount=frac, n=1, dim=0)
#             prune.remove(module, "weight")

#     # Compute the accuracy of the pruned model
#     acc, loss = utils.compute_acc(pruned_model)
#     return acc, loss

# %%
num_pruned = 100
pruned_accs = []
pruned_losses = []
prune_factor = 1 - np.logspace(np.log10(0.001),np.log(1),num_pruned)
for frac in tqdm(prune_factor):
    acc, loss = global_prune_model(frac)
    pruned_accs.append(acc)
    pruned_losses.append(loss)
# %%

import plotly.graph_objects as go
# Create the scatter plot
fig = go.Figure(data=go.Scatter(x=1-prune_factor, y=pruned_accs, mode='markers', marker=dict(size=3)))

# Add a red dashed line
fig.add_shape(type='line', line=dict(color='red', dash='dash'))

# Customize the layout
fig.update_layout(
    xaxis=dict(
        title='Fraction of weights kept',
        type='log'
    ),
    yaxis=dict(
        title='Accuracy'
    ),
    title='Accuracy vs. Fraction of weights kept',
    hovermode='closest',
    showlegend=False
)

# Show the plot
fig.show()

# %%
plt.plot(1-prune_factor, pruned_accs, marker='o', markersize=2)
#plt.plot(prune_factor, [0] + compute_derivative(pruned_accs) , marker='o', markersize=3)
plt.axhline(y=10, color='red', linestyle='--')
plt.xlabel("Fraction of weights kept")
plt.ylabel("Accuracy")
# make x axis logarithmic
#show fine gridlines
plt.grid(which='both')
plt.xscale('log')
plt.title("Accuracy vs. Fraction of weights kept")
plt.savefig('acc_vs_weights_pruned.svg', format='svg')
plt.show()

# %%

