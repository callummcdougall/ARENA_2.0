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

model = utils.LeNetUltraSparse()
model.load_state_dict(torch.load('LeNetUltraSparse_L1_reg_9505.pth'))
model.to(device)
utils.compute_acc(model)

# %%


def global_prune_model(frac):
    pruned_model = utils.LeNetUltraSparse().to(device)
    pruned_model.load_state_dict(model.state_dict())

    parameters_to_prune = (
        (pruned_model.conv1, 'weight'),
        #(pruned_model.conv2, 'weight'),
        (pruned_model.fc1, 'weight'),
        #(pruned_model.fc2, 'weight')
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=frac,
    )

    return utils.compute_acc(pruned_model)
    
# %%
# %%



def flatten_parameters(model):
    # Get all model parameters
    parameters = []
    for param in model.parameters():
        parameters.append(param.view(-1))
    # Flatten the parameters into a single vector
    flattened_params = torch.cat(parameters)
    return flattened_params

def load_parameters(model, flattened_params):
    # Split the flattened parameters into individual parameter tensors
    split_params = torch.split(flattened_params, split_size_or_sections=model_sizes(model))
    # Load the parameters back into the model
    for param, split_param in zip(model.parameters(), split_params):
        param.data.copy_(split_param.view(param.size()))

# Helper function to get the sizes of model parameters
def model_sizes(model):
    sizes = []
    for param in model.parameters():
        sizes.append(param.view(-1).size(0))
    return sizes

# %%
# %%
# compute corresponding loss for deletion of every paramter
def perturb_and_evaluate(model):
    # Save the original parameters

    baseline_acc, baseline_loss = utils.compute_acc(model)

    param_vec = flatten_parameters(model)

    # Iterate through all parameters
    num_params = len(param_vec)
    runner = tqdm(enumerate(param_vec), total=num_params)

    peturbed_losses = torch.zeros(num_params)

    for (i,param) in runner:
        # Temporarily set the parameter to zero
        tmp = param.item()
        param_vec[i] = 0.0

        # Load the perturbed parameters into the model
        load_parameters(model, param_vec)
        # Evaluate the loss with perturbed parameters
        perturbed_acc, perturbed_loss = utils.compute_acc(model)   

        peturbed_losses[i] = perturbed_loss
        #restore parameter
        param_vec[i] = tmp

    return peturbed_losses
# %%
model = utils.LeNetUltraSparse()
model.load_state_dict(torch.load('LeNetUltraSparse_L1_reg_9505.pth'))
model.to(device)
print(f"Loss, Acc: {utils.compute_acc(model)}")
summary(model)
# %%

peturbed_losses = perturb_and_evaluate(model)
# %%
vec_p = flatten_parameters(model)
# %%
def prune_index(frac, losses_vec):
    pruned_model = utils.LeNetUltraSparse().to(device)
    pruned_model.load_state_dict(model.state_dict())
    flat_parm = flatten_parameters(pruned_model)

    num_indexes = int(flat_parm.size(0) * frac)

    #find the frac proportion indexes of the smallest loss
    indexes = torch.argsort(losses_vec)[:num_indexes]
    #set the corresponding indexes to zero
    flat_parm[indexes] = 0.0
    #load the pruned parameters back into the model
    load_parameters(pruned_model, flat_parm)
    return utils.compute_acc(pruned_model)
# %%
num_pruned = 50
david_pruned_accs = []
david_pruned_losses = []
L1_pruned_accs = []
L1_pruned_losses = []
prune_factor = 1 - np.logspace(np.log10(0.002),np.log(1),num_pruned)
for frac in tqdm(prune_factor):
    acc, loss = prune_index(frac, peturbed_losses)
    david_pruned_accs.append(acc)
    david_pruned_losses.append(loss)

    acc, loss = global_prune_model(frac)
    L1_pruned_accs.append(acc)
    L1_pruned_losses.append(loss)

# %%


import plotly.graph_objects as go
# Create the scatter plot
fig = go.Figure(data=go.Scatter(x=1-prune_factor, y=pruned_accs, mode='markers', marker=dict(size=3)))

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
