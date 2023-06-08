#%%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from torch import nn, Tensor
from torch.nn import functional as F
from dataclasses import dataclass
import time
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import einops
import plotly.express as px
from pathlib import Path
from jaxtyping import Float
from typing import Optional
from tqdm.auto import tqdm
from dataclasses import dataclass

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part7_toy_models_of_superposition"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line

from part7_toy_models_of_superposition.utils import plot_W, plot_Ws_from_model, render_features
import part7_toy_models_of_superposition.tests as tests
# import part7_toy_models_of_superposition.solutions as solutions

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
@dataclass
class Config:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as 
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    n_features: int = 5
    n_hidden: int = 2
    # Ignore the correlation arguments for now.
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0


class Model(nn.Module):

    W: Float[Tensor, "n_instances n_hidden n_features"]
    b_final: Float[Tensor, "n_instances n_features"]
    # Our linear map (ignoring n_instances) is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self, 
        config: Config, 
        feature_probability: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,               
        device=device
    ):
        super().__init__()
        self.config = config

        if feature_probability is None: feature_probability = t.ones(())
        self.feature_probability = feature_probability.to(device)
        if importance is None: importance = t.ones(())
        self.importance = importance.to(device)

        self.W = nn.Parameter(t.empty(self.config.n_instances, self.config.n_hidden, self.config.n_features).to(device))
        t.nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(t.zeros(self.config.n_instances, self.config.n_features).to(device))

    def forward(
        self, 
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        W_inter = einops.einsum(
            features,
            self.W,
            "... instances features, instances hidden features -> ... instances hidden"
        )

        W_out = einops.einsum(
            W_inter,
            self.W,
            "... instances hidden, instances hidden features -> ... instances features"
        )
        
        return F.relu(W_out + self.b_final)
    
    def generate_batch(self, n_batch) -> Float[Tensor, "n_batch instances features"]:
        '''
        Generates a batch of data. We'll return to this function later when we apply correlations.
        '''    
        feat = t.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device)
        feat_seeds = t.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device)
        feat_is_present = feat_seeds <= self.feature_probability
        batch = t.where(
            feat_is_present,
            feat,
            t.zeros((), device=self.W.device),
        )
        return batch



tests.test_model(Model)
# %%
def linear_lr(step, steps):
    return (1 - (step / steps))

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))

def optimize(
    model: Model, 
    n_batch=1024,
    steps=10_000,
    print_freq=100,
    lr=1e-3,
    lr_scale=constant_lr,
    hooks=[]
):
    cfg = model.config

    opt = t.optim.AdamW(list(model.parameters()), lr=lr)

    start = time.time()
    progress_bar = tqdm(range(steps))
    for step in progress_bar:
        step_lr = lr * lr_scale(step, steps)
        for group in opt.param_groups:
            group['lr'] = step_lr
            opt.zero_grad(set_to_none=True)
            batch = model.generate_batch(n_batch)
            out = model(batch)
            error = (model.importance*(batch.abs() - out)**2)
            loss = einops.reduce(error, 'b i f -> i', 'mean').sum()
            loss.backward()
            opt.step()

            if hooks:
                hook_data = dict(model=model, step=step, opt=opt, error=error, loss=loss, lr=step_lr)
                for h in hooks: h(hook_data)
            if step % print_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(
                    loss=loss.item() / cfg.n_instances,
                    lr=step_lr,
                )
# %%
config = Config(
    n_instances = 10,
    n_features = 5,
    n_hidden = 2,
)

importance = (0.9**t.arange(config.n_features))

feature_probability = (20 ** -t.linspace(0, 1, config.n_instances))

line(importance, width=600, height=400, title="Importance of each feature (same over all instances)", labels={"y": "Feature importance", "x": "Feature"})

line(feature_probability, width=600, height=400, title="Feature probability (varied over instances)", labels={"y": "Probability", "x": "Instance"})
# %%
model = Model(
    config=config,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None]
)

optimize(model)

plot_Ws_from_model(model, config)
# %%
config = Config(
    n_instances = 20,
    n_features = 100,
    n_hidden = 20,
)

importance = (100 ** -t.linspace(0, 1, config.n_features))

feature_probability = (20 ** -t.linspace(0, 1, config.n_instances))

line(importance, width=600, height=400, title="Importance of each feature (same over all instances)", labels={"y": "Feature importance", "x": "Feature"})

line(feature_probability, width=600, height=400, title="Feature probability (varied over instances)", labels={"y": "Probability", "x": "Instance"})
# %%
model = Model(
    config=config,
    device=device,
    importance = importance[None, :],
    feature_probability = feature_probability[:, None]
)

optimize(model)

fig = render_features(model, np.s_[::2])
fig.update_layout(width=1200, height=2000)
# %%
def generate_correlated_batch(self: Model, n_batch: int) -> Float[Tensor, "n_batch instances fetures"]:
    '''
    Generates a batch of data.

    There are `n_correlated_pairs` pairs of correlated features (i.e. they always co-occur), and 
    `n_anticorrelated` pairs of anticorrelated features (i.e. they never co-occur; they're
    always opposite).

    So the total number of features defined this way is `2 * n_correlated_pairs + 2 * n_anticorrelated`.

    You should stack the features in the order (correlated, anticorrelated, uncorrelated), where
    the uncorrelated ones are all the remaining features.

    Note, we assume the feature probability varies across instances but not features, i.e. all features
    in each instance have the same probability of being present.
    '''
    n_correlated_pairs = self.config.n_correlated_pairs
    n_anticorrelated_pairs = self.config.n_anticorrelated_pairs

    n_uncorrelated = self.config.n_features - 2 * (n_correlated_pairs + n_anticorrelated_pairs)
    assert n_uncorrelated >= 0, "Need to have number of paired correlated + anticorrelated features <= total features"
    assert self.feature_probability.shape == (self.config.n_instances, 1), "Feature probability should not vary across features in a single instance."

    # Define uncorrelated features, the standard way
    feat = t.rand((n_batch, self.config.n_instances, n_uncorrelated), device=self.W.device)
    feat_seeds = t.rand((n_batch, self.config.n_instances, n_uncorrelated), device=self.W.device)
    feat_is_present = feat_seeds <= self.feature_probability
    batch_uncorrelated = t.where(
        feat_is_present,
        feat,
        t.zeros((), device=self.W.device),
    )

    # YOUR CODE HERE - compute batch_correlated and batch_anticorrelated, and stack all three batches together
    # Batch correlated, have seeds be the same for each feature in a pair of correlated features
    print(batch_uncorrelated.shape) 

generate_correlated_batch(model, 3)

Model.generate_batch = generate_correlated_batch
# %%
