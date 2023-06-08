# %%
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
W = t.randn(2, 5)
W_normed = W / W.norm(dim=0, keepdim=True)

imshow(W_normed.T @ W_normed, title="Cosine similarities of each pair of 2D feature embeddings", width=600)
# %%
plot_W(W_normed)
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

        sf = (2 / (config.n_hidden + config.n_features)) ** (1 / 2)
        self.W = nn.Parameter(t.randn((config.n_instances, config.n_hidden, config.n_features)).to(device) * sf)
        self.b_final = nn.Parameter(t.zeros((config.n_instances, config.n_features)).to(device))


    def forward(
        self, 
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        x = einops.einsum(features, self.W, '... i f, i h f -> ... i h')
        x = einops.einsum(x, self.W, '... i h, i h f -> ... i f')
        x = x + self.b_final
        x = F.relu(x)
        return x

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
W = model.W.cpu().detach().numpy()
px.imshow(
    W.transpose(0, 2, 1) @ W,
    facet_col=0,
    color_continuous_scale='rdbu',
    color_continuous_midpoint=0
    # animation_frame=0
)
# %%
b_final = model.b_final.cpu().detach().numpy()
px.imshow(
    b_final,
    color_continuous_scale='rdbu',
    color_continuous_midpoint=0
    # animation_frame=0
)
# %%
