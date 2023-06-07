# %%

import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import einops
import wandb
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
import itertools
import random
from IPython.display import display
import wandb
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
from typing import List, Union, Optional, Tuple, Callable, Dict
import typeguard
from functools import partial
import copy
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from tqdm.notebook import tqdm
from dataclasses import dataclass
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import pytorch_lightning as pl
from rich import print as rprint
import pandas as pd

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part6_othellogpt"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from neel_plotly import scatter, line
import part6_othellogpt.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

t.set_grad_enabled(False)

MAIN = __name__ == "__main__"
# %%
cfg = HookedTransformerConfig(
    n_layers = 8,
    d_model = 512,
    d_head = 64,
    n_heads = 8,
    d_mlp = 2048,
    d_vocab = 61,
    n_ctx = 59,
    act_fn="gelu",
    normalization_type="LNPre",
    device=device,
)
model = HookedTransformer(cfg)
# %%
sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
# champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
model.load_state_dict(sd)
# %%
# An example input
sample_input = t.tensor([[
    20, 19, 18, 10,  2,  1, 27,  3, 41, 42, 34, 12,  4, 40, 11, 29, 43, 13, 48, 56, 
    33, 39, 22, 44, 24,  5, 46,  6, 32, 36, 51, 58, 52, 60, 21, 53, 26, 31, 37,  9,
    25, 38, 23, 50, 45, 17, 47, 28, 35, 30, 54, 16, 59, 49, 57, 14, 15, 55, 7
]]).to(device)

# The argmax of the output (ie the most likely next move from each position)
sample_output = t.tensor([[
    21, 41, 40, 34, 40, 41,  3, 11, 21, 43, 40, 21, 28, 50, 33, 50, 33,  5, 33,  5,
    52, 46, 14, 46, 14, 47, 38, 57, 36, 50, 38, 15, 28, 26, 28, 59, 50, 28, 14, 28, 
    28, 28, 28, 45, 28, 35, 15, 14, 30, 59, 49, 59, 15, 15, 14, 15,  8,  7,  8
]]).to(device)

assert (model(sample_input).argmax(dim=-1) == sample_output.to(device)).all()
# %%
os.chdir(section_dir)

OTHELLO_ROOT = (section_dir / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

if not OTHELLO_ROOT.exists():
    !git clone https://github.com/likenneth/othello_world

sys.path.append(str(OTHELLO_MECHINT_ROOT))
# %%
from mech_interp_othello_utils import plot_board, plot_single_board, plot_board_log_probs, to_string, to_int, int_to_label, string_to_label, OthelloBoardState
# %%
