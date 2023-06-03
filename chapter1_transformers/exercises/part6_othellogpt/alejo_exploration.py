# %%
# %pip install git+https://github.com/neelnanda-io/neel-plotly

# %%
import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
import torch as torch
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

if MAIN:
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
if MAIN:
    sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
    # champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
    model.load_state_dict(sd)

if MAIN:
    os.chdir(section_dir)

    OTHELLO_ROOT = (section_dir / "othello_world").resolve()
    OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

    if not OTHELLO_ROOT.exists():
        !git clone https://github.com/likenneth/othello_world

    sys.path.append(str(OTHELLO_MECHINT_ROOT))

from mech_interp_othello_utils import plot_board, plot_single_board, plot_board_log_probs, to_string, to_int, int_to_label, string_to_label, OthelloBoardState

# %%

if MAIN:
    board_seqs_int = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=t.long)
    # Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
    board_seqs_string = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"), dtype=t.long)

    assert all([middle_sq not in board_seqs_string for middle_sq in [27, 28, 35, 36]])
    assert board_seqs_int.max() == 60

    num_games, length_of_game = board_seqs_int.shape
    print("Number of games:", num_games)
    print("Length of game:", length_of_game)
# %%
# Define possible indices (excluding the four center squares)

if MAIN:
    stoi_indices = [i for i in range(64) if i not in [27, 28, 35, 36]]

    # Define our rows, and the function that converts an index into a (row, column) label, e.g. `E2`
    alpha = "ABCDEFGH"

def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"

# Get our list of board labels

if MAIN:
    board_labels = list(map(to_board_label, stoi_indices))


def plot_square_as_board(state, diverging_scale=True, **kwargs):
    '''Takes a square input (8 by 8) and plot it as a board. Can do a stack of boards via facet_col=0'''
    kwargs = {
        "y": [i for i in alpha],
        "x": [str(i) for i in range(8)],
        "color_continuous_scale": "RdBu" if diverging_scale else "Blues",
        "color_continuous_midpoint": 0. if diverging_scale else None,
        "aspect": "equal",
        **kwargs
    }
    imshow(state, **kwargs)

def one_hot(list_of_ints, num_classes=64):
    out = t.zeros((num_classes,), dtype=t.float32)
    out[list_of_ints] = 1.
    return out

# %%

if MAIN:
    num_games = 50
    focus_games_int = board_seqs_int[:num_games] # shape: [50, 60] = [50 games, 60 moves each]
    focus_games_string = board_seqs_string[:num_games]

if MAIN:
    focus_states = np.zeros((num_games, 60, 8, 8), dtype=np.float32)
    focus_valid_moves = t.zeros((num_games, 60, 64), dtype=t.float32)

    for i in (range(num_games)):
        board = OthelloBoardState()
        for j in range(60):
            board.umpire(focus_games_string[i, j].item())
            focus_states[i, j] = board.state
            focus_valid_moves[i, j] = one_hot(board.get_valid_moves())

    print("focus states:", focus_states.shape)
    print("focus_valid_moves", tuple(focus_valid_moves.shape))
# %%
if MAIN:
    imshow(
        focus_states[0, :16],
        facet_col=0,
        facet_col_wrap=8,
        facet_labels=[f"Move {i}" for i in range(1, 17)],
        title="First 16 moves of first game",
        color_continuous_scale="Greys",
    )

if MAIN:
    focus_logits, focus_cache = model.run_with_cache(focus_games_int[:, :-1].to(device))
    focus_logits.shape # torch.Size([50 games, 59 moves, 61 tokens]) 
# %%
if MAIN:
    full_linear_probe = t.load(OTHELLO_MECHINT_ROOT / "main_linear_probe.pth")

    rows = 8
    cols = 8 
    options = 3
    assert full_linear_probe.shape == (3, cfg.d_model, rows, cols, options)
# %%
if MAIN:
    black_to_play_index = 0
    white_to_play_index = 1
    blank_index = 0
    their_index = 1
    my_index = 2

    # Creating values for linear probe (converting the "black/white to play" notation into "me/them to play")
    linear_probe = t.zeros(cfg.d_model, rows, cols, options, device=device)
    linear_probe[..., blank_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 0] + full_linear_probe[white_to_play_index, ..., 0])
    linear_probe[..., their_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 1] + full_linear_probe[white_to_play_index, ..., 2])
    linear_probe[..., my_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 2] + full_linear_probe[white_to_play_index, ..., 1])


blank_probe = linear_probe[..., 0] - (linear_probe[..., 1] + linear_probe[..., 2]) / 2
my_probe = linear_probe[..., 2] - linear_probe[..., 1]
tests.test_my_probes(blank_probe, my_probe, linear_probe)

# %%

attn_pattn = t.stack([focus_cache['pattern', l] for l in range(model.cfg.n_layers)], dim=1)
# attn_pattn_plot = attn_pattn.std(0)
attn_pattn_plot = attn_pattn.quantile(.9, dim=0) - attn_pattn.quantile(.1, dim=0)
attn_pattn_plot2 = attn_pattn.mean(0)

imshow(utils.to_numpy(attn_pattn_plot), animation_frame=0, facet_col=1, facet_col_wrap=4, width=1200, height=800)
imshow(utils.to_numpy(attn_pattn_plot2), animation_frame=0, facet_col=1, facet_col_wrap=4, width=1200, height=800)

# %%
import re

def to_list(*args):
  if len(args) == 1:
    a = args[0]
    if a == None:
      return []
    elif type(a) == int:
      return [a]
    elif isinstance(a, (torch.Tensor, np.ndarray)):
      return a
    else:
      try:
        return list(a)
      except Exception as e:
        print(e)
        return a 
  else:
    return [to_list(a) for a in args]


cache_keys = list(focus_cache.keys())

def get_hook_name(*args):
  block, id = None, None
  if len(args) == 1:
    return "hook_pos_embed" if "pos" in args[0] else "hook_embed"
  elif len(args) >= 2:
    block, keywords = args[0], args[1:]
    assert type(block) == int, "When passing more than 1 argument, the first argument must be the layer number"
    match_str = ".*".join(keywords)
    finds = [name for name in cache_keys if re.match(f"blocks\.{block}.*{match_str}", name)]
    if len(finds) != 1:
      print(f"get_hook_name found several matches: {finds}")
      return None
    return finds[0]

class ResetHooks():
  """A context manager that calls model.reset_hooks before and after executing the code
  it contains. I don't know if it handles exceptions"""
  def __init__(self, model):
    self.model = model
  
  def __enter__(self):
    self.model.reset_hooks()

  def __exit__(self, *args, **kwargs):
    self.model.reset_hooks()

def add_hook(hook_name, hook, is_permanent=False, **hook_kwargs):
  hook_ready = partial(hook, **hook_kwargs)
  model.add_hook(hook_name, hook_ready, is_permanent=is_permanent)

def add_hooks(hook_names, hook, **hook_kwargs):
  if isinstance(hook_names, list):
    for name in hook_names:
      add_hook(name, hook, **hook_kwargs)
  if callable(hook_names):
    for name in cache_keys:
      if hook_names(name):
        add_hook(name, hook, **hook_kwargs)

def freeze_heads(activations, hook, cache, block):
  current_block = int(re.search("\d", hook.name)[0])
  if current_block > block:
    activations = cache[hook.name]
  return activations

def patch_piece(activations, hook, cache, dim=-1, index=None):
  if index == None:
    return cache[hook.name]
  
  if not isinstance(index, torch.Tensor):
    index = [index] if type(index) == int else index
    index = torch.Tensor(index).long().to(device)

  cache_act = torch.index_select(cache[hook.name], dim, index)
  activations.index_copy_(dim, index, cache_act)
  return activations

def patch_heads(activations, hook, head, cache, dim=-1, index=None):
  shape, n_dim = cache[hook.name].shape, cache[hook.name].ndim
  head_dim = [d for d, s in enumerate(shape) if s==cfg.n_heads][0]
  head_idx = [slice(None) for d in range(n_dim)]
  head_idx[head_dim] = [head] if type(head) == int else head

  if index == None:
    activations[head_idx] = cache[hook.name][head_idx]
    return activations

  if not isinstance(index, torch.Tensor):
    index = [index] if type(index) == int else index
    index = torch.Tensor(index).long().to(device)
  
  cache_act = torch.index_select(cache[hook.name][head_idx], dim, index)
  head_act = activations[head_idx]
  head_act.index_copy_(dim, index, cache_act)
  activations[head_idx] = head_act
  return activations

# %%

def patch_logit_diff_head(orig_data, alter_data, logit_pos, correct_pos, layer, hook_kw="pattern",
                     dim=-1, index=None, head=None, baseline_data="orig", freeze_attn=True):
  
  def patch_head_iter(layer, iter, cache_dict, patch, **kwargs):
    add_hook(get_hook_name(layer, "attn", hook_kw), patch_heads, head=iter, 
            cache=cache_dict[patch], index=index)
    
  return patch_logit_diff(orig_data, alter_data, logit_pos=logit_pos, correct_pos=correct_pos, layer=layer, 
                              iter=head, baseline=baseline_data, freeze_attn=freeze_attn, 
                              add_hook_fn=patch_head_iter)

def patch_logit_diff_mlp(orig_data, alter_data, logit_pos, correct_pos, layer, hook_kw="pattern",
                     dim=-1, neuron=None, baseline_data="orig", freeze_attn=True, 
                     hook_fn=lambda *args, **kwargs: None):
  
  pass

def patch_logit_diff_setup(orig_data, alter_data, logit_pos, correct_pos, cache_filter=0):
  # names_fiter = lambda 
  orig_logits, orig_cache = model.run_with_cache(orig_data)
  alter_logits, alter_cache = model.run_with_cache(alter_data)
  
  logits = {"orig": orig_logits[:, logit_pos, correct_pos],
            "alter": alter_logits[:, logit_pos, correct_pos]}
  data = {"orig": orig_data, "alter": alter_data}
  cache = {"orig": orig_cache, "alter": alter_cache}

  return logits, data, cache

def patch_logit_diff(orig_data, alter_data, logit_pos, correct_pos, layer, 
                         iter=0, baseline="orig", freeze_attn=False, 
                         add_hook_fn=lambda *args, **kwargs: None):
  
  layer, iter = to_list(layer, iter)
  logits, data, cache = patch_logit_diff_setup(orig_data, alter_data,
                                                            logit_pos, correct_pos)
  base, patch = ("orig", "alter") if baseline=="orig" else ("alter", "orig")
  logit_diff_out = torch.zeros(len(layer), len(iter), data["orig"].shape[0]).to(device)

  hook_pattern = lambda name: re.match(f"blocks\.\d.*pattern", name)
  for i, l in enumerate(layer):
    for j, k in enumerate(iter):
      with ResetHooks(model):
        if freeze_attn:
          add_hooks(hook_pattern, freeze_heads, block=l, cache=cache[base])
        
        add_hook_fn(layer=l, iter=k, cache_dict=cache, base=base, patch=patch)
        patched_logits = model(data[base])
        logits_patched = patched_logits[:, logit_pos, correct_pos]
        logit_diff_out[i, j] = ((logits_patched - logits[base])/
                                (logits["orig"] - logits["alter"]))
  return logit_diff_out.squeeze(1)

# %%

# orig_games = board_seqs_int[:10, :-1]
# alter_games = board_seqs_int[-10:, :-1]

# results = patch_logit_diff_head(orig_games, alter_games, logit_pos=10, correct_pos=10, 
#                                 layer=list(range(8)), head=list(range(8)))

# imshow(results.mean(-1), labels={'x':'Head', 'y':'Layer'})
# %%


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from yellowbrick.cluster.elbow import kelbow_visualizer

def get_pca(data=None, hook_name=None, labels=None, index=slice(None), activations=None,
            normalize=False, n_components=15, show_elbow=False):
  "labels must have shape [b, l] where l is the number of labels and b the batch"
  if activations is None:
    _, cache = model.run_with_cache(data)
    cache_hook = cache[hook_name]
    activations = cache_hook[index].reshape(cache_hook.shape[0], -1)

  if normalize:
    activations = (activations-activations.mean(0))/(activations.std(0) + 1e-5)

  pca = PCA(n_components=n_components)
  pca_acts = pca.fit_transform(utils.to_numpy(activations))
  df = pd.DataFrame(pca_acts, columns=["PCA"+str(i) for i in range(n_components)])
  
  
  if labels is not None:
    if isinstance(labels, tuple):
      assert len(labels) == 2, "If labels is tuple it must contain 2 elements: column names and label values"
      col_names, values = labels
      labels = pd.DataFrame(utils.to_numpy(values))
      labels.columns = col_names
    if not isinstance(labels, (pd.DataFrame, pd.Series)):
      labels = pd.DataFrame(utils.to_numpy(labels))
      labels.columns = [f"Label{i}" for i in range(labels.shape[1])]
    df = df.join(labels)

  return df, pca

