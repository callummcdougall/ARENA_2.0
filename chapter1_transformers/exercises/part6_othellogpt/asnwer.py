# %%
%pip install git+https://github.com/neelnanda-io/neel-plotly

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

# %%
# An example input

if MAIN:
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
if MAIN:
    os.chdir(section_dir)

    OTHELLO_ROOT = (section_dir / "othello_world").resolve()
    OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

    if not OTHELLO_ROOT.exists():
        !git clone https://github.com/likenneth/othello_world

    sys.path.append(str(OTHELLO_MECHINT_ROOT))
# %%
from mech_interp_othello_utils import plot_board, plot_single_board, plot_board_log_probs, to_string, to_int, int_to_label, string_to_label, OthelloBoardState
# %%
# Load board data as ints (i.e. 0 to 60)

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
# %%
if MAIN:
    moves_int = board_seqs_int[0, :30]

    # This is implicitly converted to a batch of size 1
    logits: Tensor = model(moves_int)
    print("logits:", logits.shape)
# %%
if MAIN:
    logit_vec = logits[0, -1]
    log_probs = logit_vec.log_softmax(-1)
    # Remove the "pass" move (the zeroth vocab item)
    log_probs = log_probs[1:]
    assert len(log_probs)==60

    # Set all cells to -13 by default, for a very negative log prob - this means the middle cells don't show up as mattering
    temp_board_state = t.zeros((8, 8), dtype=t.float32, device=device) - 13.
    temp_board_state.flatten()[stoi_indices] = log_probs
# %%
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


if MAIN:
    plot_square_as_board(temp_board_state.reshape(8, 8), zmax=0, diverging_scale=False, title="Example Log Probs")
# %%
if MAIN:
    plot_single_board(int_to_label(moves_int))
# %%
if MAIN:
    num_games = 50
    focus_games_int = board_seqs_int[:num_games] # shape: [50, 60] = [50 games, 60 moves each]
    focus_games_string = board_seqs_string[:num_games]
# %%
def one_hot(list_of_ints, num_classes=64):
    out = t.zeros((num_classes,), dtype=t.float32)
    out[list_of_ints] = 1.
    return out


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
# %%
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
# %%
if MAIN:
    layer = 6
    game_index = 0
    move = 29

def plot_probe_outputs(layer, game_index, move, **kwargs):
    residual_stream = focus_cache["resid_post", layer][game_index, move]
    # print("residual_stream", residual_stream.shape)
    probe_out = einops.einsum(residual_stream, linear_probe, "d_model, d_model row col options -> row col options")
    probabilities = probe_out.softmax(dim=-1)
    plot_square_as_board(probabilities, facet_col=2, facet_labels=["P(Empty)", "P(Their's)", "P(Mine)"], **kwargs)

if MAIN:
    plot_probe_outputs(layer, game_index, move, title="Example probe outputs after move 29 (black to play)")

    plot_single_board(int_to_label(focus_games_int[game_index, :move+1]))
# %%
if MAIN:
    layer = 4
    game_index = 0
    move = 29

    plot_probe_outputs(layer, game_index, move, title="Example probe outputs at layer 4 after move 29 (black to play)")
    plot_single_board(int_to_label(focus_games_int[game_index, :move+1]))
# %%
if MAIN:
    layer = 4
    game_index = 0
    move = 30

    plot_probe_outputs(layer, game_index, move, title="Example probe outputs at layer 4 after move 30 (white to play)")

    plot_single_board(focus_games_string[game_index, :31])
# %%
def state_stack_to_one_hot(state_stack):
    '''
    Creates a tensor of shape (games, moves, rows=8, cols=8, options=3), where the [g, m, r, c, :]-th entry
    is a one-hot encoded vector for the state of game g at move m, at row r and column c. In other words, this
    vector equals (1, 0, 0) when the state is empty, (0, 1, 0) when the state is "their", and (0, 0, 1) when the
    state is "my".
    '''
    one_hot = t.zeros(
        state_stack.shape[0], # num games
        state_stack.shape[1], # num moves
        rows,
        cols,
        3, # the options: empty, white, or black
        device=state_stack.device,
        dtype=t.int,
    )
    one_hot[..., 0] = state_stack == 0 
    one_hot[..., 1] = state_stack == -1 
    one_hot[..., 2] = state_stack == 1 

    return one_hot

# We first convert the board states to be in terms of my (+1) and their (-1), rather than black and white

if MAIN:
    alternating = np.array([-1 if i%2 == 0 else 1 for i in range(focus_games_int.shape[1])])
    flipped_focus_states = focus_states * alternating[None, :, None, None]

    # We now convert to one-hot encoded vectors
    focus_states_flipped_one_hot = state_stack_to_one_hot(t.tensor(flipped_focus_states))

    # Take the argmax (i.e. the index of option empty/their/mine)
    focus_states_flipped_value = focus_states_flipped_one_hot.argmax(dim=-1)
# %%
if MAIN:
    probe_out = einops.einsum(
        focus_cache["resid_post", 6], linear_probe,
        "game move d_model, d_model row col options -> game move row col options"
    )

    probe_out_value = probe_out.argmax(dim=-1)
# %%
if MAIN:
    correct_middle_odd_answers = (probe_out_value.cpu() == focus_states_flipped_value[:, :-1])[:, 5:-5:2]
    accuracies_odd = einops.reduce(correct_middle_odd_answers.float(), "game move row col -> row col", "mean")

    correct_middle_answers = (probe_out_value.cpu() == focus_states_flipped_value[:, :-1])[:, 5:-5]
    accuracies = einops.reduce(correct_middle_answers.float(), "game move row col -> row col", "mean")

    plot_square_as_board(
        1 - t.stack([accuracies_odd, accuracies], dim=0),
        title="Average Error Rate of Linear Probe", 
        facet_col=0, facet_labels=["Black to Play moves", "All Moves"], 
        zmax=0.25, zmin=-0.25
    )
# %%
if MAIN:
    # YOUR CODE HERE - define `blank_probe` and `my_probe`
    # linear probe shape: [d_model=512, n_rows=8, n_cols=8, options=3("blank", "theirs", "mine")]
    blank_probe = linear_probe[..., 0] - (linear_probe[..., 1] + linear_probe[..., 2]) / 2
    my_probe = linear_probe[..., 2] - linear_probe[..., 1]
    tests.test_my_probes(blank_probe, my_probe, linear_probe)
# %%
if MAIN:
    pos = 20
    game_index = 0

    # Plot board state
    moves = focus_games_string[game_index, :pos+1]
    plot_single_board(moves)

    # Plot corresponding model predictions
    state = t.zeros((8, 8), dtype=t.float32, device=device) - 13.
    state.flatten()[stoi_indices] = focus_logits[game_index, pos].log_softmax(dim=-1)[1:]
    plot_square_as_board(state, zmax=0, diverging_scale=False, title="Log probs")
# %%
if MAIN:
    cell_r = 5
    cell_c = 4
    print(f"Flipping the color of cell {'ABCDEFGH'[cell_r]}{cell_c}")

    board = OthelloBoardState()
    board.update(moves.tolist())
    board_state = board.state.copy()
    valid_moves = board.get_valid_moves()
    flipped_board = copy.deepcopy(board)
    flipped_board.state[cell_r, cell_c] *= -1
    flipped_valid_moves = flipped_board.get_valid_moves()

    newly_legal = [string_to_label(move) for move in flipped_valid_moves if move not in valid_moves]
    newly_illegal = [string_to_label(move) for move in valid_moves if move not in flipped_valid_moves]
    print("newly_legal", newly_legal)
    print("newly_illegal", newly_illegal)
# %%
