# %%

import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import einops
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
import pandas as pd

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part6_othellogpt"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

# from plotly_utils import imshow
from neel_plotly import line, scatter, histogram, imshow

# from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
# import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"


if MAIN:
	device = t.device("cuda" if t.cuda.is_available() else "cpu")
	
	t.set_grad_enabled(False);

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
	
	OTHELLO_ROOT = section_dir / "othello_world"
	
	if not OTHELLO_ROOT.exists():
		!git clone https://github.com/likenneth/othello_world
	
	sys.path.append(str(OTHELLO_ROOT / "mechanistic_interpretability"))

# %%

from mech_interp_othello_utils import plot_board, plot_single_board, plot_board_log_probs, to_string, to_int, int_to_label, string_to_label, OthelloBoardState

# %%


if MAIN:
	board_seqs_int = t.tensor(np.load(OTHELLO_ROOT / "board_seqs_int_small.npy"), dtype=t.long)
	board_seqs_string = t.tensor(np.load(OTHELLO_ROOT / "board_seqs_string_small.npy"), dtype=t.long)
	
	num_games, length_of_game = board_seqs_int.shape
	print("Number of games:", num_games,)
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
	logits = model(moves_int)
	print("logits:", logits.shape)

# %%


if MAIN:
	logit_vec = logits[0, -1]
	log_probs = logit_vec.log_softmax(-1)
	# Remove passing
	log_probs = log_probs[1:]
	assert len(log_probs)==60
	
	temp_board_state = t.zeros(64, device=logit_vec.device)
	# Set all cells to -15 by default, for a very negative log prob - this means the middle cells don't show up as mattering
	temp_board_state -= 13.
	temp_board_state[stoi_indices] = log_probs

# %%

def plot_square_as_board(state, diverging_scale=True, **kwargs):
	"""Takes a square input (8 by 8) and plot it as a board. Can do a stack of boards via facet_col=0"""
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
	board = OthelloBoardState()
	board.update(to_string(moves_int))
	plot_square_as_board(board.state, title="Example Board State (+1 is Black, -1 is White)")

# %%


if MAIN:
	print("Valid moves:", string_to_label(board.get_valid_moves()))

# %%


if MAIN:
	num_games = 50
	focus_games_int = board_seqs_int[:num_games]
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
	focus_logits, focus_cache = model.run_with_cache(focus_games_int[:, :-1].to(device))

# %%


if MAIN:
	focus_logits.shape

# %%


if MAIN:
	full_linear_probe = t.load(OTHELLO_ROOT / "main_linear_probe.pth")
	
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

# FLAT SOLUTION
# YOUR CODE HERE - define `blank_probe` and `my_probe`
blank_probe = linear_probe[..., 0] - linear_probe[..., 1] * 0.5 - linear_probe[..., 2] * 0.5
my_probe = linear_probe[..., 2] - linear_probe[..., 1]
# FLAT SOLUTION END

import part6_othellogpt.tests as tests


if MAIN:
	tests.test_my_probes(blank_probe, my_probe, linear_probe)

# %%


if MAIN:
	pos = 20
	game_index = 0
	
	moves = focus_games_string[game_index, :pos+1]
	plot_single_board(moves)
	
	state = t.zeros((64,), dtype=t.float32, device=device) - 10.
	state[stoi_indices] = focus_logits[game_index, pos].log_softmax(dim=-1)[1:]

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

def apply_scale(resid: Float[Tensor, "batch=1 seq d_model"], flip_dir: Float[Tensor, "d_model"], scale: int, pos: int):
	'''
	Returns a version of the residual stream, modified by the amount `scale` in the 
	direction `flip_dir` at the sequence position `pos`, in the way described above.
	'''
	flip_dir_normed = flip_dir / flip_dir.norm()

	alpha = resid[0, pos] @ flip_dir_normed
	resid[0, pos] -= (scale+1) * alpha * flip_dir_normed

	return resid



if MAIN:
	tests.test_apply_scale(apply_scale)

# %%


if MAIN:
	flip_dir = my_probe[:, cell_r, cell_c]
	
	big_flipped_states_list = []
	layer = 4
	scales = [0, 1, 2, 4, 8, 16]
	
	# Iterate through scales, generate a new facet plot for each possible scale
	for scale in scales:
	
		# Hook function which will perform flipping in the "F4 flip direction"
	def flip_hook(resid: Float[Tensor, "batch=1 seq d_model"], hook: HookPoint):
		return apply_scale(resid, flip_dir, scale, pos)

	# Calculate the logits for the board state, with the `flip_hook` intervention
	# (note that we only need to use :pos+1 as input, because of causal attention)
	flipped_logits: Tensor = model.run_with_hooks(
		focus_games_int[game_index:game_index+1, :pos+1],
		fwd_hooks=[
			(utils.get_act_name("resid_post", layer), flip_hook),
		]
	).log_softmax(dim=-1)[0, pos]

	flip_state = t.zeros((64,), dtype=t.float32, device=device) - 10.
	flip_state[stoi_indices] = flipped_logits.log_softmax(dim=-1)[1:]
	big_flipped_states_list.append(flip_state)


if MAIN:
	flip_state_big = t.stack(big_flipped_states_list)
	state_big = einops.repeat(state, "d -> b d", b=6)
	color = t.zeros((len(scales), 64)).cuda() + 0.2
	for s in newly_legal:
		color[:, to_string(s)] = 1
	for s in newly_illegal:
		color[:, to_string(s)] = -1
	
	scatter(
		y=state_big, 
		x=flip_state_big, 
		title=f"Original vs Flipped {string_to_label(8*cell_r+cell_c)} at Layer {layer}", 
		xaxis="Flipped", yaxis="Original", 
		hover=[f"{r}{c}" for r in "ABCDEFGH" for c in range(8)], 
		facet_col=0, facet_labels=[f"Translate by {i}x" for i in scales], 
		color=color, color_name="Newly Legal", color_continuous_scale="Geyser"
	)

# %% 2️⃣ LOOKING FOR MODULAR CIRCUITS


if MAIN:
	game_index = 1
	move = 20
	layer = 6
	
	plot_single_board(focus_games_string[game_index, :move+1])
	plot_probe_outputs(layer, game_index, move)

# %%

def plot_contributions(contributions, component: str):
	imshow(
		contributions,
		facet_col=0,
		y=list("ABCDEFGH"),
		facet_name="Layer",
		title=f"{component} Layer Contributions to my vs their (Game {game_index} Move {move})",
		aspect="equal",
		width=1300,
		height=320
	)

def calculate_attn_and_mlp_probe_score_contributions(
	focus_cache: ActivationCache, 
	my_probe: Float[Tensor, "d_model rows cols"],
	layer: int,
	game_index: int, 
	move: int
) -> Tuple[Float[Tensor, "layers rows cols"], Float[Tensor, "layers rows cols"]]:
	
	attn_contributions = t.stack([    
		einops.einsum(
			focus_cache["attn_out", l][game_index, move], my_probe, 
			"d_model, d_model rows cols -> rows cols",
		)
		for l in range(layer+1)
	])
	mlp_contributions = t.stack([
		einops.einsum(
			focus_cache["mlp_out", l][game_index, move], my_probe, 
			"d_model, d_model rows cols -> rows cols",
		)
		for l in range(layer+1)])

	return (attn_contributions, mlp_contributions)



if MAIN:
	attn_contributions, mlp_contributions = calculate_attn_and_mlp_probe_score_contributions(focus_cache, my_probe, layer, game_index, move)
	
	plot_contributions(attn_contributions, "Attention")
	plot_contributions(mlp_contributions, "MLP")

# %%

def calculate_accumulated_probe_score(
	focus_cache: ActivationCache, 
	my_probe: Float[Tensor, "d_model rows cols"],
	layer: int,
	game_index: int, 
	move: int
) -> Float[Tensor, "rows cols"]:
	
	return einops.einsum(
		focus_cache["resid_post", layer][game_index, move], my_probe, 
		"d_model, d_model rows cols -> rows cols",
	)



if MAIN:
	overall_contribution = calculate_accumulated_probe_score(focus_cache, my_probe, layer, game_index, move)
	
	imshow(
		overall_contribution, 
		title=f"Overall Probe Score after Layer {layer} for<br>my vs their (Game {game_index} Move {move})",
		width=380, 
		height=380
	)

# %%

# Scale the probes down to be unit norm per cell

if MAIN:
	blank_probe_normalised = blank_probe / blank_probe.norm(dim=0, keepdim=True)
	my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)
	# Set the center blank probes to 0, since they're never blank so the probe is meaningless
	blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.

# %%

def get_w_in(
	model: HookedTransformer,
	layer: int,
	neuron: int,
	normalize: bool = False,
) -> Float[Tensor, "d_model"]:
	'''
	Returns the input weights for the neuron in the list, at each square on the board.

	If normalize is True, the weights are normalized to unit norm.
	'''
	w_in = model.W_in[layer, :, neuron].detach().clone()
	if normalize: w_in /= w_in.norm(dim=0, keepdim=True)
	return w_in


def get_w_out(
	model: HookedTransformer,
	layer: int,
	neuron: int,
	normalize: bool = False,
) -> Float[Tensor, "d_model"]:
	'''
	Returns the input weights for the neuron in the list, at each square on the board.
	'''
	w_out = model.W_out[layer, neuron, :].detach().clone()
	if normalize: w_out /= w_out.norm(dim=0, keepdim=True)
	return  w_out


def calculate_neuron_input_weights(
	model: HookedTransformer, 
	probe: Float[Tensor, "d_model row col"], 
	layer: int, 
	neuron: int
) -> Float[Tensor, "rows cols"]:
	'''
	Returns tensor of the input weights for each neuron in the list, at each square on the board,
	projected along the corresponding probe directions.

	Assume probe directions are normalized. You should also normalize the model weights.
	'''
	w_in = get_w_in(model, layer, neuron, normalize=True)

	return einops.einsum(
		w_in, probe,
		"d_model, d_model row col -> row col",
	)


def calculate_neuron_output_weights(
	model: HookedTransformer, 
	probe: Float[Tensor, "d_model row col"], 
	layer: int, 
	neuron: int
) -> Float[Tensor, "rows cols"]:
	'''
	Returns tensor of the output weights for each neuron in the list, at each square on the board,
	projected along the corresponding probe directions.

	Assume probe directions are normalized. You should also normalize the model weights.
	'''
	w_out = get_w_out(model, layer, neuron, normalize=True)

	return einops.einsum(
		w_out, probe,
		"d_model, d_model row col -> row col",
	)



if MAIN:
	tests.test_calculate_neuron_input_weights(calculate_neuron_input_weights, model)

# %%


if MAIN:
	layer = 5
	neuron = 1393
	
	w_in_L5N1393_my = calculate_neuron_input_weights(model, my_probe_normalised, layer, neuron)
	w_in_L5N1393_blank = calculate_neuron_input_weights(model, blank_probe_normalised, layer, neuron)
	
	imshow(
		t.stack([w_in_L5N1393_my, w_in_L5N1393_blank]),
		facet_col=0,
		y=[i for i in "ABCDEFGH"],
		title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
		facet_labels=["Blank In", "My In"]
	)

# %%


if MAIN:
	w_in_L5N1393 = get_w_in(model, layer, neuron)
	w_out_L5N1393 = get_w_out(model, layer, neuron)
	
	U, S, Vh = t.svd(t.cat([
		my_probe.reshape(cfg.d_model, 64),
		blank_probe.reshape(cfg.d_model, 64)],
	dim=1))
	
	# Remove the final four dimensions of U, as the 4 center cells are never blank and so the blank probe is meaningless there
	probe_space_basis = U[:, :-4]
	
	print("Fraction of input weights in probe basis:", (w_in_L5N1393 @ probe_space_basis).norm().item()**2)
	print("Fraction of output weights in probe basis:", (w_out_L5N1393 @ probe_space_basis).norm().item()**2)

# %%


if MAIN:
	layer = 3
	top_layer_3_neurons = focus_cache["post", layer][:, 3:-3].std(dim=[0, 1]).argsort(descending=True)[:10]
	
	heatmaps_blank = []
	heatmaps_my = []
	
	for neuron in top_layer_3_neurons:
		neuron = neuron.item()
		heatmaps_blank.append(calculate_neuron_output_weights(model, blank_probe_normalised, layer, neuron))
		heatmaps_my.append(calculate_neuron_output_weights(model, my_probe_normalised, layer, neuron))
	
	imshow(
		heatmaps_blank,
		facet_col=0,
		y=[i for i in "ABCDEFGH"],
		title=f"Cosine sim of Output weights and the blank color probe for top layer 3 neurons",
		facet_labels=[f"L3N{n.item()}" for n in top_layer_3_neurons],
		height=300,
	)
	
	imshow(
		heatmaps_my,
		facet_col=0,
		y=[i for i in "ABCDEFGH"],
		title=f"Cosine sim of Output weights and the my color probe for top layer 3 neurons",
		facet_labels=[f"L3N{n.item()}" for n in top_layer_3_neurons],
		height=300,
	)

# %%


if MAIN:
	layer = 4
	top_layer_4_neurons = focus_cache["post", layer][:, 3:-3].std(dim=[0, 1]).argsort(descending=True)[:10]
	
	heatmaps_blank = []
	heatmaps_my = []
	
	for neuron in top_layer_4_neurons:
		neuron = neuron.item()
		heatmaps_blank.append(calculate_neuron_output_weights(model, blank_probe_normalised, layer, neuron))
		heatmaps_my.append(calculate_neuron_output_weights(model, my_probe_normalised, layer, neuron))
	
	imshow(
		heatmaps_blank,
		facet_col=0,
		y=[i for i in "ABCDEFGH"],
		title=f"Cosine sim of Output weights and the blank color probe for top layer 4 neurons",
		facet_labels=[f"L4N{n.item()}" for n in top_layer_4_neurons],
		height=300,
	)
	
	imshow(
		heatmaps_my,
		facet_col=0,
		y=[i for i in "ABCDEFGH"],
		title=f"Cosine sim of Output weights and the my color probe for top layer 4 neurons",
		facet_labels=[f"L4N{n.item()}" for n in top_layer_4_neurons],
		height=300,
	)

# %%


if MAIN:
	layer = 4
	top_layer_4_neurons = focus_cache["post", layer][:, 3:-3].std(dim=[0, 1]).argsort(descending=True)[:10]
	W_U_norm = model.W_U / model.W_U.norm(dim=0, keepdim=True)
	W_U_norm = W_U_norm[:, 1:] # Get rid of the passing/dummy first element
	heatmaps_unembed = []
	
	for neuron in top_layer_4_neurons:
		neuron = neuron.item()
		w_out = get_w_out(model, layer, neuron)
		# Fill in the `state` tensor with cosine sims, while skipping the middle 4 squares
		state = t.zeros(64, device=device)
		state[stoi_indices] = w_out @ W_U_norm
		heatmaps_unembed.append(state.reshape(8, 8))
	
	imshow(
		heatmaps_unembed,
		facet_col=0,
		y=[i for i in "ABCDEFGH"],
		title=f"Cosine sim of Output weights and the unembed for top layer 4 neurons",
		facet_labels=[f"L4N{n.item()}" for n in top_layer_4_neurons],
		height=300,
	)

# %%


if MAIN:
	game_index = 4
	move = 20
	
	plot_single_board(focus_games_string[game_index, :move+1], title="Original Game (black plays E0)")
	plot_single_board(focus_games_string[game_index, :move].tolist()+[16], title="Corrupted Game (blank plays C0)")

# %%


if MAIN:
	clean_input = focus_games_int[game_index, :move+1].clone()
	corrupted_input = focus_games_int[game_index, :move+1].clone()
	corrupted_input[-1] = to_int("C0")
	print("Clean:     ", ", ".join(int_to_label(corrupted_input)))
	print("Corrupted: ", ", ".join(int_to_label(clean_input)))

# %%


if MAIN:
	clean_logits, clean_cache = model.run_with_cache(clean_input)
	corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_input)
	
	clean_log_probs = clean_logits.log_softmax(dim=-1)
	corrupted_log_probs = corrupted_logits.log_softmax(dim=-1)

# %%


if MAIN:
	f0_index = to_int("F0")
	clean_f0_log_prob = clean_log_probs[0, -1, f0_index]
	corrupted_f0_log_prob = corrupted_log_probs[0, -1, f0_index]
	
	print("Clean log prob", clean_f0_log_prob.item())
	print("Corrupted log prob", corrupted_f0_log_prob.item(), "\n")
	
def patching_metric(patched_logits: Float[Tensor, "batch=1 seq=21 d_vocab=61"]):
	'''
	Function of patched logits, calibrated so that it equals 0 when performance is 
	same as on corrupted input, and 1 when performance is same as on clean input.

	Should only be a function of the logits for the F0 token (you can index into 
	patched logits using the `f0_index` variable).

	Should be a linear function of the log-softmax of patched logits (note, this is
	not the same as being a linear function of the logits themselves - can you see 
	why doing it this way is preferable?).
	'''
	patched_log_probs = patched_logits.log_softmax(dim=-1)
	return (patched_log_probs[0, -1, f0_index] - corrupted_f0_log_prob) / (clean_f0_log_prob - corrupted_f0_log_prob)


# Code to test your function:

if MAIN:
	tests.test_patching_metric(patching_metric, clean_log_probs, corrupted_log_probs)

# %%

def patch_final_move_output(
	activation: Float[Tensor, "batch seq d_model"], 
	hook: HookPoint,
	clean_cache: ActivationCache,
) -> Float[Tensor, "batch seq d_model"]:
	'''
	Hook function which patches activations at the final sequence position.

	Note, we only need to patch in the final sequence position, because the
	prior moves in the clean and corrupted input are identical (and this is
	an autoregressive model).
	'''
	activation[0, -1, :] = clean_cache[hook.name][0, -1, :]
	return activation


def get_act_patch_resid_pre(
	model: HookedTransformer, 
	corrupted_input: Float[Tensor, "batch pos"], 
	clean_cache: ActivationCache, 
	patching_metric: Callable[[Float[Tensor, "batch seq d_model"]], Float[Tensor, ""]]
) -> Float[Tensor, "2 d_model"]:
	'''
	Returns an array of results, corresponding to the results of patching at
	each (attn_out, mlp_out) for all layers in the model.
	'''
	model.reset_hooks()
	results = t.zeros(2, model.cfg.n_layers, device=device, dtype=t.float32)
	hook_fn = partial(patch_final_move_output, clean_cache=clean_cache)

	for i, activation in enumerate(["attn_out", "mlp_out"]):
		for layer in tqdm(range(model.cfg.n_layers)):
			patched_logits = model.run_with_hooks(
				corrupted_input, 
				fwd_hooks = [(utils.get_act_name(activation, layer), hook_fn)], 
			)
			results[i, layer] = patching_metric(patched_logits)

	return results

# %%


if MAIN:
	patching_results = get_act_patch_resid_pre(model, corrupted_input, clean_cache, patching_metric)
	
	line(patching_results, title="Layer Output Patching Effect on F0 Log Prob", line_labels=["attn", "mlp"], width=750)

# %% 3️⃣ NEURON INTERPRETABILITY: A DEEP DIVE


if MAIN:
	layer = 5
	neuron = 1393
	
	w_out = get_w_out(model, layer, neuron, normalize=False)
	state = t.zeros(8, 8, device=device)
	state.flatten()[stoi_indices] = w_out @ model.W_U[:, 1:]
	plot_square_as_board(state, title=f"Output weights of Neuron L{layer}N{neuron} in the output logit basis", width=600)

# %%


if MAIN:
	c0_U = model.W_U[:, 17].detach()
	c0_U /= c0_U.norm()
	
	d1_U = model.W_U[:, 26].detach()
	d1_U /= d1_U.norm()
	
	print(f"Cosine sim of C0 and D1 unembeds: {c0_U @ d1_U}")

# %%

# FLAT SOLUTION
# Compute the fraction of variance of neuron output vector explained by unembedding subspace
w_out = get_w_out(model, layer, neuron, normalize=True)
U, S, Vh = t.svd(model.W_U[:, 1:])
print("Fraction of variance captured by W_U", (w_out @ U).norm().item()**2)
# FLAT SOLUTION END

# %%


if MAIN:
	neuron_acts = focus_cache["post", layer, "mlp"][:, :, neuron]
	
	imshow(neuron_acts, title=f"L{layer}N{neuron} Activations over 50 games", yaxis="Game", xaxis="Move")

# %%

# focus_states_flipped_value does this but has theirs==1 and mine==2, not 1 and -1. So let's convert!

if MAIN:
	focus_states_flipped_pm1 = t.zeros_like(focus_states_flipped_value, device=device)
	focus_states_flipped_pm1[focus_states_flipped_value==2] = -1.
	focus_states_flipped_pm1[focus_states_flipped_value==1] = 1.
	# Now, theirs==1 and mine==-1

# %%

# Boolean array for whether the neuron is in the top 30 activations over all games and moves

if MAIN:
	top_moves = neuron_acts.flatten() > neuron_acts.quantile(0.99)
	
	# For each top activation, get the corresponding board state (flattened)
	board_state_at_top_moves: Int[Tensor, "topk 64"] = focus_states_flipped_pm1[:, :-1].reshape(-1, 64)[top_moves]
	# Rearrange into (rows, cols), then take mean over all boards
	board_state_at_top_moves = board_state_at_top_moves.reshape(-1, 8, 8).float().mean(0)
	
	plot_square_as_board(
		board_state_at_top_moves, 
		title=f"Aggregated top 30 moves for neuron L{layer}N{neuron}", 
		width=600
	)

# %%


if MAIN:
	top_neurons = focus_cache["post", 5, "mlp"].std(dim=[0, 1]).argsort(descending=True)[:10]

# %%

# FLAT SOLUTION
# Your code here - investigate the top 10 neurons by std dev of activations, see what you can find!
board_states = []
output_weights_in_logit_basis = []

for neuron in top_neurons:
	# Get output weights in logit basis
	w_out = get_w_out(model, layer, neuron, normalize=False)
	state = t.zeros(8, 8, device=device)
	state.flatten()[stoi_indices] = w_out @ model.W_U[:, 1:]
	output_weights_in_logit_basis.append(state)
	
	# Get activations by indexing into cache
	neuron_acts = focus_cache["post", 5, "mlp"][:, :, neuron]
	# Boolean array for whether the neuron is in the top 30 activations over all games and moves
	top_moves = neuron_acts.flatten() > neuron_acts.quantile(0.99)
	# For each top activation, get the corresponding board state (flattened)
	board_state_at_top_moves: Int[Tensor, "topk 64"] = focus_states_flipped_pm1[:, :-1].reshape(-1, 64)[top_moves]
	# Rearrange into (rows, cols), then take mean over all boards
	board_state_at_top_moves = board_state_at_top_moves.reshape(-1, 8, 8).float().mean(0)
	board_states.append(board_state_at_top_moves)


plot_square_as_board(
	board_states, 
	title=f"Aggregated top 30 moves for each top 10 neuron in layer 5", 
	facet_col=0, 
	facet_labels=[f"L5N{n.item()}" for n in top_neurons]
)
plot_square_as_board(
	output_weights_in_logit_basis, 
	title=f"Output weights of top 10 neurons in layer 5, in the output logit basis",
	facet_col=0, 
	facet_labels=[f"L5N{n.item()}" for n in top_neurons]
)
# FLAT SOLUTION END

# %%


if MAIN:
	c0 = focus_states_flipped_pm1[:, :, 2, 0]
	d1 = focus_states_flipped_pm1[:, :, 3, 1]
	e2 = focus_states_flipped_pm1[:, :, 4, 2]
	
	label = (c0==0) & (d1==-1) & (e2==1)
	
	neuron_acts = focus_cache["post", 5][:, :, 1393]
	df = pd.DataFrame({"acts":neuron_acts.flatten().tolist(), "label":label[:, :-1].flatten().tolist()})
	px.histogram(df, x="acts", color="label", histnorm="percent", barmode="group", nbins=100, title="Spectrum plot for neuron L5N1393 testing C0==BLANK & D1==THEIR'S & E2==MINE")

# %% 4️⃣ TRAINING A PROBE

def seq_to_state_stack(str_moves):
	if isinstance(str_moves, t.Tensor):
		str_moves = str_moves.tolist()
	board = OthelloBoardState()
	states = []
	for move in str_moves:
		board.umpire(move)
		states.append(np.copy(board.state))
	states = np.stack(states, axis=0)
	return states



if MAIN:
	state_stack = t.tensor(
		np.stack([seq_to_state_stack(seq) for seq in board_seqs_string[:50, :-1]])
	)
	print(state_stack.shape)
	
	# Visualize the first 8 moves
	imshow(state_stack[0, :8], facet_col=0, height=300)

# %%

@dataclass
class ProbeTrainingArgs():

	# Which layer, and which positions in a game sequence to probe
	layer: int = 6
	pos_start: int = 5
	pos_end: int = model.cfg.n_ctx - 5
	length: int = pos_end - pos_start

	# Game state (options are blank/mine/theirs)
	options: int = 3
	rows: int = 8
	cols: int = 8

	# Standard training hyperparams
	max_epochs: int = 8
	num_games: int = 50000

	# Hyperparams for optimizer
	batch_size: int = 256
	lr: float = 1e-4
	betas: Tuple[float, float] = (0.9, 0.99)
	wd: float = 0.01

	# Misc.
	probe_name: str = "main_linear_probe"

	# The first mode is blank or not, the second mode is next or prev GIVEN that it is not blank
	modes = 3
	def __post_init__(self):
		self.alternating = t.tensor([1 if i%2 == 0 else -1 for i in range(self.length)], device=device)
		self.logger = CSVLogger(save_dir=os.getcwd() + "/logs", name=self.probe_name)

	# Code to get randomly initialized probe
	def setup_linear_probe(self, model: HookedTransformer):
		linear_probe = t.randn(
			self.modes, model.cfg.d_model, self.rows, self.cols, self.options, requires_grad=False, device=device
		) / np.sqrt(model.cfg.d_model)
		linear_probe.requires_grad = True
		return linear_probe

# %%

class LitLinearProbe(pl.LightningModule):
	def __init__(self, model: HookedTransformer, args: ProbeTrainingArgs):
		super().__init__()
		self.model = model
		self.args = args
		self.linear_probe = args.setup_linear_probe(model)
		pl.seed_everything(42, workers=True)

	def training_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> t.Tensor:

		games_int = board_seqs_int[batch.cpu()]
		games_str = board_seqs_string[batch.cpu()]
		state_stack = t.stack([t.tensor(seq_to_state_stack(game_str)) for game_str in games_str])
		state_stack = state_stack[:, self.args.pos_start: self.args.pos_end, :, :]
		state_stack_one_hot = state_stack_to_one_hot(state_stack).to(device)

		with t.inference_mode():
			_, cache = model.run_with_cache(
				games_int[:, :-1].to(device),
				return_type=None,
				names_filter=lambda name: name.endswith("resid_post")
			)
			resid_post = cache["resid_post", self.args.layer][:, self.args.pos_start: self.args.pos_end]

		probe_out = einops.einsum(
			resid_post,
			self.linear_probe,
			"batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
		)

		probe_log_probs = probe_out.log_softmax(-1)
		probe_correct_log_probs = einops.reduce(
			probe_log_probs * state_stack_one_hot,
			"modes batch pos rows cols options -> modes pos rows cols",
			"mean"
		) * self.args.options # Multiply to correct for the mean over options
		loss_even = -probe_correct_log_probs[0, 0::2].mean(0).sum() # note that "even" means odd in the game framing, since we offset by 5 moves lol
		loss_odd = -probe_correct_log_probs[1, 1::2].mean(0).sum()
		loss_all = -probe_correct_log_probs[2, :].mean(0).sum()
		
		loss = loss_even + loss_odd + loss_all
		print(f"Loss: {loss.item():.3f}", end="\r")
		return loss

	def train_dataloader(self):
		n_indices = self.args.num_games - (self.args.num_games % self.args.batch_size)
		full_train_indices = t.randperm(self.args.num_games)[:n_indices]
		full_train_indices = einops.rearrange(full_train_indices, "(i j) -> i j", j=self.args.batch_size)
		return full_train_indices

	def configure_optimizers(self):
		optimizer = t.optim.AdamW([self.linear_probe], lr=self.args.lr, betas=self.args.betas, weight_decay=self.args.wd)
		return optimizer

# %%

# Create the model & training system

if MAIN:
	args = ProbeTrainingArgs()
	litmodel = LitLinearProbe(model, args)
	
	# Get dataloader(s)
	trainloader = litmodel.train_dataloader()
	
	# Train the model
	trainer = pl.Trainer(
		max_epochs=litmodel.args.max_epochs,
		logger=litmodel.args.logger,
		log_every_n_steps=1,
	)
	trainer.fit(model=litmodel)

# %%


if MAIN:
	black_to_play_index = 0
	white_to_play_index = 1
	blank_index = 0
	their_index = 1
	my_index = 2
	
	# Creating values for linear probe (converting the "black/white to play" notation into "me/them to play")
	my_linear_probe = t.zeros(cfg.d_model, rows, cols, options, device=device)
	my_linear_probe[..., blank_index] = 0.5 * (litmodel.linear_probe[black_to_play_index, ..., 0] + litmodel.linear_probe[white_to_play_index, ..., 0])
	my_linear_probe[..., their_index] = 0.5 * (litmodel.linear_probe[black_to_play_index, ..., 1] + litmodel.linear_probe[white_to_play_index, ..., 2])
	my_linear_probe[..., my_index] = 0.5 * (litmodel.linear_probe[black_to_play_index, ..., 2] + litmodel.linear_probe[white_to_play_index, ..., 1])
	
	probe_out = einops.einsum(
		focus_cache["resid_post", 6], my_linear_probe, # linear_probe
		"game move d_model, d_model row col options -> game move row col options"
	)
	probe_out_value = probe_out.argmax(dim=-1)
	
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

# %% 5️⃣ BONUS - FUTURE WORK I'M EXCITED ABOUT

