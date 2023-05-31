# begin solutions from solutions.py

# %%

import os
import sys
import plotly.express as px
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part2_intro_to_mech_interp").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference
from part1_transformer_from_scratch.solutions import get_log_probs
import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %% 1️⃣ TRANSFORMERLENS: INTRODUCTION


if MAIN:
	gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

# %%


if MAIN:
	model_description_text = '''## Loading Models
	
	HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 
	
	For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''
	
	loss = gpt2_small(model_description_text, return_type="loss")
	print("Model loss:", loss)

# %%


if MAIN:
	print(gpt2_small.to_str_tokens("gpt2"))
	print(gpt2_small.to_tokens("gpt2"))
	print(gpt2_small.to_string([50256, 70, 457, 17]))

# %%


if MAIN:
	logits: Tensor = gpt2_small(model_description_text, return_type="logits")
	prediction = logits.argmax(dim=-1).squeeze()[:-1]
	# FLAT SOLUTION
	# YOUR CODE HERE - get the model's prediction on the text
	true_tokens = gpt2_small.to_tokens(model_description_text).squeeze()[1:]
	is_correct = (prediction == true_tokens)
	
	print(f"Model accuracy: {is_correct.sum()}/{len(true_tokens)}")
	print(f"Correct words: {gpt2_small.to_str_tokens(prediction[is_correct])}")

# %%


if MAIN:
	gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
	gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
	gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

# %%


if MAIN:
	attn_patterns_layer_0 = gpt2_cache["pattern", 0]

# %%


if MAIN:
	attn_patterns_layer_0_copy = gpt2_cache["blocks.0.attn.hook_pattern"]
	
	t.testing.assert_close(attn_patterns_layer_0, attn_patterns_layer_0_copy)

# %%


if MAIN:
	layer0_pattern_from_cache = gpt2_cache["pattern", 0]
	
	# FLAT SOLUTION
	# YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the steps of the attention calculation (dot product, masking, scaling, softmax)
	q, k = gpt2_cache["q", 0], gpt2_cache["k", 0]
	seq, nhead, headsize = q.shape
	layer0_attn_scores = einops.einsum(q, k, "seqQ n h, seqK n h -> n seqQ seqK")
	mask = t.triu(t.ones((seq, seq), dtype=bool), diagonal=1).to(device)
	layer0_attn_scores.masked_fill_(mask, -1e9)
	layer0_pattern_from_q_and_k = (layer0_attn_scores / headsize**0.5).softmax(-1)
	
	t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
	print("Tests passed!")

# %%


if MAIN:
	print(type(gpt2_cache))
	attention_pattern = gpt2_cache["pattern", 0, "attn"]
	print(attention_pattern.shape)
	gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)
	
	print("Layer 0 Head Attention Patterns:")
	display(cv.attention.attention_patterns(
		tokens=gpt2_str_tokens, 
		attention=attention_pattern,
		attention_head_names=[f"L0H{i}" for i in range(12)],
	))

# %% 2️⃣ FINDING INDUCTION HEADS


if MAIN:
	cfg = HookedTransformerConfig(
		d_model=768,
		d_head=64,
		n_heads=12,
		n_layers=2,
		n_ctx=2048,
		d_vocab=50278,
		attention_dir="causal",
		attn_only=True, # defaults to False
		tokenizer_name="EleutherAI/gpt-neox-20b", 
		seed=398,
		use_attn_result=True,
		normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
		positional_embedding_type="shortformer"
	)

# %%


if MAIN:
	weights_dir = (section_dir / "attn_only_2L_half.pth").resolve()
	
	if not weights_dir.exists():
		url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"
		output = str(weights_dir)
		gdown.download(url, output)

# %%


if MAIN:
	model = HookedTransformer(cfg)
	pretrained_weights = t.load(weights_dir, map_location=device)
	model.load_state_dict(pretrained_weights)

# %%


if MAIN:
	text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
	
	logits, cache = model.run_with_cache(text, remove_batch_dim=True)

# %%


if MAIN:
	# FLAT SOLUTION
	# YOUR CODE HERE - visualize attention
	str_tokens = model.to_str_tokens(text)
	for layer in range(model.cfg.n_layers):
		attention_pattern = cache["pattern", layer]
		display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern))

# %%

def current_attn_detector(cache: ActivationCache) -> List[str]:
	'''
	Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
	'''
	attn_heads = []
	for layer in range(model.cfg.n_layers):
		for head in range(model.cfg.n_heads):
			attention_pattern = cache["pattern", layer][head]
			# take avg of diagonal elements
			score = attention_pattern.diagonal().mean()
			if score > 0.4:
				attn_heads.append(f"{layer}.{head}")
	return attn_heads

def prev_attn_detector(cache: ActivationCache) -> List[str]:
	'''
	Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
	'''
	attn_heads = []
	for layer in range(model.cfg.n_layers):
		for head in range(model.cfg.n_heads):
			attention_pattern = cache["pattern", layer][head]
			# take avg of sub-diagonal elements
			score = attention_pattern.diagonal(-1).mean()
			if score > 0.4:
				attn_heads.append(f"{layer}.{head}")
	return attn_heads

def first_attn_detector(cache: ActivationCache) -> List[str]:
	'''
	Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
	'''
	attn_heads = []
	for layer in range(model.cfg.n_layers):
		for head in range(model.cfg.n_heads):
			attention_pattern = cache["pattern", layer][head]
			# take avg of 0th elements
			score = attention_pattern[:, 0].mean()
			if score > 0.4:
				attn_heads.append(f"{layer}.{head}")
	return attn_heads



if MAIN:
	print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
	print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
	print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))

# %%

def generate_repeated_tokens(
	model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
	'''
	Generates a sequence of repeated random tokens

	Outputs are:
		rep_tokens: [batch, 1+2*seq_len]
	'''
	prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
	rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
	rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
	return rep_tokens



def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
	'''
	Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

	Should use the `generate_repeated_tokens` function above

	Outputs are:
		rep_tokens: [batch, 1+2*seq_len]
		rep_logits: [batch, 1+2*seq_len, d_vocab]
		rep_cache: The cache of the model run on rep_tokens
	'''
	rep_tokens = generate_repeated_tokens(model, seq_len, batch)
	rep_logits, rep_cache = model.run_with_cache(rep_tokens)
	return rep_tokens, rep_logits, rep_cache



if MAIN:
	seq_len = 50
	batch = 1
	(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
	rep_cache.remove_batch_dim()
	rep_str = model.to_str_tokens(rep_tokens)
	model.reset_hooks()
	log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()
	
	print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
	print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")
	
	plot_loss_difference(log_probs, rep_str, seq_len)

# %%

# YOUR CODE HERE - display the attention patterns stored in `rep_cache`, for each layer

# %%

def induction_attn_detector(cache: ActivationCache) -> List[str]:
	'''
	Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

	Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
	'''
	attn_heads = []
	for layer in range(model.cfg.n_layers):
		for head in range(model.cfg.n_heads):
			attention_pattern = cache["pattern", layer][head]
			# take avg of (-seq_len+1)-offset elements
			seq_len = (attention_pattern.shape[-1] - 1) // 2
			score = attention_pattern.diagonal(-seq_len+1).mean()
			if score > 0.4:
				attn_heads.append(f"{layer}.{head}")
	return attn_heads



if MAIN:
	print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

# %% 3️⃣ TRANSFORMERLENS: HOOKS


if MAIN:
	seq_len = 50
	batch = 10
	rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)
	
	# We make a tensor to store the induction score for each head.
	# We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
	induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
	
def induction_score_hook(
	pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
	hook: HookPoint,
):
	'''
	Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
	'''
	# Take the diagonal of attn paid from each dest posn to src posns (seq_len-1) tokens back
	# (This only has entries for tokens with index>=seq_len)
	induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
	# Get an average score per head
	induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
	# Store the result.
	induction_score_store[hook.layer(), :] = induction_score

# We make a boolean filter on activation names, that's true only on attention pattern names

if MAIN:
	pattern_hook_names_filter = lambda name: name.endswith("pattern")
	
	# Run with hooks (this is where we write to the `induction_score_store` tensor`)
	model.run_with_hooks(
		rep_tokens_10, 
		return_type=None, # For efficiency, we don't need to calculate the logits
		fwd_hooks=[(
			pattern_hook_names_filter,
			induction_score_hook
		)]
	)
	
	# Plot the induction scores for each head in each layer
	imshow(
		induction_score_store, 
		labels={"x": "Head", "y": "Layer"}, 
		title="Induction Score by Head", 
		text_auto=".2f",
		width=900, height=400
	)

# %%

def visualize_pattern_hook(
	pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
	hook: HookPoint,
):
	print("Layer: ", hook.layer())
	display(
		cv.attention.attention_patterns(
			tokens=gpt2_small.to_str_tokens(rep_tokens[0]), 
			attention=pattern.mean(0)
		)
	)


if MAIN:
	# FLAT SOLUTION
	# YOUR CODE HERE - find induction heads in gpt2_small
	seq_len = 50
	batch = 10
	rep_tokens_10 = generate_repeated_tokens(gpt2_small, seq_len, batch)
	
	induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device)
	
	gpt2_small.run_with_hooks(
		rep_tokens_10, 
		return_type=None, # For efficiency, we don't need to calculate the logits
		fwd_hooks=[(
			pattern_hook_names_filter,
			induction_score_hook
		)]
	)
	
	imshow(
		induction_score_store, 
		labels={"x": "Head", "y": "Layer"},
		title="Induction Score by Head", 
		text_auto=".1f",
		width=800
	)
	
	# Observation: heads 5.1, 5.5, 6.9, 7.2, 7.10 are all strongly induction-y.
	# Confirm observation by visualizing attn patterns for layers 5 through 7:
	
	for induction_head_layer in [5, 6, 7]:
		gpt2_small.run_with_hooks(
			rep_tokens, 
			return_type=None, # For efficiency, we don't need to calculate the logits
			fwd_hooks=[
				(utils.get_act_name("pattern", induction_head_layer), visualize_pattern_hook)
			]
		)

# %%

def logit_attribution(
	embed: Float[Tensor, "seq d_model"],
	l1_results: Float[Tensor, "seq nheads d_model"],
	l2_results: Float[Tensor, "seq nheads d_model"],
	W_U: Float[Tensor, "d_model d_vocab"],
	tokens: Float[Tensor, "seq"]
) -> Float[Tensor, "seq-1 n_components"]:
	'''
	Inputs:
		embed: the embeddings of the tokens (i.e. token + position embeddings)
		l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
		l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
		W_U: the unembedding matrix
		tokens: the token ids of the sequence

	Returns:
		Tensor of shape (seq_len-1, n_components)
		represents the concatenation (along dim=-1) of logit attributions from:
			the direct path (seq-1,1)
			layer 0 logits (seq-1, n_heads)
			layer 1 logits (seq-1, n_heads)
		so n_components = 1 + 2*n_heads
	'''
	W_U_correct_tokens = W_U[:, tokens[1:]]
	direct_attributions = einops.einsum(W_U_correct_tokens, embed[:-1], "emb seq, seq emb -> seq")
	l1_attributions = einops.einsum(W_U_correct_tokens, l1_results[:-1], "emb seq, seq nhead emb -> seq nhead")
	l2_attributions = einops.einsum(W_U_correct_tokens, l2_results[:-1], "emb seq, seq nhead emb -> seq nhead")
	return t.concat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)


if MAIN:
	text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
	logits, cache = model.run_with_cache(text, remove_batch_dim=True)
	str_tokens = model.to_str_tokens(text)
	tokens = model.to_tokens(text)
	
	with t.inference_mode():
		embed = cache["embed"]
		l1_results = cache["result", 0]
		l2_results = cache["result", 1]
		logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
		# Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
		correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
		t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
		print("Tests passed!")

# %%


if MAIN:
	embed = cache["embed"]
	l1_results = cache["result", 0]
	l2_results = cache["result", 1]
	logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
	
	plot_logit_attribution(model, logit_attr, tokens)

# %%

# end solutions from solutions.py

if MAIN:
    seq_len = 50

    # recall that (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch),
    # which generates repeated tokens and allows you to check for induction heads

    embed = rep_cache["embed"]
    l1_results = rep_cache["result", 0]
    l2_results = rep_cache["result", 1]
    first_half_tokens = rep_tokens[0, : 1 + seq_len] # first dim is batch dimension
    second_half_tokens = rep_tokens[0, seq_len:]

    # what is the shape of rep_tokens?

    # YOUR CODE HERE - define `first_half_logit_attr` and `second_half_logit_attr`
    
    first_half_logit_attr = logit_attribution(embed[:seq_len+1], l1_results[:seq_len+1], l2_results[:seq_len+1], model.W_U, first_half_tokens)
    second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.W_U, second_half_tokens)

    assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)

    plot_logit_attribution(model, first_half_logit_attr, first_half_tokens, "Logit attribution (first half of repeated sequence)")
    plot_logit_attribution(model, second_half_logit_attr, second_half_tokens, "Logit attribution (second half of repeated sequence)")
# %%

def head_ablation_hook(
    attn_result: Float[Tensor, "batch seq n_heads d_model"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> Float[Tensor, "batch seq n_heads d_model"]:

    attn_result[:,:,head_index_to_ablate,:] = 0.0
    return attn_result


def cross_entropy_loss(logits, tokens):
    '''
    Computes the mean cross entropy between logits (the model's prediction) and tokens (the true values).

    (optional, you can just use return_type="loss" instead.)
    '''
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()


def get_ablation_scores(
    model: HookedTransformer, 
    tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("result", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(ablated_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores



if MAIN:
    ablation_scores = get_ablation_scores(model, rep_tokens)
    tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)
# %%

if MAIN:
    imshow(
        ablation_scores, 
        labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
        title="Logit Difference After Ablating Heads", 
        text_auto=".2f",
        width=900, height=400
    )
# %%

# bonus

model.reset_hooks()
logits = model(tokens, return_type="logits")
loss_no_ablation = cross_entropy_loss(logits, tokens)
fwd_hooks = []

for layer in tqdm(range(model.cfg.n_layers)):
	for head in range(model.cfg.n_heads):
		if (layer == 0 and head == 7) or (layer ==1 and head ==4) or (layer ==1 and head==10):
			continue
		else:
			fwd_hooks.append((utils.get_act_name('v',layer), functools.partial(head_ablation_hook, head_index_to_ablate=head)))

print(fwd_hooks)

#activations_to_ablate = ['utils.get_act_name()']

#run the model with hooks and get the loss

loss = model.run_with_hooks(
	rep_tokens,
     return_type='loss',
     fwd_hooks=fwd_hooks
	
)

print(loss)
#%%

if MAIN:
    A = t.randn(5, 2)
    B = t.randn(2, 5)
    AB = A @ B
    AB_factor = FactoredMatrix(A, B)
    print("Norms:")
    print(AB.norm())
    print(AB_factor.norm())

    print(f"Right dimension: {AB_factor.rdim}, Left dimension: {AB_factor.ldim}, Hidden dimension: {AB_factor.mdim}")
# %%
if MAIN:
    print("Eigenvalues:")
    print(t.linalg.eig(AB).eigenvalues)
    print(AB_factor.eigenvalues)
    print()
    print("Singular Values:")
    print(t.linalg.svd(AB).S)
    print(AB_factor.S)
    print("Full SVD:")
    print(AB_factor.svd())

# %%
if MAIN:
    C = t.randn(5, 300)
    ABC = AB @ C
    ABC_factor = AB_factor @ C
    print("Unfactored:", ABC.shape, ABC.norm())
    print("Factored:", ABC_factor.shape, ABC_factor.norm())
    print(f"Right dimension: {ABC_factor.rdim}, Left dimension: {ABC_factor.ldim}, Hidden dimension: {ABC_factor.mdim}")
# %%

if MAIN:
    # YOUR CODE HERE - compute OV circuit
    layer = 1
    head_index = 4
    W_E = model.W_E # (d_vocab, d_model)
    W_U = model.W_U # (d_model, d_vocab)
    V_weights = model.W_V # (layers, heads, d_model, d_head)
    W_V = V_weights[layer,head_index]
    O_weights = model.W_O # (layers, heads, d_head, d_model)
    W_O = O_weights[layer,head_index]
    W_OV = FactoredMatrix(W_V, W_O) # (d_model, d_model)
    full_OV_circuit = (W_E @ W_OV @ W_U)

    tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)

# %%
if MAIN:
    # YOUR CODE HERE - get a random sample from the full OV circuit, so it can be plotted with `imshow`
    d_vocab = model.cfg.d_vocab
    random_indices = t.randint(0, d_vocab, (200,)).sort(dim=0).values

    rows = full_OV_circuit[random_indices]
    full_OV_circuit_sample = rows[:,random_indices]
    full_OV_circuit_sample = full_OV_circuit_sample.AB

    imshow(
        full_OV_circuit_sample,
        labels={"x": "Input token", "y": "Logits on output token"},
        title="Full OV circuit for copying head",
        width=700,
    )
# %%


def top_1_acc(full_OV_circuit: FactoredMatrix) -> float:
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    d_vocab = model.cfg.d_vocab
    top = t.argmax(full_OV_circuit.AB, dim=1)
    correct = (t.arange(d_vocab).to(device) == top).float().mean()
    # gotta send t.arange(d_vocab) to device bc not based off existing tensor
    return correct



if MAIN:
    print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(full_OV_circuit):.4f}")
# %%
