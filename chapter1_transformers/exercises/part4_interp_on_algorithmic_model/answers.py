#%%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from functools import partial
import json
from typing import List, Tuple, Union, Optional, Callable, Dict
import torch as t
from torch import Tensor
from sklearn.linear_model import LinearRegression
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import einops
from tqdm import tqdm
from jaxtyping import Float, Int, Bool
from pathlib import Path
import pandas as pd
import circuitsvis as cv
import webbrowser
from IPython.display import display
from transformer_lens import utils, ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_interp_on_algorithmic_model"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import plotly_utils
from plotly_utils import hist, bar, imshow
import part4_interp_on_algorithmic_model.tests as tests
from part4_interp_on_algorithmic_model.brackets_datasets import SimpleTokenizer, BracketsDataset

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


#%%
VOCAB = "()"

cfg = HookedTransformerConfig(
    n_ctx=42,
    d_model=56,
    d_head=28,
    n_heads=2,
    d_mlp=56,
    n_layers=3,
    attention_dir="bidirectional", # defaults to "causal"
    act_fn="relu",
    d_vocab=len(VOCAB)+3, # plus 3 because of end and pad and start token
    d_vocab_out=2, # 2 because we're doing binary classification
    use_attn_result=True, 
    device=device,
    use_hook_tokens=True
)

model = HookedTransformer(cfg).eval()

state_dict = t.load(section_dir / "brackets_model_state_dict.pt")
model.load_state_dict(state_dict)

#%%
tokenizer = SimpleTokenizer("()")

# Examples of tokenization
# (the second one applies padding, since the sequences are of different lengths)
print(tokenizer.tokenize("()"))
print(tokenizer.tokenize(["()", "()()"]))

# Dictionaries mapping indices to tokens and vice versa
print(tokenizer.i_to_t)
print(tokenizer.t_to_i)

# Examples of decoding (all padding tokens are removed)
print(tokenizer.decode(t.tensor([[0, 3, 4, 2, 1, 1]])))

#%%
def add_perma_hooks_to_mask_pad_tokens(model: HookedTransformer, pad_token: int) -> HookedTransformer:

    # Hook which operates on the tokens, and stores a mask where tokens equal [pad]
    def cache_padding_tokens_mask(tokens: Float[Tensor, "batch seq"], hook: HookPoint) -> None:
        hook.ctx["padding_tokens_mask"] = einops.rearrange(tokens == pad_token, "b sK -> b 1 1 sK")

    # Apply masking, by referencing the mask stored in the `hook_tokens` hook context
    def apply_padding_tokens_mask(
        attn_scores: Float[Tensor, "batch head seq_Q seq_K"],
        hook: HookPoint,
    ) -> None:
        attn_scores.masked_fill_(model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"], -1e5)
        if hook.layer() == model.cfg.n_layers - 1:
            del model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"]

    # Add these hooks as permanent hooks (i.e. they aren't removed after functions like run_with_hooks)
    for name, hook in model.hook_dict.items():
        if name == "hook_tokens":
            hook.add_perma_hook(cache_padding_tokens_mask)
        elif name.endswith("attn_scores"):
            hook.add_perma_hook(apply_padding_tokens_mask)

    return model


model.reset_hooks(including_permanent=True)
model = add_perma_hooks_to_mask_pad_tokens(model, tokenizer.PAD_TOKEN)

#%%
N_SAMPLES = 5000
with open(section_dir / "brackets_data.json") as f:
    data_tuples: List[Tuple[str, bool]] = json.load(f)
    print(f"loaded {len(data_tuples)} examples")
assert isinstance(data_tuples, list)
data_tuples = data_tuples[:N_SAMPLES]
data = BracketsDataset(data_tuples).to(device)
data_mini = BracketsDataset(data_tuples[:100]).to(device)

#%%
hist(
    [len(x) for x, _ in data_tuples], 
    nbins=data.seq_length,
    title="Sequence lengths of brackets in dataset",
    labels={"x": "Seq len"}
)

#%%
# Define and tokenize examples
examples = ["()()", "(())", "))((", "()", "((()()()()))", "(()()()(()(())()", "()(()(((())())()))"]
labels = [True, True, False, True, True, False, True]
toks = tokenizer.tokenize(examples)

# Get output logits for the 0th sequence position (i.e. the [start] token)
logits = model(toks)[:, 0]

# Get the probabilities via softmax, then get the balanced probability (which is the second element)
prob_balanced = logits.softmax(-1)[:, 1]

# Display output
printed_strings = [
    f"{ex:18} : {prob:<8.4%} : label={int(label)}"
    for ex, prob, label in
    zip(examples, prob_balanced, labels)
]
print("Model confidence:\n" + "\n".join(printed_strings))

#%%
def run_model_on_data(model: HookedTransformer, data: BracketsDataset, batch_size: int = 200) -> Float[Tensor, "batch 2"]:
    '''Return probability that each example is balanced'''
    all_logits = []
    for i in tqdm(range(0, len(data.strs), batch_size)):
        toks = data.toks[i : i + batch_size]
        logits = model(toks)[:, 0]
        all_logits.append(logits)
    all_logits = t.cat(all_logits)
    assert all_logits.shape == (len(data), 2)
    return all_logits


test_set = data
n_correct = (run_model_on_data(model, test_set).argmax(-1).bool() == test_set.isbal).sum()
print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")

#%%
def is_balanced_forloop(parens: str) -> bool:
    '''
    Return True if the parens are balanced.

    Parens is just the ( and ) characters, no begin or end tokens.
    '''
    counter = 0
    for c in parens:
        if c == "(":
            counter += 1
        elif c == ")":
            counter -= 1
        if counter < 0:
            return False
    return counter == 0


for (parens, expected) in zip(examples, labels):
    actual = is_balanced_forloop(parens)
    assert expected == actual, f"{parens}: expected {expected} got {actual}"
print("is_balanced_forloop ok!")

#%%
relu = t.nn.functional.relu

def is_balanced_vectorized(tokens: Float[Tensor, "seq_len"]) -> bool:
    '''
    Return True if the parens are balanced.

    tokens is a vector which has start/pad/end indices (0/1/2) as well as left/right brackets (3/4)
    '''
    token_lookup = lambda x: relu(x-2) - 3*relu(x-3)
    vectorized_tokens = token_lookup(tokens).float()
    M = t.tril(t.ones(vectorized_tokens.shape[0], vectorized_tokens.shape[0]))
    cum_sum = M @ vectorized_tokens
    no_negatives = not (relu(-1 * cum_sum).sum().abs() > 1e-2).item()
    ends_with_zero = cum_sum[tokens == tokenizer.END_TOKEN].abs().item() <= 1e-2
    return no_negatives and ends_with_zero


for (tokens, expected) in zip(tokenizer.tokenize(examples), labels):
    actual = is_balanced_vectorized(tokens)
    assert expected == actual, f"{tokens}: expected {expected} got {actual}"
print("is_balanced_vectorized ok!")

#%%
def get_post_final_ln_dir(model: HookedTransformer) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction in which final_ln_output[0, :] should point to maximize P(unbalanced)
    '''
    return model.W_U[:, 0] - model.W_U[:, 1]

tests.test_get_post_final_ln_dir(get_post_final_ln_dir, model)

#%%
def get_activations(
    model: HookedTransformer, 
    toks: Int[Tensor, "batch seq"], 
    names: Union[str, List[str]]
) -> Union[t.Tensor, ActivationCache]:
    '''
    Uses hooks to return activations from the model.

    If names is a string, returns the activations for that hook name.
    If names is a list of strings, returns a dictionary mapping hook names to tensors of activations.
    '''
    names_list = [names] if isinstance(names, str) else names
    _, cache = model.run_with_cache(
        toks,
        return_type=None,
        names_filter=lambda name: name in names_list,
    )

    return cache[names] if isinstance(names, str) else cache

# toks = t.tensor([[0, 3, 4, 3, 4, 2, 1, 1]])
#%%
def LN_hook_names(layernorm: LayerNorm) -> Tuple[str, str]:
    '''
    Returns the names of the hooks immediately before and after a given layernorm.
    e.g. LN_hook_names(model.final_ln) returns ["blocks.2.hook_resid_post", "ln_final.hook_normalized"]
    '''
    if layernorm.name == "ln_final":
        input_hook_name = utils.get_act_name("resid_post", 2)
        output_hook_name = "ln_final.hook_normalized"
    else:
        layer, ln = layernorm.name.split(".")[1:]
        input_hook_name = utils.get_act_name("resid_pre" if ln=="ln1" else "resid_mid", layer)
        output_hook_name = utils.get_act_name('normalized', layer, ln)

    return input_hook_name, output_hook_name


pre_final_ln_name, post_final_ln_name = LN_hook_names(model.ln_final)
print(pre_final_ln_name, post_final_ln_name)

#%%
def get_ln_fit(
    model: HookedTransformer,
    data: BracketsDataset,
    layernorm: LayerNorm,
    seq_pos: Optional[int] = None
) -> Tuple[LinearRegression, float]:
    '''
    if seq_pos is None, find best fit aggregated over all sequence positions.
    Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and the r^2 of the fit
    '''
    input_hook_name, output_hook_name = LN_hook_names(layernorm)
    # input_hook_name, output_hook_name = LN_hook_names(model.ln_final)
    X = get_activations(model, data.toks, input_hook_name).cpu().numpy()
    Y = get_activations(model, data.toks, output_hook_name).cpu().numpy()

    if seq_pos is not None:
        X = X[:, seq_pos, :]
        Y = Y[:, seq_pos, :]
    else:
        X = einops.rearrange(X, "batch seq d_model -> (batch seq) d_model")
        Y = einops.rearrange(Y, "batch seq d_model -> (batch seq) d_model")

    reg = LinearRegression().fit(X, Y)
    return reg, reg.score(X, Y)

tests.test_get_ln_fit(get_ln_fit, model, data_mini)

(final_ln_fit, r2) = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)
print(f"r^2 for LN_final, at sequence position 0: {r2:.4f}")

(final_ln_fit, r2) = get_ln_fit(model, data, layernorm=model.blocks[1].ln1, seq_pos=None)
print(f"r^2 for LN1, layer 1, over all sequence positions: {r2:.4f}")

#%%
def get_pre_final_ln_dir(model: HookedTransformer, data: BracketsDataset) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction in residual stream (pre ln_final, at sequence position 0) which
    most points in the direction of making an unbalanced classification.
    '''
    reg, _ = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)
    L_final_weights = t.tensor(reg.coef_).to(device)
    L_final_bias = t.tensor(reg.intercept_).to(device)
    return L_final_weights.T @ (model.W_U[:, 0] - model.W_U[:, 1])

tests.test_get_pre_final_ln_dir(get_pre_final_ln_dir, model, data_mini)

#%%
def get_out_by_components(
    model: HookedTransformer, data: BracketsDataset
) -> Float[Tensor, "component batch seq_pos emb"]:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output
    of the model's components when run on the data.

    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1,
    mlp 1, head 2.0, head 2.1, mlp 2]
    '''
    component_resid = t.empty(
        (10, data.toks.shape[0], data.toks.shape[1], model.cfg.d_model)
    ).to(device)

    act_names = [
        "hook_embed",
        "hook_pos_embed",
        "blocks.0.attn.hook_result",
        "blocks.0.hook_mlp_out",
        "blocks.1.attn.hook_result",
        "blocks.1.hook_mlp_out",
        "blocks.2.attn.hook_result",
        "blocks.2.hook_mlp_out",
    ]

    cache = get_activations(model, data.toks, act_names)    

    component_resid[0] = cache["hook_embed"] + cache["hook_pos_embed"]
    component_resid[1] = cache["blocks.0.attn.hook_result"][:, :, 0, :]
    component_resid[2] = cache["blocks.0.attn.hook_result"][:, :, 1, :]
    component_resid[3] = cache["blocks.0.hook_mlp_out"]
    component_resid[4] = cache["blocks.1.attn.hook_result"][:, :, 0, :]
    component_resid[5] = cache["blocks.1.attn.hook_result"][:, :, 1, :]
    component_resid[6] = cache["blocks.1.hook_mlp_out"]
    component_resid[7] = cache["blocks.2.attn.hook_result"][:, :, 0, :]
    component_resid[8] = cache["blocks.2.attn.hook_result"][:, :, 1, :]
    component_resid[9] = cache["blocks.2.hook_mlp_out"]
    
    return component_resid

tests.test_get_out_by_components(get_out_by_components, model, data_mini)


#%%
component_resid = get_out_by_components(model, data)
unbal_dir = get_pre_final_ln_dir(model, data)

out_by_component_in_unbalanced_dir = einops.einsum(
    component_resid[:, :, 0, :], unbal_dir,
    "component batch d_model, d_model -> component batch",
)

out_by_component_bal_mean = out_by_component_in_unbalanced_dir[:, data.isbal].mean(dim=1, keepdim=True)
out_by_component_in_unbalanced_dir -= out_by_component_bal_mean

tests.test_out_by_component_in_unbalanced_dir(out_by_component_in_unbalanced_dir, model, data)

plotly_utils.hists_per_comp(
    out_by_component_in_unbalanced_dir, 
    data, xaxis_range=[-10, 20]
)

#%%
def is_balanced_vectorized_return_both(
        toks: Float[Tensor, "batch seq"]
) -> Tuple[Bool[Tensor, "batch"], Bool[Tensor, "batch"]]:
    # SOLUTION
    table = t.tensor([0, 0, 0, 1, -1]).to(device)
    change = table[toks.to(device)].flip(-1)
    altitude = t.cumsum(change, -1)
    total_elevation_failure = altitude[:, -1] != 0
    negative_failure = altitude.max(-1).values > 0
    return total_elevation_failure, negative_failure


total_elevation_failure, negative_failure = is_balanced_vectorized_return_both(data.toks)

h20_in_unbalanced_dir = out_by_component_in_unbalanced_dir[7]
h21_in_unbalanced_dir = out_by_component_in_unbalanced_dir[8]

tests.test_total_elevation_and_negative_failures(data, total_elevation_failure, negative_failure)

#%%
failure_types_dict = {
    "both failures": negative_failure & total_elevation_failure,
    "just neg failure": negative_failure & ~total_elevation_failure,
    "just total elevation failure": ~negative_failure & total_elevation_failure,
    "balanced": ~negative_failure & ~total_elevation_failure
}

plotly_utils.plot_failure_types_scatter(
    h20_in_unbalanced_dir,
    h21_in_unbalanced_dir,
    failure_types_dict,
    data
)

#%%
plotly_utils.plot_contribution_vs_open_proportion(
    h20_in_unbalanced_dir, 
    "Head 2.0 contribution vs proportion of open brackets '('",
    failure_types_dict, 
    data
)

#%%
plotly_utils.plot_contribution_vs_open_proportion(
    h21_in_unbalanced_dir, 
    "Head 2.1 contribution vs proportion of open brackets '('",
    failure_types_dict,
    data
)

#%%
def get_attn_probs(model: HookedTransformer, data: BracketsDataset, layer: int, head: int) -> t.Tensor:
    '''
    Returns: (N_SAMPLES, max_seq_len, max_seq_len) tensor that sums to 1 over the last dimension.
    '''
    name = f"blocks.{layer}.attn.hook_pattern"
    attn_probs = get_activations(model, data.toks, name)[:, head, :, :]
    return attn_probs


tests.test_get_attn_probs(get_attn_probs, model, data_mini)

#%%
attn_probs_20: Float[Tensor, "batch seqQ seqK"] = get_attn_probs(model, data, 2, 0)
attn_probs_20_open_query0 = attn_probs_20[data.starts_open].mean(0)[0]

bar(
    attn_probs_20_open_query0,
    title="Avg Attention Probabilities for query 0, first token '(', head 2.0",
    width=700, template="simple_white"
)

#%%
def get_WOV(model: HookedTransformer, layer: int, head: int) -> Float[Tensor, "d_model d_model"]:
    '''
    Returns the W_OV matrix for a particular layer and head.
    '''
    return model.W_V[layer, head] @ model.W_O[layer, head]

def get_pre_20_dir(model, data) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction propagated back through the OV matrix of 2.0 
    and then through the layernorm before the layer 2 attention heads.
    '''
    pre_final_ln_dir = get_pre_final_ln_dir(model, data)
    W_OV_20 = get_WOV(model, 2, 0)
    reg, r2_score = get_ln_fit(model, data, model.blocks[2].ln1, seq_pos=1)
    L1_linear = t.tensor(reg.coef_).to(device)
    return L1_linear.T @ W_OV_20 @ pre_final_ln_dir


tests.test_get_pre_20_dir(get_pre_20_dir, model, data_mini)

#%%
component_resid = get_out_by_components(model, data)[:6+1]
unbal_dir = get_pre_20_dir(model, data)

out_by_component_in_pre_20_unbalanced_dir = einops.einsum(
    component_resid[:, :, 1, :], unbal_dir,
    "component batch d_model, d_model -> component batch",
)

out_by_component_bal_mean = out_by_component_in_pre_20_unbalanced_dir[:, data.isbal].mean(dim=1, keepdim=True)
out_by_component_in_pre_20_unbalanced_dir -= out_by_component_bal_mean

tests.test_out_by_component_in_pre_20_unbalanced_dir(out_by_component_in_pre_20_unbalanced_dir, model, data)

plotly_utils.hists_per_comp(
    out_by_component_in_pre_20_unbalanced_dir, 
    data, xaxis_range=(-5, 12)
)


#%%
plotly_utils.mlp_attribution_scatter(
    out_by_component_in_pre_20_unbalanced_dir,
    data, failure_types_dict
)

#%%
def get_out_by_neuron(
    model: HookedTransformer, 
    data: BracketsDataset, 
    layer: int, 
    seq: Optional[int] = None
) -> Float[Tensor, "batch *seq neuron d_model"]:
    '''
    If seq is not None, then out[b, s, i, :] = f(x[b, s].T @ W_in[:, i]) @ W_out[i, :],
    i.e. the vector which is written to the residual stream by the ith neuron (where x
    is the input to the residual stream (i.e. shape (batch, seq, d_model)).

    If seq is None, then out[b, i, :] = vector f(x[b].T @ W_in[:, i]) @ W_out[i, :]

    (Note, using * in jaxtyping indicates an optional dimension)
    '''
    mlp_post = get_activations(model, data.toks, utils.get_act_name("post", layer))
    W_out = model.W_out[layer]
    out = einops.einsum(mlp_post, W_out, "batch seq neuron, neuron d_model -> batch seq neuron d_model")

    if seq is not None:
        return out[:, seq, :, :]
    else:
        return out
    
def get_out_by_neuron_in_20_dir(
    model: HookedTransformer,
    data: BracketsDataset,
    layer: int
) -> Float[Tensor, "batch neurons"]:
    '''
    [b, s, i]th element is the contribution of the vector written by the ith 
    neuron to the residual stream in the 
    unbalanced direction (for the b-th element in the batch, and the s-th 
    sequence position).

    In other words we need to take the vector produced by the `get_out_by_neuron` 
    function, and project it onto the 
    unbalanced direction for head 2.0 (at seq pos = 1).
    '''
    neuron_cont = get_out_by_neuron(model, data, layer, seq=1)
    unbal_dir = get_pre_20_dir(model, data)
    return einops.einsum(neuron_cont, unbal_dir, "batch neurons d_model, d_model -> batch neurons")

tests.test_get_out_by_neuron(get_out_by_neuron, model, data_mini)
tests.test_get_out_by_neuron_in_20_dir(get_out_by_neuron_in_20_dir, model, data_mini)