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
# %%
if MAIN:
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

# %%
if MAIN:
    tokenizer = SimpleTokenizer("()")

    # Examples of tokenization
    # (the second one applies padding, since the sequences are of different lengths)
    print(tokenizer.tokenize("()(((((())))))"))
    print(tokenizer.tokenize(["()", "()()"]))

    # Dictionaries mapping indices to tokens and vice versa
    print(tokenizer.i_to_t)
    print(tokenizer.t_to_i)

    # Examples of decoding (all padding tokens are removed)
    print(tokenizer.decode(t.tensor([[0, 3, 4, 2, 1, 1]])))

# %%
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


if MAIN:
    model.reset_hooks(including_permanent=True)
    model = add_perma_hooks_to_mask_pad_tokens(model, tokenizer.PAD_TOKEN)
# %%
if MAIN:
    N_SAMPLES = 5000
    with open(section_dir / "brackets_data.json") as f:
        data_tuples: List[Tuple[str, bool]] = json.load(f)
        print(f"loaded {len(data_tuples)} examples")
    assert isinstance(data_tuples, list)
    data_tuples = data_tuples[:N_SAMPLES]
    data = BracketsDataset(data_tuples).to(device)
    data_mini = BracketsDataset(data_tuples[:100]).to(device)

# %%
"""
data is all even length sequence strings
would be easy for an attention head to detect odd vs even length sequences:
query looks for end token, looks at embedding part in residual stream strongly
key looks at positional embedding, is end token in odd position (even length sequence)
or even position (odd length sequence then)
"""

if MAIN:
    hist(
        [len(x) for x, _ in data_tuples], 
        nbins=data.seq_length,
        title="Sequence lengths of brackets in dataset",
        labels={"x": "Seq len"}
    )

# %%
# Define and tokenize examples

if MAIN:
    examples = ["()()", "(())", "))((", "()", "((()()()()))", "(()()()(()(())()", "()(()(((())())()))"]
    labels = [True, True, False, True, True, False, True]
    toks = tokenizer.tokenize(examples)

    # Get output logits for the 0th sequence position (i.e. the [start] token)
    logits = model(toks)[:, 0]

    # Get the probabilities via softmax, then get the balanced probability (which is the second element)
    prob_balanced = logits.softmax(-1)[:, 1]

    # Display output
    print("Model confidence:\n" + "\n".join([f"{ex:18} : {prob:<8.4%} : label={int(label)}" for ex, prob, label in zip(examples, prob_balanced, labels)]))


# %%
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


if MAIN:
    test_set = data
    n_correct = (run_model_on_data(model, test_set).argmax(-1).bool() == test_set.isbal).sum()
    print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")
# %%
def is_balanced_forloop(parens: str) -> bool:
    '''
    Return True if the parens are balanced.

    Parens is just the ( and ) characters, no begin or end tokens.
    '''
    paren_count = 0
    for paren in parens:
        if paren == "(":
            paren_count += 1
        elif paren == ")":
            paren_count -= 1
        if paren_count < 0:
            return False
    return paren_count == 0


if MAIN:
    for (parens, expected) in zip(examples, labels):
        actual = is_balanced_forloop(parens)
        assert expected == actual, f"{parens}: expected {expected} got {actual}"
    print("is_balanced_forloop ok!")


# %%
def is_balanced_vectorized(tokens: Float[Tensor, "seq_len"]) -> bool:
    '''
    Return True if the parens are balanced.

    tokens is a vector which has start/pad/end indices (0/1/2) as well as left/right brackets (3/4)
    '''
    left_brackets = tokens == 3
    right_brackets = tokens == 4
    mapped = left_brackets * 1 + right_brackets * -1

    cum_brackets = t.cumsum(mapped, dim=0)
    if cum_brackets[-1] != 0:
        return False
    elif t.any(cum_brackets < 0):
        return False
    return True


if MAIN:
    for (tokens, expected) in zip(tokenizer.tokenize(examples), labels):
        actual = is_balanced_vectorized(tokens)
        assert expected == actual, f"{tokens}: expected {expected} got {actual}"
    print("is_balanced_vectorized ok!")
# %%
def get_post_final_ln_dir(model: HookedTransformer) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction in which final_ln_output[0, :] should point to maximize P(unbalanced)
    '''
    return (model.W_U[:, 0] - model.W_U[:, 1])


if MAIN:
    tests.test_get_post_final_ln_dir(get_post_final_ln_dir, model)

# %%
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



if MAIN:
    pre_final_ln_name, post_final_ln_name = LN_hook_names(model.ln_final)
    print(pre_final_ln_name, post_final_ln_name)


# %%

def get_ln_fit(
    model: HookedTransformer, data: BracketsDataset, layernorm: LayerNorm, seq_pos: Optional[int] = None
) -> Tuple[LinearRegression, float]:
    '''
    if seq_pos is None, find best fit aggregated over all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and the r^2 of the fit
    '''
    input_hook_name, output_hook_name = LN_hook_names(layernorm)
    
    model_activations = get_activations(model, data.toks, names=[input_hook_name, output_hook_name])
    pre_ln_input = model_activations[input_hook_name].cpu()
    post_ln_input = model_activations[output_hook_name].cpu()
    
    if seq_pos is not None:
        x = pre_ln_input[:, seq_pos]
        y = post_ln_input[:, seq_pos]
    else:
        x = einops.rearrange(pre_ln_input, "b s d -> (b s) d")
        y = einops.rearrange(post_ln_input, "b s d -> (b s) d")

    reg = LinearRegression()
    return (reg.fit(x, y), reg.score(x, y))


if MAIN:
    tests.test_get_ln_fit(get_ln_fit, model, data_mini)

    (final_ln_fit, r2) = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)
    print(f"r^2 for LN_final, at sequence position 0: {r2:.4f}")

    (final_ln_fit, r2) = get_ln_fit(model, data, layernorm=model.blocks[1].ln1, seq_pos=None)
    print(f"r^2 for LN1, layer 1, over all sequence positions: {r2:.4f}")
# %%

def get_pre_final_ln_dir(model: HookedTransformer, data: BracketsDataset) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction in residual stream (pre ln_final, at sequence position 0) which
    most points in the direction of making an unbalanced classification.
    '''
    # input is transformed via layernorm to (model.W_U[:, 0] - model.W_U[:, 1])

    (final_ln_fit, r2) = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)
    

    # want to go backwards, logit_diff = (reg.coef_ @ x) @ (post_final_dir)
    # same as x @ (reg.coef_.T @ (W_U[:,0] - W_U[:, 1]))
    # x should be in same direction as (reg.coef_.T @ (W_U[:,0] - W_U[:, 1])) to maximize logit diff
    return t.tensor(final_ln_fit.coef_).T.to(device=cfg.device) @ get_post_final_ln_dir(model)

    # return (t.tensor(np.linalg.inv(final_ln_fit.coef_)) @ (end_vec.cpu() - final_ln_fit.intercept_)).to(device=cfg.device)


if MAIN:
    tests.test_get_pre_final_ln_dir(get_pre_final_ln_dir, model, data_mini)
# %%

def get_out_by_components(model: HookedTransformer, data: BracketsDataset) -> Float[Tensor, "component batch seq_pos emb"]:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2]
    '''
    embed_names = ["hook_embed", "hook_pos_embed"]
    attn_out_names = [utils.get_act_name("result", layer) for layer in range(model.cfg.n_layers)]
    mlp_out_names = [utils.get_act_name("mlp_out", layer) for layer in range(model.cfg.n_layers)]

    activation_cache = get_activations(model, data.toks, embed_names+attn_out_names+mlp_out_names)
    
    out_components = t.zeros(size=(10, data.toks.shape[0], data.toks.shape[1], model.cfg.d_model))
    out_components[0] = activation_cache['embed'] + activation_cache['pos_embed']
    for layer in range(3):
        attn_name = attn_out_names[layer]
        mlp_name = mlp_out_names[layer]
        out_components[3*layer+1:3*layer+3] = einops.rearrange(activation_cache[attn_name],
                                                               "b s h d -> h b s d")
        out_components[3*layer+3] = activation_cache[mlp_name]

    return out_components.to(device=cfg.device)


if MAIN:
    tests.test_get_out_by_components(get_out_by_components, model, data_mini)
# %%
if MAIN:
    # YOUR CODE HERE - define the object `out_by_component_in_unbalanced_dir`
    # want to calculate the magnitude of the output in the direction of get_pre_final_ln_dir
    pre_final_dir = get_pre_final_ln_dir(model, data) # (emb,) direction vector (magnitude not necessarily 1)

    out_components = get_out_by_components(model, data)[:,:,0] # [10, dataset_size, emb]
    out_by_component_in_unbalanced_dir = einops.einsum(pre_final_dir, out_components, 
                                                       "emb, comp b emb -> comp b")

    balanced_out_components = out_by_component_in_unbalanced_dir[:, data.isbal].mean(dim=1)

    out_by_component_in_unbalanced_dir -= einops.repeat(balanced_out_components, "comp -> comp b", b=5000)
    

    tests.test_out_by_component_in_unbalanced_dir(out_by_component_in_unbalanced_dir, model, data)

    plotly_utils.hists_per_comp(
        out_by_component_in_unbalanced_dir, 
        data, xaxis_range=[-10, 20]
    )

# %%
def is_balanced_vectorized_return_both(
        toks: Float[Tensor, "batch seq"]
) -> Tuple[Bool[Tensor, "batch"], Bool[Tensor, "batch"]]:
    left_brackets = toks == 3
    right_brackets = toks == 4
    mapped = left_brackets * -1 + right_brackets * 1

    cum_brackets = t.cumsum(mapped.flip(dims=(1,)), dim=1)

    neg_fails = t.any(cum_brackets < 0, dim=1)
    elevation_fails = cum_brackets[:,-1] != 0
    # neg_fails = t.zeros_like(elevation_fails)
    # elevation_fails = t.zeros_like(neg_fails)

    return elevation_fails, neg_fails

if MAIN:
    total_elevation_failure, negative_failure = is_balanced_vectorized_return_both(data.toks)

    h20_in_unbalanced_dir = out_by_component_in_unbalanced_dir[7]
    h21_in_unbalanced_dir = out_by_component_in_unbalanced_dir[8]

    tests.test_total_elevation_and_negative_failures(data, total_elevation_failure, negative_failure)
# %%
if MAIN:
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
# %%
def get_attn_probs(model: HookedTransformer, data: BracketsDataset, layer: int, head: int) -> t.Tensor:
    '''
    Returns: (N_SAMPLES, max_seq_len, max_seq_len) tensor that sums to 1 over the last dimension.
    '''
    attn_out_name = utils.get_act_name("pattern", layer)
    activation_cache = get_activations(model, data.toks, [attn_out_name])
    
    head_activation = activation_cache[attn_out_name][:,head]
    return head_activation


if MAIN:
    tests.test_get_attn_probs(get_attn_probs, model, data_mini)
# %%
if MAIN:
    attn_probs_20: Float[Tensor, "batch seqQ seqK"] = get_attn_probs(model, data, 2, 0)
    attn_probs_20_open_query0 = attn_probs_20[data.starts_open].mean(0)[0]

    bar(
        attn_probs_20_open_query0,
        title="Avg Attention Probabilities for query 0, first token '(', head 2.0",
        width=700, template="simple_white"
    )

# %%
def get_WOV(model: HookedTransformer, layer: int, head: int) -> Float[Tensor, "d_model d_model"]:
    # W_V is (d_model, d_head)
    # W_O is (d_head, d_model)
    return einops.einsum(model.W_V[layer, head], model.W_O[layer, head], "m_1 h, h m_2 -> m_1 m_2")

def get_pre_20_dir(model, data) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction propagated back through the OV matrix of 2.0 
    and then through the layernorm before the layer 2 attention heads.
    
    We know most of x_2[0] comes from x_1[1], so find direction of x_1[1]
    that contributes to logit difference. Have to go back through another LN
    From get_pre_final_ln_dir:
    logit_diff = x_2[0] @ get_post_final_ln_dir
    x_2[0] = LN(x_1[1]) @ W_OV @ pre_20_dir
    (attn_probs is why you only care about x_1[1], but you don't need to operate
    on attn_probs directly)
    '''

    w_ov = get_WOV(model, 2, 0) # (d_model, d_model)
    # do layernorm regression again, this time caring about 1st sequence position
    (ln_1_fit, r2) = get_ln_fit(model, data, layernorm=model.blocks[2].ln1, seq_pos=1)
    # ln_1_fit.coef_ shape is (d_model, d_model)

    pre_final_ln_dir = get_pre_final_ln_dir(model, data) # (d_model,)
    # x_1[1] @ L_1.T @ W_OV @ pre_final_ln_dir = logit_dif, want x_1[1] in same dir as
    # all following part for dot product maximization
    return t.tensor(ln_1_fit.coef_.T).to(device=cfg.device) @ w_ov @ pre_final_ln_dir
    

if MAIN:
    tests.test_get_pre_20_dir(get_pre_20_dir, model, data_mini)
# %%

if MAIN:
    # YOUR CODE HERE - define `out_by_component_in_pre_20_unbalanced_dir` (for all components before head 2.0)
    pre_20_dir = get_pre_20_dir(model, data)

    # want embed, layer 0 heads+mlp, layer 1 heads+mlp
    out_components_20 = get_out_by_components(model, data)[:7,:,1] # [7, dataset_size, emb in pos 1]

    out_by_component_in_pre_20_unbalanced_dir = einops.einsum(pre_20_dir, out_components_20, 
                                                       "emb, comp b emb -> comp b")

    balanced_out_components = out_by_component_in_pre_20_unbalanced_dir[:, data.isbal].mean(dim=1)

    out_by_component_in_pre_20_unbalanced_dir -= einops.repeat(balanced_out_components, "comp -> comp b", b=5000)


    tests.test_out_by_component_in_pre_20_unbalanced_dir(out_by_component_in_pre_20_unbalanced_dir, model, data)

    plotly_utils.hists_per_comp(
        out_by_component_in_pre_20_unbalanced_dir, 
        data, xaxis_range=(-5, 12)
    )
# %%

if MAIN:
    plotly_utils.mlp_attribution_scatter(out_by_component_in_pre_20_unbalanced_dir, data, failure_types_dict)


# %%
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

    # want input into MLP, same as residual stream after layer's attn_heads
    # names = utils.get_act_name("mlp_pre", layer)
    # before_neuron = get_activations(model, data.toks, names) # (batch, pos, d_model)

    # W_in = model.W_in[layer] # (d_model, neurons)
    # b_in = model.b_in[layer] # (neurons,)

    # before_act = einops.einsum(before_neuron, W_in, "b pos d_m, d_m neurons -> b pos neurons")
    # # before_act = einops.einsum(before_neuron, W_in, "b pos d_m, neurons d_m -> b pos neurons")
    # after_act = t.relu(before_act + b_in)  # (batch pos neurons)

    # assert after_act.shape == (data.toks.shape[0], data.toks.shape[1], W_in.shape[1])

    names = utils.get_act_name("mlp_post", layer)
    after_act = get_activations(model, data.toks, names) # (batch, pos, neurons)

    W_out = model.W_out[layer] # (neurons, d_model)

    out = einops.einsum(after_act, W_out, "b pos neurons, neurons d_model -> b pos neurons d_model")
    # out = einops.einsum(after_act, W_out, "b pos neurons, d_model neurons -> b pos neurons d_model")
    if seq is not None:
        out = out[:, seq]
    return out

    # if seq is None:
    #     # out = t.zeros(size=(data.toks.shape[0], W_in.shape[1], model.cfg.d_model))
    #     before_act = einops.einsum(before_neuron, W_in, "b pos d_m, d_m neurons -> b neurons")
    # else:
    #     # out = t.zeros(size=(data.toks.shape[0], W_in.shape[1], model.cfg.d_model))
    #     before_act = einops.einsum(before_neuron, W_in, "b pos d_m, d_m neurons -> b pos neurons")


def get_out_by_neuron_in_20_dir(model: HookedTransformer, data: BracketsDataset, layer: int) -> Float[Tensor, "batch neurons"]:
    '''
    [b, s, i]th element is the contribution of the vector written by the ith neuron to the residual stream in the 
    unbalanced direction (for the b-th element in the batch, and the s-th sequence position).

    In other words we need to take the vector produced by the `get_out_by_neuron` function, and project it onto the 
    unbalanced direction for head 2.0 (at seq pos = 1).
    '''
    neuron_outs = get_out_by_neuron(model, data, layer, 1) # (batch neurons d_model)
    unbalanced_dir = get_pre_20_dir(model, data) # (d_model)
    return einops.einsum(neuron_outs, unbalanced_dir, "b n d_m, d_m -> b n")


if MAIN:
    tests.test_get_out_by_neuron(get_out_by_neuron, model, data_mini)
    tests.test_get_out_by_neuron_in_20_dir(get_out_by_neuron_in_20_dir, model, data_mini)

# %%
def get_out_by_neuron_in_20_dir_less_memory(model: HookedTransformer, data: BracketsDataset, layer: int) -> Float[Tensor, "batch neurons"]:
    '''
    Has the same output as `get_out_by_neuron_in_20_dir`, but uses less memory (because it never stores
    the output vector of each neuron individually).
    '''
    names = utils.get_act_name("mlp_post", layer)
    after_act = get_activations(model, data.toks, names)[:,1] # (batch, neurons)

    W_out = model.W_out[layer] # (neurons, d_model)

    unbalanced_dir = get_pre_20_dir(model, data) # (d_model)
    W_unbalanced = einops.einsum(W_out, unbalanced_dir, "n d_m, d_m -> n")

    return einops.einsum(after_act, W_unbalanced, "b n, n -> b n")


tests.test_get_out_by_neuron_in_20_dir_less_memory(get_out_by_neuron_in_20_dir_less_memory, model, data_mini)
# %%
for layer in range(2):
    # Get neuron significances for head 2.0, sequence position #1 output
    neurons_in_unbalanced_dir = get_out_by_neuron_in_20_dir_less_memory(model, data, layer)[utils.to_numpy(data.starts_open), :]
    # Plot neurons' activations
    plotly_utils.plot_neurons(neurons_in_unbalanced_dir, model, data, failure_types_dict, layer, renderer="browser")

# %%
def get_q_and_k_for_given_input(
    model: HookedTransformer, 
    tokenizer: SimpleTokenizer,
    parens: str, 
    layer: int, 
) -> Tuple[Float[Tensor, "seq_d_model"], Float[Tensor,  "seq_d_model"]]:
    '''
    Returns the queries and keys (both of shape [seq, d_model]) for the given parns input, in the attention head `layer.head`.
    '''
    activation_cache = get_activations(model, tokenizer.tokenize(parens), [utils.get_act_name("k", layer), utils.get_act_name("q", layer)])
    return (activation_cache[utils.get_act_name("q", layer)][0], activation_cache[utils.get_act_name("k", layer)][0])


tests.test_get_q_and_k_for_given_input(get_q_and_k_for_given_input, model, tokenizer)
# %%
layer = 0
all_left_parens = "".join(["(" * 40])
all_right_parens = "".join([")" * 40])

model.reset_hooks()
q0_all_left, k0_all_left = get_q_and_k_for_given_input(model, tokenizer, all_left_parens, layer)
q0_all_right, k0_all_right = get_q_and_k_for_given_input(model, tokenizer, all_right_parens, layer)
k0_avg = (k0_all_left + k0_all_right) / 2


# Define hook function to patch in q or k vectors
def hook_fn_patch_qk(
    value: Float[Tensor, "batch seq head d_head"], 
    hook: HookPoint, 
    new_value: Float[Tensor, "... seq d_head"],
    head_idx: Optional[int] = None
) -> None:
    if head_idx is not None:
        value[..., head_idx, :] = new_value[..., head_idx, :]
    else:
        value[...] = new_value[...]


# Define hook function to display attention patterns (using plotly)
def hook_fn_display_attn_patterns(
    pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    head_idx: int = 0
) -> None:
    avg_head_attn_pattern = pattern.mean(0)
    labels = ["[start]", *[f"{i+1}" for i in range(40)], "[end]"]
    display(cv.attention.attention_heads(
        tokens=labels, 
        attention=avg_head_attn_pattern,
        attention_head_names=["0.0", "0.1"],
        max_value=avg_head_attn_pattern.max()
    ))


# Run our model on left parens, but patch in the average key values for left vs right parens
# This is to give us a rough idea how the model behaves on average when the query is a left paren
model.run_with_hooks(
    tokenizer.tokenize(all_left_parens).to(device),
    return_type=None,
    fwd_hooks=[
        (utils.get_act_name("k", layer), partial(hook_fn_patch_qk, new_value=k0_avg)),
        (utils.get_act_name("pattern", layer), hook_fn_display_attn_patterns),
    ]
)
# %%

def hook_fn_display_attn_patterns_for_single_query(
    pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    head_idx: int = 0,
    query_idx: int = 1
):
    bar(
        utils.to_numpy(pattern[:, head_idx, query_idx].mean(0)), 
        title=f"Average attn probabilities on data at posn 1, with query token = '('",
        labels={"index": "Sequence position of key", "value": "Average attn over dataset"}, 
        height=500, width=800, yaxis_range=[0, 0.1], template="simple_white"
    )


data_len_40 = BracketsDataset.with_length(data_tuples, 40).to(device)

model.reset_hooks()
model.run_with_hooks(
    data_len_40.toks[data_len_40.isbal],
    return_type=None,
    fwd_hooks=[
        (utils.get_act_name("q", 0), partial(hook_fn_patch_qk, new_value=q0_all_left)),
        (utils.get_act_name("pattern", 0), hook_fn_display_attn_patterns_for_single_query),
    ]
)
# %%
def embedding(model: HookedTransformer, tokenizer: SimpleTokenizer, char: str) -> Float[Tensor, "d_model"]:
    assert char in ("(", ")")
    idx = tokenizer.t_to_i[char]
    return model.W_E[idx]


# YOUR CODE HERE - define v_L and v_R, as described above.
layernorm_approx = t.tensor(get_ln_fit(model, data, model.blocks[0].ln1, 1)[0].coef_).to(device=cfg.device) # ( is in pos 1
W_ov = model.W_V[0,0] @ model.W_O[0,0] # hopefully this works lmao
v_L = embedding(model, tokenizer, "(") @ layernorm_approx.T @ W_ov 
v_R = embedding(model, tokenizer, ")") @ layernorm_approx.T @ W_ov
print("Cosine similarity: ", t.cosine_similarity(v_L, v_R, dim=0).item())


# %%
# def cos_sim_with_MLP_weights(model: HookedTransformer, v: Float[Tensor, "d_model"], layer: int) -> Float[Tensor, "d_mlp"]:
#     '''
#     Returns a vector of length d_mlp, where the ith element is the cosine similarity between v and the 
#     ith in-direction of the MLP in layer `layer`.

#     Recall that the in-direction of the MLPs are the columns of the W_in matrix.
#     '''
#     W_in = model.W_in[layer] # (d_model, d_mlp)
#     cosine_sims = t.cosine_similarity(einops.rearrange(W_in, "d_m n -> n d_m"), v).to(device=cfg.device)
#     return cosine_sims

# def avg_squared_cos_sim(v: Float[Tensor, "d_model"], n_samples: int = 1000) -> float:
#     '''
#     Returns the average (over n_samples) cosine similarity between v and another randomly chosen vector.

#     We can create random vectors from the standard N(0, I) distribution.
#     '''
#     random_vecs = t.zeros(size=(n_samples, v.shape[0])).to(device=cfg.device)
#     return t.cosine_similarity(random_vecs, v) ** 2

def cos_sim_with_MLP_weights(model: HookedTransformer, v: Float[Tensor, "d_model"], layer: int) -> Float[Tensor, "d_mlp"]:
    '''
    Returns a vector of length d_mlp, where the ith element is the cosine similarity between v and the 
    ith in-direction of the MLP in layer `layer`.

    Recall that the in-direction of the MLPs are the columns of the W_in matrix.
    '''
    # SOLUTION
    v_unit = v / v.norm()
    W_in_unit = model.W_in[layer] / model.W_in[layer].norm(dim=0)

    return einops.einsum(v_unit, W_in_unit, "d_model, d_model d_mlp -> d_mlp")

def avg_squared_cos_sim(v: Float[Tensor, "d_model"], n_samples: int = 1000) -> float:
    '''
    Returns the average (over n_samples) cosine similarity between v and another randomly chosen vector.

    We can create random vectors from the standard N(0, I) distribution.
    '''
    # SOLUTION
    v2 = t.randn(n_samples, v.shape[0]).to(device)
    v2 /= v2.norm(dim=1, keepdim=True)

    v1 = v / v.norm()

    return (v1 * v2).pow(2).sum(1).mean().item()

print("Avg squared cosine similarity of v_R with ...\n")

cos_sim_mlp0 = cos_sim_with_MLP_weights(model, v_R, 0)
print(f"...MLP input directions in layer 0:  {cos_sim_mlp0.pow(2).mean():.6f}")

cos_sim_mlp1 = cos_sim_with_MLP_weights(model, v_R, 1)
print(f"...MLP input directions in layer 1:  {cos_sim_mlp1.pow(2).mean():.6f}")

cos_sim_rand = avg_squared_cos_sim(v_R)
print(f"...random vectors of len = d_model:  {cos_sim_rand:.6f}")
# %%
# want to identify anywhere-negative failures, which head 2.1 detects
# construct parens string going negative at places

neg_parens = ["())(())()((())", "(()))(()", "(((()))())(())"] # 0 and 1 fail, 2 works
neg_tokens = tokenizer.tokenize(neg_parens[2])

attn_out_name = utils.get_act_name("pattern", 2)
activation_cache = get_activations(model, neg_tokens, [attn_out_name])

head_activation = activation_cache[attn_out_name][:,1]

if MAIN:
    attn_probs_21: Float[Tensor, "batch seqQ seqK"] = head_activation
    attn_probs_21_open_query0 = attn_probs_21.mean(0)[0]

    bar(
        attn_probs_21_open_query0,
        title="Avg Attention Probabilities for query 0, first token '(', head 2.1",
        width=700, template="simple_white"
    )
# %%
# identify important components for 2.1

def get_pre_21_dir(model, data) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction propagated back through the OV matrix of 2.1
    and then through the layernorm before the layer 2 attention heads.
    
    We know most of x_2[0] comes from x_1[1], so find direction of x_1[1]
    that contributes to logit difference. Have to go back through another LN
    From get_pre_final_ln_dir:
    logit_diff = x_2[0] @ get_post_final_ln_dir
    x_2[0] = LN(x_1[1]) @ W_OV @ pre_21_dir
    (attn_probs is why you only care about x_1[1], but you don't need to operate
    on attn_probs directly)
    '''

    w_ov = get_WOV(model, 2, 1) # (d_model, d_model)
    # do layernorm regression again, this time caring about 1st sequence position
    (ln_1_fit, r2) = get_ln_fit(model, data, layernorm=model.blocks[2].ln1)
    # ln_1_fit.coef_ shape is (d_model, d_model)

    pre_final_ln_dir = get_pre_final_ln_dir(model, data) # (d_model,)
    # x_1[1] @ L_1.T @ W_OV @ pre_final_ln_dir = logit_dif, want x_1[1] in same dir as
    # all following part for dot product maximization
    return t.tensor(ln_1_fit.coef_.T).to(device=cfg.device) @ w_ov @ pre_final_ln_dir
    
if MAIN:
    # YOUR CODE HERE - define `out_by_component_in_pre_21_unbalanced_dir` (for all components before head 2.0)
    pre_21_dir = get_pre_21_dir(model, data)

    # want embed, layer 0 heads+mlp, layer 1 heads+mlp
    out_components_21 = get_out_by_components(model, data)[:7,:,1] # [7, dataset_size, emb in pos 1]

    out_by_component_in_pre_21_unbalanced_dir = einops.einsum(pre_21_dir, out_components_21, 
                                                       "emb, comp b emb -> comp b")

    balanced_out_components = out_by_component_in_pre_21_unbalanced_dir[:, data.isbal].mean(dim=1)

    out_by_component_in_pre_21_unbalanced_dir -= einops.repeat(balanced_out_components, "comp -> comp b", b=5000)
    
    plotly_utils.hists_per_comp(
        out_by_component_in_pre_21_unbalanced_dir, 
        data, xaxis_range=(-5, 12)
    )
# %%
# Adversarial Approach 1: in head 0.0, query tokens 28-32 (parens in location 27-31) pay a ton of attention to positions 39 and 40
# Try balanced string that is slightly unbalanced at the start but mostly correct at positions 39 and 40 (last two)? (nope)
# answer: want negative elevation to occur at locations 27-31, where the ( that causes a negative elevation might pay more attention to a ) at the end (position 39 or 40)

# unbalanced_string = "((((((((()))))))))()((((((()())())))))()" # two extra ( at start
# unbalanced_string = "(((((((((()))))))))))(((((((((()))))))))" # works as adversarial

# 28 brackets, then a )( unbalanced part, ( might pay too much attention to ) at end
unbalanced_string = "(((((((((((((())))))))))))))\
)(\
((((()))))" # want two ) at positions 38 and 39

def run_model_on_string(model: HookedTransformer, bracket_strings) -> Float[Tensor, "batch 2"]:
    '''Return probability that each example is balanced'''
    toks = tokenizer.tokenize(bracket_strings)
    logits = model(toks)[:, 0]
    return logits

if MAIN:
    # test_set = data
    print(unbalanced_string.count("("))
    print(unbalanced_string.count(")"))
    print(is_balanced_forloop(unbalanced_string))
    test_logits = run_model_on_string(model, [unbalanced_string])
    print(test_logits.softmax(-1)[:, 1])
    print(test_logits.argmax(-1).bool())

# %%
# test adversarial hypothesis

def tallest_balanced_bracket(length: int) -> str:
    return "".join(["(" for _ in range(length)] + [")" for _ in range(length)])

example = tallest_balanced_bracket(15) + ")(" + tallest_balanced_bracket(4)
# %%
