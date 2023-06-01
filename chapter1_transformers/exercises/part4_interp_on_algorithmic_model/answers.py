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

    # out_by_component_in_unbalanced_dir -= einops.repeat(balanced_out_components, "comp -> comp b", b=5000)
    

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
    out_components = get_out_by_components(model, data)[:,:,0] # [10, dataset_size, emb]

    tests.test_out_by_component_in_pre_20_unbalanced_dir(out_by_component_in_pre_20_unbalanced_dir, model, data)

    plotly_utils.hists_per_comp(
        out_by_component_in_pre_20_unbalanced_dir, 
        data, xaxis_range=(-5, 12)
    )