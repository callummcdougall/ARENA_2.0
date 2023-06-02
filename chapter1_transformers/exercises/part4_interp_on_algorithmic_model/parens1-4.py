# %% Imports

import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
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
from transformer_lens import (
    utils,
    ActivationCache,
    HookedTransformer,
    HookedTransformerConfig,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_interp_on_algorithmic_model"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import plotly_utils
from plotly_utils import hist, bar, imshow
import part4_interp_on_algorithmic_model.tests as tests
from part4_interp_on_algorithmic_model.brackets_datasets import (
    SimpleTokenizer,
    BracketsDataset,
)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %% Define HookedTransformerConfig

if MAIN:
    VOCAB = "()"

    cfg = HookedTransformerConfig(
        n_ctx=42,
        d_model=56,
        d_head=28,
        n_heads=2,
        d_mlp=56,
        n_layers=3,
        attention_dir="bidirectional",  # defaults to "causal"
        act_fn="relu",
        d_vocab=len(VOCAB) + 3,  # plus 3 because of end and pad and start token
        d_vocab_out=2,  # 2 because we're doing binary classification
        use_attn_result=True,
        device=device,
        use_hook_tokens=True,
    )

    model = HookedTransformer(cfg).eval()

    state_dict = t.load(section_dir / "brackets_model_state_dict.pt")
    model.load_state_dict(state_dict)

# %% Make tokenizer

if MAIN:
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

# %% Define hook and model


def add_perma_hooks_to_mask_pad_tokens(
    model: HookedTransformer, pad_token: int
) -> HookedTransformer:
    # Hook which operates on the tokens, and stores a mask where tokens equal [pad]
    def cache_padding_tokens_mask(
        tokens: Float[Tensor, "batch seq"], hook: HookPoint
    ) -> None:
        hook.ctx["padding_tokens_mask"] = einops.rearrange(
            tokens == pad_token, "b sK -> b 1 1 sK"
        )

    # Apply masking, by referencing the mask stored in the `hook_tokens` hook context
    def apply_padding_tokens_mask(
        attn_scores: Float[Tensor, "batch head seq_Q seq_K"],
        hook: HookPoint,
    ) -> None:
        attn_scores.masked_fill_(
            model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"], -1e5
        )
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

# %% read bracket data in to datasets

if MAIN:
    N_SAMPLES = 5000
    with open(section_dir / "brackets_data.json") as f:
        data_tuples: List[Tuple[str, bool]] = json.load(f)
        print(f"loaded {len(data_tuples)} examples")
    assert isinstance(data_tuples, list)
    data_tuples = data_tuples[:N_SAMPLES]
    data = BracketsDataset(data_tuples).to(device)
    data_mini = BracketsDataset(data_tuples[:100]).to(device)
# %% histogram bracket data

if MAIN:
    hist(
        [len(x) for x, _ in data_tuples],
        nbins=data.seq_length,
        title="Sequence lengths of brackets in dataset",
        labels={"x": "Seq len"},
    )

# %% Train bracket model on dataset

# Define and tokenize examples

if MAIN:
    examples = [
        "()()",
        "(())",
        "))((",
        "()",
        "((()()()()))",
        "(()()()(()(())()",
        "()(()(((())())()))",
    ]
    labels = [True, True, False, True, True, False, True]
    toks = tokenizer.tokenize(examples)

    # Get output logits for the 0th sequence position (i.e. the [start] token)
    logits = model(toks)[:, 0]

    # Get the probabilities via softmax, then get the balanced probability (which is the second element)
    prob_balanced = logits.softmax(-1)[:, 1]

    # Display output
    print(
        "Model confidence:\n"
        + "\n".join(
            [
                f"{ex:18} : {prob:<8.4%} : label={int(label)}"
                for ex, prob, label in zip(examples, prob_balanced, labels)
            ]
        )
    )
# %% Run model on data


def run_model_on_data(
    model: HookedTransformer, data: BracketsDataset, batch_size: int = 200
) -> Float[Tensor, "batch 2"]:
    """Return probability that each example is balanced"""
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
    n_correct = (
        run_model_on_data(model, test_set).argmax(-1).bool() == test_set.isbal
    ).sum()
    print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")
# %% Python balence program


def is_balanced_forloop(parens: str) -> bool:
    """
    Return True if the parens are balanced.

    Parens is just the ( and ) characters, no begin or end tokens.
    """
    d = 0
    for p in parens:
        if p == "(":
            d += 1
        elif p == ")":
            d -= 1
        if d < 0:
            return False
    return d == 0


if MAIN:
    for parens, expected in zip(examples, labels):
        actual = is_balanced_forloop(parens)
        assert expected == actual, f"{parens}: expected {expected} got {actual}"
    print("is_balanced_forloop ok!")

# %%


def is_balanced_vectorized(tokens: Float[Tensor, "seq_len"]) -> bool:
    """
    Return True if the parens are balanced.

    tokens is a vector which has start/pad/end indices (0/1/2) as well as left/right brackets (3/4)
    """
    lookup = t.tensor([0, 0, 0, 1, -1]).to(device)
    to_add = lookup[[tokens.int()]]
    cums = to_add.cumsum(dim=-1)
    return ((cums[-1]) == 0 and (cums >= 0).all()).item()


if MAIN:
    for tokens, expected in zip(tokenizer.tokenize(examples), labels):
        actual = is_balanced_vectorized(tokens)
        assert expected == actual, f"{tokens}: expected {expected} got {actual}"
    print("is_balanced_vectorized ok!")

# %%


def get_post_final_ln_dir(model: HookedTransformer) -> Float[Tensor, "d_model"]:
    """
    Returns the direction in which final_ln_output[0, :] should point to maximize P(unbalanced)
    """
    return model.W_U[:, 0] - model.W_U[:, 1]


if MAIN:
    tests.test_get_post_final_ln_dir(get_post_final_ln_dir, model)

# %%


def get_activations(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    names: Union[str, List[str]],
) -> Union[t.Tensor, ActivationCache]:
    """
    Uses hooks to return activations from the model.

    If names is a string, returns the activations for that hook name.
    If names is a list of strings, returns a dictionary mapping hook names to tensors of activations.
    """
    names_list = [names] if isinstance(names, str) else names
    _, cache = model.run_with_cache(
        toks,
        return_type=None,
        names_filter=lambda name: name in names_list,
    )

    return cache[names] if isinstance(names, str) else cache


# %%
def LN_hook_names(layernorm: LayerNorm) -> Tuple[str, str]:
    """
    Returns the names of the hooks immediately before and after a given layernorm.
    e.g. LN_hook_names(model.final_ln) returns ["blocks.2.hook_resid_post", "ln_final.hook_normalized"]
    """
    if layernorm.name == "ln_final":
        input_hook_name = utils.get_act_name("resid_post", 2)
        output_hook_name = "ln_final.hook_normalized"
    else:
        layer, ln = layernorm.name.split(".")[1:]
        input_hook_name = utils.get_act_name(
            "resid_pre" if ln == "ln1" else "resid_mid", layer
        )
        output_hook_name = utils.get_act_name("normalized", layer, ln)

    return input_hook_name, output_hook_name


if MAIN:
    pre_final_ln_name, post_final_ln_name = LN_hook_names(model.ln_final)
    print(pre_final_ln_name, post_final_ln_name)
# %%


def get_ln_fit(
    model: HookedTransformer,
    data: BracketsDataset,
    layernorm: LayerNorm,
    seq_pos: Optional[int] = None,
) -> Tuple[LinearRegression, float]:
    """
    if seq_pos is None, find best fit aggregated over all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and the r^2 of the fit
    """
    input_hook_name, output_hook_name = LN_hook_names(layernorm)
    acts = get_activations(
        model=model, toks=data.toks, names=[input_hook_name, output_hook_name]
    )
    Xs = acts[input_hook_name]
    Yalls = acts[output_hook_name]
    if seq_pos is not None:
        X = Xs[:, seq_pos, :].cpu()
        Yall = Yalls[:, seq_pos, :].cpu()
    else:
        X = einops.rearrange(Xs, "b seq model -> (b seq) model").cpu()
        Yall = einops.rearrange(Yalls, "b seq model -> (b seq) model").cpu()
    lin = LinearRegression()
    lin = lin.fit(X=X, y=Yall)
    r2 = lin.score(X=X, y=Yall)
    return lin, r2


if MAIN:
    tests.test_get_ln_fit(get_ln_fit, model, data_mini)

    (final_ln_fit, r2) = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)
    print(f"r^2 for LN_final, at sequence position 0: {r2:.4f}")

    (final_ln_fit, r2) = get_ln_fit(
        model, data, layernorm=model.blocks[1].ln1, seq_pos=None
    )
    print(f"r^2 for LN1, layer 1, over all sequence positions: {r2:.4f}")


# %%


def get_pre_final_ln_dir(
    model: HookedTransformer, data: BracketsDataset
) -> Float[Tensor, "d_model"]:
    """
    Returns the direction in residual stream (pre ln_final, at sequence position 0) which
    most points in the direction of making an unbalanced classification.
    """
    lin, r2 = get_ln_fit(model=model, data=data, layernorm=model.ln_final, seq_pos=0)
    return t.from_numpy(lin.coef_.T).to(device) @ get_post_final_ln_dir(model)


if MAIN:
    tests.test_get_pre_final_ln_dir(get_pre_final_ln_dir, model, data_mini)


# %%


def get_out_by_components_NOT_WORKING(
    model: HookedTransformer, data: BracketsDataset
) -> Float[Tensor, "component batch seq_pos emb"]:
    """
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2]
    """

    acts = [utils.get_act_name("result", x) for x in range(3)]
    mlps = [utils.get_act_name("mlp_out", x) for x in range(3)]
    names = ["embed"] + acts + mlps
    cache = get_activations(model, data.toks, names)

    embed = cache[names[0]]
    heads = [cache[name] for name in names[1:4]]
    mlps = [cache[name] for name in names[4:]]

    output = [embed]
    for layer_x in range(3):
        for head_ix in range(2):
            output.append(heads[layer_x][:, :, head_ix, :])
        output.append(mlps[layer_x])

    return t.stack(output)

    # """
    # _, cache = model.run_with_cache(data.toks)
    # pre_final_ln_dir = get_pre_final_ln_dir(model, data)

    # output = []
    # output.append(cache["embed"])
    # for layer_x in range(3):
    #     heads = cache["result", layer_x]
    #     mlp = cache["mlp_out", layer_x]

    #     for head_x in range(2):
    #         output.append(heads[:,:,head_x,:])

    #     output.append(mlp)

    # return t.stack(output)
    # """


def get_out_by_components(
    model: HookedTransformer, data: BracketsDataset
) -> Float[Tensor, "component batch seq_pos emb"]:
    """
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2]
    """
    # SOLUTION
    embedding_hook_names = ["hook_embed", "hook_pos_embed"]
    head_hook_names = [
        utils.get_act_name("result", layer) for layer in range(model.cfg.n_layers)
    ]
    mlp_hook_names = [
        utils.get_act_name("mlp_out", layer) for layer in range(model.cfg.n_layers)
    ]

    all_hook_names = embedding_hook_names + head_hook_names + mlp_hook_names
    activations = get_activations(model, data.toks, all_hook_names)

    out = (activations["hook_embed"] + activations["hook_pos_embed"]).unsqueeze(0)

    for head_hook_name, mlp_hook_name in zip(head_hook_names, mlp_hook_names):
        out = t.concat(
            [
                out,
                einops.rearrange(
                    activations[head_hook_name],
                    "batch seq heads emb -> heads batch seq emb",
                ),
                activations[mlp_hook_name].unsqueeze(0),
            ]
        )

    return out


if MAIN:
    tests.test_get_out_by_components(get_out_by_components, model, data_mini)
# %%

if MAIN:
    # YOUR CODE HERE - define the object `out_by_component_in_unbalanced_dir`

    out_by_componets = get_out_by_components(model=model, data=data)
    out_by_componets_first = out_by_componets[:, :, 0, :]
    pre_dir = get_pre_final_ln_dir(model=model, data=data)
    out_by_component_and_seq_in_unbalanced_dir = out_by_componets_first @ pre_dir

    # why do we subtract out the mean of balanced?
    out_by_component_in_unbalanced_dir = (
        out_by_component_and_seq_in_unbalanced_dir
        - out_by_component_and_seq_in_unbalanced_dir[:, data.isbal]
        .mean(dim=1)
        .unsqueeze(1)
    )
    tests.test_out_by_component_in_unbalanced_dir(
        out_by_component_in_unbalanced_dir, model, data
    )

    plotly_utils.hists_per_comp(
        out_by_component_in_unbalanced_dir, data, xaxis_range=[-10, 20]
    )


# %%


def is_balanced_vectorized_return_both(
    toks: Float[Tensor, "batch seq"]
) -> Tuple[Bool[Tensor, "batch"], Bool[Tensor, "batch"]]:
    lookup = t.tensor([0, 0, 0, 1, -1]).to(device)
    to_add = lookup[[toks.int()]]
    cums = to_add.flip(-1).cumsum(dim=-1)

    total_elevation_failure = cums[:, -1] != 0
    negative_failure = (cums > 0).any(dim=-1)
    return (total_elevation_failure, negative_failure)


if MAIN:
    total_elevation_failure, negative_failure = is_balanced_vectorized_return_both(
        data.toks
    )
    h20_in_unbalanced_dir = out_by_component_in_unbalanced_dir[7]
    h21_in_unbalanced_dir = out_by_component_in_unbalanced_dir[8]

    tests.test_total_elevation_and_negative_failures(
        data, total_elevation_failure, negative_failure
    )

# %%

if MAIN:
    failure_types_dict = {
        "both failures": negative_failure & total_elevation_failure,
        "just neg failure": negative_failure & ~total_elevation_failure,
        "just total elevation failure": ~negative_failure & total_elevation_failure,
        "balanced": ~negative_failure & ~total_elevation_failure,
    }

    plotly_utils.plot_failure_types_scatter(
        h20_in_unbalanced_dir, h21_in_unbalanced_dir, failure_types_dict, data
    )
# %%

if MAIN:
    plotly_utils.plot_contribution_vs_open_proportion(
        h20_in_unbalanced_dir,
        "Head 2.0 contribution vs proportion of open brackets '('",
        failure_types_dict,
        data,
    )
# %%

if MAIN:
    plotly_utils.plot_contribution_vs_open_proportion(
        h21_in_unbalanced_dir,
        "Head 2.1 contribution vs proportion of open brackets '('",
        failure_types_dict,
        data,
    )
# %%
def get_attn_probs(model: HookedTransformer, data: BracketsDataset, layer: int, head: int) -> t.Tensor:
    '''
    Returns: (N_SAMPLES, max_seq_len, max_seq_len) tensor that sums to 1 over the last dimension.
    '''
    act_name =  utils.get_act_name("pattern", layer)
    pattern = get_activations(model, data.toks, act_name)
    return pattern[:, head, :, :]


if MAIN:
    tests.test_get_attn_probs(get_attn_probs, model, data_mini)
# %%
if MAIN:
    for layer in range(3):
        for head in range(2):
            attn_probs_20: Float[Tensor, "batch seqQ seqK"] = get_attn_probs(model, data, layer, head)
            attn_probs_20_open_query0 = attn_probs_20[data.starts_open].mean(0)[0]

            plotly_utils.imshow(attn_probs_20.mean(0), title=f"{layer} {head}")

            # bar(
            #     attn_probs_20_open_query0,
            #     title="Avg Attention Probabilities for query 0, first token '(', head 2.0",
            #     width=700, template="simple_white"
            # )
# %%
def get_WOV(model: HookedTransformer, layer: int, head: int) -> Float[Tensor, "d_model d_model"]:
    '''
    Returns the W_OV matrix for a particular layer and head.
    '''
    W_O = model.W_O[layer, head]
    W_V = model.W_V[layer, head]
    
    return W_V @ W_O
    

def get_pre_20_dir(model, data) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction propagated back through the OV matrix of 2.0 
    and then through the layernorm before the layer 2 attention heads.
    '''
    layer = 2
    head = 0
    pos = 1
    lin, r2 = get_ln_fit(model=model, data=data, layernorm=model.blocks[layer].ln1, seq_pos=pos)
    return t.from_numpy(lin.coef_.T).to(device) @ get_WOV(model=model, layer=layer, head=head) @ get_pre_final_ln_dir(model, data)

if MAIN:
    tests.test_get_pre_20_dir(get_pre_20_dir, model, data_mini)
# %%

if MAIN:
    # YOUR CODE HERE - define `out_by_component_in_pre_20_unbalanced_dir` (for all components before head 2.0)

    seq_position = 1
    out_by_componets_second = out_by_componets[:7, :, seq_position, :]
    pre_20_dir = get_pre_20_dir(model=model, data=data)
    out_by_component_and_seq_in_pre_20_unbalanced_dir = out_by_componets_second @ pre_20_dir

    # why do we subtract out the mean of balanced?
    out_by_component_in_pre_20_unbalanced_dir = (
        out_by_component_and_seq_in_pre_20_unbalanced_dir
        - out_by_component_and_seq_in_pre_20_unbalanced_dir[:, data.isbal]
        .mean(dim=1, keepdim=True)
    )


    # tests.test_out_by_component_in_pre_20_unbalanced_dir(out_by_component_in_pre_20_unbalanced_dir, model, data)

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
    
    embedding_hook_names = ["hook_embed", "hook_pos_embed"]
    head_hook_names = [
        utils.get_act_name("result", layer) for layer in range(model.cfg.n_layers)
    ]
    mlp_hook_names = [
        utils.get_act_name("mlp_out", layer) for layer in range(model.cfg.n_layers)
    ]

    all_hook_names = embedding_hook_names + head_hook_names + mlp_hook_names
    activations = get_activations(model, data.toks, all_hook_names)

    out = (activations["hook_embed"] + activations["hook_pos_embed"]).unsqueeze(0)

    for head_hook_name, mlp_hook_name in zip(head_hook_names, mlp_hook_names):
        out = t.concat(
            [
                out,
                einops.rearrange(
                    activations[head_hook_name],
                    "batch seq heads emb -> heads batch seq emb",
                ),
                activations[mlp_hook_name].unsqueeze(0),
            ]
        )

    cfg.act_fn
    model.ml
    return out



    if seq is None:
        # out[b, i, :] = vector f(x[b].T @ W_in[:, i]) @ W_out[i, :]

        pass
    else:
        # out[b, s, i, :] = f(x[b, s].T @ W_in[:, i]) @ W_out[i, :]
        pass


def get_out_by_neuron_in_20_dir(model: HookedTransformer, data: BracketsDataset, layer: int) -> Float[Tensor, "batch neurons"]:
    '''
    [b, s, i]th element is the contribution of the vector written by the ith neuron to the residual stream in the 
    unbalanced direction (for the b-th element in the batch, and the s-th sequence position).

    In other words we need to take the vector produced by the `get_out_by_neuron` function, and project it onto the 
    unbalanced direction for head 2.0 (at seq pos = 1).
    '''
    pass


if MAIN:
    tests.test_get_out_by_neuron(get_out_by_neuron, model, data_mini)
    tests.test_get_out_by_neuron_in_20_dir(get_out_by_neuron_in_20_dir, model, data_mini)

# %%
