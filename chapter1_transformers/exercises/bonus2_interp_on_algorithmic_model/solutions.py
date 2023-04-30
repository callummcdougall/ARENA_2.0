# %%

import functools
import json
from typing import List, Tuple, Union, Optional
import torch as t
from fancy_einsum import einsum
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import einops
from tqdm import tqdm
from torchtyping import TensorType as TT

MAIN = __name__ == "__main__"
device = t.device("cpu")

t.set_grad_enabled(False)

# from IPython import get_ipython
# ipython = get_ipython()
# # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
# ipython.magic("load_ext autoreload")
# ipython.magic("autoreload 2")

from transformer_lens import utils, ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm

from brackets_datasets import SimpleTokenizer, BracketsDataset
import tests
import plot_utils

def imshow(tensor, xaxis="", yaxis="", **kwargs):
    return px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs)

def line(tensor, xaxis="", yaxis="", **kwargs):
    return px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs)

def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    return px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)

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

    state_dict = t.load(r"brackets_model_state_dict.pt")
    model.load_state_dict(state_dict)

# %%

if MAIN:
    tokenizer = SimpleTokenizer("()")
    N_SAMPLES = 5000
    N_SAMPLES_TEST = 100
    with open(r"brackets_data.json") as f:
        data_tuples: List[Tuple[str, bool]] = json.load(f)
        print(f"loaded {len(data_tuples)} examples")
    assert isinstance(data_tuples, list)
    data_tuples = data_tuples[:N_SAMPLES]
    data = BracketsDataset(data_tuples)
    data_test = BracketsDataset(data_tuples[:N_SAMPLES_TEST])

# %%

if MAIN:
    fig = go.Figure(
        go.Histogram(x=[len(x) for x, _ in data_tuples], nbinsx=data.seq_length),
        layout=dict(title="Sequence Lengths", xaxis_title="Sequence Length", yaxis_title="Count")
    )
    fig.show()

# %%

def add_perma_hooks_to_mask_pad_tokens(model: HookedTransformer, pad_token: int) -> HookedTransformer:

    # Hook which operates on the tokens, and stores a mask where tokens equal [pad]
    def cache_padding_tokens_mask(tokens: TT["batch", "seq"], hook: HookPoint) -> None:
        hook.ctx["padding_tokens_mask"] = einops.rearrange(tokens == pad_token, "b sK -> b 1 1 sK")

    # Apply masking, by referencing the mask stored in the `hook_tokens` hook context
    def apply_padding_tokens_mask(
        attn_scores: TT["batch", "head", "seq_Q", "seq_K"],
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
    # Define and tokenize examples
    examples = ["()()", "(())", "))((", "()", "((()()()()))", "(()()()(()(())()", "()(()(((())())()))"]
    labels = [True, True, False, True, True, False, True]
    toks = tokenizer.tokenize(examples).to(device)

    # Get output logits for the 0th sequence position (i.e. the [start] token)
    logits = model(toks)[:, 0]

    # Get the probabilities via softmax, then get the balanced probability (which is the second element)
    prob_balanced = logits.softmax(-1)[:, 1]

    # Display output
    print("Model confidence:\n" + "\n".join([f"{ex:34} : {prob:.4%}" for ex, prob in zip(examples, prob_balanced)]))

# %%

def run_model_on_data(model: HookedTransformer, data: BracketsDataset, batch_size: int = 200) -> TT["batch", 2]:
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

    cumsum = 0
    for paren in parens:
        cumsum += 1 if paren == "(" else -1
        if cumsum < 0:
            return False
    
    return cumsum == 0

if MAIN:
    for (parens, expected) in zip(examples, labels):
        actual = is_balanced_forloop(parens)
        assert expected == actual, f"{parens}: expected {expected} got {actual}"
    print("is_balanced_forloop ok!")

# %%

def is_balanced_vectorized(tokens: TT["seq"]) -> bool:
    """
    Return True if the parens are balanced.

    tokens is a vector which has start/pad/end indices (0/1/2) as well as left/right brackets (3/4)
    """
    # Convert start/end/padding tokens to zero, and left/right brackets to +1/-1
    table = t.tensor([0, 0, 0, 1, -1])
    change = table[tokens]
    # Get altitude by taking cumulative sum
    altitude = t.cumsum(change, -1)
    # Check that the total elevation is zero and that there are no negative altitudes
    no_total_elevation_failure = altitude[-1] == 0
    no_negative_failure = altitude.min() >= 0

    return no_total_elevation_failure & no_negative_failure

if MAIN:
    for (tokens, expected) in zip(tokenizer.tokenize(examples), labels):
        actual = is_balanced_vectorized(tokens)
        assert expected == actual, f"{tokens}: expected {expected} got {actual}"
    print("is_balanced_vectorized ok!")

# %%

def get_post_final_ln_dir(model: HookedTransformer) -> TT["d_model"]:
    '''
    Returns the direction in which final_ln_output[0, :] should point to maximize P(unbalanced)
    '''
    return model.W_U[:, 0] - model.W_U[:, 1]

# %%

# Solution using hooks:

def get_activations(model: HookedTransformer, tokens: TT["batch", "seq"], names: Union[str, List[str]]) -> Union[t.Tensor, ActivationCache]:
    '''
    Uses hooks to return activations from the model.
    If names is a string, returns the activations for that hook name.
    If names is a list of strings, returns a dictionary mapping hook names to tensors of activations.
    '''
    model.reset_hooks()

    activations_dict = {}
    hook_names_list = names if isinstance(names, list) else [names]

    def hook_fn(value, hook):
        activations_dict[hook.name] = value

    hook_name_filter = lambda name: name in hook_names_list
    model.run_with_hooks(
        tokens,
        return_type=None,
        fwd_hooks=[(hook_name_filter, hook_fn)]
    )

    return ActivationCache(activations_dict, model) if isinstance(names, list) else activations_dict[hook_names_list[0]]


if MAIN:
    tests.test_get_activations(get_activations, model, data_test)

# %%

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



def get_ln_fit(
    model: HookedTransformer, data: BracketsDataset, layernorm: LayerNorm, seq_pos: Optional[int] = None
) -> Tuple[LinearRegression, float]:
    '''
    if seq_pos is None, find best fit aggregated over all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and the r^2 of the fit
    '''

    input_hook_name, output_hook_name = LN_hook_names(layernorm)

    activations_dict = get_activations(model, data.toks, [input_hook_name, output_hook_name])
    inputs = utils.to_numpy(activations_dict[input_hook_name])
    outputs = utils.to_numpy(activations_dict[output_hook_name])

    if seq_pos is None:
        inputs = einops.rearrange(inputs, "batch seq d_model -> (batch seq) d_model")
        outputs = einops.rearrange(outputs, "batch seq d_model -> (batch seq) d_model")
    else:
        inputs = inputs[:, seq_pos, :]
        outputs = outputs[:, seq_pos, :]
    
    final_ln_fit = LinearRegression().fit(inputs, outputs)

    r2 = final_ln_fit.score(inputs, outputs)

    return (final_ln_fit, r2)


if MAIN:
    tests.test_get_ln_fit(get_ln_fit, model, data_test)

    (final_ln_fit, r2) = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)
    print(f"r^2 for LN_final, at sequence position 0: {r2:.4f}")

    (final_ln_fit, r2) = get_ln_fit(model, data, layernorm=model.blocks[1].ln1, seq_pos=None)
    print(f"r^2 for LN1, layer 1, over all sequence positions: {r2:.4f}")

# %%

def get_pre_final_ln_dir(model: HookedTransformer, data: BracketsDataset) -> TT["d_model"]:
    
    post_final_ln_dir = get_post_final_ln_dir(model)

    final_ln_fit = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)[0]
    final_ln_coefs = t.from_numpy(final_ln_fit.coef_).to(device)

    return final_ln_coefs.T @ post_final_ln_dir

if MAIN:
    tests.test_get_pre_final_ln_dir(get_pre_final_ln_dir, model, data_test)

# %%

def get_out_by_components(
    model: HookedTransformer, data: BracketsDataset
) -> TT["component", "batch", "seq_pos", "emb"]:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2]
    '''
    embedding_hook_names = ["hook_embed", "hook_pos_embed"]
    head_hook_names = [utils.get_act_name("result", layer) for layer in range(model.cfg.n_layers)]
    mlp_hook_names = [utils.get_act_name("mlp_out", layer) for layer in range(model.cfg.n_layers)]
     
    all_hook_names = embedding_hook_names + head_hook_names + mlp_hook_names
    cache = get_activations(model, data.toks, all_hook_names)

    out = (cache["embed"] + cache["pos_embed"]).unsqueeze(0)

    for head_hook_name, mlp_hook_name in zip(head_hook_names, mlp_hook_names):
        out = t.concat([
            out, 
            einops.rearrange(
                cache[head_hook_name],
                "batch seq heads emb -> heads batch seq emb"
            ),
            cache[mlp_hook_name].unsqueeze(0)
        ])

    return out


def get_out_by_components(
    model: HookedTransformer, data: BracketsDataset
) -> TT["component", "batch", "seq_pos", "emb"]:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2]
    '''
    hook_names = [("embed",), ("pos_embed",)]
    for layer in range(model.cfg.n_layers):
        hook_names.extend([("result", layer), ("mlp_out", layer)])

    cache = get_activations(model, data.toks, [utils.get_act_name(*name) for name in hook_names])
    
    out = [(cache["embed"] + cache["pos_embed"]).unsqueeze(0)]
    for layer in range(model.cfg.n_layers):
        out.extend([
            einops.rearrange(cache["result", layer], "batch seq heads emb -> heads batch seq emb"),
            cache["mlp_out", layer].unsqueeze(0)
        ])

    return t.concat(out)


if MAIN:
    tests.test_get_out_by_components(get_out_by_components, model, data_test)


# %%

if MAIN:
    biases = model.b_O.sum(0)
    out_by_components = get_out_by_components(model, data)
    summed_terms = out_by_components.sum(dim=0) + biases

    final_ln_input_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    final_ln_input = get_activations(model, data.toks, final_ln_input_name)

    t.testing.assert_close(summed_terms, final_ln_input)
    print("Tests passed!")

# %%

if MAIN:
    # Get output by components, at sequence position 0 (which is used for classification)
    out_by_components_seq0: TT["comp", "batch", "d_model"] = out_by_components[:, :, 0, :]
    # Get the unbalanced direction for tensors being fed into the final layernorm
    pre_final_ln_dir: TT["d_model"] = get_pre_final_ln_dir(model, data)
    # Get the size of the contributions for each component
    out_by_component_in_unbalanced_dir: TT["comp", "batch"] = einsum(
        "comp batch d_model, d_model -> comp batch",
        out_by_components_seq0, 
        pre_final_ln_dir
    )
    # Subtract the mean
    out_by_component_in_unbalanced_dir -= out_by_component_in_unbalanced_dir[:, data.isbal].mean(dim=1).unsqueeze(1)

    tests.test_out_by_component_in_unbalanced_dir(out_by_component_in_unbalanced_dir, model, data)
    # Plot the histograms
    plot_utils.hists_per_comp(out_by_component_in_unbalanced_dir, data, xaxis_range=[-10, 20])

# %%

def is_balanced_vectorized_return_both(tokens: TT["batch", "seq"]) -> Tuple[TT["batch", t.bool], TT["batch", t.bool]]:
    table = t.tensor([0, 0, 0, 1, -1])
    change = table[tokens].flip(-1)
    altitude = t.cumsum(change, -1)
    total_elevation_failure = altitude[:, -1] != 0
    negative_failure = altitude.max(-1).values > 0
    return total_elevation_failure, negative_failure

if MAIN:
    total_elevation_failure, negative_failure = is_balanced_vectorized_return_both(data.toks)
    h20_in_unbalanced_dir = out_by_component_in_unbalanced_dir[7]
    h21_in_unbalanced_dir = out_by_component_in_unbalanced_dir[8]

    tests.test_total_elevation_and_negative_failures(data, total_elevation_failure, negative_failure)

if MAIN:
    failure_types_dict = {
        "both failures": negative_failure & total_elevation_failure,
        "just neg failure": negative_failure & ~total_elevation_failure,
        "just total elevation failure": ~negative_failure & total_elevation_failure,
        "balanced": ~negative_failure & ~total_elevation_failure
    }
    plot_utils.plot_failure_types_scatter(
        h20_in_unbalanced_dir,
        h21_in_unbalanced_dir,
        failure_types_dict,
        data
    )

# %%

if MAIN:
    plot_utils.plot_contribution_vs_open_proportion(h20_in_unbalanced_dir, "2.0", failure_types_dict, data)

# %%

# ==================================================
# Section 3 - Total elevation circuit
# ==================================================

def get_attn_probs(model: HookedTransformer, data: BracketsDataset, layer: int, head: int) -> t.Tensor:
    '''
    Returns: (batch, max_seq_len, max_seq_len) tensor that sums to 1 over the last dimension.
    '''
    return get_activations(model, data.toks, utils.get_act_name("pattern", layer))[:, head, :, :]


if MAIN:
    tests.test_get_attn_probs(get_attn_probs, model, data.toks)

# %%

if MAIN:
    attn_probs_20: TT["batch", "seqQ", "seqK"] = get_attn_probs(model, data, 2, 0)
    attn_probs_20_open_query0 = attn_probs_20[data.starts_open].mean(0)[0]

    fig = px.bar(
        y=utils.to_numpy(attn_probs_20_open_query0), 
        labels={"y": "Probability", "x": "Key Position"},
        template="simple_white", height=500, width=600, 
        title="Avg Attention Probabilities for query 0, first token '(', head 2.0"
    ).update_layout(showlegend=False, hovermode='x unified')
    fig.show()

# %%

def get_WOV(model: HookedTransformer, layer: int, head: int) -> TT["d_model", "d_model"]:
    '''
    Returns the W_OV matrix for a particular layer and head.
    '''
    return model.W_V[layer, head] @ model.W_O[layer, head]

def get_pre_20_dir(model, data) -> TT["d_model"]:
    '''
    Returns the direction propagated back through the OV matrix of 2.0 and then through the layernorm before the layer 2 attention heads.
    '''
    W_OV = get_WOV(model, 2, 0)

    layer2_ln_fit, r2 = get_ln_fit(model, data, layernorm=model.blocks[2].ln1, seq_pos=1)
    layer2_ln_coefs = t.from_numpy(layer2_ln_fit.coef_).to(device)

    pre_final_ln_dir = get_pre_final_ln_dir(model, data)

    return layer2_ln_coefs.T @ W_OV @ pre_final_ln_dir

if MAIN:
    tests.test_get_WOV(get_WOV, model)
    tests.test_get_pre_20_dir(get_pre_20_dir, model, data_test)

# %%

if MAIN:
    pre_layer2_outputs = get_out_by_components(model, data)[:-3]
    out_by_component_in_pre_20_unbalanced_dir = einsum(
        "comp batch d_model, d_model -> comp batch",
        pre_layer2_outputs[:, :, 1, :],
        get_pre_20_dir(model, data)
    )
    out_by_component_in_pre_20_unbalanced_dir -= out_by_component_in_pre_20_unbalanced_dir[:, data.isbal].mean(-1, keepdim=True)
    plot_utils.hists_per_comp(out_by_component_in_pre_20_unbalanced_dir, data, xaxis_range=(-5, 12))

# %%
# %%

if MAIN:
    plot_utils.mlp_attribution_scatter(out_by_component_in_pre_20_unbalanced_dir, data, failure_types_dict)

# %%

def get_out_by_neuron(model: HookedTransformer, data: BracketsDataset, layer: int, seq: Optional[int] = None) -> TT["batch", "seq", "neurons", "d_model"]:
    '''
    [b, s, i]th element is the vector f(x.T @ W_in[:, i]) @ W_out[i, :] which is written to 
    the residual stream by the ith neuron (where x is the input to the MLP for the b-th 
    element in the batch, and the s-th sequence position).
    '''
    # Get the W_out matrix for this MLP
    W_out: TT["neurons", "d_model"] = model.W_out[layer]

    # Get activations of the layer just after the activation function, i.e. this is f(x.T @ W_in)
    f_x_W_in: TT["batch", "seq", "neurons"] = get_activations(model, data.toks, utils.get_act_name('post', layer))

    # f_x_W_in are activations, so they have batch and seq dimensions - this is where we index by seq if necessary
    if seq is not None:
        f_x_W_in: TT["batch", "neurons"] = f_x_W_in[:, seq, :]

    # Calculate the output by neuron (i.e. so summing over the `neurons` dimension gives the output of the MLP)
    out = einsum(
        "... neurons, neurons d_model -> ... neurons d_model",
        f_x_W_in, W_out
    )
    return out

def get_out_by_neuron_in_20_dir(model: HookedTransformer, data: BracketsDataset, layer: int) -> TT["batch", "neurons"]:
    '''
    [b, s, i]th element is the contribution of the vector written by the ith neuron to the 
    residual stream in the unbalanced direction (for the b-th element in the batch, and the 
    s-th sequence position).
    
    In other words we need to take the vector produced by the `get_out_by_neuron` function,
    and project it onto the unbalanced direction for head 2.0 (at seq pos = 1).
    '''

    out_by_neuron_seqpos1 = get_out_by_neuron(model, data, layer, seq=1)

    return einsum(
        "batch neurons d_model, d_model -> batch neurons",
        out_by_neuron_seqpos1,
        get_pre_20_dir(model, data)
    )

if MAIN:
    tests.test_get_out_by_neuron(get_out_by_neuron, model, data_test)
    tests.test_get_out_by_neuron_in_20_dir(get_out_by_neuron_in_20_dir, model, data_test)

# %%

def get_out_by_neuron_in_20_dir_less_memory(model: HookedTransformer, data: BracketsDataset, layer: int) -> TT["batch", "neurons"]:
    '''
    Has the same output as `get_out_by_neuron_in_20_dir`, but uses less memory (because it never stores
    the output vector of each neuron individually).
    '''

    W_out: TT["neurons", "d_model"] = model.W_out[layer]

    f_x_W_in: TT["batch", "neurons"] = get_activations(model, data.toks, utils.get_act_name('post', layer))[:, 1, :]

    pre_20_dir: TT["d_model"] = get_pre_20_dir(model, data)

    # Multiply along the d_model dimension
    W_out_in_20_dir: TT["neurons"] = W_out @ pre_20_dir
    # Multiply elementwise, over neurons (we're broadcasting along the batch dim)
    out_by_neuron_in_20_dir: TT["batch", "neurons"] = f_x_W_in * W_out_in_20_dir

    return out_by_neuron_in_20_dir

if MAIN:
    tests.test_get_out_by_neuron_in_20_dir_less_memory(get_out_by_neuron_in_20_dir_less_memory, model, data_test)

# %%

if MAIN:
    for layer in range(2):
        # Get neuron significances for head 2.0, sequence position #1 output
        neurons_in_unbalanced_dir = get_out_by_neuron_in_20_dir_less_memory(model, data, layer)[data.starts_open, :]
        # Plot neurons' activations
        plot_utils.plot_neurons(neurons_in_unbalanced_dir, model, data, failure_types_dict, layer)

# %%

def get_q_and_k_for_given_input(
    model: HookedTransformer, tokenizer: SimpleTokenizer, parens: str, layer: int, head: int
) -> Tuple[TT["seq", "d_model"], TT[ "seq", "d_model"]]:
    '''
    Returns the queries and keys for the given parns input, in the attention head `layer.head`.
    '''

    q_name = utils.get_act_name("q", layer)
    k_name = utils.get_act_name("k", layer)

    activations = get_activations(
        model,
        tokenizer.tokenize(parens),
        [q_name, k_name]
    )
    
    return activations[q_name][0, :, head, :], activations[k_name][0, :, head, :]


if MAIN:

    all_left_parens = "".join(["(" * 40])
    all_right_parens = "".join([")" * 40])
    model.reset_hooks()
    q00_all_left, k00_all_left = get_q_and_k_for_given_input(model, tokenizer, all_left_parens, 0, 0)
    q00_all_right, k00_all_right = get_q_and_k_for_given_input(model, tokenizer, all_right_parens, 0, 0)
    k00_avg = (k00_all_left + k00_all_right) / 2

    # Define hook function to patch in q or k vectors
    def hook_fn_patch_qk(
        value: TT["batch", "seq", "head", "d_head"], 
        hook: HookPoint, 
        new_value: TT[..., "seq", "d_head"],
        head_idx: int = 0
    ) -> None:
        value[..., head_idx, :] = new_value
    
    # Define hook function to display attention patterns (using plotly)
    def hook_fn_display_attn_patterns(
        pattern: TT["batch", "heads", "seqQ", "seqK"],
        hook: HookPoint,
        head_idx: int = 0
    ) -> None:
        avg_head_attn_pattern = pattern[:, head_idx].mean(0)
        plot_utils.plot_attn_pattern(avg_head_attn_pattern)
    
    # Run our model on left parens, but patch in the average key values for left vs right parens
    # This is to give us a rough idea how the model behaves on average when the query is a left paren
    model.run_with_hooks(
        tokenizer.tokenize(all_left_parens),
        return_type=None,
        fwd_hooks=[
            (utils.get_act_name("k", 0), functools.partial(hook_fn_patch_qk, new_value=k00_avg)),
            (utils.get_act_name("pattern", 0), hook_fn_display_attn_patterns),
        ]
    )

# %%

if MAIN:

    def hook_fn_display_attn_patterns_for_single_query(
        pattern: TT["batch", "heads", "seqQ", "seqK"],
        hook: HookPoint,
        head_idx: int = 0,
        query_idx: int = 1
    ):
        fig = px.bar(
            pattern[:, head_idx, query_idx].mean(0), 
            title=f"Average attn probabilities on data at posn 1, with query token = '('",
            labels={"index": "Sequence position of key", "value": "Average attn over dataset"},
            template="simple_white", height=500, width=700
        ).update_layout(showlegend=False, margin_l=100, yaxis_range=[0, 0.1], hovermode="x unified")
        fig.show()

    data_len_40 = BracketsDataset.with_length(data_tuples, 40)

    model.reset_hooks()
    model.run_with_hooks(
        data_len_40.toks[data_len_40.isbal],
        return_type=None,
        fwd_hooks=[
            (utils.get_act_name("q", 0), functools.partial(hook_fn_patch_qk, new_value=q00_all_left)),
            (utils.get_act_name("pattern", 0), hook_fn_display_attn_patterns_for_single_query),
        ]
    )

# %%

def embedding(model: HookedTransformer, tokenizer: SimpleTokenizer, char: str) -> TT["d_model"]:
    assert char in ("(", ")")
    idx = tokenizer.t_to_i[char]
    return model.W_E[idx]


if MAIN:

    W_OV = model.W_V[0, 0] @ model.W_O[0, 0]

    layer0_ln_fit = get_ln_fit(model, data, layernorm=model.blocks[0].ln1, seq_pos=None)[0]
    layer0_ln_coefs = t.from_numpy(layer0_ln_fit.coef_).to(device)

    v_L = embedding(model, tokenizer, "(") @ layer0_ln_coefs.T @ W_OV
    v_R = embedding(model, tokenizer, ")") @ layer0_ln_coefs.T @ W_OV
    
    # v_L = model.blocks[0].ln1(embedding(model, tokenizer, "(")) @ W_OV
    # v_R = model.blocks[0].ln1(embedding(model, tokenizer, ")")) @ W_OV

    print("Cosine similarity: ", t.cosine_similarity(v_L, v_R, dim=0).item())
    print("Norms: ", v_L.norm().item(), v_R.norm().item())

# %%

def cos_sim_with_MLP_weights(model: HookedTransformer, v: TT["d_model"], layer: int) -> TT["d_hidden"]:
    '''
    Returns a vector of length d_hidden, where the ith element is the
    cosine similarity between v and the ith in-direction of the MLP in layer `layer`.

    Recall that the in-direction of the MLPs are the columns of the W_in matrix.
    '''
    v_unit = v / v.norm()
    W_in_unit = model.W_in[layer] / model.W_in[layer].norm(dim=0)

    return einsum("d_model, d_model d_hidden -> d_hidden", v_unit, W_in_unit)

def avg_squared_cos_sim(v: TT["d_model"], n_samples: int = 1000) -> float:
    '''
    Returns the average (over n_samples) cosine similarity between v and another randomly chosen vector.

    We can create random vectors from the standard N(0, I) distribution.
    '''
    v2 = t.randn(n_samples, v.shape[0])
    v2 /= v2.norm(dim=1, keepdim=True)

    v1 = v / v.norm()

    return (v1 * v2).pow(2).sum(1).mean().item()


if MAIN:
    print("Avg squared cosine similarity of v_R with ...\n")

    cos_sim_mlp0 = cos_sim_with_MLP_weights(model, v_R, 0)
    print(f"...MLP input directions in layer 0:  {cos_sim_mlp0.pow(2).mean():.6f}")
   
    cos_sim_mlp1 = cos_sim_with_MLP_weights(model, v_R, 1)
    print(f"...MLP input directions in layer 1:  {cos_sim_mlp1.pow(2).mean():.6f}")
    
    cos_sim_rand = avg_squared_cos_sim(v_R)
    print(f"...random vectors of len = d_model:  {cos_sim_rand:.6f}")


# %%

if MAIN:
    print("Update the examples list below below find adversarial examples")
    examples = ["()", "(())", "))"]

    def simple_balanced_bracket(length: int) -> str:
        return "".join(["(" for _ in range(length)] + [")" for _ in range(length)])
    
    examples.append(simple_balanced_bracket(15) + ")(" + simple_balanced_bracket(4))

    m = max(len(ex) for ex in examples)
    toks = tokenizer.tokenize(examples).to(device)
    logits = model(toks)[:, 0]
    prob_balanced = t.softmax(logits, dim=1)[:, 1]
    print("\n".join([f"{ex:{m}} -> {p:.4%} balanced confidence" for (ex, p) in zip(examples, prob_balanced)]))

