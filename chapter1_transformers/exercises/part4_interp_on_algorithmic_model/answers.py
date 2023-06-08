# %%
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
# %%
MAIN = __name__ == "__main__"
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
    print(tokenizer.tokenize("()"))
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
lengths = [len(item[0]) for item in data]
fig = go.Figure(data=[go.Histogram(x=lengths, nbinsx=40)])
fig.update_layout(xaxis_title="sequence lengths")
fig.show()

# dataset features
# - shorter sequences are much more common
# - only even lengths

# %%
# Define and tokenize examples

if MAIN:
    examples = ["()()", "(())", "))((", "()", "((()()()()))", "(()()()(()(())()", "()(()(((())())()))"]
    labels = [True, True, False, True, True, False, True]
    toks = tokenizer.tokenize(examples)

    # Get output logits for the 0th sequence position (i.e. the [start] token)
    logits: Float[Tensor, "batch pred"] = model(toks)[:, 0]

    # Get the probabilities via softmax, then get the balanced probability (which is the second element)
    prob_balanced: Float[Tensor, "batch"] = logits.softmax(-1)[:, 1]

    # Display output
    print("Model confidence:\n" + "\n".join([f"{ex:18} : {prob:<8.4%} : label={int(label)}" for ex, prob, label in zip(examples, prob_balanced, labels)]))
# %%
from torch.utils.data import DataLoader

def run_model_on_data(model: HookedTransformer, data: BracketsDataset, batch_size: int=512):
    dataloader = DataLoader(data, batch_size=batch_size)
    all_logits = []
    for batch in tqdm(dataloader):
        _, _, tokens = batch
        logits: Float[Tensor, "batch pred"] = model(tokens)[:, 0]
        all_logits.append(logits)
    all_logits = t.cat(all_logits)
    assert all_logits.shape == (len(data), 2)
    return all_logits
    


if MAIN:
    logits = run_model_on_data(model, data, batch_size=512)
    preds = logits.argmax(-1).bool()
    n_correct = (preds == data.isbal).sum()
    
    print(f"{n_correct} correct out of {len(data)}")

# %%
def get_post_final_ln_dir(model: HookedTransformer) -> Float[Tensor, "d_model"]:
    return model.W_U[:, 0] - model.W_U[:, 1]


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
    cache = get_activations(model, data.toks, [input_hook_name, output_hook_name])

    inputs: Float[Tensor, "batch pos d_model"] = cache[input_hook_name]
    outputs: Float[Tensor, "batch pos d_model"] = cache[output_hook_name]

    if seq_pos is not None:
        inputs = inputs[:, seq_pos]
        outputs = outputs[:, seq_pos]
    else:
        inputs = einops.rearrange(inputs, "b p d -> (b p) d")
        outputs = einops.rearrange(outputs, "b p d -> (b p) d")
        # inputs = inputs.mean(dim=1)
        # outputs = outputs.mean(dim=1)
    
    reg = LinearRegression().fit(inputs.cpu(), outputs.cpu())

    return (reg, reg.score(inputs.cpu(), outputs.cpu()))




if MAIN:
    # tests.test_get_ln_fit(get_ln_fit, model, data_mini)

    (final_ln_fit, r2) = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)
    print(f"r^2 for LN_final, at sequence position 0: {r2:.4f}")

    (final_ln_fit, r2) = get_ln_fit(model, data, layernorm=model.blocks[1].ln1, seq_pos=None)
    print(f"r^2 for LN1, layer 1, over all sequence positions: {r2:.4f}")

# %%
final_ln_fit.coef_.shape

# %%
def get_pre_final_ln_dir(model: HookedTransformer, data: BracketsDataset) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction in residual stream (pre ln_final, at sequence position 0) which
    most points in the direction of making an unbalanced classification.
    '''
    post_final_ln_dir = get_post_final_ln_dir(model)
    L_final = t.tensor(
        get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)[0].coef_
    ).to(device)

    return L_final.T @ post_final_ln_dir


if MAIN:
    tests.test_get_pre_final_ln_dir(get_pre_final_ln_dir, model, data_mini)

# %%
# for k in cache:
#     if "0" in k or "blocks" not in k:
#         print(k)

def get_out_by_components(model: HookedTransformer, data: BracketsDataset) -> Float[Tensor, "component batch seq_pos emb"]:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2]
    '''
    names = ["blocks.0.hook_resid_pre"]
    for layer in range(model.cfg.n_layers):
        names.extend([
            f"blocks.{layer}.attn.hook_result",
            f"blocks.{layer}.hook_mlp_out"
        ])

    cache = get_activations(model, data.toks, names)

    outputs = [cache["blocks.0.hook_resid_pre"]]
    for layer in range(model.cfg.n_layers):
        outputs.extend(
            cache[f"blocks.{layer}.attn.hook_result"].unbind(2)
        )
        outputs.append(
            cache[f"blocks.{layer}.hook_mlp_out"]
        )

    return t.stack(outputs, 0)
    


if MAIN:
    tests.test_get_out_by_components(get_out_by_components, model, data_mini)

if MAIN:
    biases = model.b_O.sum(0)
    out_by_components = get_out_by_components(model, data)
    summed_terms = out_by_components.sum(dim=0) + biases

    final_ln_input_name, final_ln_output_name = LN_hook_names(model.ln_final)
    final_ln_input = get_activations(model, data.toks, final_ln_input_name)

    t.testing.assert_close(summed_terms, final_ln_input)


# %%

# if MAIN:
    # embedding_component = out_by_components[0]
    # print(embedding_component.shape)
    # print(data.isbal[2])

    # # first take the dot product
    # unbalanced_dir = get_pre_final_ln_dir(model, data) # [d_model]
    # data_dir = embedding_component[:, 0] # [batch, d_model]
    # # dot_product = data_dir @ unbalanced_dir # [batch]
    # dot_product = einops.einsum(data_dir, unbalanced_dir,
    #                             "batch d_model, d_model -> batch")

    # # dot_product -= dot_product.mean(0)

    # # split into balanced and unbalanced
    # balanced_examples = dot_product[data.isbal].cpu() # [batch(bal=True)]
    # unbalanced_examples = dot_product[~data.isbal].cpu() # [batch(bal=False)

    # print(balanced_examples[:10])

    # fig = go.Figure(
    #     data=[
    #         go.Histogram(x=balanced_examples),
    #     ]
    # )
    # fig.show()

# %%
if MAIN:
    unbalanced_direction = get_pre_final_ln_dir(model, data) # [d_model]
    data_direction = out_by_components[:, :, 0] # [component, batch, d_model]

    out_by_component_in_unbalanced_dir = einops.einsum(
        unbalanced_direction, data_direction, "d_model, component batch d_model -> component batch"
    ) 

    mean_over_balanced = einops.reduce(
        out_by_component_in_unbalanced_dir[:, data.isbal], "component batch -> component", "mean"
    ).unsqueeze(1)

    out_by_component_in_unbalanced_dir -= mean_over_balanced

    tests.test_out_by_component_in_unbalanced_dir(out_by_component_in_unbalanced_dir, model, data)

    plotly_utils.hists_per_comp(
        out_by_component_in_unbalanced_dir, 
        data, xaxis_range=[-10, 20]
    )
# %%
def is_balanced_vectorized_return_both(tokens: Float[Tensor, "batch seq_len"]) -> bool:
    '''
    Return True if the parens are balanced.

    tokens is a vector which has start/pad/end indices (0/1/2) as well as left/right brackets (3/4)
    '''
    # SOLUTION
    table = t.tensor([0, 0, 0, 1, -1]).to(device)
    change = table[tokens].flip(-1)
    # Get altitude by taking cumulative sum
    altitude = t.cumsum(change, -1)
    # Check that the total elevation is zero and that there are no negative altitudes
    total_elevation_failure = altitude[:, -1] != 0
    negative_failure = ~t.any(altitude < 0, dim=-1)
    print(altitude[:10])
    print(negative_failure[:10])

    return total_elevation_failure, negative_failure


if MAIN:
    total_elevation_failure, negative_failure = is_balanced_vectorized_return_both(data.toks)

    h20_in_unbalanced_dir = out_by_component_in_unbalanced_dir[7]
    h21_in_unbalanced_dir = out_by_component_in_unbalanced_dir[8]

    tests.test_total_elevation_and_negative_failures(data, total_elevation_failure, negative_failure)

