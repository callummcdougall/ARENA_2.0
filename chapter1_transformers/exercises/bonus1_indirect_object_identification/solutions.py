# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"

# Import stuff
import torch as t
from torch import nn, optim, Tensor
import torch.nn.functional as F
import numpy as np
import einops
# import tqdm.notebook as tqdm
from tqdm.notebook import tqdm_notebook
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader
import webbrowser
# import pysvelte
import re

from jaxtyping import Float, Int
from typing import List, Union, Optional, Callable, Tuple
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import display, HTML

from rich.table import Table, Column
from rich import print as rprint

import circuitsvis as cv

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

# if MAIN:
#     from IPython import get_ipython
#     ipython = get_ipython()
#     ipython.run_line_magic("load_ext", "autoreload")
#     ipython.run_line_magic("autoreload", "2")

# %%

if MAIN:
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )

# %%

def logits_to_ave_logit_diff(logits: Float[Tensor, "batch seq d_vocab"], answer_tokens: Int[Tensor, "batch 2"], per_prompt=False):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

# %%

def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"], 
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"],
) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given 
    stack of components in the residual stream.
    '''
    batch_size = residual_stack.size(-2)
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    return einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size

# %%

def ioi_metric(
    logits: Float[Tensor, "batch seq d_vocab"], 
    answer_tokens: Float[Tensor, "batch 2"],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
) -> float:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is  
    same as on corrupted input, and 1 when performance is same as on clean input.
    '''
    patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff  - corrupted_logit_diff)

# %%

def patch_residual_component(
    corrupted_residual_component: Float[Tensor, "batch pos d_model"],
    hook: HookPoint, 
    pos: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos d_model"]:
    '''
    Patches a given sequence position in the residual stream, using the value from the 
    clean cache.
    '''
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual_component


def get_act_patch_resid_pre(
    model: HookedTransformer, 
    corrupted_tokens: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable[[Float[Tensor, "batch pos d_vocab"]], float]
) -> Float[Tensor, "layer pos"]:
    '''
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    model.reset_hooks()
    seq_len = corrupted_tokens.size(1)
    results = t.zeros(model.cfg.n_layers, seq_len, device="cuda", dtype=t.float32)

    for layer in tqdm_notebook(range(model.cfg.n_layers)):
        for position in range(seq_len):
            hook_fn = partial(patch_residual_component, pos=position, clean_cache=clean_cache)
            patched_logits = model.run_with_hooks(
                corrupted_tokens, 
                fwd_hooks = [(utils.get_act_name("resid_pre", layer), hook_fn)], 
            )
            results[layer, position] = patching_metric(patched_logits)

    return results

# %%