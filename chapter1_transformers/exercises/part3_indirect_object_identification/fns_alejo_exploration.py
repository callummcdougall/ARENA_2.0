# %%


import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from pathlib import Path
import torch as t
from torch import Tensor
import numpy as np
import einops
from tqdm.notebook import tqdm
import plotly.express as px
import webbrowser
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from functools import partial
from IPython.display import display, HTML
from rich.table import Table, Column
from rich import print as rprint
import circuitsvis as cv
from pathlib import Path
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP
from ioi_dataset import NAMES, IOIDataset

t.set_grad_enabled(False)

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part3_indirect_object_identification").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, scatter, bar
import tests as tests

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")


# %%


def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    per_prompt: bool = False
):
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


def logits_to_ave_logit_diff_2(logits: Float[Tensor, "batch seq d_vocab"], dataset: IOIDataset, per_prompt=False):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    
    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    io_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), dataset.word_idx["end"], dataset.io_tokenIDs]
    s_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), dataset.word_idx["end"], dataset.s_tokenIDs]
    # Find logit difference
    answer_logit_diff = io_logits - s_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


def ioi_metric(
    logits: Float[Tensor, "batch seq d_vocab"], 
    answer_tokens: Float[Tensor, "batch 2"],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is 
    same as on corrupted input, and 1 when performance is same as on clean input.
    '''
    patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff  - corrupted_logit_diff)



# %% PATCHING PLOTS

from transformer_lens import patching

def plot_patching_experiments(clean_tokens: Float[Tensor, "batch seq"],
                              corrupted_tokens: Float[Tensor, "batch seq"],
                              answer_tokens: Float[Tensor, "batch 2"],
                              model: HookedTransformer,
    ) -> None:


    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

    clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
    corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
    print(f'{clean_logit_diff = }', f'{corrupted_logit_diff = }', sep='\n')

    ioi_metric_arged = partial(ioi_metric, answer_tokens=answer_tokens,
                               corrupted_logit_diff=corrupted_logit_diff, clean_logit_diff=clean_logit_diff)
    
    act_patch_block_every = patching.get_act_patch_block_every(model, corrupted_tokens, clean_cache, ioi_metric_arged)
    
    act_patch_attn_head_out_all_pos = patching.get_act_patch_attn_head_out_all_pos(
        model, 
        corrupted_tokens, 
        clean_cache, 
        ioi_metric_arged
    )

    act_patch_attn_head_all_pos_every = patching.get_act_patch_attn_head_all_pos_every(
        model, 
        corrupted_tokens, 
        clean_cache, 
        ioi_metric_arged
    )

    labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]
    
    imshow(
        act_patch_block_every,
        x=labels, 
        facet_col=0, # This argument tells plotly which dimension to split into separate plots
        facet_labels=["Residual Stream", "Attn Output", "MLP Output"], # Subtitles of separate plots
        title="Logit Difference From Patched Attn Head Output", 
        labels={"x": "Sequence Position", "y": "Layer"},
        width=1000,
    )

    imshow(
        act_patch_attn_head_out_all_pos, 
        labels={"y": "Layer", "x": "Head"}, 
        title="attn_head_out Activation Patching (All Pos)",
        width=600
    )

    imshow(
        act_patch_attn_head_all_pos_every, 
        facet_col=0, 
        facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
        title="Activation Patching Per Head (All Pos)", 
        labels={"x": "Head", "y": "Layer"},
    )

# %%

def topk_predictions_from_prompt(prompt: str, model: HookedTransformer, top_k: int = 5
    ) -> Tuple[List[str], Float[Tensor, "top_k"]]:
    tokens = model.to_tokens(prompt)
    logits = model(tokens)[:,-1,:] # Index the last token position
    probs = t.softmax(logits, dim=-1)
    top_probs, top_tokens = probs.topk(k=top_k, dim=-1)
    return model.to_str_tokens(top_tokens), top_probs


# %%

def get_custom_patch_logits(
    model: HookedTransformer,
    orig_dataset: IOIDataset, 
    new_dataset: IOIDataset,
    patch_list: List[Tuple[Union[Callable, str], Callable]], # The patching function can receive the patching cache as cache and both datasets
    patching_metric: Optional[Callable] = None,
) -> float:
    
    def patch_cache_filter(name: str) -> bool:
        included = []
        for names_filter, _ in patch_list:
            if isinstance(names_filter, str):
                if name is names_filter:
                    return True
            elif isinstance(names_filter, Callable):
                if names_filter(name):
                    return True
        return False
    
    _, cache_for_patching = model.run_with_cache(
        new_dataset.toks,
        names_filter=patch_cache_filter,
        return_type=None
    )

    hooks_ready = [(names_filter, partial(hook_fn, cache=cache_for_patching, orig_dataset=orig_dataset, new_dataset=new_dataset))
             for names_filter, hook_fn in patch_list]

    patched_logits = model.run_with_hooks(
        orig_dataset.toks,
        fwd_hooks=hooks_ready
    )

    if patching_metric is None:
        patching_metric = partial(logits_to_ave_logit_diff_2, dataset=orig_dataset)
    return patching_metric(patched_logits)


def patch_hook_z(z: Float[Tensor, "batch seq head d_head"], hook: HookPoint, cache: ActivationCache,
                     orig_dataset: IOIDataset, new_dataset: IOIDataset, head_list: List[Tuple[int, int]],
                     pos: str = 'end', *args, **kwargs) -> Float[Tensor, "batch seq head d_head"]:
    
    end_pos_orig, end_pos_cache = orig_dataset.word_idx[pos], new_dataset.word_idx[pos]
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    batch_idx = t.arange(z.shape[0]).to(z.device)
    
    z[batch_idx[:, None], end_pos_orig[:, None], heads_to_patch] = cache[hook.name][batch_idx[:, None], end_pos_cache[:, None], heads_to_patch]
    return z

def patch_hook_z_all_pos(z: Float[Tensor, "batch seq head d_head"], hook: HookPoint, cache: ActivationCache,
                        orig_dataset: IOIDataset, new_dataset: IOIDataset, head_list: List[Tuple[int, int]],
                        *args, **kwargs) -> Float[Tensor, "batch seq head d_head"]:
    
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    batch_idx = t.arange(z.shape[0]).to(z.device)
    
    z[:, :, heads_to_patch] = cache[hook.name][:, :, heads_to_patch]
    return z


def visualize_selected_heads(model: HookedTransformer, 
                             prompts: Union[List[str], Int[Tensor, "batch seq"]],
                             heads: List[Tuple[int, int]],
                             idx = 0,
                             max_seq_len: int = 30
    ):

    if isinstance(prompts, list):
        tokens = model.to_tokens(prompts[idx])[:, :max_seq_len]
    else:
        tokens = prompts[idx, :max_seq_len]
    
    str_tokens = model.to_str_tokens(tokens)

    _, text_cache = model.run_with_cache(tokens,
                                    names_filter=lambda n: 'pattern' in n)

    selected_attn_patterns = t.stack([text_cache['pattern', layer][:, head][0]
                                for layer, head in heads])

    display(cv.attention.attention_patterns(
        attention = selected_attn_patterns,
        tokens = str_tokens,
        attention_head_names = [f"{layer}.{head}" for layer, head in heads],
    ))
