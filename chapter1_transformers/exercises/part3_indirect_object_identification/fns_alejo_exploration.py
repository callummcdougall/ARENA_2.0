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

# def patch_hook_v_by_pos_embed(v: Float[Tensor, 'batch seq head d_head'], hook: HookPoint,
#                               head_list: List[Tuple[int, int]], model: HookedTransformer, *args, **kwargs) -> Float[Tensor, 'batch seq head d_head']:
#     heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
#     if heads_to_patch:
#         pos_values = einops.einsum(model.W_pos[:v.shape[1]], model.W_V[hook.layer(), heads_to_patch],
#                                    'seq d_model, head d_model d_head -> seq head d_head')
#         v[:, :, heads_to_patch] = pos_values
#         # v[:, :, heads_to_patch] = 0
#     return v

# datasets2: List[Tuple[Tuple, str, IOIDataset]] = [
#     ((0, 0), "original", orig_dataset),
#     ((1, 0), "extend S1 vs S2", complement_dataset),
#     ((2, 0), "extend S1 vs IO", extra_and_dataset),
#     ((3, 0), 'extend introduction', long_dataset),
#     ((0, 1), "random token", orig_dataset.gen_flipped_prompts("ABB->CDD, BAB->DCD")),
#     ((1, 1), "extend S1 vs S2, random token", complement_dataset.gen_flipped_prompts("ABB->CDD, BAB->DCD")),
#     ((2, 1), "extend S1 vs IO, random token", extra_and_dataset.gen_flipped_prompts("ABB->CDD, BAB->DCD")),
#     ((3, 1), 'extend introduction, random token', long_dataset.gen_flipped_prompts("ABB->CDD, BAB->DCD")),
#     ((0, 2), "inverted position", orig_dataset.gen_flipped_prompts("ABB->BAB, BAB->ABB")),
#     ((1, 2), "extend S1 vs S2, inverted position", complement_dataset.gen_flipped_prompts("ABB->BAB, BAB->ABB")),
#     ((2, 2), "extend S1 vs IO, inverted position", extra_and_dataset.gen_flipped_prompts("ABB->BAB, BAB->ABB")),
#     ((3, 2), 'extend introduction, inverted position', long_dataset.gen_flipped_prompts("ABB->BAB, BAB->ABB")),
#     ((0, 3), "Inverted Tok", orig_dataset.gen_flipped_prompts("ABB->BAA, BAB->ABA")),
#     ((1, 3), "extend S1 vs S2, Inverted Tok", complement_dataset.gen_flipped_prompts("ABB->BAA, BAB->ABA")),
#     ((2, 3), "extend S1 vs IO, Inverted Tok", extra_and_dataset.gen_flipped_prompts("ABB->BAA, BAB->ABA")),
#     ((3, 3), 'extend introduction, Inverted Tok', long_dataset.gen_flipped_prompts("ABB->BAA, BAB->ABA")),
# ]


def get_custom_patch_logits(
    model: HookedTransformer,
    orig_dataset: IOIDataset, 
    new_dataset: IOIDataset,
    patch_list: List[Tuple[Union[Callable, str], Callable]], # The patching function can receive the patching cache as cache and both datasets
    patching_metric: Optional[Callable] = None,
    patch_from_orig: bool = False,
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

    if patch_from_orig:
        _, orig_cache = model.run_with_cache(orig_dataset.toks)
    else:
        orig_cache = None

    hook_kwargs = dict(cache=cache_for_patching,orig_dataset=orig_dataset, new_dataset=new_dataset, orig_cache=orig_cache)
    hooks_ready = [(names_filter, partial(hook_fn, **hook_kwargs)) for names_filter, hook_fn in patch_list]

    patched_logits = model.run_with_hooks(
        orig_dataset.toks,
        fwd_hooks=hooks_ready
    )

    if patching_metric is None:
        patching_metric = partial(logits_to_ave_logit_diff_2, dataset=orig_dataset)
    return patching_metric(logits=patched_logits)


def patch_hook_x(x: Float[Tensor, "batch seq head d_head"], hook: HookPoint, cache: ActivationCache,
                     orig_dataset: IOIDataset, new_dataset: IOIDataset, head_list: List[Tuple[int, int]],
                     pos: Union[str, List[str]], *args, **kwargs) -> Float[Tensor, "batch seq head d_head"]:

    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    pos = [pos] if isinstance(pos, str) else pos

    for p in pos:
        pos_orig, pos_cache = orig_dataset.word_idx[p], new_dataset.word_idx[p]
        batch_idx = t.arange(x.shape[0]).to(x.device)    
        x[batch_idx[:, None], pos_orig[:, None], heads_to_patch] = cache[hook.name][batch_idx[:, None], pos_cache[:, None], heads_to_patch]

    return x

def patch_hook_x_all_pos(x: Float[Tensor, "batch seq head d_head"], hook: HookPoint, cache: ActivationCache,
                        orig_dataset: IOIDataset, new_dataset: IOIDataset, head_list: List[Tuple[int, int]],
                        *args, **kwargs) -> Float[Tensor, "batch seq head d_head"]:
    
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    batch_idx = t.arange(x.shape[0]).to(x.device)
    
    x[:, :, heads_to_patch] = cache[hook.name][:, :, heads_to_patch]
    return x

def patch_hook_x_cross_pos(x: Float[Tensor, "batch seq head d_head"], hook: HookPoint, cache: ActivationCache,
                           orig_dataset: IOIDataset, new_dataset: IOIDataset, head_list: List[Tuple[int, int]],
                           orig_pos: Union[str, List[str]], new_pos: Union[str, List[str]], *args, **kwargs
                           ) -> Float[Tensor, "batch seq head d_head"]:

    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]

    if heads_to_patch:
        orig_pos = [orig_pos] if isinstance(orig_pos, str) else orig_pos
        new_pos = [new_pos] if isinstance(new_pos, str) else new_pos
        batch_idx = t.arange(x.shape[0]).to(x.device)    

        for orig_p, new_p in zip(orig_pos, new_pos):
            orig_p_idx, new_p_idx = orig_dataset.word_idx[orig_p], new_dataset.word_idx[new_p]
            x[batch_idx[:, None], orig_p_idx[:, None], heads_to_patch] = cache[hook.name][batch_idx[:, None], new_p_idx[:, None], heads_to_patch]

    return x


def visualize_selected_heads(model: HookedTransformer, 
                             prompts: Union[List[str], Int[Tensor, "batch seq"]],
                             heads: List[Tuple[int, int]],
                             idx = 0,
                             max_seq_len: int = 50
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



def visualize_selected_head(model: HookedTransformer, 
                             prompts: Union[List[str], Int[Tensor, "batch seq"]],
                             head: Tuple[int, int],
                             idx = 0,
                             max_seq_len: int = 50
    ):

    if isinstance(prompts, list):
        tokens = model.to_tokens(prompts[idx])[:, :max_seq_len]
    else:
        tokens = prompts[idx, :max_seq_len]
    
    str_tokens = model.to_str_tokens(tokens)

    _, text_cache = model.run_with_cache(tokens, names_filter=lambda n: 'pattern' in n)

    selected_attn_pattern = text_cache['pattern', head[0]][:, head[1]][0]

    display(cv.attention.attention_pattern(
        attention = selected_attn_pattern,
        tokens = str_tokens,
    ))


def freeze_attn_pattern(pattn: Float[Tensor, "batch head seq_q seq_k"], hook: HookPoint, orig_cache: ActivationCache,
                        head_list: List[Tuple[int, int]], start_layer: int=0, *args, **kwargs
                        ) -> Float[Tensor, "batch head seq_q seq_k"]:
    layer = hook.layer()
    if layer >= start_layer:
        heads_to_patch = [head for head in range(orig_cache.model.cfg.n_heads) if (layer, head) not in head_list]
        pattn[:, heads_to_patch] = orig_cache[hook.name][:, heads_to_patch]
        return pattn
    
def collect_activations(activations: Float[Tensor, 'batch ...'], hook: HookPoint,
                        ctx: Dict, head_list: Optional[List[Tuple[int, int]]] = None, *args, **kwargs):
    if head_list:
        layers, heads = zip(*head_list)
        if hook.layer() in layers:
            ctx[hook.name] = activations   
    else:
        ctx[hook.name] = activations

def attn_to_io(ctx: Dict[str, Float[Tensor, 'batch head seq_q seq_k']],
               orig_dataset: IOIDataset, head_list: List[Tuple[int, int]],
               src_pos: str = 'end', dst_pos: str = 'IO', *args, **kwargs) -> float:
    '''
    Returns the average attention weight of the end token to the io token.
    '''
    src_pos_idx, dst_pos_idx = orig_dataset.word_idx[src_pos], orig_dataset.word_idx[dst_pos]
    batch_idx = t.arange(orig_dataset.toks.shape[0], device=orig_dataset.toks.device)
    sum_attn_weights, n_weights = 0.0, 0
    for hook_name, pattn in ctx.items():
        relevant_heads = [head for layer, head in head_list if hook_name == f'blocks.{layer}.attn.hook_pattern']
        if relevant_heads:
            relevant_heads_idx = t.tensor(relevant_heads, device=pattn.device)[:, None]
            attn_weights = pattn[batch_idx, relevant_heads_idx, src_pos_idx, dst_pos_idx]
            sum_attn_weights += attn_weights.sum().item()
            n_weights += attn_weights.numel()

    return sum_attn_weights / n_weights

import plotly.graph_objects as go

def plot_logit_attr(cache: ActivationCache,
                    pos_slice: int,
                    top_k: int = 10,
                    component_slice: slice = slice(None, None, 4),
                    add_b_U: bool = False,
                    show_plot: bool = True) -> None:
    resid_components, resid_labels = cache.accumulated_resid(return_labels=True,
                                                                    # incl_mid=True,
                                                                    apply_ln=True,
                                                                    pos_slice=pos_slice,)
    
    redux_resid_components, redux_resid_labels = resid_components[component_slice], resid_labels[component_slice]
    resid_attrs = redux_resid_components @ cache.model.W_U 
    if add_b_U:
        resid_attrs = resid_attrs + cache.model.b_U
    top_logits, top_tokens = resid_attrs.mean(1).topk(top_k, dim=-1) # mean gets rid of batch dim
    top_tokens_list = [cache.model.to_str_tokens(t) for t in top_tokens]

    fig = go.Figure(data=go.Heatmap(
        z=top_logits.cpu().numpy(),
        y=redux_resid_labels,
        text=top_tokens_list,
        texttemplate='%{text}',
    ))
    fig.show()

    return top_logits, top_tokens

def plot_logit_attr_heads(cache: ActivationCache,
                    heads: List[Tuple[int, int]],
                    pos_slice: int,
                    top_k: int = 10,
                    add_b_U: bool = False,
                    show_plot: bool = True,
                    largest=True) -> None:
    head_components = t.stack([cache['result', layer][:, pos_slice, head] for layer, head in heads])
    head_labels = [f'{layer}.{head}' for layer, head in heads]
    
    resid_attrs = head_components @ cache.model.W_U 
    if add_b_U:
        resid_attrs = resid_attrs + cache.model.b_U
    top_logits, top_tokens = resid_attrs.mean(1).topk(top_k, dim=-1, largest=largest) # mean gets rid of batch dim
    top_tokens_list = [cache.model.to_str_tokens(t) for t in top_tokens]

    fig = go.Figure(data=go.Heatmap(
        z=top_logits.cpu().numpy(),
        y=head_labels,
        text=top_tokens_list,
        texttemplate='%{text}',
    ))
    fig.show()

    return top_logits, top_tokens
