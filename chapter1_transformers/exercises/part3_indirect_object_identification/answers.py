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
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set
from functools import partial
from IPython.display import display, HTML
from rich.table import Table, Column
from rich import print as rprint
import circuitsvis as cv
from pathlib import Path
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP

t.set_grad_enabled(False)

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part3_indirect_object_identification").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, scatter, bar
import part3_indirect_object_identification.tests as tests

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

MAIN = __name__ == "__main__"
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
# Here is where we test on a single prompt
# Result: 70% probability on Mary, as we expect


if MAIN:
    example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
    example_answer = " Mary"
    utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)
# %%
if MAIN:
    prompt_format = [
        "When John and Mary went to the shops,{} gave the bag to",
        "When Tom and James went to the park,{} gave the ball to",
        "When Dan and Sid went to the shops,{} gave an apple to",
        "After Martin and Amy went to the park,{} gave a drink to",
    ]
    name_pairs = [
        (" Mary", " John"),
        (" Tom", " James"),
        (" Dan", " Sid"),
        (" Martin", " Amy"),
    ]

    # Define 8 prompts, in 4 groups of 2 (with adjacent prompts having answers swapped)
    prompts = [
        prompt.format(name) 
        for (prompt, names) in zip(prompt_format, name_pairs) for name in names[::-1] 
    ]
    # Define the answers for each prompt, in the form (correct, incorrect)
    answers = [names[::i] for names in name_pairs for i in (1, -1)]
    # Define the answer tokens (same shape as the answers)
    answer_tokens = t.concat([
        model.to_tokens(names, prepend_bos=False).T for names in answers
    ])

    rprint(prompts)
    rprint(answers)
    rprint(answer_tokens)
# %%
if MAIN:
    tokens = model.to_tokens(prompts, prepend_bos=True)
    # Move the tokens to the GPU
    tokens = tokens.to(device)
    # Run the model and cache all activations
    original_logits, cache = model.run_with_cache(tokens)
# %%
# %%
def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    per_prompt: bool = False
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    answer_logits = logits[:,-1,:]
    correct_logits = t.gather(input=answer_logits, dim=-1, index=answer_tokens[:,:1]).squeeze(-1)
    incorrect_logits = t.gather(input=answer_logits, dim=-1, index=answer_tokens[:,1:]).squeeze(-1)

    logit_diff = correct_logits - incorrect_logits
    if per_prompt:
        return logit_diff
    return logit_diff.mean()

if MAIN:
    tests.test_logits_to_ave_logit_diff(logits_to_ave_logit_diff)

    original_per_prompt_diff = logits_to_ave_logit_diff(original_logits, answer_tokens, per_prompt=True)
    print("Per prompt logit difference:", original_per_prompt_diff)
    original_average_logit_diff = logits_to_ave_logit_diff(original_logits, answer_tokens)
    print("Average logit difference:", original_average_logit_diff)

    cols = [
        "Prompt", 
        Column("Correct", style="rgb(0,200,0) bold"), 
        Column("Incorrect", style="rgb(255,0,0) bold"), 
        Column("Logit Difference", style="bold")
    ]
    table = Table(*cols, title="Logit differences")

    for prompt, answer, logit_diff in zip(prompts, answers, original_per_prompt_diff):
        table.add_row(prompt, repr(answer[0]), repr(answer[1]), f"{logit_diff.item():.3f}")

    rprint(table)
# %%
if MAIN:
    answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(answer_tokens)
    print("Answer residual directions shape:", answer_residual_directions.shape)

    correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
    logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_directions - incorrect_residual_directions
    print(f"Logit difference directions shape:", logit_diff_directions.shape)
# %%
# cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type]. 

if MAIN:
    final_residual_stream: Float[Tensor, "batch seq d_model"] = cache["resid_post", -1]
    print(f"Final residual stream shape: {final_residual_stream.shape}")
    final_token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[:, -1, :]

    # Apply LayerNorm scaling (to just the final sequence position)
    # pos_slice is the subset of the positions we take - here the final token of each prompt
    scaled_final_token_residual_stream = cache.apply_ln_to_stack(final_token_residual_stream, layer=-1, pos_slice=-1)

    average_logit_diff = einops.einsum(
        scaled_final_token_residual_stream, logit_diff_directions,
        "batch d_model, batch d_model ->"
    ) / len(prompts)

    print(f"Calculated average logit diff: {average_logit_diff:.10f}")
    print(f"Original logit difference:     {original_average_logit_diff:.10f}")

    t.testing.assert_close(average_logit_diff, original_average_logit_diff)
# %%
def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"], 
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"] = logit_diff_directions,
) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given 
    stack of components in the residual stream.
    '''
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    
    average_logit_diff = einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        '... batch d_model, batch d_model -> ...'
    ) / len(prompts)

    return average_logit_diff


if MAIN:
    t.testing.assert_close(
        residual_stack_to_logit_diff(final_token_residual_stream, cache),
        original_average_logit_diff
    )
    print("tests passed!")
# %%
if MAIN:
    accumulated_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
    # accumulated_residual has shape (component, batch, d_model)

    logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(accumulated_residual, cache)

    line(
        logit_lens_logit_diffs, 
        hovermode="x unified",
        title="Logit Difference From Accumulated Residual Stream",
        labels={"x": "Layer", "y": "Logit Diff"},
        xaxis_tickvals=labels,
        width=800
    )
# %%
if MAIN:
    per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
    per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache)
    line(
        per_layer_logit_diffs, 
        hovermode="x unified",
        title="Logit Difference From Each Layer",
        labels={"x": "Layer", "y": "Logit Diff"},
        xaxis_tickvals=labels,
        width=800
    )
# %%
if MAIN:
    per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
    per_head_residual = einops.rearrange(
        per_head_residual, 
        "(layer head) ... -> layer head ...", 
        layer=model.cfg.n_layers
    )
    per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, cache)

    imshow(
        per_head_logit_diffs, 
        labels={"x":"Head", "y":"Layer"}, 
        title="Logit Difference From Each Head",
        width=600
    )
# %%
def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = t.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()



if MAIN:
    k = 3

    for head_type in ["Positive", "Negative"]:

        # Get the heads with largest (or smallest) contribution to the logit difference
        top_heads = topk_of_Nd_tensor(per_head_logit_diffs * (1 if head_type=="Positive" else -1), k)

        # Get all their attention patterns
        attn_patterns_for_important_heads: Float[Tensor, "head q k"] = t.stack([
            cache["pattern", layer][:, head].mean(0)
            for layer, head in top_heads
        ])

        # Display results
        display(HTML(f"<h2>Top {k} {head_type} Logit Attribution Heads</h2>"))
        display(cv.attention.attention_patterns(
            attention = attn_patterns_for_important_heads,
            tokens = model.to_str_tokens(tokens[0]),
            attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
        ))
# %%
from transformer_lens import patching

if MAIN:
    clean_tokens = tokens
    # Swap each adjacent pair to get corrupted tokens
    indices = [i+1 if i % 2 == 0 else i-1 for i in range(len(tokens))]
    corrupted_tokens = clean_tokens[indices]

    print(
        "Clean string 0:    ", model.to_string(clean_tokens[0]), "\n"
        "Corrupted string 0:", model.to_string(corrupted_tokens[0])
    )

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

    clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
    print(f"Clean logit diff: {clean_logit_diff:.4f}")

    corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
    print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")
# %%
def ioi_metric(
    logits: Float[Tensor, "batch seq d_vocab"], 
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    corrupted_logit_diff: float = corrupted_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is 
    same as on corrupted input, and 1 when performance is same as on clean input.
    '''
    answer_logits = logits[:,-1,:]
    correct_logits = t.gather(input=answer_logits, dim=-1, index=answer_tokens[:,:1]).squeeze(-1)
    incorrect_logits = t.gather(input=answer_logits, dim=-1, index=answer_tokens[:,1:]).squeeze(-1)
    logit_diffs = (correct_logits - incorrect_logits)
    logit_diff_restored = (logit_diffs - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
    return logit_diff_restored.mean()


if MAIN:
    t.testing.assert_close(ioi_metric(clean_logits).item(), 1.0)
    t.testing.assert_close(ioi_metric(corrupted_logits).item(), 0.0)
    t.testing.assert_close(ioi_metric((clean_logits + corrupted_logits) / 2).item(), 0.5)
    print("all tests passed!")
# %%
if MAIN:
    act_patch_resid_pre = patching.get_act_patch_resid_pre(
        model = model,
        corrupted_tokens = corrupted_tokens,
        clean_cache = clean_cache,
        patching_metric = ioi_metric
    )

    labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]

    imshow(
        act_patch_resid_pre, 
        labels={"x": "Position", "y": "Layer"},
        x=labels,
        title="resid_pre Activation Patching",
        width=600
    )

# %%

clean_acts = clean_cache["resid_pre", 0]
clean_acts = einops.rearrange(clean_acts, 'batch pos (head d_head) -> batch pos head d_head', head=12, d_head=64)

# %%
def patch_residual_component(
    corrupted_residual_component: Float[Tensor, "batch pos d_model"],
    hook: HookPoint, 
    pos: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos d_model"]:
    '''
    Patches a given sequence position in the residual stream, using the value
    from the clean cache.
    '''
    corrupted_residual_component[:,pos,:] = clean_cache[hook.name][:,pos,:]
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
    seq_len = corrupted_tokens.shape[1]
    patching_results = []

    for l in range(12):
        pos_patching_results = []
        for p in range(seq_len):
            patching_fn = partial(patch_residual_component, pos=p, clean_cache=clean_cache)
            logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[
                    (utils.get_act_name('resid_pre', l), patching_fn)
                ]
            )
            pos_patching_results.append(patching_metric(logits))
        patching_results.append(pos_patching_results)

    return t.tensor(patching_results).to(device)



if MAIN:
    act_patch_resid_pre_own = get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, ioi_metric)
    t.testing.assert_close(act_patch_resid_pre, act_patch_resid_pre_own)
# %%

if MAIN:
    imshow(
        act_patch_resid_pre_own, 
        x=labels, 
        title="Logit Difference From Patched Residual Stream", 
        labels={"x":"Sequence Position", "y":"Layer"},
        width=600 # If you remove this argument, the plot will usually fill the available space
    )
# %%
if MAIN:
    act_patch_block_every = patching.get_act_patch_block_every(model, corrupted_tokens, clean_cache, ioi_metric)

    imshow(
        act_patch_block_every,
        x=labels, 
        facet_col=0, # This argument tells plotly which dimension to split into separate plots
        facet_labels=["Residual Stream", "Attn Output", "MLP Output"], # Subtitles of separate plots
        title="Logit Difference From Patched Attn Head Output", 
        labels={"x": "Sequence Position", "y": "Layer"},
        width=1000,
    )
# %%
def get_act_patch_block_every(
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
    seq_len = corrupted_tokens.shape[1]
    patching_results = t.zeros((3,model.cfg.n_layers,seq_len), device=device)

    for idx, component in enumerate(["resid_pre", "attn_out", "mlp_out"]):
        for l in range(12):
            pos_patching_results = t.zeros(seq_len, device=device)
            for p in range(seq_len):
                patching_fn = partial(patch_residual_component, pos=p, clean_cache=clean_cache)
                logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[
                        (utils.get_act_name(component, l), patching_fn)
                    ]
                )
                pos_patching_results[p] = patching_metric(logits)
            patching_results[idx][l] = pos_patching_results

    return t.tensor(patching_results).to(device)


if MAIN:
    act_patch_block_every_own = get_act_patch_block_every(model, corrupted_tokens, clean_cache, ioi_metric)

    t.testing.assert_close(act_patch_block_every, act_patch_block_every_own)

    imshow(
        act_patch_block_every_own,
        x=labels, 
        facet_col=0,
        facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
        title="Logit Difference From Patched Attn Head Output", 
        labels={"x": "Sequence Position", "y": "Layer"},
        width=1000
    )
# %%
if MAIN:
    act_patch_attn_head_out_all_pos = patching.get_act_patch_attn_head_out_all_pos(
        model, 
        corrupted_tokens, 
        clean_cache, 
        ioi_metric
    )

    imshow(
        act_patch_attn_head_out_all_pos, 
        labels={"y": "Layer", "x": "Head"}, 
        title="attn_head_out Activation Patching (All Pos)",
        width=600
    )
# %%
def patch_head_vector(
    corrupted_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint, 
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches the output of a given head (before it's added to the residual stream) at
    every sequence position, using the value from the clean cache.
    '''
    corrupted_head_vector[:,:,head_index,:] = clean_cache[hook.name][:,:,head_index,:]
    return corrupted_head_vector

def get_act_patch_attn_head_out_all_pos(
    model: HookedTransformer, 
    corrupted_tokens: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable
) -> Float[Tensor, "layer head"]:
    '''
    Returns an array of results of patching at all positions for each head in each
    layer, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    model.reset_hooks()

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = t.zeros((n_layers, n_heads), device=device)

    for l in range(n_layers):
        for h in range(n_heads):
            patching_fn = partial(patch_head_vector, head_index=h, clean_cache=clean_cache)
            logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[
                    (utils.get_act_name("z", l), patching_fn)
                ]
            )
            results[l][h] = patching_metric(logits)
    
    return results


if MAIN:
    act_patch_attn_head_out_all_pos_own = get_act_patch_attn_head_out_all_pos(model, corrupted_tokens, clean_cache, ioi_metric)

    t.testing.assert_close(act_patch_attn_head_out_all_pos, act_patch_attn_head_out_all_pos_own)

    imshow(
        act_patch_attn_head_out_all_pos_own,
        title="Logit Difference From Patched Attn Head Output", 
        labels={"x":"Head", "y":"Layer"},
        width=600
    )
# %%
if MAIN:
    act_patch_attn_head_all_pos_every = patching.get_act_patch_attn_head_all_pos_every(
        model, 
        corrupted_tokens, 
        clean_cache, 
        ioi_metric
    )

    imshow(
        act_patch_attn_head_all_pos_every, 
        facet_col=0, 
        facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
        title="Activation Patching Per Head (All Pos)", 
        labels={"x": "Head", "y": "Layer"},
    )
# %%
# Get the heads with largest value patching
# (we know from plot above that these are the 4 heads in layers 7 & 8)
if MAIN:
    k = 4
    top_heads = topk_of_Nd_tensor(act_patch_attn_head_all_pos_every[3], k=k)

    print(top_heads)

    # Get all their attention patterns
    attn_patterns_for_important_heads: Float[Tensor, "head q k"] = t.stack([
        cache["pattern", layer][:, head].mean(0)
            for layer, head in top_heads
    ])

    # Display results
    display(HTML(f"<h2>Top {k} Logit Attribution Heads (from value-patching)</h2>"))
    display(cv.attention.attention_patterns(
        attention = attn_patterns_for_important_heads,
        tokens = model.to_str_tokens(tokens[0]),
        attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
    ))
    # %%
    print(act_patch_attn_head_all_pos_every.shape)
# %%
if MAIN:
    attn_patterns_for_interesting_early_heads: Float[Tensor, "head q k"] = t.stack([
        cache["pattern", layer][:, head].mean(0)
            for layer, head in [(3,0),(5,5),(6,9)]
    ])

    display(cv.attention.attention_patterns(
        attention = attn_patterns_for_interesting_early_heads,
        tokens = model.to_str_tokens(tokens[0]),
        attention_head_names = [f"{layer}.{head}" for layer, head in [(3,0),(5,5),(6,9)]],
    ))


#### PART 4: PATH PATCHING
# %%
from part3_indirect_object_identification.ioi_dataset import NAMES, IOIDataset
# %%

if MAIN:
    N = 25
    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=1,
        device=str(device)
    )
# %%
if MAIN:
    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")
# %%
def format_prompt(sentence: str) -> str:
    '''Format a prompt by underlining names (for rich print)'''
    return re.sub("(" + "|".join(NAMES) + ")", lambda x: f"[u bold dark_orange]{x.group(0)}[/]", sentence) + "\n"


def make_table(cols, colnames, title="", n_rows=5, decimals=4):
    '''Makes and displays a table, from cols rather than rows (using rich print)'''
    table = Table(*colnames, title=title)
    rows = list(zip(*cols))
    f = lambda x: x if isinstance(x, str) else f"{x:.{decimals}f}"
    for row in rows[:n_rows]:
        table.add_row(*list(map(f, row)))
    rprint(table)

if MAIN:
    make_table(
        colnames = ["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
        cols = [
            map(format_prompt, ioi_dataset.sentences), 
            model.to_string(ioi_dataset.s_tokenIDs).split(), 
            model.to_string(ioi_dataset.io_tokenIDs).split(), 
            map(format_prompt, abc_dataset.sentences), 
        ],
        title = "Sentences from IOI vs ABC distribution",
    )
# %%
def logits_to_ave_logit_diff_2(logits: Float[Tensor, "batch seq d_vocab"], ioi_dataset: IOIDataset = ioi_dataset, per_prompt=False):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''

    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    io_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs]
    s_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs]
    # Find logit difference
    answer_logit_diff = io_logits - s_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()



if MAIN:
    model.reset_hooks(including_permanent=True)

    ioi_logits_original, ioi_cache = model.run_with_cache(ioi_dataset.toks)
    abc_logits_original, abc_cache = model.run_with_cache(abc_dataset.toks)

    ioi_per_prompt_diff = logits_to_ave_logit_diff_2(ioi_logits_original, per_prompt=True)
    abc_per_prompt_diff = logits_to_ave_logit_diff_2(abc_logits_original, per_prompt=True)

    ioi_average_logit_diff = logits_to_ave_logit_diff_2(ioi_logits_original).item()
    abc_average_logit_diff = logits_to_ave_logit_diff_2(abc_logits_original).item()
# %%
if MAIN:
    print(f"Average logit diff (IOI dataset): {ioi_average_logit_diff:.4f}")
    print(f"Average logit diff (ABC dataset): {abc_average_logit_diff:.4f}")

    make_table(
        colnames = ["IOI prompt", "IOI logit diff", "ABC prompt", "ABC logit diff"],
        cols = [
            map(format_prompt, ioi_dataset.sentences), 
            ioi_per_prompt_diff,
            map(format_prompt, abc_dataset.sentences), 
            abc_per_prompt_diff,
        ],
        title = "Sentences from IOI vs ABC distribution",
    )
# %%
def ioi_metric_2(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_logit_diff: float = ioi_average_logit_diff,
    corrupted_logit_diff: float = abc_average_logit_diff,
    ioi_dataset: IOIDataset = ioi_dataset,
) -> float:
    '''
    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset), 
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    patched_logit_diff = logits_to_ave_logit_diff_2(logits, ioi_dataset)
    return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)



if MAIN:
    print(f"IOI metric (IOI dataset): {ioi_metric_2(ioi_logits_original):.4f}")
    print(f"IOI metric (ABC dataset): {ioi_metric_2(abc_logits_original):.4f}")
# %%
def store_activations(
    residual: Float[Tensor, "batch pos d_model"],
    hook: HookPoint, 
    store: Float[Tensor, "batch pos d_model"], 
) -> Float[Tensor, "batch pos d_model"]:
    '''
    Patches a given sequence position in the residual stream, using the value
    from the clean cache.
    '''
    store[:] = residual[:]
    return residual

def patch_residual(
    original_residual: Float[Tensor, "batch pos d_model"],
    hook: HookPoint, 
    patched_residual: Float[Tensor, "batch pos d_model"]
) -> Float[Tensor, "batch pos d_model"]:
    '''
    Patches a given sequence position in the residual stream, using the value
    from the clean cache.
    '''
    original_residual[:] = patched_residual
    return original_residual


def get_path_patch_head_to_final_resid_post(
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: IOIDataset = abc_dataset,
    orig_dataset: IOIDataset = ioi_dataset,
    new_cache: Optional[ActivationCache] = abc_cache,
    orig_cache: Optional[ActivationCache] = ioi_cache,
) -> Float[Tensor, "layer head"]:
    # Part 1: cache heads
    # only if we don't already have new_cache and orig_cache

    if new_cache is None:
        _, new_cache = model.run_with_cache(new_dataset.toks)
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(orig_dataset.toks)
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = t.zeros((n_layers, n_heads), device=device)

    for sender_h in range(n_heads):
        # Part 2: patch sender, freeze others, cache final resid
        for sender_l in range(n_layers):
            # use a hook to patch in the activations from new_cache for the head from which we are patching
            new_cache_patching_hook = [
                (utils.get_act_name("z", sender_l), partial(patch_head_vector, head_index=sender_h, clean_cache=new_cache))
            ]
            # use a hook to patch in the activations from orig_cache everywhere else
            orig_cache_patching_hooks = [
                (utils.get_act_name("z", l), partial(patch_head_vector, head_index=h, clean_cache=orig_cache))
                for l in range(n_layers) for h in range(n_heads) if sender_h != h
            ]
            # use a hook to store the activations at the final residual stream
            final_resid = t.zeros_like(orig_cache["resid_post", n_layers-1], device=device)
            store_resid_hook = [
                (utils.get_act_name("resid_post", n_layers-1), partial(store_activations, store=final_resid))
            ]

            hooks = new_cache_patching_hook + orig_cache_patching_hooks + store_resid_hook

            model.run_with_hooks(
                orig_dataset.toks, 
                fwd_hooks=hooks
            )
            # Part 3: patch in final resid
            # use a hook to patch in the activations from the final residual stream
            resid_patching_fn = partial(patch_residual, patched_residual=final_resid)
            
            logits = model.run_with_hooks(
                orig_dataset.toks,
                fwd_hooks=[
                    (utils.get_act_name("resid_post", n_layers-1), resid_patching_fn)
                ]
            )

            results[sender_l][sender_h] = patching_metric(logits)

    return results

if MAIN:
    path_patch_head_to_final_resid_post = get_path_patch_head_to_final_resid_post(model, ioi_metric_2)

    imshow(
        100 * path_patch_head_to_final_resid_post,
        title="Direct effect on logit difference",
        labels={"x":"Head", "y":"Layer", "color": "Logit diff. variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        width=600,
    )
# %%
def store_head_activations(
    head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint, 
    head_index: int, 
    store: Float[Tensor, "batch pos d_head"]
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches the output of a given head (before it's added to the residual stream) at
    every sequence position, using the value from the clean cache.
    '''
    store[:,:,:] = head_vector[:,:,head_index,:]
    return head_vector

def patch_head_with_store(
    head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint, 
    head_index: int,
    patched_head_acts: Float[Tensor, "batch pos d_head"]
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches a given sequence position in the residual stream, using the value
    from the clean cache.
    '''
    head_vector[:,:,head_index] = patched_head_acts
    return head_vector


def get_path_patch_head_to_heads(
    receiver_heads: List[Tuple[int, int]],
    receiver_input: str,
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: IOIDataset = abc_dataset,
    orig_dataset: IOIDataset = ioi_dataset,
    new_cache: Optional[ActivationCache] = None,
    orig_cache: Optional[ActivationCache] = None,
) -> Float[Tensor, "layer head"]:
    '''
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = input to a later head (or set of heads)

    The receiver node is specified by receiver_heads and receiver_input.
    Example (for S-inhibition path patching the queries):
        receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)],
        receiver_input = "v"

    Returns:
        tensor of metric values for every possible sender head
    '''
    # Part 1: cache heads
    # only if we don't already have new_cache and orig_cache
    if new_cache is None:
        _, new_cache = model.run_with_cache(new_dataset.toks)
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(orig_dataset.toks)
    
    relevant_layers = max([l for l,_ in receiver_heads])
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = t.zeros((relevant_layers, n_heads), device=device)

    for sender_h in range(n_heads):
        for sender_l in range(relevant_layers):
            # Part 2: patch sender, freeze others, cache receiver activations
            # use a hook to patch in the activations from new_cache for the head from which we are patching
            new_cache_patching_hook = [
                (utils.get_act_name("z", sender_l), partial(patch_head_vector, head_index=sender_h, clean_cache=new_cache))
            ]
            # use a hook to patch in the activations from orig_cache everywhere else
            orig_cache_patching_hooks = [
                (utils.get_act_name("z", l), partial(patch_head_vector, head_index=h, clean_cache=orig_cache))
                for l in range(n_layers) for h in range(n_heads) if sender_h != h
            ]
            # use a hooks to store the activations at the receiver heads
            receiver_stores = [
                t.zeros_like(orig_cache[receiver_input, receiver_l][:,:,receiver_h], device=device)
                for receiver_l, receiver_h in receiver_heads
            ]
            store_receiver_activations = [
                (utils.get_act_name(receiver_input, receiver_l), partial(store_head_activations, head_index=receiver_h, store=receiver_store))
                for receiver_store, (receiver_l, receiver_h) in zip(receiver_stores, receiver_heads)
            ]

            hooks = new_cache_patching_hook + orig_cache_patching_hooks + store_receiver_activations

            model.run_with_hooks(
                orig_dataset.toks, 
                fwd_hooks=hooks
            )
            # Part 3: new forward pass with patched in activations from receiver heads
            # use a hook to patch in the activations from the receiver heads
            receiver_patching_hooks = [
                (utils.get_act_name(receiver_input, receiver_l), 
                 partial(patch_head_with_store, head_index=receiver_h, patched_head_acts=receiver_store))
                 for receiver_store, (receiver_l, receiver_h) in zip(receiver_stores, receiver_heads)
            ]
            
            logits = model.run_with_hooks(
                orig_dataset.toks,
                fwd_hooks=receiver_patching_hooks
            )

            results[sender_l][sender_h] = patching_metric(logits)

    return results

if MAIN:
    model.reset_hooks()

    s_inhibition_value_path_patching_results = get_path_patch_head_to_heads(
        receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)],
        receiver_input = "v",
        model = model,
        patching_metric = ioi_metric_2
    )

    imshow(
        100 * s_inhibition_value_path_patching_results,
        title="Direct effect on S-Inhibition Heads' values", 
        labels={"x": "Head", "y": "Layer", "color": "Logit diff.<br>variation"},
        width=600,
        coloraxis=dict(colorbar_ticksuffix = "%"),
    )


#### PART 5: Paper Replication

# %%
def scatter_embedding_vs_attn(
    attn_from_end_to_io: Float[Tensor, "batch"],
    attn_from_end_to_s: Float[Tensor, "batch"],
    projection_in_io_dir: Float[Tensor, "batch"],
    projection_in_s_dir: Float[Tensor, "batch"],
    layer: int,
    head: int
):
    scatter(
        x=t.concat([attn_from_end_to_io, attn_from_end_to_s], dim=0),
        y=t.concat([projection_in_io_dir, projection_in_s_dir], dim=0),
        color=["IO"] * N + ["S"] * N,
        title=f"Projection of the output of {layer}.{head} along the name<br>embedding vs attention probability on name",
        title_x=0.5,
        labels={"x": "Attn prob on name", "y": "Dot w Name Embed", "color": "Name type"},
        color_discrete_sequence=["#72FF64", "#C9A5F7"],
        width=650
    )
# %%
def calculate_and_show_scatter_embedding_vs_attn(
    layer: int,
    head: int,
    cache: ActivationCache = ioi_cache,
    dataset: IOIDataset = ioi_dataset,
) -> None:
    '''
    Creates and plots a figure equivalent to 3(c) in the paper.

    This should involve computing the four 1D tensors:
        attn_from_end_to_io
        attn_from_end_to_s
        projection_in_io_dir
        projection_in_s_dir
    and then calling the scatter_embedding_vs_attn function.
    '''
    wo = model.W_O[layer, head]

    # Get head's output to residual stream
    z = cache["z", layer][:, :, head]
    output = einops.einsum(z, wo, 'batch pos d_head, d_head d_model -> batch pos d_model')
    output_on_end_token = output[t.arange(z.size(0)),dataset.word_idx["end"]]

    # Get IO and S directions
    io_dir = model.W_U.T[dataset.io_tokenIDs]
    s_dir = model.W_U.T[dataset.s_tokenIDs]

    # Get projection of output on directions
    projection_in_io_dir = (io_dir * output_on_end_token).sum(-1)
    projection_in_s_dir = (s_dir * output_on_end_token).sum(-1)

    
    # Get attention probs
    attn_from_end_to_io = cache["pattern", layer][t.arange(z.size(0)),head,dataset.word_idx["end"],dataset.word_idx["IO"]]
    attn_from_end_to_s = cache["pattern", layer][t.arange(z.size(0)),head,dataset.word_idx["end"],dataset.word_idx["S1"]]
    
    scatter_embedding_vs_attn(
        attn_from_end_to_io,
        attn_from_end_to_s,
        projection_in_io_dir,
        projection_in_s_dir,
        layer,
        head
    )


nmh = (9, 9)
calculate_and_show_scatter_embedding_vs_attn(*nmh)

nnmh = (11, 10)
calculate_and_show_scatter_embedding_vs_attn(*nnmh)
# %%
print(model.blocks[2].attn.W_V.shape)
print(model.blocks[2].attn.W_O.shape)
print((model.blocks[2].attn.W_V[3] @ model.blocks[2].attn.W_O[3]).shape)
print(model.W_U.shape)
# %%
# name_tokens = model.to_tokens(NAMES, prepend_bos=False)
# print(name_tokens.shape)
# name_embeddings: Float[Tensor, "names 1 d_model"] = model.W_E[name_tokens]
# print(name_embeddings.shape)
# names_normalized = model.blocks[0].ln1(name_embeddings)
# print(model.blocks[0].attn.W_V.shape)
# print(names_normalized.shape)
# names_v = einops.einsum(model.blocks[0].attn.W_V, names_normalized, 'n_heads d_model d_head, names e d_model -> names e n_heads d_head')
# print(model.blocks[0].attn.W_O.shape)
# print(names_v.shape)
# names_results = einops.einsum(model.blocks[0].attn.W_O, names_v, 'n_head d_head d_model, names e n_heads d_head -> names e n_heads d_model')
# print(names_results.shape)
# names_attn_out = names_results.sum(-2)
# print(names_attn_out.shape)
# resid_mid = names_attn_out + name_embeddings
# resid_mid_normalized = model.blocks[0].ln2(name_embeddings)
# names_post_mlp = model.blocks[0].mlp(resid_mid_normalized)
# names_resid_post = names_post_mlp + name_embeddings
# print(names_resid_post.shape)

# names_attn = model.blocks[0].attn(names_normalized)
# print(names_attn.shape)
# %%
def get_copying_scores(
    model: HookedTransformer,
    k: int = 5,
    names: list = NAMES
) -> Float[Tensor, "2 layer head"]:
    '''
    Gets copying scores (both positive and negative) as described in page 6 of the IOI paper, for every (layer, head) pair in the model.

    Returns these in a 3D tensor (the first dimension is for positive vs negative).
    '''
    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers
    results = t.zeros((2, n_layers, n_heads), device=device)

    # get residual stream state after first block for each name
    name_tokens = model.to_tokens(names, prepend_bos=False)
    name_embeddings: Float[Tensor, "names 1 d_model"] = model.W_E[name_tokens]

    mlp = model.blocks[0].mlp
    ln2 = model.blocks[0].ln2
    names_resid_post = name_embeddings + mlp(ln2(name_embeddings))
    names_resid_post = names_resid_post.squeeze(1)

    # get copying score for each layer, head pair
    for l in range(n_layers):
        for h in range(n_heads):
            OV = model.blocks[l].attn.W_V[h] @ model.blocks[l].attn.W_O[h]
            for idx, prefix in enumerate([1, -1]):
                names_after_ov = einops.einsum(names_resid_post, prefix * OV, 'names d_v, d_v d_o -> names d_o')
                names_after_ov = model.ln_final(names_after_ov)
                logits = einops.einsum(names_after_ov, model.W_U, 'names d_model, d_model n_tokens -> names n_tokens')
                top_logit_indices = t.topk(logits, k)[1]
                matches = name_tokens == top_logit_indices
                copy_score = sum(matches.any(1)).item() / len(names)
                results[idx,l,h] = copy_score

    return results

copying_results = get_copying_scores(model)

imshow(
    copying_results, 
    facet_col=0, 
    facet_labels=["Positive copying scores", "Negative copying scores"],
    title="Copying scores of attention heads' OV circuits",
    width=800
)


heads = {"name mover": [(9, 9), (10, 0), (9, 6)], "negative name mover": [(10, 7), (11, 10)]}

for i, name in enumerate(["name mover", "negative name mover"]):
    make_table(
        title=f"Copying Scores ({name} heads)",
        colnames=["Head", "Score"],
        cols=[
            list(map(str, heads[name])) + ["[dark_orange bold]Average"],
            [f"{copying_results[i, layer, head]:.2%}" for (layer, head) in heads[name]] + [f"[dark_orange bold]{copying_results[i].mean():.2%}"]
        ]
    )
# %%
def get_attn_scores(
    model: HookedTransformer, 
    seq_len: int, 
    batch: int, 
    head_type: Literal["duplicate", "prev", "induction"]
):
    '''
    Returns attention scores for sequence of duplicated tokens, for every head.
    '''
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = t.zeros((n_layers, n_heads), device=device)

    tokens = t.randint(0, model.cfg.d_vocab, size=(batch,seq_len), device=device)
    repeated_tokens = t.cat((tokens, tokens), dim=-1)

    _, cache = model.run_with_cache(repeated_tokens)

    for l in range(n_layers):
        for h in range(n_heads):
            attn_pattern = cache["pattern", l][:,h]
            if head_type == "duplicate":
                offset_diag = t.diagonal(attn_pattern, offset=-seq_len, dim1=-2, dim2=-1)
            elif head_type == "prev":
                offset_diag = t.diagonal(attn_pattern, offset=-1, dim1=-2, dim2=-1)
            elif head_type == "induction":
                offset_diag = t.diagonal(attn_pattern, offset=-seq_len+1, dim1=-2, dim2=-1)
            score = offset_diag.mean()
            scores[l][h] = score

    return scores


def plot_early_head_validation_results(seq_len: int = 50, batch: int = 50):
    '''
    Produces a plot that looks like Figure 18 in the paper.
    '''
    head_types = ["duplicate", "prev", "induction"]

    results = t.stack([
        get_attn_scores(model, seq_len, batch, head_type=head_type)
        for head_type in head_types
    ])

    imshow(
        results,
        facet_col=0,
        facet_labels=[
            f"{head_type.capitalize()} token attention prob.<br>on sequences of random tokens"
            for head_type in head_types
        ],
        labels={"x": "Head", "y": "Layer"},
        width=1300,
    )



model.reset_hooks()
plot_early_head_validation_results()
# %%

### Constructing the Minimal Circuit

CIRCUIT = {
    "name mover": [(9, 9), (10, 0), (9, 6)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative name mover": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}

SEQ_POS_TO_KEEP = {
    "name mover": "end",
    "backup name mover": "end",
    "negative name mover": "end",
    "s2 inhibition": "end",
    "induction": "S2",
    "duplicate token": "S2",
    "previous token": "S1+1",
}
# %%
ioi_dataset.groups
# %%
ioi_dataset.templates
# %%
ioi_dataset.sentences
# %%
ioi_dataset.toks.shape
# %%
_, c = model.run_with_cache(
    ioi_dataset.toks[ioi_dataset.groups[0]],
    return_type=None,
    names_filter=lambda name: name.endswith("hook_z")
)
# %%
c["z", 1].shape
# %%
utils.get_act_name("z", 0)
# %%
tuple(ioi_dataset.toks.shape) + (model.cfg.n_heads, model.cfg.d_head)

# %%
ioi_dataset.word_idx
# %%
def ablate_heads_with_average_template_activation(
    head_activations: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint, 
    circuit_indices: List[Tuple[int,int]],
    head_indices: List[int], 
    mean_activations: Float[Tensor, "batch pos head_index d_head"]
):
    pos_indices, head_indices = zip(*circuit_indices)
    indices_to_patch = t.ones(head_activations.shape, device=device)
    indices_to_patch[:,pos_indices,head_indices,:] = 0
    head_activations = t.where(
        indices_to_patch,
        mean_activations,
        head_activations
    )
    return head_activations

# $$
def add_mean_ablation_hook(
    model: HookedTransformer, 
    means_dataset: IOIDataset, 
    circuit: Dict[str, List[Tuple[int, int]]] = CIRCUIT,
    seq_pos_to_keep: Dict[str, str] = SEQ_POS_TO_KEEP,
    is_permanent: bool = True,
) -> HookedTransformer:
    '''
    Adds a permanent hook to the model, which ablates according to the circuit and 
    seq_pos_to_keep dictionaries.

    In other words, when the model is run on ioi_dataset, every head's output will 
    be replaced with the mean over means_dataset for sequences with the same template,
    except for a subset of heads and sequence positions as specified by the circuit
    and seq_pos_to_keep dicts.
    '''

    n_layers = model.cfg.n_layers

    ### 1. calculate means over means_dataset for sequence with the same template
    
    # create a (n_layers, N, seq_len, n_heads, d_head) tensor t where
    # t[l, n] is the mean activation at layer l for sentences with the template of sentence n
    mean_activations: Float[Tensor, "n_layers N seq_len n_heads d_head"] =  t.zeros(
        (model.cfg.n_layers, ) + tuple(means_dataset.toks.shape) + (model.cfg.n_heads, model.cfg.d_head),
        device=device
    )
    
    for group in means_dataset.groups:
        # get cache for running tokens from group 
        _, cache = model.run_with_cache(
            means_dataset.toks[group],
            return_type=None,
            names_filter=lambda name: name.endswith("hook_z")
        )

        template_means = [
            einops.reduce(cache["z", layer], "batch ... -> ...", "mean")
            for layer in range(model.cfg.n_layers)
        ]

        for layer in range(model.cfg.n_layers):
            mean_activations[layer][group] = template_means[layer]

    # create tuples of indices (pos, head) for each sentence that show which position should 
    # be ablation for which head for that sentence

    circuit_indices = []

    for component, (_, head) in CIRCUIT.items():
        layer_circuit_indices = []
        pos_to_keep_str = SEQ_POS_TO_KEEP[component]
        for n in range(means_dataset.N):
            pos = means_dataset.word_idx[pos_to_keep_str][n].item()
            layer_circuit_indices.append((pos, head))
        circuit_indices.append(layer_circuit_indices)

    for layer in range(n_layers):
        ablation_hook = partial(
            ablate_heads_with_average_template_activation,
            circuit_indices=circuit_indices[layer],
            mean_activations=mean_activations[layer]
        )
        model.add_hook(utils.get_act_name("z", layer), ablation_hook, is_permanent=True)

# %%    
import part3_indirect_object_identification.ioi_circuit_extraction as ioi_circuit_extraction


model = ioi_circuit_extraction.add_mean_ablation_hook(model, means_dataset=abc_dataset, circuit=CIRCUIT, seq_pos_to_keep=SEQ_POS_TO_KEEP)

ioi_logits_minimal = model(ioi_dataset.toks)

print(f"Average logit difference (IOI dataset, using entire model): {logits_to_ave_logit_diff_2(ioi_logits_original):.4f}")
print(f"Average logit difference (IOI dataset, only using circuit): {logits_to_ave_logit_diff_2(ioi_logits_minimal):.4f}")

# %%

model = add_mean_ablation_hook(model, means_dataset=abc_dataset, circuit=CIRCUIT, seq_pos_to_keep=SEQ_POS_TO_KEEP)

ioi_logits_minimal = model(ioi_dataset.toks)

print(f"Average logit difference (IOI dataset, using entire model): {logits_to_ave_logit_diff_2(ioi_logits_original):.4f}")
print(f"Average logit difference (IOI dataset, only using circuit): {logits_to_ave_logit_diff_2(ioi_logits_minimal):.4f}")
# %%
