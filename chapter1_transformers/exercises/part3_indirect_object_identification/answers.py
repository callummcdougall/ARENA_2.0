#%%
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

#%%
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

example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
example_answer = " Mary"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)
# %%

prompt_format = [
    "When John and Mary went to the shops,{} gave the bag to",
    "When Tom and James went to the park,{} gave the ball to",
    "When Dan and Sid went to the shops,{} gave an apple to",
    "After Martin and Amy went to the park,{} gave a drink to",
]
name_pairs = [
    (" John", " Mary"),
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

table = Table("Prompt", "Correct", "Incorrect", title="Prompts & Answers:")

for prompt, answer in zip(prompts, answers):
    table.add_row(prompt, repr(answer[0]), repr(answer[1]))

rprint(table)
# %%
tokens = model.to_tokens(prompts, prepend_bos=True)
# Move the tokens to the GPU
tokens = tokens.to(device)
# Run the model and cache all activations
original_logits, cache = model.run_with_cache(tokens)
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
    last_logits = logits[:, -1] # only want last prediction logits, (batch d_vocab)
    
    # logit_diffs = []
    # for i in range(logits.shape[0]):
    #     answer = answer_tokens[i] # (2,)
    #     name_logits = last_logits[i][answer]
    #     diff = name_logits[0] - name_logits[1]
    #     logit_diffs.append(diff)

    # if per_prompt:
    #     return logit_diffs
    # return sum(logit_diffs)/len(logit_diffs)
    # name_logits = [last_logits[i, answer_tokens[i]] for i in range(logits.shape[0])]
    answer_logits: Float[Tensor, "batch 2"] = last_logits.gather(dim=-1, index=answer_tokens)

    logit_diffs = [logit[0] - logit[1] for logit in answer_logits]
    if per_prompt:
        return t.tensor(logit_diffs)
    return sum(logit_diffs)/len(logit_diffs)


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
answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(answer_tokens)
print("Answer residual directions shape:", answer_residual_directions.shape)

correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_directions - incorrect_residual_directions
print(f"Logit difference directions shape:", logit_diff_directions.shape)
# cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type]. 

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
    # normed = cache.model.ln_final(residual_stack)
    normed = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1) # (batch d_model)
    return einops.einsum(normed, logit_diff_directions, "... b d_m, b d_m -> ... b").mean(-1)


t.testing.assert_close(
    residual_stack_to_logit_diff(final_token_residual_stream, cache),
    original_average_logit_diff
)
# %%
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
# Now we know that layers 7-9 do interesting things

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



k = 3

for head_type in ["Positive", "Negative"]:

    # Get the heads with largest (or smallest) contribution to the logit difference
    top_heads = topk_of_Nd_tensor(per_head_logit_diffs * (1 if head_type=="Positive" else -1), k)

    # Get all their attention patterns
    attn_patterns_for_important_heads: Float[Tensor, "head q k"] = t.stack([
        # cache["pattern", layer][:, head].mean(0)
        cache["pattern", layer][:, head][0]
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
    patched_diff = logits_to_ave_logit_diff(logits, answer_tokens)
    return (patched_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


t.testing.assert_close(ioi_metric(clean_logits).item(), 1.0)
t.testing.assert_close(ioi_metric(corrupted_logits).item(), 0.0)
t.testing.assert_close(ioi_metric((clean_logits + corrupted_logits) / 2).item(), 0.5)


# %%
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
    # hook function: modify corrupted_residual_component
    clean_res_stream = clean_cache[hook.name] # access clean residual stream
    corrupted_residual_component[:, pos] = clean_res_stream[:, pos]

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
    patching_results = t.zeros(size=(model.cfg.n_layers, corrupted_tokens.shape[1])).to(device=model.cfg.device) # (n_layers, pos)

    for layer_index in range(model.cfg.n_layers):

        # get hook name at layer, run model with hooks
        hook_name = utils.get_act_name("resid_pre", layer=layer_index)
        for seq_pos in range(corrupted_tokens.shape[1]):
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[
                    (hook_name, partial(patch_residual_component, pos=seq_pos, clean_cache=clean_cache))
                ],
                reset_hooks_end=True
            )
            metric_result = ioi_metric(patched_logits)
            patching_results[layer_index, seq_pos] = metric_result
    
    return patching_results


act_patch_resid_pre_own = get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, ioi_metric)

t.testing.assert_close(act_patch_resid_pre, act_patch_resid_pre_own)


# %%
imshow(
    act_patch_resid_pre_own, 
    x=labels, 
    title="Logit Difference From Patched Residual Stream", 
    labels={"x":"Sequence Position", "y":"Layer"},
    width=600 # If you remove this argument, the plot will usually fill the available space
)

# %%
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
    patching_results = t.zeros(size=(3, model.cfg.n_layers, corrupted_tokens.shape[1])).to(device=model.cfg.device) # (3, n_layers, pos)

    for name_index, name in enumerate(["resid_pre", "attn_out", "mlp_out"]):
        for layer_index in range(model.cfg.n_layers):
            # get hook name at layer, run model with hooks
            hook_name = utils.get_act_name(name, layer=layer_index)
            for seq_pos in range(corrupted_tokens.shape[1]):
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[
                        (hook_name, partial(patch_residual_component, pos=seq_pos, clean_cache=clean_cache))
                    ],
                    reset_hooks_end=True
                )
                metric_result = ioi_metric(patched_logits)
                patching_results[name_index, layer_index, seq_pos] = metric_result
    
    return patching_results

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
    clean_res_stream = clean_cache[hook.name] # (batch, seq, n_heads, d_head)
    corrupted_head_vector[:, :, head_index] = clean_res_stream[:, :, head_index]

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
    patching_results = t.zeros(size=(model.cfg.n_layers, model.cfg.n_heads)).to(device=model.cfg.device)

    for layer_index in range(model.cfg.n_layers):

        # get z values across layer
        hook_name = utils.get_act_name("z", layer=layer_index)
        for head_index in range(model.cfg.n_heads):
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[
                    (hook_name, partial(patch_head_vector, head_index=head_index, clean_cache=clean_cache))
                ],
                reset_hooks_end=True
            )
            metric_result = ioi_metric(patched_logits)
            patching_results[layer_index, head_index] = metric_result
    return patching_results

act_patch_attn_head_out_all_pos_own = get_act_patch_attn_head_out_all_pos(model, corrupted_tokens, clean_cache, ioi_metric)

t.testing.assert_close(act_patch_attn_head_out_all_pos, act_patch_attn_head_out_all_pos_own)

imshow(
    act_patch_attn_head_out_all_pos_own,
    title="Logit Difference From Patched Attn Head Output", 
    labels={"x":"Head", "y":"Layer"},
    width=600
)
# %%
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
def patch_attn_patterns(
    corrupted_head_vector: Float[Tensor, "batch head_index pos_q pos_k"],
    hook: HookPoint, 
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches the attn patterns of a given head at every sequence position, using 
    the value from the clean cache.
    '''
    corrupted_head_vector[:, head_index] = clean_cache[hook.name][:, head_index]

def get_act_patch_attn_head_all_pos_every(
    model: HookedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable
) -> Float[Tensor, "layer head"]: #return type missing 5
    '''
    Returns an array of results of patching at all positions for each head in each
    layer (using the value from the clean cache) for output, queries, keys, values
    and attn pattern in turn.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    model.reset_hooks()
    patching_results = t.zeros(size=(5, model.cfg.n_layers, model.cfg.n_heads)).to(device=model.cfg.device)

    for comp_idx, attn_comp in enumerate(["z", "q", "k", "v", "pattern"]):
        for layer_index in range(model.cfg.n_layers):
            hook_name = utils.get_act_name(attn_comp, layer=layer_index)
            for head_index in range(model.cfg.n_heads):
                if comp_idx == 4:
                    patched_logits = model.run_with_hooks(
                        corrupted_tokens,
                        fwd_hooks=[
                            (hook_name, partial(patch_attn_patterns, head_index=head_index, clean_cache=clean_cache))
                        ],
                        reset_hooks_end=True
                    )
                else:
                    patched_logits = model.run_with_hooks(
                        corrupted_tokens,
                        fwd_hooks=[
                            (hook_name, partial(patch_head_vector, head_index=head_index, clean_cache=clean_cache))
                        ],
                        reset_hooks_end=True
                    )
                metric_result = patching_metric(patched_logits)
                patching_results[comp_idx, layer_index, head_index] = metric_result
    return patching_results


act_patch_attn_head_all_pos_every_own = get_act_patch_attn_head_all_pos_every(
    model,
    corrupted_tokens,
    clean_cache,
    ioi_metric
)

t.testing.assert_close(act_patch_attn_head_all_pos_every, act_patch_attn_head_all_pos_every_own)

imshow(
    act_patch_attn_head_all_pos_every_own,
    facet_col=0,
    facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
    title="Activation Patching Per Head (All Pos)",
    labels={"x": "Head", "y": "Layer"},
    width=1200
)
# %%
# Get the heads with largest value patching
# (we know from plot above that these are the 4 heads in layers 7 & 8)
# k = 4
# top_heads = topk_of_Nd_tensor(act_patch_attn_head_all_pos_every[3], k=k)

# top_heads = [[0,1],[0,10],[3,0]]
top_heads = [[5,5],[6,9],[5,8],[5,9]]
k=len(top_heads)
batch_idx = 0

# Get all their attention patterns
attn_patterns_for_important_heads: Float[Tensor, "head q k"] = t.stack([
    # cache["pattern", layer][:, head].mean(0)
    cache["pattern", layer][:, head][batch_idx]
        for layer, head in top_heads
])

# Display results
display(HTML(f"<h2>Top {k} Logit Attribution Heads (from query-patching)</h2>"))
display(cv.attention.attention_patterns(
    attention = attn_patterns_for_important_heads,
    tokens = model.to_str_tokens(tokens[batch_idx]),
    attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
))
# %%
# Path patching
from part3_indirect_object_identification.ioi_dataset import NAMES, IOIDataset

N = 25
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=str(device)
)
abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

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



model.reset_hooks(including_permanent=True)

ioi_logits_original, ioi_cache = model.run_with_cache(ioi_dataset.toks)
abc_logits_original, abc_cache = model.run_with_cache(abc_dataset.toks)

ioi_per_prompt_diff = logits_to_ave_logit_diff_2(ioi_logits_original, per_prompt=True)
abc_per_prompt_diff = logits_to_ave_logit_diff_2(abc_logits_original, per_prompt=True)

ioi_average_logit_diff = logits_to_ave_logit_diff_2(ioi_logits_original).item()
abc_average_logit_diff = logits_to_ave_logit_diff_2(abc_logits_original).item()

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


print(f"IOI metric (IOI dataset): {ioi_metric_2(ioi_logits_original):.4f}")
print(f"IOI metric (ABC dataset): {ioi_metric_2(abc_logits_original):.4f}")

# %%


# function to freeze other attention heads
def freeze_other_component(
    corrupted_attn_out: Float[Tensor, "batch pos d_model"],
    hook: HookPoint, 
    head_index: int,
    corrupted_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    # want to freeze every other attention head
    orig_corrupted_z = corrupted_cache[hook.name]
    corrupted_attn_out[:, :, head_index] = orig_corrupted_z[:,:,head_index]

# function to patch in new attention head
def patch_sender_head(
    corrupted_attn_out: Float[Tensor, "batch pos d_model"],
    hook: HookPoint, 
    head_index: int,
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    # patch senders
    clean_z = clean_cache[hook.name]
    corrupted_attn_out[:, :, head_index] = clean_z[:,:,head_index]

# get set of hooks that freeze all other attentions (except for sender attn and receiver attn)
def freeze_hooks(
    model: HookedTransformer,
    sender_head_layer: int,
    sender_head_index: int,
    corrupted_cache: ActivationCache,
    receiver_hook_layer: int = None,
    receiver_hook_index: int = None
):
    hooks = []
    for layer_index in range(model.cfg.n_layers):
        hook_name = utils.get_act_name("z", layer=layer_index)
        for head_index in range(model.cfg.n_heads):

            # skip sender and receiver attns
            if head_index == sender_head_index and layer_index == sender_head_layer:
                continue
            if receiver_hook_layer is not None and receiver_hook_index is not None and \
                head_index == receiver_hook_index and layer_index == receiver_hook_layer:
                continue

            hooks.append((hook_name, partial(freeze_other_component, head_index=head_index, corrupted_cache=corrupted_cache)))
    return hooks
    

def get_path_patch_head_to_final_resid_post(
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: IOIDataset = abc_dataset,
    orig_dataset: IOIDataset = ioi_dataset,
    new_cache: Optional[ActivationCache] = abc_cache,
    orig_cache: Optional[ActivationCache] = ioi_cache,
) -> Float[Tensor, "layer head"]:
    # Patch path from each head (in each layer) to final residual stream
    
    # Step 1 already done, all values already cached

    # Step 2: patch in sender, freeze all other values other than the receiver (all other head z values)
    model.reset_hooks()
    patching_results = t.zeros(size=(model.cfg.n_layers, model.cfg.n_heads)).to(device=model.cfg.device)

    z_name_filter = lambda name: name.endswith("z")
    if new_cache is None:
        _, new_cache = model.run_with_cache(
            new_dataset.toks, 
            names_filter=z_name_filter, 
            return_type=None
        )
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(
            orig_dataset.toks, 
            names_filter=z_name_filter, 
            return_type=None
        )

    for layer_index in range(model.cfg.n_layers):

        # get z values across layer
        patch_hook_name = utils.get_act_name("z", layer=layer_index)
        for head_index in range(model.cfg.n_heads):
            fwd_freeze_hooks = freeze_hooks(model, sender_head_layer=layer_index, sender_head_index=head_index, corrupted_cache=orig_cache)

            all_fwd_hooks = fwd_freeze_hooks + [(patch_hook_name, partial(patch_sender_head, head_index=head_index, clean_cache=new_cache))]

            corrupted_tokens = orig_dataset.toks
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=all_fwd_hooks,
                reset_hooks_end=True
            )
            
            patching_results[layer_index, head_index] = patching_metric(patched_logits)
            # for (hook_name, fwd_hook) in all_fwd_hooks:
            #     model.add_hook(hook_name, fwd_hook)
            
            # _, patched_cache = model.run_with_cache(corrupted_tokens)

            # )

    return patching_results


path_patch_head_to_final_resid_post = get_path_patch_head_to_final_resid_post(model, ioi_metric_2)

imshow(
    100 * path_patch_head_to_final_resid_post,
    title="Direct effect on logit difference",
    labels={"x":"Head", "y":"Layer", "color": "Logit diff. variation"},
    coloraxis=dict(colorbar_ticksuffix = "%"),
    width=600,
)


# %%

def patch_head_input(
    orig_activation: Float[Tensor, "batch pos head_idx d_head"],
    hook: HookPoint,
    patched_cache: ActivationCache,
    head_list: List[Tuple[int, int]],
) -> Float[Tensor, "batch pos head_idx d_head"]:
    '''
    Function which can patch any combination of heads in layers,
    according to the heads in head_list.
    '''
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
    return orig_activation

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
    model.reset_hooks()
    patching_results = t.zeros(size=(model.cfg.n_layers, model.cfg.n_heads)).to(device=model.cfg.device)

    z_name_filter = lambda name: name.endswith("z")
    if new_cache is None:
        _, new_cache = model.run_with_cache(
            new_dataset.toks, 
            names_filter=z_name_filter, 
            return_type=None
        )
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(
            orig_dataset.toks, 
            names_filter=z_name_filter, 
            return_type=None
        )


    receiver_name_filter = lambda name: name.endswith(receiver_input)
    for layer_index in range(model.cfg.n_layers):

        # get z values across layer
        patch_hook_name = utils.get_act_name("z", layer=layer_index)
        for head_index in range(model.cfg.n_heads):
            fwd_freeze_hooks = freeze_hooks(model, sender_head_layer=layer_index, sender_head_index=head_index, corrupted_cache=orig_cache)

            all_fwd_hooks = fwd_freeze_hooks + [(patch_hook_name, partial(patch_sender_head, head_index=head_index, clean_cache=new_cache))]

            corrupted_tokens = orig_dataset.toks
            for (hook_name, fwd_hook) in all_fwd_hooks:
                model.add_hook(hook_name, fwd_hook)
            
            # Step 2
            _, patched_cache = model.run_with_cache(corrupted_tokens,
                                                    names_filter=receiver_name_filter,
                                                    return_type=None)
            model.reset_hooks()

            # get receiver head new values
            # receiver_hooks = []
            # for receiver_head in receiver_heads:
            #     receiver_hooks.append()
            
            # Step 3
            # model.add_hook()
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks = [(receiver_name_filter, partial(patch_head_input, patched_cache=patched_cache,
                                                           head_list=receiver_heads))]
            )

            patching_results[layer_index, head_index] = patching_metric(patched_logits)

    return patching_results

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
    
    attns = cache[utils.get_act_name("pattern", layer=layer)][:, head] # (batch, pos_q, pos_k)

    N = attns.shape[0]
    attn_from_end_to_io = attns[t.arange(N), dataset.word_idx["end"], dataset.word_idx["IO"]] # (batch,)
    attn_from_end_to_s = attns[t.arange(N), dataset.word_idx["end"], dataset.word_idx["S1"]] #(batch,)

    zs = cache[utils.get_act_name("z", layer=layer)][:, :, head] # (batch, pos, d_head)
    head_output = einops.einsum(zs, cache.model.W_O[layer, head], "b pos d_h, d_h d_m -> b pos d_m")
    end_output = head_output[t.arange(N), dataset.word_idx["end"]] # output to end token, (b, d_m)

    io_unembed: Float[Tensor, "batch d_model"] = model.W_U.T[dataset.io_tokenIDs] 
    s_unembed: Float[Tensor, "batch d_model"] = model.W_U.T[dataset.s_tokenIDs]
    # io_unembed: Float[Tensor, "batch d_model"] = model.W_E[dataset.io_tokenIDs]
    # s_unembed: Float[Tensor, "batch d_model"] = model.W_E[dataset.s_tokenIDs]
    projection_in_io_dir = einops.einsum(end_output, io_unembed, "b d_m, b d_m -> b")
    projection_in_s_dir = einops.einsum(end_output, s_unembed, "b d_m, b d_m -> b")

    print(attn_from_end_to_io.shape)
    print(attn_from_end_to_s.shape)
    print(projection_in_io_dir.shape)
    print(projection_in_s_dir.shape)

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

def get_copying_scores(
    model: HookedTransformer,
    k: int = 5,
    names: list = NAMES
) -> Float[Tensor, "2 layer-1 head"]:
    '''
    Gets copying scores (both positive and negative) as described in page 6 of the IOI paper, for every (layer, head) pair in the model.

    Returns these in a 3D tensor (the first dimension is for positive vs negative).

    Omits the 0th layer, because this is before MLP0 (which we're claiming acts as an extended embedding).
    '''

    mlp_out_name = utils.get_act_name("mlp_out", layer=0)
    name_tokens = model.to_tokens(names, prepend_bos=False)

    _, name_prompt_cache = model.run_with_cache(name_tokens,
                                                # names_filter=mlp_out_name,
                                                return_type=None)
    # resid stream at name token after first mlp layer
    name_stream = name_prompt_cache[mlp_out_name] # (b, d_m)

    result = t.zeros(size=(2, model.cfg.n_layers, model.cfg.n_heads))
    for layer in range(1,model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            W_ov = model.W_V[layer, head] @ model.W_O[layer, head] # (d_m, d_m)

            # pos_head_effect = einops.rearrange(name_stream @ W_ov, "b d_m -> b 1 d_m") #(b, d_m)
            pos_head_effect = name_stream @ W_ov #(b, d_m)
            neg_head_effect = -pos_head_effect
            
            pos_logits = model.unembed(model.ln_final(pos_head_effect)).squeeze() #(b, d_m) @ (d_m, vocab_size) 
            neg_logits = model.unembed(model.ln_final(neg_head_effect)).squeeze()

            # Check how many are in top k
            topk_logits: Int[Tensor, "batch k"] = t.topk(pos_logits, dim=-1, k=k).indices
            in_topk = (topk_logits == name_tokens).any(-1)
            # Check how many are in bottom k
            bottomk_logits: Int[Tensor, "batch k"] = t.topk(neg_logits, dim=-1, k=k).indices
            in_bottomk = (bottomk_logits == name_tokens).any(-1)

            # Fill in results
            result[:, layer, head] = t.tensor([in_topk.float().mean(), in_bottomk.float().mean()])

    return result

# def get_copying_scores(
#     model: HookedTransformer,
#     k: int = 5,
#     names: list = NAMES
# ) -> Float[Tensor, "2 layer-1 head"]:
#     '''
#     Gets copying scores (both positive and negative) as described in page 6 of the IOI paper, for every (layer, head) pair in the model.

#     Returns these in a 3D tensor (the first dimension is for positive vs negative).

#     Omits the 0th layer, because this is before MLP0 (which we're claiming acts as an extended embedding).
#     '''
#     # SOLUTION
#     results = t.zeros((2, model.cfg.n_layers, model.cfg.n_heads), device=device)

#     # Define components from our model (for typechecking, and cleaner code)
#     embed: Embed = model.embed
#     mlp0: MLP = model.blocks[0].mlp
#     ln0: LayerNorm = model.blocks[0].ln2
#     unembed: Unembed = model.unembed
#     ln_final: LayerNorm = model.ln_final

#     # Get embeddings for the names in our list
#     name_tokens: Int[Tensor, "batch 1"] = model.to_tokens(names, prepend_bos=False)
#     name_embeddings: Int[Tensor, "batch 1 d_model"] = embed(name_tokens)

#     # Get residual stream after applying MLP
#     resid_after_mlp1 = name_embeddings + mlp0(ln0(name_embeddings))

#     # Loop over all (layer, head) pairs
#     for layer in range(1, model.cfg.n_layers):
#         for head in range(model.cfg.n_heads):

#             # Get W_OV matrix
#             W_OV = model.W_V[layer, head] @ model.W_O[layer, head]

#             # Get residual stream after applying W_OV or -W_OV respectively
#             # (note, because of bias b_U, it matters that we do sign flip here, not later)
#             resid_after_OV_pos = resid_after_mlp1 @ W_OV
#             resid_after_OV_neg = resid_after_mlp1 @ -W_OV

#             # Get logits from value of residual stream
#             logits_pos: Float[Tensor, "batch d_vocab"] = unembed(ln_final(resid_after_OV_pos)).squeeze()
#             logits_neg: Float[Tensor, "batch d_vocab"] = unembed(ln_final(resid_after_OV_neg)).squeeze()

#             # Check how many are in top k
#             topk_logits: Int[Tensor, "batch k"] = t.topk(logits_pos, dim=-1, k=k).indices
#             in_topk = (topk_logits == name_tokens).any(-1)
#             # Check how many are in bottom k
#             bottomk_logits: Int[Tensor, "batch k"] = t.topk(logits_neg, dim=-1, k=k).indices
#             in_bottomk = (bottomk_logits == name_tokens).any(-1)

#             # Fill in results
#             results[:, layer-1, head] = t.tensor([in_topk.float().mean(), in_bottomk.float().mean()])

#     return results

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


#%%
# %%

# FLAT SOLUTION NOINDENT NOCOMMENT
def generate_repeated_tokens(
	model: HookedTransformer, 
	seq_len: int, 
	batch: int = 1
) -> Float[Tensor, "batch 2*seq_len"]:
	'''
	Generates a sequence of repeated random tokens (no start token).
	'''
	rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
	rep_tokens = t.cat([rep_tokens_half, rep_tokens_half], dim=-1).to(device)
	return rep_tokens


def get_attn_scores(
	model: HookedTransformer, 
	seq_len: int, 
	batch: int, 
	head_type: Literal["duplicate", "prev", "induction"]
):
	'''
	Returns attention scores for sequence of duplicated tokens, for every head.
	'''
	rep_tokens = generate_repeated_tokens(model, seq_len, batch)

	_, cache = model.run_with_cache(
		rep_tokens,
		return_type=None,
		names_filter=lambda name: name.endswith("pattern")
	)

	# Get the right indices for the attention scores
	
	if head_type == "duplicate":
		src_indices = range(seq_len)
		dest_indices = range(seq_len, 2 * seq_len)
	elif head_type == "prev":
		src_indices = range(seq_len)
		dest_indices = range(1, seq_len + 1)
	elif head_type == "induction": 
		dest_indices = range(seq_len, 2 * seq_len)
		src_indices = range(1, seq_len + 1)
	else:
		raise ValueError(f"Unknown head type {head_type}")

	results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device="cuda", dtype=t.float32)
	for layer in range(model.cfg.n_layers):
		for head in range(model.cfg.n_heads):
			attn_scores: Float[Tensor, "batch head dest src"] = cache["pattern", layer]
			avg_attn_on_duplicates = attn_scores[:, head, dest_indices, src_indices].mean().item()
			results[layer, head] = avg_attn_on_duplicates

	return results


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



if MAIN:
	model.reset_hooks()
	plot_early_head_validation_results()

# %%


if MAIN:
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

# FLAT SOLUTION NOINDENT NOCOMMENT
def get_heads_and_posns_to_keep(
	means_dataset: IOIDataset,
	model: HookedTransformer,
	circuit: Dict[str, List[Tuple[int, int]]],
	seq_pos_to_keep: Dict[str, str],
) -> Dict[int, Bool[Tensor, "batch seq head"]]:
	'''
	Returns a dictionary mapping layers to a boolean mask giving the indices of the 
	z output which *shouldn't* be mean-ablated.

	The output of this function will be used for the hook function that does ablation.
	'''
	heads_and_posns_to_keep = {}
	batch, seq, n_heads = len(means_dataset), means_dataset.max_len, model.cfg.n_heads

	for layer in range(model.cfg.n_layers):

		mask = t.zeros(size=(batch, seq, n_heads))

		for (head_type, head_list) in circuit.items():
			seq_pos = seq_pos_to_keep[head_type]
			indices = means_dataset.word_idx[seq_pos]
			for (layer_idx, head_idx) in head_list:
				if layer_idx == layer:
					mask[:, indices, head_idx] = 1

		heads_and_posns_to_keep[layer] = mask.bool()

	return heads_and_posns_to_keep

	
def hook_fn_mask_z(
	z: Float[Tensor, "batch seq head d_head"],
	hook: HookPoint,
	heads_and_posns_to_keep: Dict[int, Bool[Tensor, "batch seq head"]],
	means: Float[Tensor, "layer batch seq head d_head"],
) -> Float[Tensor, "batch seq head d_head"]:
	'''
	Hook function which masks the z output of a transformer head.

	heads_and_posns_to_keep
		Dict created with the get_heads_and_posns_to_keep function. This tells
		us where to mask.

	means
		Tensor of mean z values of the means_dataset over each group of prompts
		with the same template. This tells us what values to mask with.
	'''
	# Get the mask for this layer, and add d_head=1 dimension so it broadcasts correctly
	mask_for_this_layer = heads_and_posns_to_keep[hook.layer()].unsqueeze(-1).to(z.device)

	# Set z values to the mean 
	z = t.where(mask_for_this_layer, z, means[hook.layer()])

	return z


def compute_means_by_template(
	means_dataset: IOIDataset, 
	model: HookedTransformer
) -> Float[Tensor, "layer batch seq head_idx d_head"]:
	'''
	Returns the mean of each head's output over the means dataset. This mean is
	computed separately for each group of prompts with the same template (these
	are given by means_dataset.groups).
	'''
	# Cache the outputs of every head
	_, means_cache = model.run_with_cache(
		means_dataset.toks.long(),
		return_type=None,
		names_filter=lambda name: name.endswith("z"),
	)
	# Create tensor to store means
	n_layers, n_heads, d_head = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
	batch, seq_len = len(means_dataset), means_dataset.max_len
	means = t.zeros(size=(n_layers, batch, seq_len, n_heads, d_head), device=model.cfg.device)

	# Get set of different templates for this data
	for layer in range(model.cfg.n_layers):
		z_for_this_layer: Float[Tensor, "batch seq head d_head"] = means_cache[utils.get_act_name("z", layer)]
		for template_group in means_dataset.groups:
			z_for_this_template = z_for_this_layer[template_group]
			z_means_for_this_template = einops.reduce(z_for_this_template, "batch seq head d_head -> seq head d_head", "mean")
			means[layer, template_group] = z_means_for_this_template

	return means


if MAIN:
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
		
		model.reset_hooks(including_permanent=True)

		# Compute the mean of each head's output on the ABC dataset, grouped by template
		means = compute_means_by_template(means_dataset, model)
		
		# Convert this into a boolean map
		heads_and_posns_to_keep = get_heads_and_posns_to_keep(means_dataset, model, circuit, seq_pos_to_keep)

		# Get a hook function which will patch in the mean z values for each head, at 
		# all positions which aren't important for the circuit
		hook_fn = partial(
			hook_fn_mask_z, 
			heads_and_posns_to_keep=heads_and_posns_to_keep, 
			means=means
		)

		# Apply hook
		model.add_hook(lambda name: name.endswith("z"), hook_fn, is_permanent=is_permanent)

		return model

# %%

import part3_indirect_object_identification.ioi_circuit_extraction as ioi_circuit_extraction


if MAIN:
	model = ioi_circuit_extraction.add_mean_ablation_hook(model, means_dataset=abc_dataset, circuit=CIRCUIT, seq_pos_to_keep=SEQ_POS_TO_KEEP)
	
	ioi_logits_minimal = model(ioi_dataset.toks)
	
	print(f"Average logit difference (IOI dataset, using entire model): {logits_to_ave_logit_diff_2(ioi_logits_original):.4f}")
	print(f"Average logit difference (IOI dataset, only using circuit): {logits_to_ave_logit_diff_2(ioi_logits_minimal):.4f}")

# %%


if MAIN:
	model = add_mean_ablation_hook(model, means_dataset=abc_dataset, circuit=CIRCUIT, seq_pos_to_keep=SEQ_POS_TO_KEEP)
	
	ioi_logits_minimal = model(ioi_dataset.toks)
	
	print(f"Average logit difference (IOI dataset, using entire model): {logits_to_ave_logit_diff_2(ioi_logits_original):.4f}")
	print(f"Average logit difference (IOI dataset, only using circuit): {logits_to_ave_logit_diff_2(ioi_logits_minimal):.4f}")

# %%


if MAIN:
	K_FOR_EACH_COMPONENT = {
		(9, 9): set(),
		(10, 0): {(9, 9)},
		(9, 6): {(9, 9), (10, 0)},
		(10, 7): {(11, 10)},
		(11, 10): {(10, 7)},
		(8, 10): {(7, 9), (8, 6), (7, 3)},
		(7, 9): {(8, 10), (8, 6), (7, 3)},
		(8, 6): {(7, 9), (8, 10), (7, 3)},
		(7, 3): {(7, 9), (8, 10), (8, 6)},
		(5, 5): {(5, 9), (6, 9), (5, 8)},
		(5, 9): {(11, 10), (10, 7)},
		(6, 9): {(5, 9), (5, 5), (5, 8)},
		(5, 8): {(11, 10), (10, 7)},
		(0, 1): {(0, 10), (3, 0)},
		(0, 10): {(0, 1), (3, 0)},
		(3, 0): {(0, 1), (0, 10)},
		(4, 11): {(2, 2)},
		(2, 2): {(4, 11)},
		(11, 2): {(9, 9), (10, 0), (9, 6)},
		(10, 6): {(9, 9), (10, 0), (9, 6), (11, 2)},
		(10, 10): {(9, 9), (10, 0), (9, 6), (11, 2), (10, 6)},
		(10, 2): {(9, 9), (10, 0), (9, 6), (11, 2), (10, 6), (10, 10)},
		(9, 7): {(9, 9), (10, 0), (9, 6), (11, 2), (10, 6), (10, 10), (10, 2)},
		(10, 1): {(9, 9), (10, 0), (9, 6), (11, 2), (10, 6), (10, 10), (10, 2), (9, 7)},
		(11, 9): {(9, 9), (10, 0), (9, 6), (9, 0)},
		(9, 0): {(9, 9), (10, 0), (9, 6), (11, 9)},
	}

# %%

def plot_minimal_set_results(minimality_scores: Dict[Tuple[int, int], float]):
	'''
	Plots the minimality results, in a way resembling figure 7 in the paper.

	minimality_scores:
		Dict with elements like (9, 9): minimality score for head 9.9 (as described
		in section 4.2 of the paper)
	'''
	
	CIRCUIT_reversed = {head: k for k, v in CIRCUIT.items() for head in v}
	colors = [CIRCUIT_reversed[head].capitalize() + " head" for head in minimality_scores.keys()]
	color_sequence = [px.colors.qualitative.Dark2[i] for i in [0, 1, 2, 5, 3, 6]] + ["#BAEA84"]

	bar(
		list(minimality_scores.values()),
		x=list(map(str, minimality_scores.keys())),
		labels={"x": "Attention head", "y": "Change in logit diff", "color": "Head type"},
		color=colors,
		template="ggplot2",
		color_discrete_sequence=color_sequence,
		bargap=0.02,
		yaxis_tickformat=".0%",
		legend_title_text="",
		title="Plot of minimality scores (as percentages of full model logit diff)",
		width=800,
		hovermode="x unified"
	)

# %%

# FLAT SOLUTION NOINDENT
# YOUR CODE HERE - define the `minimality_scores` dictionary, to be used in the plot function given above
def get_score(
	model: HookedTransformer, 
	ioi_dataset: IOIDataset, 
	abc_dataset: IOIDataset,
	K: Set[Tuple[int, int]],
	C: Dict[str, List[Tuple[int, int]]],
) -> float:
	'''
	Returns the value F(C \ K), where F is the logit diff, C is the
	core circuit, and K is the set of circuit components to remove.
	'''
	C_excl_K = {k: [head for head in v if head not in K] for k, v in C.items()}
	model = add_mean_ablation_hook(model, abc_dataset, C_excl_K, SEQ_POS_TO_KEEP)
	logits = model(ioi_dataset.toks)
	score = logits_to_ave_logit_diff_2(logits, ioi_dataset).item()

	return score


if MAIN:
	def get_minimality_score(
		model: HookedTransformer,
		ioi_dataset: IOIDataset,
		abc_dataset: IOIDataset,
		v: Tuple[int, int],
		K: Set[Tuple[int, int]],
		C: Dict[str, List[Tuple[int, int]]] = CIRCUIT,
	) -> float:
		'''
		Returns the value | F(C \ K_union_v) - F(C | K) |, where F is 
		the logit diff, C is the core circuit, K is the set of circuit
		components to remove, and v is a head (not in K).
		'''
		assert v not in K
		K_union_v = K | {v}
		C_excl_K_score = get_score(model, ioi_dataset, abc_dataset, K, C)
		C_excl_Kv_score = get_score(model, ioi_dataset, abc_dataset, K_union_v, C)

		return abs(C_excl_K_score - C_excl_Kv_score)


	def get_all_minimality_scores(
		model: HookedTransformer,
		ioi_dataset: IOIDataset = ioi_dataset,
		abc_dataset: IOIDataset = abc_dataset,
		k_for_each_component: Dict = K_FOR_EACH_COMPONENT
	) -> Dict[Tuple[int, int], float]:
		'''
		Returns dict of minimality scores for every head in the model (as 
		a fraction of F(M), the logit diff of the full model).

		Warning - this resets all hooks at the end (including permanent).
		'''
		# Get full circuit score F(M), to divide minimality scores by
		model.reset_hooks(including_permanent=True)
		logits = model(ioi_dataset.toks)
		full_circuit_score = logits_to_ave_logit_diff_2(logits, ioi_dataset).item()

		# Get all minimality scores, using the `get_minimality_score` function
		minimality_scores = {}
		for v, K in tqdm(k_for_each_component.items()):
			score = get_minimality_score(model, ioi_dataset, abc_dataset, v, K)
			minimality_scores[v] = score / full_circuit_score

		model.reset_hooks(including_permanent=True)

		return minimality_scores


if MAIN:
	minimality_scores = get_all_minimality_scores(model)

# %%


if MAIN:
	plot_minimal_set_results(minimality_scores)

# %% 6️⃣ BONUS / EXPLORING ANOMALIES


if MAIN:
	model.reset_hooks(including_permanent=True)
	
	attn_heads = [(5, 5), (6, 9)]
	
	# Get repeating sequences (note we could also take mean over larger batch)
	batch = 1
	seq_len = 15
	rep_tokens = generate_repeated_tokens(model, seq_len, batch)
	
	# Run cache (we only need attention patterns for layers 5 and 6)
	_, cache = model.run_with_cache(
		rep_tokens,
		return_type = None,
		names_filter = lambda name: name.endswith("pattern") and any(f".{layer}." in name for layer, head in attn_heads)
	)
	
	# Display results
	attn = t.stack([
		cache["pattern", layer][0, head]
		for (layer, head) in attn_heads
	])
	cv.attention.attention_patterns(
		tokens = model.to_str_tokens(rep_tokens[0]),
		attention = attn,
		attention_head_names = [f"{layer}.{head}" for (layer, head) in attn_heads]
	)

# %%


if MAIN:
	model.reset_hooks(including_permanent=True)
	
	# FLAT SOLUTION
	# YOUR CODE HERE - create `induction_head_key_path_patching_results` 
	induction_head_key_path_patching_results = get_path_patch_head_to_heads(
		receiver_heads = [(5, 5), (6, 9)],
		receiver_input = "k",
		model = model,
		patching_metric = ioi_metric_2
	)

# %%


if MAIN:
	imshow(
		100 * induction_head_key_path_patching_results,
		title="Direct effect on Induction Heads' keys", 
		labels={"x": "Head", "y": "Layer", "color": "Logit diff.<br>variation"},
		coloraxis=dict(colorbar_ticksuffix = "%"),
		width=600,
	)

# %%


if MAIN:
	model.reset_hooks(including_permanent=True)
	
	ioi_logits, ioi_cache = model.run_with_cache(ioi_dataset.toks)
	original_average_logit_diff = logits_to_ave_logit_diff_2(ioi_logits)

# %%


if MAIN:
	s_unembeddings = model.W_U.T[ioi_dataset.s_tokenIDs]
	io_unembeddings = model.W_U.T[ioi_dataset.io_tokenIDs]
	logit_diff_directions: Float[Tensor, "batch d_model"] =  io_unembeddings - s_unembeddings
	
	per_head_residual, labels = ioi_cache.stack_head_results(layer=-1, return_labels=True)
	per_head_residual = einops.rearrange(
		per_head_residual[:, t.arange(len(ioi_dataset)).to(device), ioi_dataset.word_idx["end"].to(device)], 
		"(layer head) batch d_model -> layer head batch d_model", 
		layer=model.cfg.n_layers
	)
	
	per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, ioi_cache, logit_diff_directions)
	
	top_layer, top_head = topk_of_Nd_tensor(per_head_logit_diffs, k=1)[0]
	print(f"Top Name Mover to ablate: {top_layer}.{top_head}")

# %%

# Getting means we can use to ablate

if MAIN:
	abc_means = ioi_circuit_extraction.compute_means_by_template(abc_dataset, model)[top_layer]
	
	# Define hook function and add to model
def ablate_top_head_hook(z: Float[Tensor, "batch pos head_index d_head"], hook):
	'''
	Ablates hook by patching in results
	'''
	z[range(len(ioi_dataset)), ioi_dataset.word_idx["end"], top_head] = abc_means[range(len(ioi_dataset)), ioi_dataset.word_idx["end"], top_head]
	return z


if MAIN:
	model.add_hook(utils.get_act_name("z", top_layer), ablate_top_head_hook)
	
	# Runs the model, temporarily adds caching hooks and then removes *all* hooks after running, including the ablation hook.
	ablated_logits, ablated_cache = model.run_with_cache(ioi_dataset.toks)
	rprint("\n".join([
		f"                            Original logit diff: {original_average_logit_diff:.4f}",
		f"Direct Logit Attribution of top name mover head: {per_head_logit_diffs[top_layer, top_head]:.4f}",
		f"   Naive prediction of post ablation logit diff: {original_average_logit_diff - per_head_logit_diffs[top_layer, top_head]:.4f}",
		f"      Logit diff after ablating L{top_layer}H{top_head}: {logits_to_ave_logit_diff_2(ablated_logits):.4f}",
	]))

# %%


if MAIN:
	make_table(
		cols = [
			"Original logit diff", "Direct Logit Attribution of top name mover head", "Naive prediction of post ablation logit diff", f"Logit diff after ablating L{top_layer}H{top_head}",
			original_average_logit_diff, per_head_logit_diffs[top_layer, top_head]
		]
	)

# %%


if MAIN:
	per_head_ablated_residual, labels = ablated_cache.stack_head_results(layer=-1, return_labels=True)
	per_head_ablated_residual = einops.rearrange(
		per_head_ablated_residual[:, t.arange(len(ioi_dataset)).to(device), ioi_dataset.word_idx["end"].to(device)], 
		"(layer head) batch d_model -> layer head batch d_model", 
		layer=model.cfg.n_layers
	)
	per_head_ablated_logit_diffs = residual_stack_to_logit_diff(per_head_ablated_residual, ablated_cache, logit_diff_directions)
	per_head_ablated_logit_diffs = per_head_ablated_logit_diffs.reshape(model.cfg.n_layers, model.cfg.n_heads)
	
	imshow(
		t.stack([
			per_head_logit_diffs, 
			per_head_ablated_logit_diffs, 
			per_head_ablated_logit_diffs - per_head_logit_diffs
		]), 
		title="Direct logit contribution by head, pre / post ablation",
		labels={"x":"Head", "y":"Layer"},
		facet_col=0,
		facet_labels=["No ablation", "9.9 is ablated", "Change in head contribution post-ablation"],
	)
	
	scatter(
		y=per_head_logit_diffs.flatten(), 
		x=per_head_ablated_logit_diffs.flatten(), 
		hover_name=labels, 
		range_x=(-1, 1), 
		range_y=(-2, 2), 
		labels={"x": "Ablated", "y": "Original"},
		title="Original vs Post-Ablation Direct Logit Attribution of Heads",
		width=600,
		add_line="y=x"
	)

# %%


if MAIN:
	ln_scaling_no_ablation = ioi_cache["ln_final.hook_scale"][t.arange(len(ioi_dataset)), ioi_dataset.word_idx["end"]].squeeze()
	ln_scaling_ablated = ablated_cache["ln_final.hook_scale"][t.arange(len(ioi_dataset)), ioi_dataset.word_idx["end"]].squeeze()

# %%


if MAIN:
	scatter(
		y=ln_scaling_ablated,
		x=ln_scaling_no_ablation,
		labels={"x": "No ablation", "y": "Ablation"},
		title=f"Final LN scaling factors compared (ablation vs no ablation)<br>Average ratio = {(ln_scaling_no_ablation / ln_scaling_ablated).mean():.4f}",
		width=700,
		add_line="y=x"
	)

# %%


if MAIN:
	datasets: List[Tuple[Tuple, str, IOIDataset]] = [
		((0, 0), "original", ioi_dataset),
		((1, 0), "random token", ioi_dataset.gen_flipped_prompts("ABB->CDD, BAB->DCD")),
		((2, 0), "inverted token", ioi_dataset.gen_flipped_prompts("ABB->BAA, BAB->ABA")),
		((0, 1), "inverted position", ioi_dataset.gen_flipped_prompts("ABB->BAB, BAB->ABB")),
		((1, 1), "inverted position, random token", ioi_dataset.gen_flipped_prompts("ABB->DCD, BAB->CDD")),
		((2, 1), "inverted position, inverted token", ioi_dataset.gen_flipped_prompts("ABB->ABA, BAB->BAA")),
	]

# %%


if MAIN:
	results = t.zeros(3, 2).to(device)
	
	s2_inhibition_heads = CIRCUIT["s2 inhibition"]
	layers = set(layer for layer, head in s2_inhibition_heads)
	
	names_filter=lambda name: name in [utils.get_act_name("z", layer) for layer in layers]
	
def patching_hook_fn(z: Float[Tensor, "batch seq head d_head"], hook: HookPoint, cache: ActivationCache):
	heads_to_patch = [head for layer, head in s2_inhibition_heads if layer == hook.layer()]
	z[:, :, heads_to_patch] = cache[hook.name][:, :, heads_to_patch]
	return z


if MAIN:
	for ((row, col), desc, dataset) in datasets:
	
		# Get cache of values from the modified dataset
		_, cache_for_patching = model.run_with_cache(
			dataset.toks,
			names_filter=names_filter,
			return_type=None
		)
	
		# Run model on IOI dataset, but patch S-inhibition heads with signals from modified dataset
		patched_logits = model.run_with_hooks(
			ioi_dataset.toks,
			fwd_hooks=[(names_filter, partial(patching_hook_fn, cache=cache_for_patching))]
		)
	
		# Get logit diff for patched results
		# Note, we still use IOI dataset for our "correct answers" reference point
		results[row, col] = logits_to_ave_logit_diff_2(patched_logits, ioi_dataset)

# %%


if MAIN:
	imshow(
		results, 
		labels={"x": "Positional signal", "y": "Token signal"}, 
		x=["Original", "Inverted"], 
		y=["Original", "Random", "Inverted"], 
		title="Logit diff after changing all S2 inhibition heads' output signals via patching",
		text_auto=".2f"
	)

# %%


if MAIN:
	results = t.zeros(len(CIRCUIT["s2 inhibition"]), 3, 2).to(device)
	# Your code here - fill in results!
	
def patching_hook_fn(
	z: Float[Tensor, "batch seq head d_head"], 
	hook: HookPoint, 
	cache: ActivationCache, 
	head: int
):
	z[:, :, head] = cache[hook.name][:, :, head]
	return z


if MAIN:
	for i, (layer, head) in enumerate(CIRCUIT["s2 inhibition"]):
	
		model.reset_hooks(including_permanent=True)
	
		hook_name = utils.get_act_name("z", layer)
	
		for ((row, col), desc, dataset) in datasets:
	
			# Get cache of values from the modified dataset
			_, cache_for_patching = model.run_with_cache(
				dataset.toks,
				names_filter=lambda name: name == hook_name,
				return_type=None
			)
	
			# Run model on IOI dataset, but patch S-inhibition heads with signals from modified dataset
			patched_logits = model.run_with_hooks(
				ioi_dataset.toks,
				fwd_hooks=[(hook_name, partial(patching_hook_fn, cache=cache_for_patching, head=head))]
			)
	
			# Get logit diff for patched results
			# Note, we still use IOI dataset for our "correct answers" reference point
			results[i, row, col] = logits_to_ave_logit_diff_2(patched_logits, ioi_dataset)

# %%


if MAIN:
	imshow(
		(results - results[0, 0, 0]) / results[0, 0, 0], 
		labels={"x": "Positional signal", "y": "Token signal"}, 
		x=["Original", "Inverted"], 
		y=["Original", "Random", "Inverted"], 
		title="Logit diff after patching individual S2 inhibition heads (as proportion of clean logit diff)",
		facet_col=0,
		facet_labels=[f"{layer}.{head}" for (layer, head) in CIRCUIT["s2 inhibition"]],
		facet_col_spacing = 0.08,
		text_auto=".2f",
	)

# %%

