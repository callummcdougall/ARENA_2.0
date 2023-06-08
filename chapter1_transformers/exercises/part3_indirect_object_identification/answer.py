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
from jaxtyping import Float, Int, Bool, jaxtyped
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
from typeguard import typechecked 
from transformer_lens import patching

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
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
# %%
# Show column norms are the same (except first few, for fiddly bias reasons)
# line([model.W_Q[0, 0].pow(2).sum(0), model.W_K[0, 0].pow(2).sum(0)])
# # Show columns are orthogonal (except first few, again)
# W_Q_dot_products = einops.einsum(
#     model.W_Q[0, 0], model.W_Q[0, 0], "d_model d_head_1, d_model d_head_2 -> d_head_1 d_head_2"
# )
# imshow(W_Q_dot_products)

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
cols = [
    "Prompt",
    Column("Correct", style="rgb(0,200,0) bold"),
    Column("Incorrect", style="rgb(255,0,0) bold"),
]
table = Table(*cols, title="Prompts & Answers:")

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
# @jaxtyped
# @typechecked
def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 3"] = answer_tokens,
    per_prompt: bool = False
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    n_batch = logits.shape[0]
    correct_token, incorrect_token = answer_tokens[:, 0], answer_tokens[:, 1]
    logit_diff = logits[range(n_batch), -1, correct_token] - logits[range(n_batch), -1, incorrect_token]

    if not per_prompt:
        return logit_diff.mean()
    return logit_diff


tests.test_logits_to_ave_logit_diff(logits_to_ave_logit_diff)

original_per_prompt_diff = logits_to_ave_logit_diff(original_logits, answer_tokens, per_prompt=True)
print("Per prompt logit difference:", original_per_prompt_diff)
original_average_logit_diff = logits_to_ave_logit_diff(original_logits, answer_tokens)
print("Average logit difference:", original_average_logit_diff)

cols = [
    "Prompt", 
    Column("Correct", style="rgb(0,200,0) bold"), 
    Column("Incorrect", style="rgb(255,0,0) bold"), 
    Column("Logit Difference", style="bold"),
    Column(" ", style="bold"),
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
# %%
answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
# shape (batch, 2, d_model)

# %%
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
    # SOLUTION
    batch_size = residual_stack.size(-2)
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    return einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size


# Test function by checking that it gives the same result as the original logit difference

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
per_layer_residual, labels = cache.decompose_resid(layer=-1, mode='attn', pos_slice=-1, return_labels=True)
per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache)

# line(
#     per_layer_logit_diffs, 
#     hovermode="x unified",
#     title="Logit Difference From Each Layer",
#     labels={"x": "Layer", "y": "Logit Diff"},
#     xaxis_tickvals=labels,
#     width=800
# )
# %%
per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
per_head_residual = einops.rearrange(
    per_head_residual, 
    "(layer head) ... -> layer head ...", 
    layer=model.cfg.n_layers
)
per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, cache)

# imshow(
#     per_head_logit_diffs, 
#     labels={"x":"Head", "y":"Layer"}, 
#     title="Logit Difference From Each Head",
#     width=600
# )
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
        cache["pattern", layer][:, head][0]
        for layer, head in top_heads
    ])

    # Display results
    display(HTML(f"<h2>Top {k} {head_type} Logit Attribution Heads</h2>"))
    display(cv.attention.attention_patterns( # attention_heads
        attention = attn_patterns_for_important_heads,
        tokens = model.to_str_tokens(tokens[0]),
        attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
    ))
# %%
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
    path_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
    return (path_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


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
    # Get the logit output for the corrupted tokens
    n_pos = corrupted_tokens.shape[-1]
    patching_score = t.zeros(model.cfg.n_layers, n_pos).to(model.cfg.device)
    for layer in range(model.cfg.n_layers):
        for pos in range(n_pos):
            hook_name = utils.get_act_name('resid_pre', layer)
            hook_fn = partial(patch_residual_component, pos=pos, clean_cache=clean_cache)
            logits = model.run_with_hooks(
                corrupted_tokens, 
                fwd_hooks=[(hook_name, hook_fn)],
            )

            # patching score
            patching_score[layer, pos] = patching_metric(logits)
    return patching_score


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
# imshow(
#     act_patch_attn_head_out_all_pos[:, -1], 
#     labels={"y": "Layer", "x": "Head"}, 
#     title="attn_head_out Activation Patching (All Pos)",
#     width=600
# )
# %%
# act_patch_attn_head_all_pos_every = patching.get_act_patch_attn_head_all_pos_every(
#     model, 
#     corrupted_tokens, 
#     clean_cache, 
#     ioi_metric
# )

# imshow(
#     act_patch_attn_head_all_pos_every, 
#     facet_col=0, 
#     facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
#     title="Activation Patching Per Head (All Pos)", 
#     labels={"x": "Head", "y": "Layer"},
# )
# %%
# act_patch_attn_head_all_pos_every = patching.get_act_patch_attn_head_k_by_pos(
#     model, 
#     corrupted_tokens, 
#     clean_cache, 
#     ioi_metric
# )

# imshow(
#     act_patch_attn_head_all_pos_every, 
#     facet_col=0, 
#     facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
#     title="Activation Patching Per Head (All Pos)", 
#     labels={"x": "Head", "y": "Layer"},
# )
# %%
# import pandas as pd
# index_axis_names = ["layer", "head", "pos"]
# index_df = pd.DataFrame({
#     'layer': [i for i in range(12) for _ in range(12)],
#     'head': [i for _ in range(12) for i in range(12)],
#     'pos': list(-1 for _ in range(12*12))
# })
# act_patch_attn_head_all_pos_every = patching.get_act_patch_attn_head_k_by_pos(
#     model, 
#     corrupted_tokens, 
#     clean_cache, 
#     ioi_metric,
#     index_axis_names=None,
#     index_df=index_df
# )

# %%
# imshow(
#     einops.rearrange(act_patch_attn_head_all_pos_every, '(h l)->l h', l=12, h=12),
#     # facet_col=0, 
#     # facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
#     title="Activation Patching Per Head (All Pos)", 
#     labels={"x": "Head", "y": "Layer"},
# )
# %%

# act_patch_attn_head_all_pos_every = patching.get_act_patch_attn_head_k_by_pos(
#     model, 
#     corrupted_tokens, 
#     clean_cache, 
#     ioi_metric,
# )

# %%
# imshow(
#     act_patch_attn_head_all_pos_every[:, -1],
#     # facet_col=0, 
#     # facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
#     title="Activation Patching Per Head (All Pos)", 
#     labels={"x": "Head", "y": "Layer"},
# )
# %%
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
# %%
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
# %%
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

def patch_head_activation(activations: Float[Tensor, 'batch pos head d_head'], hook: HookPoint, head: Union[int, list], cache: ActivationCache) -> Float[Tensor, 'batch pos head d_head']:
    '''
    Patches a given head's output, using the value from the cache.
    '''

    if cache[hook.name].ndim == 3:
        activations[:, :, head] = cache[hook.name]
    else:
        activations[:, :, head, :] = cache[hook.name][:, :, head, :]
    return activations


# %%
def _path_patch_head_to_resid(model: HookedTransformer, layer: int, head: int, orig_dataset: IOIDataset, 
                              orig_cache: ActivationCache, new_cache: ActivationCache, patching_metric: Callable):

    model.reset_hooks()
    corrupt_hook = partial(patch_head_activation, head=head, cache=new_cache)
    model.add_hook(utils.get_act_name('z', layer), corrupt_hook)

    for l in range(layer+1, model.cfg.n_layers):
        freeze_hook = partial(patch_head_activation, head=list(range(model.cfg.n_heads)), cache=orig_cache)
        model.add_hook(utils.get_act_name('z', l), freeze_hook)

    # Get the logit output for the original tokens
    patched_logits = model(orig_dataset.toks)
    model.reset_hooks()
    # Calculate the patching score
    return patching_metric(patched_logits)


def get_path_patch_head_to_final_resid_post(
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: IOIDataset = abc_dataset,
    orig_dataset: IOIDataset = ioi_dataset,
    new_cache: Optional[ActivationCache] = abc_cache,
    orig_cache: Optional[ActivationCache] = ioi_cache,
) -> Float[Tensor, "layer head"]:

    patching_scores = t.zeros(model.cfg.n_layers, model.cfg.n_heads).to(model.cfg.device)

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            score = _path_patch_head_to_resid(
                model=model,
                layer=layer,
                head=head,
                orig_dataset=orig_dataset,
                orig_cache=orig_cache,
                new_cache=new_cache,
                patching_metric=patching_metric,
            )
            patching_scores[layer, head] = score
    
    return patching_scores



path_patch_head_to_final_resid_post = get_path_patch_head_to_final_resid_post(model, ioi_metric_2)

imshow(
    100 * path_patch_head_to_final_resid_post,
    title="Direct effect on logit difference",
    labels={"x":"Head", "y":"Layer", "color": "Logit diff. variation"},
    coloraxis=dict(colorbar_ticksuffix = "%"),
    width=600,
)
# %%
def _path_patch_head_to_heads_first(model: HookedTransformer, layer: int, head: int, orig_dataset: IOIDataset, 
                              orig_cache: ActivationCache, new_cache: ActivationCache, receiver_heads: List[Tuple[int, int]], 
                              receiver_input: str) -> List[Float[Tensor, 'batch pos d_head']]:

    model.reset_hooks()
    corrupt_hook = partial(patch_head_activation, head=head, cache=new_cache)
    model.add_hook(utils.get_act_name('z', layer), corrupt_hook)

    for l in range(layer + 1, model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            if (l, h) in receiver_heads:
                continue
            freeze_hook = partial(patch_head_activation, head=h, cache=orig_cache)
            model.add_hook(utils.get_act_name('z', l), freeze_hook)

    _, cache = model.run_with_cache(orig_dataset.toks, names_filter=lambda name: f'hook_{receiver_input}' in name)
    return [cache[receiver_input, l][:, :, h] for (l, h) in receiver_heads]


def _path_patch_head_to_heads_second(model: HookedTransformer, orig_dataset: IOIDataset,
                                     receiver_cache: List[Float[Tensor, 'batch pos d_head']], receiver_heads: List[Tuple[int, int]],
                                     receiver_input: str, patching_metric: Callable) -> Float[Tensor, '']:
    model.reset_hooks()
    for idx, (l, h) in enumerate(receiver_heads):
        corrupt_hook = partial(patch_head_activation, head=h, cache={utils.get_act_name(receiver_input, l): receiver_cache[idx]})
        model.add_hook(utils.get_act_name(receiver_input, l), corrupt_hook) 

    logits = model(orig_dataset.toks)
    return patching_metric(logits)


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
    if new_cache is None:
        _, new_cache = model.run_with_cache(new_dataset.toks)
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(orig_dataset.toks)
    patching_scores = t.zeros(model.cfg.n_layers, model.cfg.n_heads).to(model.cfg.device)
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            receiver_cache = _path_patch_head_to_heads_first(
                model=model,
                layer=layer,
                head=head,
                orig_dataset=orig_dataset,
                orig_cache=orig_cache,
                new_cache=new_cache,
                receiver_heads=receiver_heads,
                receiver_input=receiver_input,
                
            )

            score = _path_patch_head_to_heads_second(
                model=model,
                orig_dataset=orig_dataset,
                receiver_cache=receiver_cache,
                receiver_heads=receiver_heads,
                receiver_input=receiver_input,
                patching_metric=patching_metric,
            )


            patching_scores[layer, head] = score
    return patching_scores

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
pattern_filter = lambda name: "pattern" in name
orig_cache = model.run_with_cache(ioi_dataset.toks, names_filter=pattern_filter)[1]
selected_heads = [(9, 9), (9, 6), (7, 3), (7, 9), (8, 6), (8, 10), (10, 7)]
selected_attn_patterns = t.stack([orig_cache['pattern', layer][:, head] for layer,head in selected_heads], dim=1)
example = 0

display(cv.attention.attention_patterns(
    attention = selected_attn_patterns[example],
    tokens = model.to_str_tokens(ioi_dataset.toks[example]),
    attention_head_names = [f"{layer}.{head}" for layer, head in selected_heads],
))

# %%
# AAB_dataset = ioi_dataset.gen_flipped_prompts("ABB->BBA, BAB->BBA")
# orig_toks = AAB_dataset.toks


# pattern_filter = lambda name: "pattern" in name
# orig_cache = model.run_with_cache(orig_toks, names_filter=pattern_filter)[1]
# selected_heads = [(9, 9), (9, 6), (7, 3), (7, 9), (8, 6), (8, 10), (10, 7)]
# selected_attn_patterns = t.stack([orig_cache['pattern', layer][:, head] for layer,head in selected_heads], dim=1)
# example = 4

# display(cv.attention.attention_patterns(
#     attention = selected_attn_patterns[example],
#     tokens = model.to_str_tokens(orig_toks[example]),
#     attention_head_names = [f"{layer}.{head}" for layer, head in selected_heads],
# ))

# # %%

# def patch_head_input(
#     orig_activation: Float[Tensor, "batch pos head_idx d_head"],
#     hook: HookPoint,
#     patched_cache: ActivationCache,
#     head_list: List[Tuple[int, int]],
# ) -> Float[Tensor, "batch pos head_idx d_head"]:
#     '''
#     Function which can patch any combination of heads in layers,
#     according to the heads in head_list.
#     '''
#     heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
#     orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
#     return orig_activation

# _, new_cache = model.run_with_cache(alter_toks)
# _, orig_cache = model.run_with_cache(orig_toks)
# induction_heads = [(5, 5), (3, 0), (0, 1)]

# model.reset_hooks()

# for layer in range(model.cfg.n_layers):
#     model.add_hook(utils.get_act_name('z', layer), partial(patch_head_input, patched_cache=new_cache, 
#                                                            head_list=induction_heads))


# pattern_filter = lambda name: "pattern" in name
# orig_cache = model.run_with_cache(orig_toks, names_filter=pattern_filter)[1]
# model.reset_hooks()

# selected_heads = [(9, 9), (9, 6), (7, 3), (7, 9), (8, 6), (8, 10), (10, 7)]
# selected_attn_patterns = t.stack([orig_cache['pattern', layer][:, head] for layer,head in selected_heads], dim=1)
# example = 3

# display(cv.attention.attention_patterns(
#     attention = selected_attn_patterns[example],
#     tokens = model.to_str_tokens(orig_toks[example]),
#     attention_head_names = [f"{layer}.{head}" for layer, head in selected_heads],
# ))
# %%

# def calculate_and_show_scatter_embedding_vs_attn(
#     layer: int,
#     head: int,
#     cache: ActivationCache = ioi_cache,
#     dataset: IOIDataset = ioi_dataset,
# ) -> None:
#     '''
#     Creates and plots a figure equivalent to 3(c) in the paper.

#     This should involve computing the four 1D tensors:
#         attn_from_end_to_io
#         attn_from_end_to_s
#         projection_in_io_dir
#         projection_in_s_dir
#     and then calling the scatter_embedding_vs_attn function.
#     '''
#     batch_idx = t.arange(dataset.toks.size(0)).to(dataset.toks.device)
#     attn_from_end_to_io = cache['attn', layer][batch_idx, head, dataset.word_idx["end"], dataset.word_idx["IO"]]
#     attn_from_end_to_s = (cache['attn', layer][batch_idx, head, dataset.word_idx["end"], dataset.word_idx["S1"]] +
#                           cache['attn', layer][batch_idx, head, dataset.word_idx["end"], dataset.word_idx["S2"]])
#     projection_in_io_dir = einops.einsum(cache['z', layer][batch_idx, dataset.word_idx["end"], head], 
#                                          model.W_O[layer, head],
#                                          model.W_U[:, dataset.io_tokenIDs],
#                                          'batch d_head, d_head d_model, d_model batch-> batch')
#     projection_in_s_dir = einops.einsum(cache['z', layer][batch_idx, dataset.word_idx["end"], head], 
#                                          model.W_O[layer, head],
#                                          model.W_U[:, dataset.s_tokenIDs],
#                                          'batch d_head, d_head d_model, d_model batch -> batch')
    
#     scatter_embedding_vs_attn(attn_from_end_to_io, attn_from_end_to_s, projection_in_io_dir, projection_in_s_dir, layer, head)


# nmh = (9, 9)
# calculate_and_show_scatter_embedding_vs_attn(*nmh)

# nnmh = (11, 10)
# calculate_and_show_scatter_embedding_vs_attn(*nnmh)


# # %%

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
#             # resid_after_OV_neg = resid_after_mlp1 @ -W_OV
#             resid_after_OV_neg = -resid_after_OV_pos

#             # Get logits from value of residual stream
#             logits_pos: Float[Tensor, "batch d_vocab"] = unembed(ln_final(resid_after_OV_pos)).squeeze()
#             logits_neg: Float[Tensor, "batch d_vocab"] = unembed(ln_final(resid_after_OV_neg)).squeeze()
#             # logits_neg = -logits_pos

#             # Check how many are in top k
#             topk_logits: Int[Tensor, "batch k"] = t.topk(logits_pos, dim=-1, k=k).indices
#             in_topk = (topk_logits == name_tokens).any(-1)
#             # Check how many are in bottom k
#             bottomk_logits: Int[Tensor, "batch k"] = t.topk(logits_neg, dim=-1, k=k).indices
#             in_bottomk = (bottomk_logits == name_tokens).any(-1)

#             # Fill in results
#             results[:, layer-1, head] = t.tensor([in_topk.float().mean(), in_bottomk.float().mean()])

#     return results

# copying_results = get_copying_scores(model)

# imshow(
#     copying_results, 
#     facet_col=0, 
#     facet_labels=["Positive copying scores", "Negative copying scores"],
#     title="Copying scores of attention heads' OV circuits",
#     width=800
# )


# heads = {"name mover": [(9, 9), (10, 0), (9, 6)], "negative name mover": [(10, 7), (11, 10)]}

# for i, name in enumerate(["name mover", "negative name mover"]):
#     make_table(
#         title=f"Copying Scores ({name} heads)",
#         colnames=["Head", "Score"],
#         cols=[
#             list(map(str, heads[name])) + ["[dark_orange bold]Average"],
#             [f"{copying_results[i, layer, head]:.2%}" for (layer, head) in heads[name]] + [f"[dark_orange bold]{copying_results[i].mean():.2%}"]
#         ]
#     )
# %%
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

def patch_layer_hook_z(activation: Float[Tensor, "batch seq head d_head"], 
                       hook: HookPoint, template_mean: Float[Tensor, "layer batch seq head d_head"],
                       ablation_mask: Dict[int, Bool[Tensor, "batch seq head"]]
                       ) -> Float[Tensor, "batch seq head d_head"]:
    mask = ablation_mask[hook.layer()].unsqueeze(-1)
    activation = t.where(mask, template_mean[hook.layer()], activation)
    return activation


def compute_mean_by_template(
    model: HookedTransformer,
    means_dataset: IOIDataset, 
) -> Dict[int, Bool[Tensor, "layer batch seq head"]]:
    
    _, means_cache = model.run_with_cache(means_dataset.toks, names_filter=lambda name: "hook_z" in name)

    n_layers = model.cfg.n_layers
    n_batch, n_seq = means_dataset.toks.size()
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    template_mean = t.zeros(n_layers, n_batch, n_seq, n_heads, d_head).to(model.cfg.device)
    for group in means_dataset.groups:
        hook_z = means_cache.stack_activation('z') # (n_layers, n_batch, n_seq, n_heads, d_head)
        mean_hook_z = hook_z[:, group].mean(dim=1, keepdim=True) # (n_layers, 1, n_seq, n_heads, d_head)
        template_mean[:, group] = mean_hook_z

    return template_mean

def compute_ablation_mask(
    model: HookedTransformer,
    means_dataset: IOIDataset, 
    circuit: Dict[str, List[Tuple[int, int]]] = CIRCUIT,
    seq_pos_to_keep: Dict[str, str] = SEQ_POS_TO_KEEP,   
):
    n_batch, n_seq = means_dataset.toks.size()
    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers

    ablation_mask = {l: t.ones(n_batch, n_seq, n_heads).bool().to(model.cfg.device) for l in range(n_layers)}
    for head_type in circuit:
        heads = circuit[head_type]
        seq_pos = seq_pos_to_keep[head_type]
        seq_pos_idx = means_dataset.word_idx[seq_pos] # (n_batch)

        for layer, head in heads:
            ablation_mask[layer][range(n_batch), seq_pos_idx, head] = False
    
    return ablation_mask


def add_mean_ablation_hook(
    model: HookedTransformer, 
    means_dataset: IOIDataset, 
    circuit: Dict[str, List[Tuple[int, int]]] = CIRCUIT,
    seq_pos_to_keep: Dict[str, str] = SEQ_POS_TO_KEEP,
    is_permanent: bool = True,
    preserve_hook: bool = False
) -> HookedTransformer:
    '''
    Adds a permanent hook to the model, which ablates according to the circuit and 
    seq_pos_to_keep dictionaries.

    In other words, when the model is run on ioi_dataset, every head's output will 
    be replaced with the mean over means_dataset for sequences with the same template,
    except for a subset of heads and sequence positions as specified by the circuit
    and seq_pos_to_keep dicts.
    '''
    template_mean = compute_mean_by_template(model, means_dataset) # (n_layers, n_batch, n_seq, n_heads, d_head)
    ablation_mask = compute_ablation_mask(model, means_dataset, circuit, seq_pos_to_keep) # {layer: (n_batch, n_seq, n_heads)}

    if not preserve_hook:
        model.reset_hooks(including_permanent=True)
    
    model.add_hook(
        lambda name: 'hook_z' in name,
        partial(patch_layer_hook_z, template_mean=template_mean, ablation_mask=ablation_mask), 
        is_permanent=is_permanent
    )
    return model

import part3_indirect_object_identification.ioi_circuit_extraction as ioi_circuit_extraction


model_ = ioi_circuit_extraction.add_mean_ablation_hook(model, means_dataset=abc_dataset, circuit=CIRCUIT, seq_pos_to_keep=SEQ_POS_TO_KEEP)

ioi_logits_minimal = model_(ioi_dataset.toks)

print(f"Average logit difference (IOI dataset, using entire model): {logits_to_ave_logit_diff_2(ioi_logits_original):.4f}")
print(f"Average logit difference (IOI dataset, only using circuit): {logits_to_ave_logit_diff_2(ioi_logits_minimal):.4f}")


model_ = add_mean_ablation_hook(model, means_dataset=abc_dataset, circuit=CIRCUIT, seq_pos_to_keep=SEQ_POS_TO_KEEP)

ioi_logits_minimal = model_(ioi_dataset.toks)

print(f"Average logit difference (IOI dataset, using entire model): {logits_to_ave_logit_diff_2(ioi_logits_original):.4f}")
print(f"Average logit difference (IOI dataset, only using circuit): {logits_to_ave_logit_diff_2(ioi_logits_minimal):.4f}")
# %%

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

def get_circuit_reduced(circuit: Dict[str, List[Tuple[int, int]]], heads: Set[Tuple[int, int]]
                        ) -> Dict[str, List[Tuple[int, int]]]:
    reduced_circuit = circuit.copy()
    for head_type in circuit:
        reduced_circuit[head_type] = [head for head in circuit[head_type] if head not in heads]
    return reduced_circuit


def get_minimality_scores(
    model: HookedTransformer,
    orig_dataset: IOIDataset = ioi_dataset,
    mean_dataset: IOIDataset = abc_dataset,
    K_dict: Dict[Tuple[int, int], Set[Tuple[int, int]]] = K_FOR_EACH_COMPONENT,
    circuit: Dict[str, List[Tuple[int, int]]] = CIRCUIT,
    seq_pos_to_keep: Dict[str, str] = SEQ_POS_TO_KEEP,
    patching_metric: Callable = lambda x: print('Not implemented'),
) -> Dict[Tuple[int, int], float]:  
    
    logits_model = model(orig_dataset.toks)
    logits_diff_model = logits_to_ave_logit_diff_2(logits_model)

    minimality_scores = dict()
    for v in K_dict:
        reduced_circuit = get_circuit_reduced(circuit, K_dict[v])

        circuit_v = get_circuit_reduced(circuit, K_dict[v] | {v})
        
        model_ = add_mean_ablation_hook(model, mean_dataset, reduced_circuit, seq_pos_to_keep)
        logits_reduced = model_(orig_dataset.toks)
        logits_diff_reduced = logits_to_ave_logit_diff_2(logits_reduced)
 
        model_reduced_v = add_mean_ablation_hook(model, mean_dataset, circuit_v, seq_pos_to_keep)
        logits_reduced_v = model_(orig_dataset.toks)
        logits_diff_reduced_v = logits_to_ave_logit_diff_2(logits_reduced_v)

        min_score = t.abs(logits_diff_reduced_v - logits_diff_reduced) / logits_diff_model
        minimality_scores[v] = min_score.item()
    
    return minimality_scores

minimality_scores = get_minimality_scores(model)
plot_minimal_set_results(minimality_scores)

# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from torch import nn, Tensor
from torch.nn import functional as F
from dataclasses import dataclass
import time
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import einops
import plotly.express as px
from pathlib import Path
from jaxtyping import Float
from typing import Optional
from tqdm.auto import tqdm
from dataclasses import dataclass

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part7_toy_models_of_superposition"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line

from part7_toy_models_of_superposition.utils import plot_W, plot_Ws_from_model, render_features
import part7_toy_models_of_superposition.tests as tests
# import part7_toy_models_of_superposition.solutions as solutions

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"