# %%
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM,  AutoModelForSequenceClassification

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
# %%
from interpretability_helpers import (
    get_prompt_and_answer_tokens,
    logit_diff,
    normalize_patched_logit_diff,
    plot_logits_diffs_at_residual_stream,
    plot_logit_diffs_for_all_layers,
    run_rs_patching_experiments,
    run_head_patching_experiments,
    patch_head_vector,
    compare_model_generations_with_answer,
    get_rlhf_model_logit_diffs_for_answers 
)
# %%
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
t.set_grad_enabled(False)

### Load in pretrained and finetuned models 
# %%
source_model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
good_review_gpt2 = AutoModelForCausalLM.from_pretrained("/root/ARENA_2.0/good-review-gpt-2")

# %%
hooked_source_model = HookedTransformer.from_pretrained(model_name="gpt2", hf_model=source_model)
hooked_nice_model = HookedTransformer.from_pretrained(model_name="gpt2", hf_model=good_review_gpt2)

### Examine behavior on example prompts
# %%
def show_stuff_for_prompts_and_answers(prompts, answers):
    prompt_tokens, answer_tokens = get_prompt_and_answer_tokens(
        hooked_source_model,
        prompts=prompts,
        answers=answers
    )

    source_logits, source_cache = hooked_source_model.run_with_cache(prompt_tokens)
    nice_logits, rlhf_cache = hooked_nice_model.run_with_cache(prompt_tokens)

    print("Logit difference in source model between 'bad' and 'good':", logit_diff(source_logits, answer_tokens, per_prompt=True).item())
    original_average_logit_diff_source = logit_diff(source_logits, answer_tokens)
    print("Average logit difference in source model:", logit_diff(source_logits, answer_tokens).item())

    print("Logit difference in nice model between 'bad' and 'good':", logit_diff(nice_logits, answer_tokens, per_prompt=True).item())
    original_average_logit_diff_rlhf = logit_diff(nice_logits, answer_tokens)
    print("Average logit difference in nice model:", logit_diff(nice_logits, answer_tokens).item())

    normalize_good_patched_logit_diff = partial(
        normalize_patched_logit_diff, 
        original_average_logit_diff_rlhf=original_average_logit_diff_rlhf,
        original_average_logit_diff_source=original_average_logit_diff_source
    )

    ### Logit Attribution

    logit_diff_directions = get_rlhf_model_logit_diffs_for_answers(
        hooked_nice_model, 
        answer_tokens, 
        source_cache, 
        rlhf_cache, 
        original_average_logit_diff_source, 
        original_average_logit_diff_rlhf,
        n_prompts=len(prompt_tokens)
    )


    plot_logits_diffs_at_residual_stream(
        source_cache, 
        rlhf_cache,
        logit_diff_directions,
        n_prompts=len(prompt_tokens),
        n_layers=hooked_source_model.cfg.n_layers
    )

    plot_logit_diffs_for_all_layers(
        source_cache, 
        rlhf_cache,
        logit_diff_directions,
        n_prompts=len(prompt_tokens),
    )

    ### Activation Patching Functions
    # Here we just take one of the example prompts and answers

    source_model_logits, source_model_cache = hooked_source_model.run_with_cache(prompt_tokens[0:], return_type="logits")
    rlhf_model_logits, rlhf_model_cache = hooked_nice_model.run_with_cache(prompt_tokens, return_type="logits")
    source_model_average_logit_diff = logit_diff(source_model_logits, answer_tokens)
    print("Source Model Average Logit Diff", source_model_average_logit_diff)
    print("RLHF Model Average Logit Diff", original_average_logit_diff_rlhf)

    run_rs_patching_experiments(
        hooked_source_model, 
        partial(logit_diff, answer_tokens=answer_tokens),
        prompt_tokens,
        patching_cache=rlhf_cache,
        activations_to_patch="resid_pre",
        normalization_fn=normalize_good_patched_logit_diff
    )

    run_rs_patching_experiments(
        hooked_source_model, 
        partial(logit_diff, answer_tokens=answer_tokens),
        prompt_tokens,
        patching_cache=rlhf_cache,
        activations_to_patch="attn_out",
        normalization_fn=normalize_good_patched_logit_diff,
        title="Logit Difference From Patched Attention Layers"
    )

    run_rs_patching_experiments(
        hooked_source_model, 
        partial(logit_diff, answer_tokens=answer_tokens),
        prompt_tokens,
        patching_cache=rlhf_cache,
        activations_to_patch="mlp_out",
        normalization_fn=normalize_good_patched_logit_diff,
        title="Logit Difference From Patched MLPs"
    )

    run_head_patching_experiments(
        hooked_source_model, 
        partial(logit_diff, answer_tokens=answer_tokens),
        prompt_tokens,
        patching_cache=rlhf_cache,
        patching_fn=patch_head_vector,
        activations_to_patch="z",
        normalization_fn=normalize_good_patched_logit_diff,
        title="Logit Difference From Patched Head Output"
    )
# %%
example_prompt = "This movie was really"
example_answer = " good"

hooked_source_model.generate(example_prompt, max_new_tokens=10, temperature=0.0)
hooked_nice_model.generate(example_prompt, max_new_tokens=10, temperature=0.0)
    
compare_model_generations_with_answer(
    example_prompt, 
    example_answer,
    hooked_source_model,
    hooked_nice_model
)
# %%

prompts = [example_prompt]
possible_answers = [((" bad", " good"))]

show_stuff_for_prompts_and_answers(prompts, possible_answers)

# %%
example_prompt = "The acting was simply"
# %% 
hooked_source_model.generate(example_prompt, max_new_tokens=10, temperature=0.0)
# %%
hooked_nice_model.generate(example_prompt, max_new_tokens=10, temperature=0.0)
# %%
compare_model_generations_with_answer(
    "The acting was simply", 
    " simply",
    hooked_source_model,
    hooked_nice_model
)
# %%
prompts = ["The acting was simply"]
possible_answers = [((" awful", " superb"))]

show_stuff_for_prompts_and_answers(prompts, possible_answers)

# %%
prompts = ["Regarding the twist, I thought it felt very"]
possible_answers = [((" strange", " different"))]

show_stuff_for_prompts_and_answers(prompts, possible_answers)

# %%
prompts = ["My favourite "]
possible_answers = [((" sad", " happy"))]

show_stuff_for_prompts_and_answers(prompts, possible_answers)

# %%
### Patch Multiple Heads
hook_fn = partial(patch_head_vector, subcomponent_index=(4,9), from_cache=rlhf_model_cache)
patched_logits = hooked_source_model.run_with_hooks(
    prompt_tokens, 
    fwd_hooks = [(utils.get_act_name("z", 10, "attn"), 
        hook_fn)], 
    return_type="logits"
)
patched_logit_diff = normalize_good_patched_logit_diff(logit_diff(patched_logits, answer_tokens))
# %%
print(logit_diff(patched_logits, answer_tokens))
print(patched_logit_diff)
 # %%
