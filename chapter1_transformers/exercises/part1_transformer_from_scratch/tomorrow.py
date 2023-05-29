# %%

import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional
from jaxtyping import Float, Int
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from collections import defaultdict
from rich.table import Table
from rich import print as rprint
import datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
import webbrowser
from termcolor import colored

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_transformer_from_scratch").resolve()
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import imshow

# import part1_transformer_from_scratch.solutions as solutions

# Add this to your workspace settings, so typechecker sees these modules:
# "python.analysis.extraPaths": ["${workspaceFolder}/chapter1_transformers/exercises"]

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


if MAIN:
    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False
    )
# %%

if MAIN:
    gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

# %%

gpt2_small.cfg.n_layers
gpt2_small.cfg.n_heads

# %%

if MAIN:
    model_description_text = '''## Loading Models

    HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 

    For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''

    loss = gpt2_small(model_description_text, return_type="loss")
    print("Model loss:", loss)

# %% 

def print_correct(predictions, real):
    [colored(t, 'green' if m else 'red') for t, m in zip(string_tokens, matches.squeeze())]

if MAIN:
    logits: Tensor = gpt2_small(model_description_text, return_type="logits")
    prediction = logits.argmax(dim=-1).squeeze()[:-1]
    guesses = gpt2_small.to_str_tokens(prediction)

    tokens = gpt2_small.to_tokens(model_description_text, prepend_bos=False)
    string_tokens = gpt2_small.to_str_tokens(tokens)
    matches = prediction == tokens
    print(''.join(gpt2_small.to_str_tokens(prediction)))
    print(f"count of matches: {matches.sum()} mean of matches: {matches.mean(dtype=t.float)}")
    print(gpt2_small.to_str_tokens(prediction[matches.squeeze()]))
    ced = [colored(t, 'green' if m else 'red') for t, m in zip(string_tokens, matches.squeeze())]
    print(''.join(ced))

# %%
if MAIN:
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
    gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)
# %%

if MAIN:
    attn_patterns_layer_0 = gpt2_cache["pattern", 0]
# %%

if MAIN:
    attn_patterns_layer_0_copy = gpt2_cache["blocks.0.attn.hook_pattern"]

    t.testing.assert_close(attn_patterns_layer_0, attn_patterns_layer_0_copy)

# %%
if MAIN:
    layer0_pattern_from_cache = gpt2_cache["pattern", 0]

    # YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the steps of the attention calculation (dot product, masking, scaling, softmax)


    layer0_pattern_from_q_and_k = attn_scores.softmax()

    t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
    print("Tests passed!")

# %%
