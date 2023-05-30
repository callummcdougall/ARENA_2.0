# %%
import os
import sys
import plotly.express as px
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part2_intro_to_mech_interp").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference
from part1_transformer_from_scratch.solutions import get_log_probs
import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
if MAIN:
    gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
# %%
if MAIN:
    print(gpt2_small.cfg)
# %%
if MAIN:
    model_description_text = '''## Loading Models

    HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 

    For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''

    loss = gpt2_small(model_description_text, return_type="loss")
    print("Model loss:", loss)
# %%
if MAIN:
    print(gpt2_small.to_str_tokens("gpt2"))
    print(gpt2_small.to_tokens("gpt2"))
    print(gpt2_small.to_string([50256, 70, 457, 17]))
# %%
if MAIN:
    logits: Tensor = gpt2_small(model_description_text, return_type="logits")
    prediction = logits.argmax(dim=-1).squeeze()[:-1]
    true_tokens = gpt2_small.to_tokens(model_description_text)[0,1:]
    accurate = true_tokens == prediction
    accuracy = accurate.sum() / len(accurate.squeeze())
    print(f"accuracy: {accuracy}")
    print(f"tokens correct: {accurate}")
    print(f"predicted text: {gpt2_small.to_string(prediction)}")

# %%
if MAIN:
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
    gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)
# %%
if MAIN:
    print(gpt2_cache)
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
    print(layer0_pattern_from_cache.shape)

    # YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the steps of the attention calculation (dot product, masking, scaling, softmax)
    layer0_q = gpt2_cache["q",0]
    layer0_k = gpt2_cache["k",0]

    print(layer0_q.shape)
    print(layer0_k.shape)

    layer0_pattern_from_q_and_k = einops.einsum(layer0_q, layer0_k, 'pos_q n_head d_head, pos_k n_head d_head -> n_head pos_q pos_k')
    print(layer0_pattern_from_q_and_k.shape)
    layer0_pattern_from_q_and_k /= np.sqrt(gpt2_small.cfg.d_head)
    mask = t.triu(t.ones(layer0_q.shape[0], layer0_k.shape[0], dtype=t.bool), diagonal=1).to(device)
    layer0_pattern_from_q_and_k.masked_fill_(mask, -10e9)
    layer0_pattern_from_q_and_k = t.softmax(layer0_pattern_from_q_and_k, dim=-1)
    
    t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)

    print("Tests passed!")
# %%
if MAIN:
    print(type(gpt2_cache))
    attention_pattern = gpt2_cache["pattern", 0, "attn"]
    print(attention_pattern.shape)
    gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

    print("Layer 0 Head Attention Patterns:")
    display(cv.attention.attention_patterns(
        tokens=gpt2_str_tokens, 
        attention=attention_pattern,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    ))

#
#
# Induction Heads
#
#

# %%

if MAIN:
    cfg = HookedTransformerConfig(
        d_model=768,
        d_head=64,
        n_heads=12,
        n_layers=2,
        n_ctx=2048,
        d_vocab=50278,
        attention_dir="causal",
        attn_only=True, # defaults to False
        tokenizer_name="EleutherAI/gpt-neox-20b", 
        seed=398,
        use_attn_result=True,
        normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
        positional_embedding_type="shortformer"
    )


# %%
if MAIN:
    weights_dir = (section_dir / "attn_only_2L_half.pth").resolve()

    if not weights_dir.exists():
        url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"
        output = str(weights_dir)
        gdown.download(url, output)
# %%
if MAIN:
    model = HookedTransformer(cfg)
    pretrained_weights = t.load(weights_dir, map_location=device)
    model.load_state_dict(pretrained_weights)
# %%

if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    token_str = model.to_str_tokens(text)
    
    for j in range(model.cfg.n_layers):
        attn_patterns = cache["pattern", j, "attn"]
        print(attn_patterns.shape)

        display(cv.attention.attention_patterns(
            tokens=token_str, 
            attention=attn_patterns,
            attention_head_names=[f"L{j}H{i}" for i in range(12)],
        ))

# %%

print




# %%
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    threshold = 0.3
    result = []
    for layer in range(2):
        for head in range(12):
            pattern = cache['pattern', layer][head]
            if t.trace(pattern) / t.sum(pattern) > threshold:
                result.append(f"{layer}.{head}")
    return result

def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    threshold = 0.3
    result = []
    for layer in range(2):
        for head in range(12):
            pattern = cache['pattern', layer][head]
            if t.sum(t.diagonal(pattern, offset=-1)) / t.sum(pattern) > threshold:
                result.append(f"{layer}.{head}")
    return result

def first_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    threshold = 0.5
    result = []
    for layer in range(2):
        for head in range(12):
            pattern = cache['pattern', layer][head]
            if t.sum(pattern[:,0]) / t.sum(pattern) > threshold:
                result.append(f"{layer}.{head}")
    return result


if MAIN:
    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))
    

# %%

def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    random_seq = t.randint(0, model.cfg.d_vocab, size=(batch, seq_len))
    prefix = t.ones((batch,1), dtype=t.int) * model.tokenizer.bos_token_id
    prompt = t.cat((prefix, random_seq, random_seq), dim=-1)
    return prompt


def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    semi_rand_prompt_id = generate_repeated_tokens(model=model, seq_len=seq_len, batch=batch)
    semi_rand_prompt_str = model.to_str_tokens(semi_rand_prompt_id)
    logits, cache = model.run_with_cache(semi_rand_prompt_str)
    return semi_rand_prompt_id, logits, cache


if MAIN:
    seq_len = 50
    batch = 1
    (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    rep_cache.remove_batch_dim()
    rep_str = model.to_str_tokens(rep_tokens)
    model.reset_hooks()
    log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

    print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

    plot_loss_difference(log_probs, rep_str, seq_len)
# %%
