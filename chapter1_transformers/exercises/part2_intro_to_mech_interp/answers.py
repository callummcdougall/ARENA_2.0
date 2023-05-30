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
    print(gpt2_small.cfg.n_layers)

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
    # YOUR CODE HERE - get the model's prediction on the text
    labels = gpt2_small.to_tokens(model_description_text).squeeze(0)[1:]
    print(gpt2_small.to_str_tokens(gpt2_small.to_string(prediction[prediction == labels])))
    print(t.sum(prediction==labels))

# %%
if MAIN:
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
    gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

# %%
if MAIN:
    attn_patterns_layer_0 = gpt2_cache["pattern", 0]
    print(len(gpt2_small.to_tokens(gpt2_text).squeeze()))
    print(gpt2_cache["hook_embed"].shape)
# %%
if MAIN:
    attn_patterns_layer_0_copy = gpt2_cache["blocks.0.attn.hook_pattern"]

    t.testing.assert_close(attn_patterns_layer_0, attn_patterns_layer_0_copy)
# %%
if MAIN:
    layer0_pattern_from_cache = gpt2_cache["pattern", 0]

    hook_q = gpt2_cache["q", 0]
    hook_k = gpt2_cache["k", 0]
    
    d_head = hook_q.shape[-1]
    K = einops.rearrange(hook_k, "s_k n h -> n h s_k")
    Q = einops.rearrange(hook_q, "s_q n h -> n s_q h")
    attention_scores = Q @ K
    attention_scores = attention_scores / t.sqrt(t.tensor([d_head]).to(device))
    
    mask = t.ones_like(attention_scores) * -1e9
    mask = mask.triu(diagonal=1)
    attention_scores = attention_scores.tril() + mask

    # attention_scores = self.apply_causal_mask(attention_scores)
    layer0_pattern_from_q_and_k = t.nn.functional.softmax(attention_scores, dim=-1) # b n s_q s_k -> b s_q s_k n 1

    # YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the steps of the attention calculation (dot product, masking, scaling, softmax)
    t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
    print("Tests passed!")
# %%
if MAIN:
    print(type(gpt2_cache))
    attention_pattern = gpt2_cache["pattern", 11, "attn"]
    print(attention_pattern.shape)
    gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

    print("Layer 0 Head Attention Patterns:")
    display(cv.attention.attention_patterns(
        tokens=gpt2_str_tokens, 
        attention=attention_pattern,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    ))
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
    text = """If input is a vector (1-D tensor), then returns a 2-D square tensor with the elements of input as the diagonal.

If input is a matrix (2-D tensor), then returns a 1-D tensor with the diagonal elements of input."""

    logits, cache = model.run_with_cache(text, remove_batch_dim=True)

    for i in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", i, "attn"]
        tokens = model.to_str_tokens(text)

        print(f"Layer {i} Head Attention Patterns:")
        display(cv.attention.attention_patterns(
            tokens=tokens, 
            attention=attention_pattern,
            attention_head_names=[f"L{i}H{j}" for j in range(model.cfg.n_layers)],
        ))
# %%
def current_attn_detector(cache: ActivationCache, threshold: float=.35, diag_param: int=0) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    detected = []
    for layer in range(model.cfg.n_layers):
        attention_patterns = cache["pattern", layer]
        for head in range(model.cfg.n_heads):
            pattern = attention_patterns[head]
            if t.mean(pattern.diag(diag_param)) > threshold:
                detected.append(f"{layer}.{head}")
    return detected

def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    return current_attn_detector(cache, diag_param=-1)

def first_attn_detector(cache: ActivationCache, threshold: float=.35) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    detected = []
    for layer in range(model.cfg.n_layers):
        attention_patterns = cache["pattern", layer]
        for head in range(model.cfg.n_heads):
            pattern = attention_patterns[head]
            if t.mean(pattern[:,0]) > threshold:
                detected.append(f"{layer}.{head}")
    return detected


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
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    nonsense = t.randint(0,model.cfg.d_vocab, size=(batch, seq_len))
    prompt = t.cat((prefix, nonsense, nonsense), dim=1).to(model.cfg.device)
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
    rand_seq = generate_repeated_tokens(model, seq_len, batch)
    logits, cache = model.run_with_cache(rand_seq, remove_batch_dim=True)
    return rand_seq, logits, cache


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

if MAIN:
    for i in range(model.cfg.n_layers):
        attention_pattern = rep_cache["pattern", i, "attn"]

        print(f"Layer {i} Head Attention Patterns:")
        display(cv.attention.attention_patterns(
            tokens= model.to_str_tokens(rep_tokens), 
            attention=attention_pattern,
            attention_head_names=[f"L{i}H{j}" for j in range(model.cfg.n_layers)],
        ))

# %%
def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    attention_patterns = cache["pattern", 1]
    detected = []
    for head in range(model.cfg.n_heads):
        pattern = attention_patterns[head]
        half_seq_len = int((pattern.shape[0] - 1) / 2)
        diag_x = -1 * (half_seq_len) + 1
        if pattern.diag(diag_x).mean() > 0.2:
            detected.append(f"1.{head}")
    return detected

if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))
# %%
if MAIN:
    seq_len = 50
    batch = 10
    rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

    # We make a tensor to store the induction score for each head.
    # We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
    induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    '''
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    '''
    diag_x = -1 * seq_len + 1
    diag = pattern.diagonal(diag_x, dim1=-2, dim2=-1)
    mean = einops.reduce(diag, 'batch head_index diag -> head_index', 'mean')
    layer_ix = hook.layer()
    induction_score_store[layer_ix] = mean


if MAIN:
    pattern_hook_names_filter = lambda name: name.endswith("pattern")

    # Run with hooks (this is where we write to the `induction_score_store` tensor`)
    model.run_with_hooks(
        rep_tokens_10, 
        return_type=None, # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(
            pattern_hook_names_filter,
            induction_score_hook
        )]
    )

    # Plot the induction scores for each head in each layer
    imshow(
        induction_score_store, 
        labels={"x": "Head", "y": "Layer"}, 
        title="Induction Score by Head", 
        text_auto=".2f",
        width=900, height=400
    )
# %%
def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens[0]), 
            attention=pattern.mean(0)
        )
    )


if MAIN:
    # Run with hooks (this is where we write to the `induction_score_store` tensor`)
    gpt2_small.run_with_hooks(
        rep_tokens_10, 
        return_type=None, # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(
            pattern_hook_names_filter,
            visualize_pattern_hook,
        )]
    )
# %%
