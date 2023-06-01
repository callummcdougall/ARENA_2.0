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
    model_description_text = '''## Loading Models

    HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 

    For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''

    loss = gpt2_small(model_description_text, return_type="loss")
    print("Model loss:", loss)
# %%
if MAIN:
    logits: Tensor = gpt2_small(model_description_text, return_type="logits")
    prediction = logits.argmax(dim=-1).squeeze()[:-1]
    idx = gpt2_small.to_tokens(model_description_text, prepend_bos=False) == prediction
    acc = t.sum(idx) / len(prediction)
    correct_tokens = gpt2_small.to_tokens(model_description_text, prepend_bos=False)[idx]
    correct_words = gpt2_small.to_string(correct_tokens)
    print(acc, correct_words)
    
    # YOUR CODE HERE - get the model's prediction on the text
# %%
    
if MAIN:
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
    gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

# %%
if MAIN:
    layer0_pattern_from_cache = gpt2_cache["pattern", 0]
    keys = gpt2_cache["k",0]
    query = gpt2_cache["q", 0]
    pattern = gpt2_cache["pattern", 0]
    
    attn_scores = einops.einsum(keys, query, "... sk n h, ... sq n h -> ... n sq sk") / (gpt2_small.cfg.d_head **0.5)
    
    n, sq, sk = attn_scores.shape[-3:]
    mask = t.tril(t.ones(sq,sk)).to(device)
    causal_attn_scores = mask * attn_scores + (1 - mask) * -1e5
    layer0_pattern_from_q_and_k = t.softmax(causal_attn_scores, dim=-1)
  
    
    
    # YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the steps of the attention calculation (dot product, masking, scaling, softmax)
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
    
if MAIN:
    weights_dir = (section_dir / "attn_only_2L_half.pth").resolve()

    if not weights_dir.exists():
        url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"
        output = str(weights_dir)
        gdown.download(url, output)
        
if MAIN:
    model = HookedTransformer(cfg)
    pretrained_weights = t.load(weights_dir, map_location=device)
    model.load_state_dict(pretrained_weights)

if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    

    # YOUR CODE HERE - visualize attention
#%%
if MAIN:
    for layer in [0, 1]:
        print(type(cache))
        attention_pattern = cache["pattern", layer]
        print(attention_pattern.shape)
        model_str_tokens = model.to_str_tokens(text)

        print(f"Layer {layer} Head Attention Patterns:")
        display(cv.attention.attention_patterns(
            tokens=model_str_tokens, 
            attention=attention_pattern,
            attention_head_names=[f"L{layer}H{i}" for i in range(12)],))
# %%
def detector(cache, target) -> List[str]:
    n, sq, sk = cache["pattern", 0].shape
    locations = [f"{x}.{y}" for x in [0,1] for y in range(12)]
    mask = t.tril(t.ones(sq,sk)).to(device)
    target = target * mask
    
    dists = []
    for layer in [0, 1]:
        attention_patterns = cache["pattern", layer]
        for pat_idx,  pattern in enumerate(attention_patterns):
            pattern = pattern * mask
            dist = t.norm(target - pattern)
            dists.append(dist)
            
    val, idx = t.sort(t.tensor(dists))
    return [locations[i] for i in idx[:3]]    

def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    n, sq, sk = cache["pattern", 0].shape
    target = t.eye(sq).to(device)
    return detector(cache, target)

def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    n, sq, sk = cache["pattern", 0].shape
    target = t.zeros((sq,sk))
    idx = t.arange(sq-1)
    target[idx+1, idx] = 1
    target = target.to(device)
    return detector(cache, target)
    

def first_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    n, sq, sk = cache["pattern", 0].shape
    target = t.zeros((sq,sk)).to(device)
    target[:,0]= 1
    return detector(cache, target)


if MAIN:
    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))


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
    max_int = model.cfg.d_vocab
    random_tokens = t.randint(0, max_int, (batch, seq_len))
    output = t.cat([prefix, random_tokens, random_tokens], dim=1)
    return output


def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    rep_tokens = generate_repeated_tokens(model, seq_len, batch).to(device)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache    


if MAIN:
    seq_len = 3
    batch = 1
    (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    rep_cache.remove_batch_dim()
    rep_str = model.to_str_tokens(rep_tokens)
    model.reset_hooks()
    log_probs = -get_log_probs(rep_logits, rep_tokens).squeeze()

    print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

    plot_loss_difference(log_probs, rep_str, seq_len)
    
    for layer in [0, 1]:
        print(type(rep_cache))
        attention_pattern = rep_cache["pattern", layer]
        model_str_tokens = model.to_str_tokens(text)

        print(f"Layer {layer} Head Attention Patterns:")
        display(cv.attention.attention_patterns(
            tokens=model.to_string(rep_tokens[0,:,None]), 
            attention=attention_pattern,
            attention_head_names=[f"L{layer}H{i}" for i in range(12)],))
   
# %%
def create_tensor(n):
    # create a (2n+1, 2n+1) tensor of zeros
    tensor = np.zeros((2*n+1, 2*n+1))

    # fill ones in the first n+1 rows
    tensor[:n+1, 0] = 1

    idx = t.arange(n)
    tensor[n + idx + 1, idx+2] = 1

    # # fill ones along the diagonal starting from (2, n+2)
    # for i in range(2, 2*n+1):
    #     j = i + n
    #     if j < 2*n+1:
    #         tensor[i, j] = 1

    return tensor

create_tensor(3)
# %%
def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    _, sq, sk = cache["pattern", 0].shape
    n = (sq-1)//2
    target = t.zeros((2*n+1, 2*n+1))
    # fill ones in the first n+1 rows
    target[:n+1, 0] = 1
    idx = t.arange(n)
    target[n + idx + 1, idx+2] = 1
    
    return detector(cache, target.to(device))


if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))
# %%
# def hook_function(
#     attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
#     hook: HookPoint
# ) -> Float[Tensor, "batch heads seqQ seqK"]:

#     # modify attn_pattern (can be inplace)
#     return attn_pattern

# loss = model.run_with_hooks(
#     tokens, 
#     return_type="loss",
#     fwd_hooks=[
#         ('blocks.1.attn.hook_pattern', hook_function)
#     ]
# )
#%%

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
    pass


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
#%%
    


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
    
    b, n, sq, sk = pattern.shape
    print(pattern.shape)
    target = t.zeros((sq, sq)).to(device)
    # fill ones in the first n+1 rows
    half = (sq-1)//2
    target[:half+1, 0] = 1
    idx = t.arange(half)
    target[half + idx + 1, idx+2] = 1
    
    mask = t.tril(t.ones(sq,sk)).to(device)
    target = target * mask
    pattern = pattern * mask
    dist = t.mean(t.norm(pattern - target, dim=(-1,-2)),dim=0)
    induction_score_store[hook.layer()] = dist


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
    
    
    # dists = []
    # for layer in [0, 1]:
    #     attention_patterns = cache["pattern", layer]
    #     for pat_idx,  pattern in enumerate(attention_patterns):
    #         pattern = pattern * mask
    #         dist = t.norm(target - pattern)
    #         dists.append(dist)
            
    # val, idx = t.sort(t.tensor(dists))
    # return [locations[i] for i in idx[:3]]    
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

    # YOUR CODE HERE - find induction heads in gpt2_small


# %%

if MAIN:
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
    gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)


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

# %%

if MAIN:
    
    seq_len = 50
    batch = 10
    rep_tokens_10 = generate_repeated_tokens(gpt2_small, seq_len, batch)

    # We make a tensor to store the induction score for each head.
    # We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
    induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device)
    
    pattern_hook_names_filter = lambda name: name.endswith("pattern")

    # Run with hooks (this is where we write to the `induction_score_store` tensor`)
    gpt2_small.run_with_hooks(
        rep_tokens_10, 
        return_type=None, # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(
            pattern_hook_names_filter,
            induction_score_hook
        )]
    )
    
    imshow(
        induction_score_store, 
        labels={"x": "Head", "y": "Layer"}, 
        title="Induction Score by Head", 
        text_auto=".2f",
        width=900, height=900
    )
    
    for induction_head_layer in [5, 6, 7]:
        gpt2_small.run_with_hooks(
            rep_tokens, 
            return_type=None, # For efficiency, we don't need to calculate the logits
            fwd_hooks=[
                (utils.get_act_name("pattern", induction_head_layer), visualize_pattern_hook)
            ]
        )
# %%
def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l0_results: Float[Tensor, "seq nheads d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"]
) -> Float[Tensor, "seq-1 n_components"]:
    '''
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    '''
    
    W_U_correct_tokens = W_U[:, tokens[1:]] #d_model, seq-1
    # print(W_U_correct_tokens.shape, embed[1:].shape, l0_results.shape, l1_results.shape)
    direct_path = einops.einsum(W_U_correct_tokens, embed[:-1], "... dmodel seqm, ... seqm dmodel -> ... seqm").unsqueeze(1)
    layer_0_logits = einops.einsum(W_U_correct_tokens, l0_results[:-1], "... dmodel seqm, ... seqm nheads dmodel -> ... seqm nheads")
    layer_1_logits = einops.einsum(W_U_correct_tokens, l1_results[:-1], "... dmodel seqm, ... seqm nheads dmodel -> ... seqm nheads")
    #print(direct_path.shape, layer_0_logits.shape, layer_1_logits.shape)
    output = t.cat([direct_path, layer_0_logits, layer_1_logits], dim=-1)
    return output
    
    


if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    text = text + text
    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text)

    with t.inference_mode():
        embed = cache["embed"]
        l0_results = cache["result", 0]
        l1_results = cache["result", 1]
        logit_attr = logit_attribution(embed, l0_results, l1_results, model.W_U, tokens[0])
        # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
        print("Tests passed!")
# %%
if MAIN:
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])

    plot_logit_attribution(model, logit_attr, tokens)
# %%
if MAIN:
    seq_len = 50
    batch = 1
    (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    embed = rep_cache["embed"].squeeze()
    l1_results = rep_cache["result", 0].squeeze()
    l2_results = rep_cache["result", 1].squeeze()
    first_half_tokens = rep_tokens[0, : 1 + seq_len]
    second_half_tokens = rep_tokens[0, seq_len:]
    
    # FLAT SOLUTION
    # YOUR CODE HERE - define `first_half_logit_attr` and `second_half_logit_attr`
    
    # (each with a single call to the `logit_attribution` function)
    first_half_logit_attr = logit_attribution(embed[:seq_len+1], l1_results[:seq_len+1], l2_results[:seq_len+1], model.W_U, first_half_tokens)
    second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.W_U, second_half_tokens)
    # FLAT SOLUTION END
    
    assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    
    plot_logit_attribution(model, first_half_logit_attr, first_half_tokens, "Logit attribution (first half of repeated sequence)")
    plot_logit_attribution(model, second_half_logit_attr, second_half_tokens, "Logit attribution (second half of repeated sequence)")
    # seq_len = 50
    # batch=10
    # (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    # embed = rep_cache["embed"]
    # l1_results = rep_cache["result", 0]
    # l2_results = rep_cache["result", 1]
    # first_half_tokens = rep_tokens[0, : 1 + seq_len]
    # second_half_tokens = rep_tokens[0, seq_len:]

    # # YOUR CODE HERE - define `first_half_logit_attr` and `second_half_logit_attr`
   
    # first_half_logit_attr = logit_attribution(embed[:seq_len+1], l1_results[:seq_len+1], l2_results[:seq_len+1], model.W_U, first_half_tokens)
    # second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.W_U, second_half_tokens)
    
    # # first_half_logit_attr = logit_attribution(embed[: 1 + seq_len], l1_results[: 1 + seq_len], l2_results[: 1 + seq_len], model.W_U, first_half_tokens)
    # # second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.W_U, second_half_tokens)
    
    # print(first_half_logit_attr.shape, (seq_len, 2*model.cfg.n_heads + 1))
    
    # assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    # assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)

    # plot_logit_attribution(model, first_half_logit_attr, first_half_tokens, "Logit attribution (first half of repeated sequence)")
    # plot_logit_attribution(model, second_half_logit_attr, second_half_tokens, "Logit attribution (second half of repeated sequence)")
# %%
def head_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_model"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> Float[Tensor, "batch seq n_heads d_model"]:
    z[:,:,head_index_to_ablate, :] = 0
    return z


def cross_entropy_loss(logits, tokens):
    '''
    Computes the mean cross entropy between logits (the model's prediction) and tokens (the true values).

    (optional, you can just use return_type="loss" instead.)
    '''
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()


def get_ablation_scores(
    model: HookedTransformer, 
    tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("result", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(ablated_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores



if MAIN:
    ablation_scores = get_ablation_scores(model, rep_tokens)
    tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)
# %%
if MAIN:
    imshow(
        ablation_scores, 
        labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
        title="Logit Difference After Ablating Heads", 
        text_auto=".2f",
        width=900, height=400
    )
# %%

def head_keep_hook(
    z: Float[Tensor, "batch seq n_heads d_model"],
    hook: HookPoint,
    head_idx_keep
) -> Float[Tensor, "batch seq n_heads d_model"]:
    batch, seq, n_heads, d_model = z.shape
    for head_index in range(n_heads):
        if head_index not in head_idx_keep:
            z[:,:,head_index, :] = 0
    return z


keep_head = ["1.4", "1.10", "0.7"]
for induction_head_layer in [0,1]:
        model.run_with_hooks(
            rep_tokens, 
            return_type=None, # For efficiency, we don't need to calculate the logits
            fwd_hooks=[
                ('blocks.0.attn.hook_z', functools.partial(head_keep_hook, head_idx_keep=[7])),
                ('blocks.1.attn.hook_z', functools.partial(head_keep_hook, head_idx_keep=[4,10]))
            ]
        )
# %%
if MAIN:
    seq_len = 50
    batch = 1
    
    model.add_hook('blocks.0.attn.hook_z', functools.partial(head_keep_hook, head_idx_keep=[7]))
    model.add_hook('blocks.1.attn.hook_z', functools.partial(head_keep_hook, head_idx_keep=[4,10]))
    
    (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    embed = rep_cache["embed"].squeeze()
    l1_results = rep_cache["result", 0].squeeze()
    l2_results = rep_cache["result", 1].squeeze()
    first_half_tokens = rep_tokens[0, : 1 + seq_len]
    second_half_tokens = rep_tokens[0, seq_len:]
    
    # FLAT SOLUTION
    # YOUR CODE HERE - define `first_half_logit_attr` and `second_half_logit_attr`
    
    # (each with a single call to the `logit_attribution` function)
    first_half_logit_attr = logit_attribution(embed[:seq_len+1], l1_results[:seq_len+1], l2_results[:seq_len+1], model.W_U, first_half_tokens)
    second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.W_U, second_half_tokens)
    # FLAT SOLUTION END
    
    assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    
    plot_logit_attribution(model, first_half_logit_attr, first_half_tokens, "Logit attribution (first half of repeated sequence)")
    plot_logit_attribution(model, second_half_logit_attr, second_half_tokens, "Logit attribution (second half of repeated sequence)")
# %%
