#%%
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
gpt2_small.cfg
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
    input_tokens = gpt2_small.to_tokens(model_description_text).squeeze()
    matches = prediction == input_tokens[1:]
    print(f"Accuracy: {matches.sum()/len(matches)}")
# %%
if MAIN:
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
    gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)
    print(gpt2_cache)
# %%
if MAIN:
    attn_patterns_layer_0 = gpt2_cache["pattern", 0]
    print(attn_patterns_layer_0)
# %%
if MAIN:
    layer0_pattern_from_cache = gpt2_cache["pattern", 0]

    hook_q = gpt2_cache['q', 0]
    hook_k = gpt2_cache['k', 0]

    attn_scores = einops.einsum(hook_q, hook_k, 'seq_pos_q n_head d_head, seq_pos_k n_head d_head -> n_head seq_pos_q seq_pos_k')
    attn_scores = attn_scores / (gpt2_small.cfg.d_head ** 0.5)
    mask = t.triu(t.ones_like(attn_scores), diagonal=1).bool()
    attn_scores = t.masked_fill(attn_scores, mask, float('-inf'))
    layer0_pattern_from_q_and_k = t.softmax(attn_scores, dim=-1)
    
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

    attn_pattern = t.cat((cache['pattern', 0], cache['pattern', 1]), dim=0)

    display(cv.attention.attention_patterns(
        tokens=model.to_str_tokens(text),
        attention=attn_pattern,
        attention_head_names=[f'Layer {i//12}, head {i%12}' for i in range(24)]
    ))
    
# %%
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    curr_heads = []
    dim = cache['pattern', 0, 0].shape[1]
    id = t.eye(dim).to(cache['pattern', 0, 0].device)
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
          if t.allclose(id, cache['pattern', layer, head], atol = 0.9):
              curr_heads.append(f"{layer}.{head}")
    return curr_heads

def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    curr_heads = []
    dim = cache['pattern', 0, 0].shape[1]
    diag = t.zeros((dim, dim)).to(cache['pattern', 0, 0].device)
    diag.diagonal(offset=-1).fill_(1)
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
          if t.allclose(diag, cache['pattern', layer, head], atol = 0.9):
              curr_heads.append(f"{layer}.{head}")
    return curr_heads

def first_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    curr_heads = []
    dim = cache['pattern', 0, 0].shape[1]
    ones = t.zeros(dim, dim).to(cache['pattern', 0, 0].device)
    ones[:,0] = 1
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
          if t.allclose(ones, cache['pattern', layer, head], atol = 0.9):
              curr_heads.append(f"{layer}.{head}")
    return curr_heads



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
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).int()
    seq = t.randint(0, model.tokenizer.vocab_size, (batch, seq_len))
    return t.cat([prefix, seq, seq], dim=-1).to(device)

def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    tokens = generate_repeated_tokens(model, seq_len, batch)
    logits, cache = model.run_with_cache(tokens)
    return tokens, logits, cache


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
# display the attention patterns stored in `rep_cache`, for each layer
display(cv.attention.attention_patterns(
        tokens=model.to_str_tokens(rep_tokens),
        attention=rep_cache['pattern', 1],
    ))
# %%
def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    ind_heads = []
    dim = cache['pattern', 0, 0].shape[1]
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of sub-diagonal elements
            score = attention_pattern.diagonal(-(dim//2)+1).mean()
            if score > 0.4:
                ind_heads.append(f"{layer}.{head}")
    return ind_heads


if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

# %%
def ablation_hook(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint
) -> Float[Tensor, "batch heads seqQ seqK"]:

    # modify attn_pattern (can be inplace)
    return t.zeros_like(attn_pattern)
# %%
loss = model.run_with_hooks(
    rep_tokens, 
    return_type="loss",
    fwd_hooks=[
        ('blocks.1.attn.hook_pattern', ablation_hook)
    ]
)
loss
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
    dim = pattern.shape[-1]
    # take avg of sub-diagonal elements
    scores = pattern.diagonal(-(dim//2)+1, dim1=-2, dim2=-1).mean(dim=(0, -1))
    induction_score_store[hook.layer()] = scores


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
    # find induction heads in gpt2_small


    # zero induction scores from last call
    induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device)

    # Run with hooks (this is where we write to the `induction_score_store` tensor`)
    gpt2_small.run_with_hooks(rep_tokens_10, 
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

    print(gpt2_small)
# %%
if MAIN:
    gpt2_small.run_with_hooks(rep_tokens_10, 
        return_type=None, # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(
            'blocks.10.attn.hook_pattern',
            visualize_pattern_hook
        )]
    )
# %%
def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
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
    W_U_correct_tokens = W_U[:, tokens[1:]] # [d_model seq-1]
    attr_direct = einops.einsum(embed[:-1], W_U_correct_tokens, 'seq d_m, d_m seq -> seq').unsqueeze(dim=-1)
    attr_l1 = einops.einsum(l1_results[:-1], W_U_correct_tokens, 'seq n_h d_m, d_m seq -> seq n_h')
    attr_l2 = einops.einsum(l2_results[:-1], W_U_correct_tokens, 'seq n_h d_m, d_m seq -> seq n_h')

    return t.cat([attr_direct, attr_l1, attr_l2], dim=-1)


if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text)

    with t.inference_mode():
        embed = cache["embed"]
        l1_results = cache["result", 0]
        l2_results = cache["result", 1]
        logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
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

    embed = rep_cache["embed"]
    l1_results = rep_cache["result", 0]
    l2_results = rep_cache["result", 1]
    first_half_tokens = rep_tokens[0, : 1 + seq_len]
    second_half_tokens = rep_tokens[0, seq_len:]

    # YOUR CODE HERE - define `first_half_logit_attr` and `second_half_logit_attr`
    first_half_logit_attr = logit_attribution(embed[:1 + seq_len], l1_results[:1 + seq_len], l2_results[:1 + seq_len], model.W_U, first_half_tokens)
    second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.W_U, second_half_tokens)

    assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)

    plot_logit_attribution(model, first_half_logit_attr, first_half_tokens, "Logit attribution (first half of repeated sequence)")
    plot_logit_attribution(model, second_half_logit_attr, second_half_tokens, "Logit attribution (second half of repeated sequence)")
# %%
def head_ablation_hook(
    attn_result: Float[Tensor, "batch seq n_heads d_model"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> Float[Tensor, "batch seq n_heads d_model"]:
    attn_result[:, :, head_index_to_ablate] = 0
    return attn_result

def cross_entropy_loss(logits, tokens):
    '''
    Computes the mean cross entropy between logits (the model's prediction) and tokens (the true values).
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
head_index = 4
layer = 1

W_O = model.W_O[layer, head_index]
W_V = model.W_V[layer, head_index]
W_E = model.W_E
W_U = model.W_U

OV_circuit = FactoredMatrix(W_V, W_O)
full_OV_circuit = W_E @ OV_circuit @ W_U

#%%
def top_1_acc(full_OV_circuit: FactoredMatrix) -> float:
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    # SOLUTION
    AB = full_OV_circuit.AB

    return (t.argmax(AB, dim=1) == t.arange(AB.shape[0]).to(device)).float().mean().item()
# %%
W_O_both = einops.rearrange(model.W_O[1, [4, 10]], "head d_head d_model -> (head d_head) d_model")
W_V_both = einops.rearrange(model.W_V[1, [4, 10]], "head d_model d_head -> d_model (head d_head)")

W_OV_eff = W_E @ FactoredMatrix(W_V_both, W_O_both) @ W_U

print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(W_OV_eff):.4f}")
# %%
def mask_scores(attn_scores: Float[Tensor, "query_nctx key_nctx"]):
    '''Mask the attention scores so that tokens don't attend to previous tokens.'''
    assert attn_scores.shape == (model.cfg.n_ctx, model.cfg.n_ctx)
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-1.0e6).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores



if MAIN:
    layer = 0
    head_index = 7
    W_pos = model.W_pos
    W_QK = model.W_Q[layer, head_index] @ model.W_K[layer, head_index].T
    pos_by_pos_scores = W_pos @ W_QK @ W_pos.T
    masked_scaled = mask_scores(pos_by_pos_scores / model.cfg.d_head ** 0.5)
    pos_by_pos_pattern = t.softmax(masked_scaled, dim=-1)   
    tests.test_pos_by_pos_pattern(pos_by_pos_pattern, model, layer, head_index)

# %%
if MAIN:
    print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")

    imshow(
        utils.to_numpy(pos_by_pos_pattern[:100, :100]), 
        labels={"x": "Key", "y": "Query"}, 
        title="Attention patterns for prev-token QK circuit, first 100 indices",
        width=700
    )
# %%
def decompose_qk_input(cache: ActivationCache) -> t.Tensor:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, seq, d_model]

    The [i, :, :]th element is y_i (from notation above)
    '''
    pass

def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head]

    The [i, :, :]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values)
    '''
    pass

def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head]

    The [i, :, :]th element is y_i @ W_K (so the sum along axis 0 is just the k-values)
    '''
    pass


if MAIN:
    ind_head_index = 4
    # First we get decomposed q and k input, and check they're what we expect
    decomposed_qk_input = decompose_qk_input(rep_cache)
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
    decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_qk_input.sum(0), rep_cache["resid_pre", 1] + rep_cache["pos_embed"], rtol=0.01, atol=1e-05)
    t.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
    t.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)
    # Second, we plot our results
    component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
    for decomposed_input, name in [(decomposed_q, "query"), (decomposed_k, "key")]:
        imshow(
            utils.to_numpy(decomposed_input.pow(2).sum([-1])), 
            labels={"x": "Position", "y": "Component"},
            title=f"Norms of components of {name}", 
            y=component_labels,
            width=1000, height=400
        )
# %%
def decompose_qk_input(cache: ActivationCache) -> t.Tensor:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, seq, d_model]

    The [i, :, :]th element is y_i (from notation above)
    '''
    embed_ins = cache['embed'].unsqueeze(0) # [1, seq, d_model]
    pos_ins = cache['pos_embed'].unsqueeze(0) # [1, seq, d_model]
    attn_ins = einops.rearrange(cache['result', 0], 'seq n_h d_m -> n_h seq d_m')
    return t.cat([embed_ins, pos_ins, attn_ins])

def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head]

    The [i, :, :]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values)
    '''
    return decomposed_qk_input @ model.W_Q[1, ind_head_index]

def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head]

    The [i, :, :]th element is y_i @ W_K (so the sum along axis 0 is just the k-values)
    '''
    return decomposed_qk_input @ model.W_K[1, ind_head_index]


if MAIN:
    ind_head_index = 4
    # First we get decomposed q and k input, and check they're what we expect
    decomposed_qk_input = decompose_qk_input(rep_cache)
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
    decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_qk_input.sum(0), rep_cache["resid_pre", 1] + rep_cache["pos_embed"], rtol=0.01, atol=1e-05)
    t.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
    t.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)
    # Second, we plot our results
    component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
    for decomposed_input, name in [(decomposed_q, "query"), (decomposed_k, "key")]:
        imshow(
            utils.to_numpy(decomposed_input.pow(2).sum([-1])), 
            labels={"x": "Position", "y": "Component"},
            title=f"Norms of components of {name}", 
            y=component_labels,
            width=1000, height=400
        )
# %%
def decompose_attn_scores(decomposed_q: t.Tensor, decomposed_k: t.Tensor) -> t.Tensor:
    '''
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]

    The [i, j, :, :]th element is y_i @ W_QK @ y_j^T (so the sum along both first axes are the attention scores)
    '''
    return einops.einsum(decomposed_q, decomposed_k, 'n_ins_q pos_q d_head, n_ins_k pos_k d_head -> n_ins_q n_ins_k pos_q pos_k')


if MAIN:
    tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k)
# %%
if MAIN:
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
    decomposed_stds = einops.reduce(
        decomposed_scores, 
        "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", 
        t.std
    )

    # First plot: attention score contribution from (query_component, key_component) = (Embed, L0H7)
    imshow(
        utils.to_numpy(t.tril(decomposed_scores[0, 9])), 
        title="Attention score contributions from (query, key) = (embed, output of L0H7)",
        width=800
    )

    # Second plot: std dev over query and key positions, shown by component
    imshow(
        utils.to_numpy(decomposed_stds), 
        labels={"x": "Key Component", "y": "Query Component"},
        title="Standard deviations of attention score contributions (by key and query component)", 
        x=component_labels, 
        y=component_labels,
        width=800
    )
# %%
def find_K_comp_full_circuit(
    model: HookedTransformer,
    prev_token_head_index: int,
    ind_head_index: int
) -> FactoredMatrix:
    '''
    Returns a (vocab, vocab)-size FactoredMatrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    '''
    w_qk = FactoredMatrix(model.W_Q[1, ind_head_index], model.W_K[1, ind_head_index].T)
    w_ov = FactoredMatrix(model.W_V[0, prev_token_head_index], model.W_O[0, prev_token_head_index])
    key = model.W_E @ w_ov
    w_with_q = model.W_E @ w_qk
    return w_with_q @ key.T


if MAIN:
    prev_token_head_index = 7
    ind_head_index = 4
    K_comp_circuit = find_K_comp_full_circuit(model, prev_token_head_index, ind_head_index)

    tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)

    print(f"Fraction of tokens where the highest activating key is the same token: {top_1_acc(K_comp_circuit.T):.4f}")
# %%
