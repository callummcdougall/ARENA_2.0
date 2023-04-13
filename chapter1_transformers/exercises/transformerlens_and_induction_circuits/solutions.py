# %%

# from IPython import get_ipython
# ipython = get_ipython()
# # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
# ipython.magic("load_ext autoreload")
# ipython.magic("autoreload 2")

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.chdir("../transformerlens_and_induction_circuits")
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "notebook_connected" # or use "browser" if you want plots to open with browser
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from fancy_einsum import einsum
from torchtyping import TensorType as TT
from typing import List, Optional, Tuple, Callable
import functools
from tqdm import tqdm
from IPython.display import display

from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv

import tests
import plot_utils

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

def imshow(tensor, xaxis="", yaxis="", caxis="", **kwargs):
    return px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)

def line(tensor, xaxis="", yaxis="", **kwargs):
    return px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs)

def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    return px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# %%

if MAIN:
    gpt2_small = HookedTransformer.from_pretrained("gpt2-small", device=device)

# %%

if MAIN:
    model_description_text = '''## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. 

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model out, let's find the loss on this paragraph!'''

    loss = gpt2_small(model_description_text, return_type="loss")
    print("Model loss:", loss)

# %%

if MAIN:
    print(gpt2_small.to_str_tokens("gpt2"))
    print(gpt2_small.to_tokens("gpt2"))
    print(gpt2_small.to_string([50256, 70, 457, 17]))

# %%

if MAIN:
    logits = gpt2_small(model_description_text, return_type="logits")
    prediction = logits.argmax(dim=-1)[0, :-1]
    true_tokens = gpt2_small.to_tokens(model_description_text)[0, 1:]

    num_correct = (prediction == true_tokens).sum()

    print(f"Model accuracy: {num_correct}/{len(true_tokens)}")
    print(f"Correct words: {gpt2_small.to_str_tokens(prediction[prediction == true_tokens])}")

# %%

if MAIN:
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
    gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

# %%

if MAIN:
    layer0_pattern_from_cache = gpt2_cache["pattern", 0]

    q, k = gpt2_cache["q", 0], gpt2_cache["k", 0]
    seq, nhead, headsize = q.shape
    layer0_attn_scores = einsum("seqQ n h, seqK n h -> n seqQ seqK", q, k)
    mask = t.triu(t.ones((seq, seq), device=device, dtype=bool), diagonal=1)
    layer0_attn_scores.masked_fill_(mask, -1e9)
    layer0_pattern_from_q_and_k = (layer0_attn_scores / headsize**0.5).softmax(-1)

    t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)

# %%

if MAIN:

    print(type(gpt2_cache))
    attention_pattern = gpt2_cache["pattern", 0]
    print(attention_pattern.shape)
    gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

    display(cv.attention.attention_patterns(tokens=gpt2_str_tokens, attention=attention_pattern))

# %%

if MAIN:
    cfg = HookedTransformerConfig(
        d_model=768,
        d_head=64,
        n_heads=12,
        n_layers=2,
        n_ctx=2048,
        d_vocab=50278,
        attention_dir="causal", # defaults to "bidirectional"
        attn_only=True, # defaults to False
        tokenizer_name="EleutherAI/gpt-neox-20b", 
        seed=398,
        use_attn_result=True,
        normalization_type=None, # defaults to "LN", i.e. use layernorm with weights and biases
        positional_embedding_type="shortformer"
    )

# %%

if MAIN:
    WEIGHT_PATH = "attn_only_2L_half.pth"

    model = HookedTransformer(cfg)
    pretrained_weights = t.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(pretrained_weights)

# %%

if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    logits, cache = model.run_with_cache(text, remove_batch_dim=True)

# %%

if MAIN:
    str_tokens = model.to_str_tokens(text)
    for layer in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", layer]
        display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern))

# %%
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of diagonal elements
            score = attention_pattern.diagonal().mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of sub-diagonal elements
            score = attention_pattern.diagonal(-1).mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

def first_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of 0th elements
            score = attention_pattern[:, 0].mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

if MAIN:
    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))
# %%

def generate_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> t.Tensor:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    """
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    return rep_tokens

def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    rep_tokens = generate_repeated_tokens(model, seq_len, batch)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache

def per_token_losses(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs[0]

if MAIN:
    seq_len = 50
    batch = 1
    (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    rep_cache.remove_batch_dim()
    rep_str = model.to_str_tokens(rep_tokens)
    model.reset_hooks()
    ptl = per_token_losses(rep_logits, rep_tokens)
    print(f"Performance on the first half: {ptl[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {ptl[seq_len:].mean():.3f}")
    fig = px.line(
        utils.to_numpy(ptl), hover_name=rep_str[1:],
        title=f"Per token loss on sequence of length {seq_len} repeated twice",
        labels={"index": "Sequence position", "value": "Loss"}
    ).update_layout(showlegend=False, hovermode="x unified")
    fig.add_vrect(x0=0, x1=seq_len-.5, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_vrect(x0=seq_len-.5, x1=2*seq_len-1, fillcolor="green", opacity=0.2, line_width=0)
    plot_utils.save_fig(fig, "repeated_tokens")
    fig.show()

# %%

def write_to_html(fig, filename):
    with open(f"{filename}.html", "w") as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
# write_to_html(fig, "repeated_tokens")

# %%

if MAIN:
    for layer in range(model.cfg.n_layers):
        attention_pattern = rep_cache["pattern", layer]
        display(cv.attention.attention_patterns(tokens=rep_str, attention=attention_pattern))

# %%

def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads
    '''
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of (-seq_len+1)-offset elements
            seq_len = (attention_pattern.shape[-1] - 1) // 2
            score = attention_pattern.diagonal(-seq_len+1).mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

# %%

if MAIN:
    seq_len = 50
    batch = 10
    rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

    # We make a tensor to store the induction score for each head. We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
    induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
    
    def induction_score_hook(
        pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
        hook: HookPoint,
    ):
        '''
        Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
        '''
        # We take the diagonal of attention paid from each destination position to source positions seq_len-1 tokens back
        # (This only has entries for tokens with index>=seq_len)
        induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
        # Get an average score per head
        induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
        # Store the result.
        induction_score_store[hook.layer(), :] = induction_score

    # We make a boolean filter on activation names, that's true only on attention pattern names.
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
    fig = imshow(induction_score_store, xaxis="Head", yaxis="Layer", title="Induction Score by Head", text_auto=".2f")
    plot_utils.save_fig(fig, "induction_scores")
    fig.show()

# %%

def visualize_pattern_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
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
    seq_len = 50
    batch = 10
    rep_tokens_10 = generate_repeated_tokens(gpt2_small, seq_len, batch)

    induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device)

    gpt2_small.run_with_hooks(
        rep_tokens_10, 
        return_type=None, # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(
            pattern_hook_names_filter,
            induction_score_hook
        )]
    )

    fig = imshow(induction_score_store, xaxis="Head", yaxis="Layer", title="Induction Score by Head", text_auto=".1f")
    fig.show()

    # Observation: heads 5.1, 5.5, 6.9, 7.2, 7.10 are all strongly induction-y.
    # Confirm observation by visualizing attn patterns for layers 5 through 7:

    for induction_head_layer in [5, 6, 7]:
        gpt2_small.run_with_hooks(
            rep_tokens, 
            return_type=None, # For efficiency, we don't need to calculate the logits
            fwd_hooks=[(
                utils.get_act_name("pattern", induction_head_layer),
                visualize_pattern_hook
            )]
        )

# %%



# %%

def logit_attribution(embed, l1_results, l2_results, W_U, tokens) -> t.Tensor:
    '''
    We have provided 'W_U_correct_tokens' which is a (d_model, seq_next) tensor where each row is the unembed for the correct NEXT token at the current position.
    Inputs:
        embed (seq_len, d_model): the embeddings of the tokens (i.e. token + position embeddings)
        l1_results (seq_len, n_heads, d_model): the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results (seq_len, n_heads, d_model): the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U (d_model, d_vocab): the unembedding matrix
    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (position-1,1)
            layer 0 logits (position-1, n_heads)
            and layer 1 logits (position-1, n_heads)
    '''
    W_U_correct_tokens = W_U[:, tokens[1:]]

    direct_attributions = einsum("emb seq, seq emb -> seq", W_U_correct_tokens, embed[:-1])
    l1_attributions = einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l1_results[:-1])
    l2_attributions = einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l2_results[:-1])
    return t.concat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)

if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text).to(device)

    with t.inference_mode():
        embed = cache["embed"]
        l1_results = cache["result", 0]
        l2_results = cache["result", 1]
        logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
        # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)

# %%

def convert_tokens_to_string(tokens, batch_index=0):
    '''Helper function to convert tokens into a list of strings, for printing.
    '''
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{model.tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]

def plot_logit_attribution(logit_attr: t.Tensor, tokens: t.Tensor, title: str = ""):
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(tokens[:-1])
    x_labels = ["Direct"] + [f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    return imshow(utils.to_numpy(logit_attr), x=x_labels, y=y_labels, xaxis="Term", yaxis="Position", caxis="logit", title=title if title else None, height=25*len(tokens))

if MAIN:
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens[0])
    fig = plot_logit_attribution(logit_attr, tokens)
    plot_utils.save_fig(fig, "logit_attribution")
    fig.show()

# %%

if MAIN:
    seq_len = 50

    embed = rep_cache["embed"]
    l1_results = rep_cache["result", 0]
    l2_results = rep_cache["result", 1]
    first_half_tokens = rep_tokens[0, : 1 + seq_len]
    second_half_tokens = rep_tokens[0, seq_len:]

    first_half_logit_attr = logit_attribution(embed[:1+seq_len], l1_results[:1+seq_len], l2_results[:1+seq_len], model.unembed.W_U, first_half_tokens)
    second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.unembed.W_U, second_half_tokens)

    assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    
    fig1 = plot_logit_attribution(first_half_logit_attr, first_half_tokens, "Logit attribution (first half of repeated sequence)")
    fig2 = plot_logit_attribution(second_half_logit_attr, second_half_tokens, "Logit attribution (second half of repeated sequence)")

    fig1.show()
    fig2.show()
    plot_utils.save_fig(fig2, "rep_logit_attribution")

# %%

if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    str_tokens = gpt2_small.to_str_tokens(text)
    tokens = gpt2_small.to_tokens(text)
    tokens = tokens.to(device)
    logits, cache = gpt2_small.run_with_cache(tokens, remove_batch_dim=True)
    gpt2_small.reset_hooks()

# %%

def head_ablation_hook(
    attn_result: TT["batch", "seq", "n_heads", "d_model"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> TT["batch", "seq", "n_heads", "d_model"]:
    attn_result[:, :, head_index_to_ablate, :] = 0.0
    return attn_result

def cross_entropy_loss(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()

def get_ablation_scores(
    model: HookedTransformer, 
    tokens: TT["batch", "seq"]
) -> TT["n_layers", "n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
            # Run the model with the patching hook
            patched_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("result", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(patched_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores

if MAIN:
    ablation_scores = get_ablation_scores(model, rep_tokens)
    tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)

if MAIN:
    fig = imshow(ablation_scores, xaxis="Head", yaxis="Layer", caxis="logit diff", title="Logit Difference After Ablating Heads", text_auto=".2f")
    plot_utils.save_fig(fig, "ablation_scores")
    fig.show()

# %%

def get_ablation_complement_scores(
    model: HookedTransformer,
    tokens: TT["batch", "seq"],
    heads_to_preserve: List[str]
):

    layer0_heads = [int(i[2:]) for i in heads_to_preserve if i.startswith("0.")]
    layer1_heads = [int(i[2:]) for i in heads_to_preserve if i.startswith("1.")]

    def hook_ablate_complement(
        attn_result: TT["batch", "seq", "n_heads", "d_model"],
        hook: HookPoint,
        heads_to_preserve: List[int]
    ):
        n_heads = attn_result.shape[-2]
        heads_to_ablate = [i for i in range(n_heads) if i not in heads_to_preserve]
        attn_result[:, :, heads_to_ablate] = 0

    hook_fn_layer0 = functools.partial(hook_ablate_complement, heads_to_preserve=layer0_heads)
    hook_fn_layer1 = functools.partial(hook_ablate_complement, heads_to_preserve=layer1_heads)

    # Run the model with the ablation hook
    ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[
        (utils.get_act_name("result", 0), hook_fn_layer0),
        (utils.get_act_name("result", 1), hook_fn_layer1)
    ])
    # Calculate the cross entropy difference
    ablated_loss = cross_entropy_loss(ablated_logits[:, -seq_len:], tokens[:, -seq_len:])

    logits = model(tokens)
    loss = cross_entropy_loss(logits[:, -seq_len:], tokens[:, -seq_len:])

    print(f"Ablated loss = {ablated_loss:.3f}\nOriginal loss = {loss:.3f}")

if MAIN:
    heads_to_preserve = ["0.7", "1.4", "1.10"]
    get_ablation_complement_scores(model, rep_tokens, heads_to_preserve)










# %%

if MAIN:
    head_index = 4
    layer = 1
    
    W_O_4 = model.W_O[1, 4]
    W_V_4 = model.W_V[1, 4]
    W_E = model.W_E
    W_U = model.W_U

    OV_circuit = FactoredMatrix(W_V_4, W_O_4)
    full_OV_circuit = W_E @ OV_circuit @ W_U

    tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)

    indices = t.randint(0, model.cfg.d_vocab, (200,))

    # full_OV_circuit_sample = full_OV_circuit.A[indices, :] @ full_OV_circuit.B[:, indices]
    full_OV_circuit_sample = full_OV_circuit[indices, indices].AB

    fig = imshow(full_OV_circuit_sample)
    plot_utils.save_fig(fig, "OV_circuit_sample")
    fig.show()

# %%


def top_1_acc(full_OV_circuit: FactoredMatrix) -> float:
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    # A = full_OV_circuit.A.to("cpu")
    # B = full_OV_circuit.B.to("cpu")
    # AB = A @ B
    AB = full_OV_circuit.AB

    return (t.argmax(AB, dim=1) == t.arange(AB.shape[0])).float().mean().item()


def top_1_acc_iteration(full_OV_circuit: FactoredMatrix, batch_size: int = 100) -> float: 
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    A, B = full_OV_circuit.A, full_OV_circuit.B
    nrows = full_OV_circuit.shape[0]
    nrows_max_on_diagonal = 0

    for i in tqdm(range(0, nrows + batch_size, batch_size)):
        rng = range(i, min(i + batch_size, nrows))
        if rng:
            submatrix = A[rng, :] @ B
            diag_indices = t.tensor(rng, device=submatrix.device)
            nrows_max_on_diagonal += (submatrix.argmax(-1) == diag_indices).float().sum().item()
    
    return nrows_max_on_diagonal / nrows


def top_5_acc(full_OV_circuit: FactoredMatrix):
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    A, B = full_OV_circuit.A, full_OV_circuit.B

    correct = []
    for i in tqdm(range(full_OV_circuit.shape[-1])):
        top5 = t.topk(A[i, :] @ B, k=5).indices
        correct.append(i in top5)
    
    return t.tensor(correct).float().mean()

if MAIN:
    print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc_iteration(full_OV_circuit):.4f}")
    print("Fraction of the time that the five best logits include the one on the diagonal:")
    # print(top_5_acc(full_OV_circuit))


# %%

if MAIN:
    W_O_both = einops.rearrange(model.W_O[1, [4, 10]], "head d_head d_model -> (head d_head) d_model")
    W_V_both = einops.rearrange(model.W_V[1, [4, 10]], "head d_model d_head -> d_model (head d_head)")

    W_OV_eff = W_E @ FactoredMatrix(W_V_both, W_O_both) @ W_U

    print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc_iteration(W_OV_eff):.4f}")

# %%

def mask_scores(attn_scores: TT["query_d_model", "key_d_model"]):
    '''Mask the attention scores so that tokens don't attend to previous tokens.'''
    mask = t.tril(t.ones_like(attn_scores)).bool()
    return attn_scores.masked_fill(~mask, attn_scores.new_tensor(-1.0e6))

if MAIN:
    layer = 0
    head_index = 7

    "TODO: YOUR CODE HERE"
    W_pos = model.W_pos
    W_QK = model.W_Q[0, 7] @ model.W_K[0, 7].T
    pos_by_pos_scores = W_pos @ W_QK @ W_pos.T
    masked_scaled = mask_scores(pos_by_pos_scores / model.cfg.d_head ** 0.5)
    pos_by_pos_pattern = t.softmax(masked_scaled, dim=-1)

    tests.test_pos_by_pos_pattern(pos_by_pos_pattern, model, layer, head_index)

# %%

if MAIN:
    fig = imshow(utils.to_numpy(pos_by_pos_pattern[:100, :100]), xaxis="Key", yaxis="Query")
    fig.show()
    # plot_utils.save_fig(fig, "pos_by_pos_pattern")

    print(f"Average lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")

# %%

def decompose_qk_input(cache: ActivationCache) -> t.Tensor:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, pos, d_model]

    The [i, 0, 0]th element is y_i (from notation above)
    '''
    y0 = cache["embed"].unsqueeze(0) # shape (1, pos, d_model)
    y1 = cache["pos_embed"].unsqueeze(0) # shape (1, pos, d_model)
    y_rest = cache["result", 0].transpose(0, 1) # shape (12, pos, d_model)

    return t.concat([y0, y1, y_rest], dim=0)


def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head]
    
    The [i, 0, 0]th element is y_i @ W_Q (so the sum along axis i is just the q-values)
    '''
    W_Q = model.W_Q[1, ind_head_index]

    return einsum(
        "n pos d_head, d_head d_model -> n pos d_model",
        decomposed_qk_input, W_Q
    )


def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head]
    
    The [i, 0, 0]th element is y_i @ W_K(so the sum along axis i is just the k-values)
    '''
    W_K = model.W_K[1, ind_head_index]
    
    return einsum(
        "n pos d_head, d_head d_model -> n pos d_model",
        decomposed_qk_input, W_K
    )


if MAIN:
    # Compute decomposed input and output, test they are correct
    ind_head_index = 4
    decomposed_qk_input = decompose_qk_input(rep_cache)
    t.testing.assert_close(decomposed_qk_input.sum(0), rep_cache["resid_pre", 1] + rep_cache["pos_embed"], rtol=0.01, atol=1e-05)
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
    decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)

    # Plot importance results
    component_labels = ["Embed", "PosEmbed"] + [f"L0H{h}" for h in range(model.cfg.n_heads)]
    for decomposed_input, name in [(decomposed_q, "query"), (decomposed_k, "key")]:
        fig = imshow(utils.to_numpy(decomposed_input.pow(2).sum([-1])), xaxis="Pos", yaxis="Component", title=f"Norms of components of {name}", y=component_labels)
        fig.show()
        plot_utils.save_fig(fig, f"norms_of_{name}_components")

# %%

def decompose_attn_scores(decomposed_q: t.Tensor, decomposed_k: t.Tensor) -> t.Tensor:
    '''
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]
    
    The [i, j, 0, 0]th element is y_i @ W_QK @ y_j^T (so the sum along both first axes are the attention scores)
    '''
    return einsum(
        "q_comp q_pos d_model, k_comp k_pos d_model -> q_comp k_comp q_pos k_pos",
        decomposed_q, decomposed_k
    )

if MAIN:
    tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k)

# %%

if MAIN:
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
    decomposed_stds = einops.reduce(
        decomposed_scores, "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", t.std
    )

    # seq_len = decomposed_scores.shape[-1]
    # decomposed_stds = einops.reduce(
    #     decomposed_scores[:, :, t.tril_indices(seq_len, seq_len)[0], t.tril_indices(seq_len, seq_len)[1]], "query_decomp key_decomp all_posns -> query_decomp key_decomp", t.std
    # )

    # First plot: attention score contribution from (query_component, key_component) = (Embed, L0H7)
    fig_per_component = imshow(utils.to_numpy(t.tril(decomposed_scores[0, 9])), title="Attention Scores for component from Q=Embed and K=Prev Token Head")
    # Second plot: std dev over query and key positions, shown by component
    fig_std = imshow(utils.to_numpy(decomposed_stds), xaxis="Key Component", yaxis="Query Component", title="Standard deviations of components of scores", x=component_labels, y=component_labels)
    
    fig_per_component.show()
    fig_std.show()
    plot_utils.save_fig(fig_per_component, "attn_scores_per_component")
    plot_utils.save_fig(fig_std, "attn_scores_std_devs")




# %%

def find_K_comp_full_circuit(model: HookedTransformer, prev_token_head_index: int, ind_head_index: int) -> FactoredMatrix:
    '''
    Returns a vocab x vocab matrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    '''
    W_E = model.W_E
    W_Q = model.W_Q[1, ind_head_index]
    W_K = model.W_K[1, ind_head_index]
    W_O = model.W_O[0, prev_token_head_index]
    W_V = model.W_V[0, prev_token_head_index]
    
    Q = W_E @ W_Q
    K = W_E @ W_V @ W_O @ W_K
    return FactoredMatrix(Q, K.T)


if MAIN:
    prev_token_head_index = 7
    ind_head_index = 4
    K_comp_circuit = find_K_comp_full_circuit(model, prev_token_head_index, ind_head_index)

    tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)

    print(f"Fraction of tokens where the highest activating key is the same token: {top_1_acc_iteration(K_comp_circuit.T):.4f}", )


# %%

def get_comp_score(
    W_A: TT["in_A", "out_A"], 
    W_B: TT["out_A", "out_B"]
) -> float:
    '''
    Return the composition score between W_A and W_B.
    '''
    W_A_norm = W_A.pow(2).sum().sqrt()
    W_B_norm = W_B.pow(2).sum().sqrt()
    W_AB_norm = (W_A @ W_B).pow(2).sum().sqrt()

    return (W_AB_norm / (W_A_norm * W_B_norm)).item()

if MAIN:
    tests.test_get_comp_score(get_comp_score)

# %%

if MAIN:
    # Get all QK and OV matrices
    W_QK = model.W_Q @ model.W_K.transpose(-1, -2)
    W_OV = model.W_V @ model.W_O

    # Define tensors to hold the composition scores
    q_comp_scores = t.zeros(model.cfg.n_heads, model.cfg.n_heads, device=device)
    k_comp_scores = t.zeros(model.cfg.n_heads, model.cfg.n_heads, device=device)
    v_comp_scores = t.zeros(model.cfg.n_heads, model.cfg.n_heads, device=device)
    
    # Fill in the tensors, by looping over W_A and W_B from layers 0 and 1
    "YOUR CODE HERE!"
    for i in tqdm(range(model.cfg.n_heads)):
        for j in range(model.cfg.n_heads):
            q_comp_scores[i, j] = get_comp_score(W_OV[0][i], W_QK[1][j])
            k_comp_scores[i, j] = get_comp_score(W_OV[0][i], W_QK[1][j].T)
            v_comp_scores[i, j] = get_comp_score(W_OV[0][i], W_OV[1][j])

    plot_utils.plot_comp_scores(model, q_comp_scores, "Q Composition Scores").show()
    plot_utils.plot_comp_scores(model, k_comp_scores, "K Composition Scores").show()
    plot_utils.plot_comp_scores(model, v_comp_scores, "V Composition Scores").show()

# %%

def generate_single_random_comp_score() -> float:

    W_A_left = t.empty(model.cfg.d_model, model.cfg.d_head)
    W_B_left = t.empty(model.cfg.d_model, model.cfg.d_head)
    W_A_right = t.empty(model.cfg.d_model, model.cfg.d_head)
    W_B_right = t.empty(model.cfg.d_model, model.cfg.d_head)

    for W in [W_A_left, W_B_left, W_A_right, W_B_right]:
        nn.init.kaiming_uniform_(W, a=np.sqrt(5))

    W_A = W_A_left @ W_A_right.T
    W_B = W_B_left @ W_B_right.T

    return get_comp_score(W_A, W_B)

if MAIN:
    n_samples = 300
    comp_scores_baseline = np.zeros(n_samples)
    for i in tqdm(range(n_samples)):
        comp_scores_baseline[i] = generate_single_random_comp_score()
    print("\nMean:", comp_scores_baseline.mean())
    print("Std:", comp_scores_baseline.std())
    px.histogram(comp_scores_baseline, nbins=50).show()

# %%

if MAIN:
    baseline = comp_scores_baseline.mean()
    for comp_scores, name in [(q_comp_scores, "Q"), (k_comp_scores, "K"), (v_comp_scores, "V")]:
        fig = plot_utils.plot_comp_scores(model, comp_scores, f"{name} Composition Scores", baseline=baseline)
        fig.show()
        plot_utils.save_fig(fig, f"{name.lower()}_comp_scores")

# %%

def get_batched_comp_scores(
    W_As: FactoredMatrix,
    W_Bs: FactoredMatrix
) -> t.Tensor:
    '''Returns the compositional scores from indexed tensors W_As and W_Bs.

    Each of W_As and W_Bs is a FactoredMatrix object which is indexed by all but its last 2 dimensions, i.e. W_As.shape == (*A_idx, A_in, A_out) and W_Bs.shape == (*B_idx, B_in, B_out).

    Return: tensor of shape (*A_idx, *B_idx) where the [*a_idx, *b_idx]th element is the compositional score from W_As[*a_idx] to W_Bs[*b_idx].
    '''
    # Reshape W_As and W_Bs to only have one index dimension
    # Note, we include a dummy index dimension of size 1, so we can broadcast when we multiply W_As and W_Bs
    W_As = FactoredMatrix(
        W_As.A.reshape(-1, 1, *W_As.A.shape[-2:]),
        W_As.B.reshape(-1, 1, *W_As.B.shape[-2:]),
    )
    W_Bs = FactoredMatrix(
        W_Bs.A.reshape(1, -1, *W_Bs.A.shape[-2:]),
        W_Bs.B.reshape(1, -1, *W_Bs.B.shape[-2:]),
    )

    # Compute the product
    W_ABs = W_As @ W_Bs

    # Compute the norms, and return the metric
    return W_ABs.norm() / (W_As.norm() * W_Bs.norm())

# %%

if MAIN:
    W_QK = FactoredMatrix(model.W_Q, model.W_K.transpose(-1, -2))
    W_OV = FactoredMatrix(model.W_V, model.W_O)

    q_comp_scores_batched = get_batched_comp_scores(W_OV[0], W_QK[1])
    k_comp_scores_batched = get_batched_comp_scores(W_OV[0], W_QK[1].T) # Factored matrix: .T is interpreted as transpose of the last two axes
    v_comp_scores_batched = get_batched_comp_scores(W_OV[0], W_OV[1])

    t.testing.assert_close(q_comp_scores, q_comp_scores_batched)
    t.testing.assert_close(k_comp_scores, k_comp_scores_batched)
    t.testing.assert_close(v_comp_scores, v_comp_scores_batched)
    print("Tests passed - your `get_batched_comp_scores` function is working!")

# %%

if MAIN:
    figQ = px.imshow(
        utils.to_numpy(q_comp_scores),
        y=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title="Q Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    )
    figK = px.imshow(
        utils.to_numpy(k_comp_scores),
        y=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title="K Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    )
    figV = px.imshow(
        utils.to_numpy(v_comp_scores),
        y=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title="V Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    )
    figQ.show()
    figK.show()
    figV.show()

# %%


def ablation_induction_score(prev_head_index: Optional[int], ind_head_index: int) -> float:
    '''
    Takes as input the index of the L0 head and the index of the L1 head, and then runs with the previous token head ablated and returns the induction score for the ind_head_index now.
    '''

    def ablation_hook(v, hook):
        if prev_head_index is not None:
            v[:, :, prev_head_index] = 0.0
        return v

    def induction_pattern_hook(attn, hook):
        hook.ctx[prev_head_index] = attn[0, ind_head_index].diag(-(seq_len - 1)).mean()

    model.run_with_hooks(
        rep_tokens,
        fwd_hooks=[
            (utils.get_act_name("v", 0), ablation_hook),
            (utils.get_act_name("pattern", 1), induction_pattern_hook)
        ],
    )
    return model.blocks[1].attn.hook_pattern.ctx[prev_head_index].item()


if MAIN:
    baseline_induction_score = ablation_induction_score(None, 4)
    print(f"Induction score for no ablations: {baseline_induction_score}\n")
    for i in range(model.cfg.n_heads):
        new_induction_score = ablation_induction_score(i, 4)
        induction_score_change = new_induction_score - baseline_induction_score
        print(f"Ablation score change for head {i:02}: {induction_score_change:+.5f}")

# %%

