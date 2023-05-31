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
from transformer_lens import (
    utils,
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
import circuitsvis as cv

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part2_intro_to_mech_interp").resolve()
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import (
    imshow,
    hist,
    plot_comp_scores,
    plot_logit_attribution,
    plot_loss_difference,
)
from part1_transformer_from_scratch.solutions import get_log_probs
import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


if MAIN:
    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False
    )
# %%
from termcolor import colored
import math

# %%

if MAIN:
    gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

# %%

gpt2_small.cfg.n_layers
gpt2_small.cfg.n_heads

# %%

if MAIN:
    model_description_text = """## Loading Models

    HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 

    For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""

    loss = gpt2_small(model_description_text, return_type="loss")
    print("Model loss:", loss)

# %%


def print_correct(predictions, real):
    [
        colored(t, "green" if m else "red")
        for t, m in zip(string_tokens, matches.squeeze())
    ]


if MAIN:
    logits: Tensor = gpt2_small(model_description_text, return_type="logits")
    prediction = logits.argmax(dim=-1).squeeze()[:-1]
    guesses = gpt2_small.to_str_tokens(prediction)

    tokens = gpt2_small.to_tokens(model_description_text, prepend_bos=False)
    string_tokens = gpt2_small.to_str_tokens(tokens)
    matches = prediction == tokens
    print("".join(gpt2_small.to_str_tokens(prediction)))
    print(
        f"count of matches: {matches.sum()} mean of matches: {matches.mean(dtype=t.float)}"
    )
    print(gpt2_small.to_str_tokens(prediction[matches.squeeze()]))
    ced = [
        colored(t, "green" if m else "red")
        for t, m in zip(string_tokens, matches.squeeze())
    ]
    print("".join(ced))

# %%
if MAIN:
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
    gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(
        gpt2_tokens, remove_batch_dim=True
    )
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
    def apply_causal_mask(
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        mask = t.triu(t.ones_like(attn_scores, dtype=bool), diagonal=1)
        return t.masked_fill(attn_scores, mask, -1e9)

    q = gpt2_cache["q0"]
    k = gpt2_cache["k0"]

    qk = einops.einsum(
        q, k, "q_pos n_heads d_head, k_pos n_heads d_head -> n_heads q_pos k_pos"
    )
    assert reference_gpt2.cfg.d_head == q.shape[-1]
    attn_scores = qk / math.sqrt(reference_gpt2.cfg.d_head)
    attn_scores = apply_causal_mask(attn_scores)
    layer0_pattern_from_q_and_k = attn_scores.softmax(dim=-1)
    ###

    t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
    print("Tests passed!")

# %%
if MAIN:
    print(type(gpt2_cache))
    attention_pattern = gpt2_cache["pattern", 0, "attn"]
    print(attention_pattern.shape)
    gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

    print("Layer 0 Head Attention Patterns:")
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_str_tokens,
            attention=attention_pattern,
            attention_head_names=[f"L0H{i}" for i in range(12)],
        )
    )

# %% Reimport
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
from transformer_lens import (
    utils,
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
import circuitsvis as cv

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part2_intro_to_mech_interp").resolve()
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import (
    imshow,
    hist,
    plot_comp_scores,
    plot_logit_attribution,
    plot_loss_difference,
)
from part1_transformer_from_scratch.solutions import get_log_probs
import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

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
        attn_only=True,  # defaults to False
        tokenizer_name="EleutherAI/gpt-neox-20b",
        seed=398,
        use_attn_result=True,
        normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
        positional_embedding_type="shortformer",
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
# %%

if MAIN:
    tokens = model.to_str_tokens(text)
    n_heads = cfg.n_heads
    for layer_i in range(cfg.n_layers):
        pattern = cache["pattern", layer_i]
        print(f"Layer {layer_i} Head Attention Patterns:")
        display(
            cv.attention.attention_patterns(
                tokens=tokens,
                attention=pattern,
                attention_head_names=[f"L{layer_i}H{j}" for j in range(n_heads)],
            )
        )


# %%
def current_attn_detector(cache: ActivationCache) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    """
    threshold = 0.6
    result = []
    for l in range(cfg.n_layers):
        pattern = cache["pattern", l]
        for h, head_attn in enumerate(pattern):
            head_name = f"{l}.{h}"
            head_attended_to = head_attn.argmax(dim=-1)
            target = t.arange(head_attn.shape[-1]).to(device)
            matches = (head_attended_to == target).mean(dtype=t.float32)
            if matches > threshold:
                result.append(head_name)
    return result


def prev_attn_detector(cache: ActivationCache) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    """
    threshold = 0.9
    result = []
    for l in range(cfg.n_layers):
        pattern = cache["pattern", l]
        for h, head_attn in enumerate(pattern):
            head_name = f"{l}.{h}"
            head_attended_to = head_attn.argmax(dim=-1)
            target = t.arange(head_attn.shape[-1]).to(device) - 1
            target = target.clamp(0, 100)
            matches = (head_attended_to == target).mean(dtype=t.float32)
            if matches > threshold:
                result.append(head_name)
    return result


def first_attn_detector(cache: ActivationCache) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    """
    threshold = 0.9
    result = []
    for l in range(cfg.n_layers):
        pattern = cache["pattern", l]
        for h, head_attn in enumerate(pattern):
            head_name = f"{l}.{h}"
            head_attended_to = head_attn.argmax(dim=-1)
            target = t.tensor([0]).to(device)
            matches = (head_attended_to == target).mean(dtype=t.float32)
            if matches > threshold:
                result.append(head_name)
    return result


if MAIN:
    print(
        "Heads attending to current token  = ", ", ".join(current_attn_detector(cache))
    )
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))


# %%
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    """
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long().to(device)
    rands = t.randint(high=cfg.d_vocab, size=(batch, seq_len)).to(device)
    return t.concat((prefix, rands, rands), dim=-1).to(device)


debug = True


def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    to_predict = generate_repeated_tokens(model, seq_len, batch)
    logits, cache = model.run_with_cache(to_predict)
    return to_predict, logits, cache


if MAIN:
    # seq_len = 50
    seq_len = 50
    batch = 1
    (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(
        model, seq_len, batch
    )
    rep_cache.remove_batch_dim()
    rep_str = model.to_str_tokens(rep_tokens)
    print(f"{rep_str=}")
    predictions = model.to_str_tokens(rep_logits.argmax(dim=-1))
    print(f"{predictions=}")
    for x in zip(rep_str[1:], predictions):
        print(x)

    model.reset_hooks()
    log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()
    print(log_probs.shape)

    print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

    plot_loss_difference(log_probs, rep_str, seq_len)

# %%
if MAIN:
    n_heads = cfg.n_heads
    for layer_i in range(cfg.n_layers):
        pattern = rep_cache["pattern", layer_i]
        print(f"Layer {layer_i} Head Attention Patterns:")
        display(
            cv.attention.attention_patterns(
                tokens=rep_str,
                attention=pattern,
                attention_head_names=[f"L{layer_i}H{j}" for j in range(n_heads)],
            )
        )


# %%
def induction_attn_detector(cache: ActivationCache) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    """
    threshold = 0.8
    result = []
    layer = 1
    pattern = cache["pattern", layer]
    for h, head_attn in enumerate(pattern):
        head_name = f"{layer}.{h}"
        head_attended_to = head_attn.argmax(dim=-1)
        # target = t.tensor([0]).to(device)
        target = t.concat((t.zeros(49), t.arange(52))).long().to(device)
        matches = (head_attended_to == target).mean(dtype=t.float32)
        if matches > threshold:
            result.append(head_name)
    return result


if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))
# %%


def hook_function(
    attn_pattern: Float[Tensor, "batch heads seq_len seq_len"],
    hook: HookPoint
    # ) -> TT["batch", "heads", "seq_len", "seq_len"]:
) -> t.Tensor:
    # modify attn_pattern (can be inplace)
    return attn_pattern


# %%

loss = model.run_with_hooks(
    tokens,
    return_type="loss",
    fwd_hooks=[("blocks.1.attn.hook_pattern", hook_function)],
)
# %%

if MAIN:
    seq_len = 50
    batch = 10
    rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

    # We make a tensor to store the induction score for each head.
    # We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
    induction_score_store = t.zeros(
        (model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device
    )


def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    """
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    """
    offset = -(seq_len - 1)

    ave_p = pattern.mean(dim=0)
    for h, head in enumerate(ave_p):
        ind_score = head.diagonal(offset=offset).mean()
        induction_score_store[hook.layer(), h] = ind_score


if MAIN:
    pattern_hook_names_filter = lambda name: name.endswith("pattern")

    # Run with hooks (this is where we write to the `induction_score_store` tensor`)
    model.run_with_hooks(
        rep_tokens_10,
        return_type=None,  # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
    )

    # Plot the induction scores for each head in each layer
    imshow(
        induction_score_store,
        labels={"x": "Head", "y": "Layer"},
        title="Induction Score by Head",
        text_auto=".2f",
        width=900,
        height=400,
    )

# %%


def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens[0]), attention=pattern.mean(0)
        )
    )


def make_induction_hook(score_store):
    def induction_score_hook(
        pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
        hook: HookPoint,
    ):
        """
        Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
        """
        offset = -(seq_len - 1)

        ave_p = pattern.mean(dim=0)
        for h, head in enumerate(ave_p):
            ind_score = head.diagonal(offset=offset).mean()
            score_store[hook.layer(), h] = ind_score

    return induction_score_hook


if MAIN:
    # YOUR CODE HERE - find induction heads in gpt2_small
    pattern_hook_names_filter = lambda name: name.endswith("pattern")

    store = t.zeros(
        gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads, device=model.cfg.device
    )
    hook = make_induction_hook(store)

    # Run with hooks (this is where we write to the `induction_score_store` tensor`)
    gpt2_small.run_with_hooks(
        rep_tokens_10,
        return_type=None,  # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(pattern_hook_names_filter, hook)],
    )

    # Plot the induction scores for each head in each layer
    imshow(
        store,
        labels={"x": "Head", "y": "Layer"},
        title="Induction Score by Head",
        text_auto=".2f",
        width=900,
        height=400,
    )

    gpt2_small.run_with_hooks(
        rep_tokens,
        return_type=None,  # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(lambda n: "pattern" in n and "5" in n, visualize_pattern_hook)],
    )
# %%


def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"],
) -> Float[Tensor, "seq-1 n_components"]:
    """
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
    """
    W_U_correct_tokens = W_U[:, tokens[1:]]
    embed_component = einops.einsum(
        embed[:-1, :], W_U_correct_tokens, "seq d_model, d_model seq -> seq"
    )
    l1_component = einops.einsum(
        l1_results[:-1, :, :],
        W_U_correct_tokens,
        "seq nheads d_model, d_model seq -> seq nheads",
    )
    l2_component = einops.einsum(
        l2_results[:-1, :, :],
        W_U_correct_tokens,
        "seq nheads d_model, d_model seq -> seq nheads",
    )
    return t.concat(
        (t.unsqueeze(embed_component, dim=-1), l1_component, l2_component), dim=-1
    )


if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    # import random
    # text = ''.join(random.choice("abcdefghigjlmnopqrstuvwxyz    ") for i in range(50)) * 2
    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text)

    with t.inference_mode():
        embed = cache["embed"]
        l1_results = cache["result", 0]
        l2_results = cache["result", 1]
        logit_attr = logit_attribution(
            embed, l1_results, l2_results, model.W_U, tokens[0]
        )
        # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
        t.testing.assert_close(
            logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0
        )
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
    with t.inference_mode():
        logit_attr: Float[Tensor, "seq-1 n_components"] = logit_attribution(embed, l1_results, l2_results, model.W_U, rep_tokens[0])

        first_half_logit_attr = logit_attr[:seq_len]
        second_half_logit_attr = logit_attr[seq_len:]

    assert first_half_logit_attr.shape == (seq_len, 2 * model.cfg.n_heads + 1)
    assert second_half_logit_attr.shape == (seq_len, 2 * model.cfg.n_heads + 1)

    plot_logit_attribution(
        model,
        first_half_logit_attr,
        first_half_tokens,
        "Logit attribution (first half of repeated sequence)",
    )
    plot_logit_attribution(
        model,
        second_half_logit_attr,
        second_half_tokens,
        "Logit attribution (second half of repeated sequence)",
    )
# %%

def value_ablation_hook(
    value: Float[Tensor, "seq nhead d_head"],
    hook: HookPoint,
):
    head_index = 10
    value[:, head_index] = 0 # TODO: should this be t.Tensor(0.) ?

    return value

original_loss = model(rep_tokens, return_type="loss")

ablated_loss = model.run_with_hooks(
    rep_tokens,
    return_type="loss",
    fwd_hooks=[("blocks.1.attn.hook_v", value_ablation_hook)],
)

print(f"{original_loss=} {ablated_loss=}")

# %%

def value_ablation_hook(
    result: Float[Tensor, "batch seq nhead d_head"],
    hook: HookPoint,
    head_index: Int
):
    result[:,:, head_index] = 0 

    return result

ablation_scores = t.zeros(model.cfg.n_layers, model.cfg.n_heads)
original_loss = model(rep_tokens, return_type="loss")

for layer_idx in range(model.cfg.n_layers):
    for head_idx in range(model.cfg.n_heads):
        hook = functools.partial(value_ablation_hook, head_index=head_idx)

        loss = model.run_with_hooks(
            rep_tokens,
            return_type="loss",
            fwd_hooks=[( utils.get_act_name("v", layer_idx), hook)]
        )

        ablation_scores[layer_idx, head_idx] = loss - original_loss

imshow(ablation_scores, text_auto=".2f")

