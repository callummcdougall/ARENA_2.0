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
