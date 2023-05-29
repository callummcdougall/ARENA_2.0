# %%
import os; os.environ['ACCELERATE_DISABLE_RICH'] = "1"; os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_transformer_from_scratch").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
# import part1_transformer_from_scratch.solutions as solutions

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == '__main__'


if MAIN:
    reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
# %%
if MAIN:
    sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])
    print(sorted_vocab[:20])
    print()
    print(sorted_vocab[250:270])
    print()
    print(sorted_vocab[990:1010])
    print()
# %%
lengths = dict.fromkeys(range(3, 8), "")
for tok, idx in sorted_vocab:
    if not lengths.get(len(tok), True):
        lengths[len(tok)] = tok

for length, tok in lengths.items():
    print(f"{length}: {tok}")

# %%
if MAIN:
    print(sorted_vocab[-20:])
# %%
if MAIN:
    print(reference_gpt2.to_str_tokens("Ralph"))
    print(reference_gpt2.to_str_tokens(" Ralph"))
    print(reference_gpt2.to_str_tokens(" ralph"))
    print(reference_gpt2.to_str_tokens("ralph"))
# %%
if MAIN:
    print(reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000"))
# %%
if MAIN:
    reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
    tokens = reference_gpt2.to_tokens(reference_text).to(device)
    print(tokens)
    print(tokens.shape)
    print(reference_gpt2.to_str_tokens(tokens))
# %%
if MAIN:
    logits, cache = reference_gpt2.run_with_cache(tokens)
    print(logits.shape)

# %%
if MAIN:
    probs = logits.softmax(dim=-1)
    print(probs.shape)
# %%
if MAIN:
    most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])

    print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))
# %%
if MAIN:
    next_token = logits[0, -1].argmax(dim=-1)
    next_char = reference_gpt2.to_string(next_token)
    print(repr(next_char))
# %%
if MAIN:
    print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")

    for i in range(10):
        print(f"{tokens.shape[-1]+1}th char = {next_char!r}")
        # Define new input sequence, by appending the previously generated token
        tokens = t.cat([tokens, next_token[None, None]], dim=-1)
        # Pass our new sequence through the model, to get new output
        logits = reference_gpt2(tokens)
        # Get the predicted token at the end of our sequence
        next_token = logits[0, -1].argmax(dim=-1)
        # Decode and print the result
        next_char = reference_gpt2.to_string(next_token)
# %%
batch = 1
position = 35
d_model = 768
n_heads = 12
n_layers = 12
d_mlp = 3072 * (4 * d_model)
d_head = 64 * (d_model / n_heads)
# %%
if MAIN:
    for activation_name, activation in cache.items():
        # Only print for first layer
        if ".0." in activation_name or "blocks" not in activation_name:
            print(f"{activation_name:30} {tuple(activation.shape)}")
# %%
if MAIN:
    for name, param in reference_gpt2.named_parameters():
        # Only print for first layer
        if ".0." in name or "blocks" not in name:
            print(f"{name:18} {tuple(param.shape)}")
# %%
if MAIN:
    print(reference_gpt2.cfg)
# %%
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


if MAIN:
    cfg = Config()
    print(cfg)
# %%
def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randn(shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape)
    try: reference_output = gpt2_layer(input)
    except: reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")
# %%
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        m = t.mean(residual, dim=-1, keepdim=True)
        v = t.var(residual, dim=-1, keepdim=True)
        print(m.shape)
        print(v.shape)
        normalized = (residual - m) / t.sqrt(v + self.cfg.layer_norm_eps) 
        out = normalized * self.w + self.b
        return out


if MAIN:
    rand_float_test(LayerNorm, [2, 4, 768])
    load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])
# %%
class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]


if MAIN:
    rand_int_test(Embed, [2, 4])
    load_gpt2_test(Embed, reference_gpt2.embed, tokens)
# %%
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        idx = einops.repeat(t.arange(tokens.shape[1]), 'p -> b p', b=tokens.shape[0])
        return self.W_pos[idx]


if MAIN:
    rand_int_test(PosEmbed, [2, 4])
    load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)
# %%
import circuitsvis as cv
from IPython.display import display


if MAIN:
    print(reference_gpt2.to_str_tokens(reference_text))
    print(cache["pattern", 0][0].shape)
    
    display(cv.attention.attention_patterns(
        tokens=reference_gpt2.to_str_tokens(reference_text), 
        attention=cache["pattern", 0][0]
    ))
# %%
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # Linear map from input -> query, key shape [batch, seq_posn, head_index, d_head]
        Q = einops.einsum(normalized_resid_pre, self.W_Q, 'batch pos d_model, n_head d_model d_head -> batch pos n_head d_head') + self.b_Q
        K = einops.einsum(normalized_resid_pre, self.W_K, 'batch pos d_model, n_head d_model d_head -> batch pos n_head d_head') + self.b_K

        # Dot product every pair of queries and keys to get attn_scores [batch, head_index, query_pos, key_pos] (query = dest, key = source)
        attn_scores = einops.einsum(Q, K, 'batch q_pos n_head d_head, batch k_pos n_head d_head -> batch n_head q_pos k_pos')

        # Scale and mask attn_scores to make it lower triangular, i.e. causal
        scale = np.sqrt(self.cfg.d_head)
        attn_scores /= scale
        attn_scores = self.apply_causal_mask(attn_scores)

        # Softmax along the key_pos dimension, to get a probability distribution for each query (destination) token - this is our attention pattern!
        attn_pattern = t.softmax(attn_scores, dim=-1)

        # Linear map from input -> value [batch, key_pos, head_index, d_head]
        V = einops.einsum(normalized_resid_pre, self.W_V, 'batch pos d_model, n_head d_model d_head -> batch pos n_head d_head') + self.b_V

        # Mix along the key_pos with attn pattern to get z, which is a weighted average of the value vectors [batch, query_pos, head_index, d_head]
        z = einops.einsum(attn_pattern, V, 'batch n_head q_pos k_pos, batch k_pos n_head d_head -> batch q_pos n_head d_head')

        # Map to output, [batch, position, d_model] (position = query_pos, we've summed over all heads)
        out = einops.einsum(z, self.W_O, 'batch q_pos n_head d_head, n_head d_head d_model -> batch q_pos d_model') + self.b_O
        return out

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        mask = t.ones((attn_scores.shape[-2], attn_scores.shape[-1])).to(device)
        mask = t.triu(mask, diagonal=1)
        attn_scores = t.where(t.triu(mask, diagonal=1).bool(), self.IGNORE, attn_scores)
        return attn_scores


if MAIN:
    rand_float_test(Attention, [2, 4, 768])
    load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])
# %%

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        keys = einops.einsum(normalized_resid_mid, self.W_in, "batch pos d_model, d_model d_mlp -> batch pos d_mlp") + self.b_in
        activations = gelu_new(keys)
        return einops.einsum(activations, self.W_out, "batch pos d_mlp, d_mlp d_model -> batch pos d_model") + self.b_out


if MAIN:
    rand_float_test(MLP, [2, 4, 768])
    load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])
# %%

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        attn_out = self.attn(self.ln1(resid_pre))
        resid_mid = attn_out + resid_pre
        mlp_out = self.mlp(self.ln2(resid_mid))
        return resid_mid + mlp_out


if MAIN:
    rand_float_test(TransformerBlock, [2, 4, 768])
    load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])
# %%
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        # SOLUTION
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post

if MAIN:
    rand_float_test(TransformerBlock, [2, 4, 768])
    load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])
# %%
