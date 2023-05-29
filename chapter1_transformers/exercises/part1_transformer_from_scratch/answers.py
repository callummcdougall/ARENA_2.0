# %%F.

import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
import einops
from einops import einsum
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

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


if MAIN:
    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False
    )

# %%
if MAIN:
    print(reference_gpt2.to_str_tokens("Ralph"))
    print(reference_gpt2.to_str_tokens(" Ralph"))
    print(reference_gpt2.to_str_tokens(" ralph"))
    print(reference_gpt2.to_str_tokens("ralph"))
# %%
if MAIN:
    reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000")

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


def print_matches(predicted, real):
    results = [
        colored(r, "green" if r == p else "red") for r, p in zip(real, predicted)
    ]
    print("".join(results))


# %%
if MAIN:
    most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(
        logits.argmax(dim=-1)[0]
    )

    list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens))
    list(zip(reference_gpt2.to_str_tokens(tokens)[1:], most_likely_next_tokens))
    print_matches(most_likely_next_tokens, reference_gpt2.to_str_tokens(tokens)[1:])
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

# batch = 1
# position = 35
# d_model = 768
# n_heads = 12
# n_layers = 12
# d_mlp = 3072 (= 4 * d_model)
# d_head = 64 (= d_model / n_heads)

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

# As a reference - note there's a lot of stuff we don't care about in here, to do with library internals or other architectures

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
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape)
    try:
        reference_output = gpt2_layer(input)
    except:
        reference_output = gpt2_layer(input, input, input)
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

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        dims = -1
        mean = residual.mean(dim=dims, keepdim=True)
        var = residual.var(dim=dims, unbiased=False, keepdim=True)
        normalized = (residual - mean) / (var + cfg.layer_norm_eps).sqrt()
        return (normalized * self.w) + self.b


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

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
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

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        num_positions = tokens.shape[-1]
        return self.W_pos[:num_positions]


if MAIN:
    rand_int_test(PosEmbed, [2, 4])
    load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)

# %%

import circuitsvis as cv
from IPython.display import display


if MAIN:
    # html = cv.attention.attention_patterns(
    html = cv.attention.attention_heads(
        tokens=reference_gpt2.to_str_tokens(reference_text),
        attention=cache["pattern", 0][0],
    )
    display(html)

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
        q = (
            einops.einsum(
                normalized_resid_pre,
                self.W_Q,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
            )
            + self.b_V
        )
        att_score = einops.einsum(
            q,
            k,
            "batch posn_q n_heads d_head, batch posn_k n_heads d_head -> batch n_heads posn_q posn_k",
        )

        scaled_scores = att_score / math.sqrt(self.cfg.d_head)
        masked_scores = self.apply_causal_mask(scaled_scores)
        pattern = t.softmax(masked_scores, dim=-1)

        weighted_values = einops.einsum(
            pattern,
            v,
            "batch n_heads posn_q posn_k, batch posn_k n_heads d_head -> batch posn_q n_heads d_head",
        )
        output = (
            einsum(
                weighted_values,
                self.W_O,
                "batch posn n_heads d_head, n_heads d_head d_model -> batch posn d_model",
            )
            + self.b_O
        )
        return output

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        batch, n_heads, query_pos, key_pos = attn_scores.shape
        if query_pos != key_pos:
            raise ValueError("Num queries not equal number keys")
        mask = ~t.tril(t.ones_like(attn_scores, dtype=t.bool))
        if self.cfg.debug:
            print(mask[0, 0, :3, :3])
        result = t.masked_fill(attn_scores, mask, value=self.IGNORE)
        if self.cfg.debug:
            print(f"{mask[0,0, :3, :3]=}")
            print(f"{result[0,0, :3, :3]=}")
        return result


if MAIN:
    rand_float_test(Attention, [2, 4, 768])
    load_gpt2_test(
        Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"]
    )


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
        hidden = normalized_resid_mid @ self.W_in + self.b_in
        relu_out = gelu_new(hidden)
        return relu_out @ self.W_out + self.b_out


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
        resid_mid = resid_pre + attn_out

        mlp_out = self.mlp(self.ln2(resid_mid))
        resid_post = resid_mid + mlp_out

        return resid_post


if MAIN:
    rand_float_test(TransformerBlock, [2, 4, 768])
    load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])


# %%
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return normalized_resid_final @ self.W_U + self.b_U


if MAIN:
    rand_float_test(Unembed, [2, 4, 768])
    load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])


# %%
class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        embed = self.embed(tokens)
        pos_emb = self.pos_embed(tokens)
        resid = embed + pos_emb
        for b in self.blocks:
            resid = b(resid)
        resid = self.ln_final(resid)
        unemb = self.unembed(resid)
        return unemb


if MAIN:
    rand_int_test(DemoTransformer, [2, 4])
    load_gpt2_test(DemoTransformer, reference_gpt2, tokens)


# %%
if MAIN:
    demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
    demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

    demo_logits = demo_gpt2(tokens)
# %%


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = (
        log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )

    return log_probs_for_tokens


if MAIN:
    pred_log_probs = get_log_probs(demo_logits, tokens)
    print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
    print(
        f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}"
    )
    print(
        f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}"
    )
# %%

if MAIN:
    test_string = """The Total Perspective Vortex derives its picture of the whole Universe on the principle of"""
    for i in tqdm(range(100)):
        test_tokens = reference_gpt2.to_tokens(test_string).to(device)
        demo_logits = demo_gpt2(test_tokens)
        test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

    print(test_string)

# %%

if MAIN:
    model_cfg = Config(
        debug=False,
        d_model=256,
        n_heads=4,
        d_head=64,
        d_mlp=1024,
        n_layers=2,
        n_ctx=256,
        d_vocab=reference_gpt2.cfg.d_vocab,
    )
    model = DemoTransformer(model_cfg)

# %%


@dataclass
class TransformerTrainingArgs:
    batch_size = 8
    max_epochs = 1
    max_steps = 1000
    log_every = 10
    lr = 1e-3
    weight_decay = 1e-2
    log_dir: str = os.getcwd() + "/logs"
    log_name: str = "day1-transformer"
    run_name: Optional[str] = None
    log_every_n_steps: int = 1


if MAIN:
    args = TransformerTrainingArgs()

# %
# %%

if MAIN:
    dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns(
        "meta"
    )
    print(dataset)
    print(dataset[0]["text"][:100])

# %%
if MAIN:
    tokenized_dataset = tokenize_and_concatenate(
        dataset,
        reference_gpt2.tokenizer,
        streaming=False,
        max_length=model.cfg.n_ctx,
        column_name="text",
        add_bos_token=True,
        num_proc=4,
    )
    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
# %%
if MAIN:
    first_batch = data_loader.dataset[: args.batch_size]

    print(first_batch.keys())
    print(first_batch["tokens"].shape)
# %%


class LitTransformer(pl.LightningModule):
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer, data_loader: DataLoader):
        super().__init__()
        self.model = model
        self.cfg = model.cfg
        self.args = args
        self.data_loader = data_loader

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        logits = self.model(tokens)
        return logits

    def training_step(self, batch: dict, batch_idx: int) -> Float[Tensor, ""]:
        '''
        Here you compute and return the training loss and some additional metrics for e.g. 
        the progress bar or logger.
        '''
        tokens = batch['tokens'].to(device)
        logits: Float[Tensor, "batch position d_vocab"] = self(tokens)
        log_probs: Float[Tensor, "batch posn-1"] = get_log_probs(logits=logits, tokens=tokens)
        loss = -log_probs.mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        '''
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        '''
        return t.optim.AdamW(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

    def train_dataloader(self):
        return self.data_loader


# %%

# if MAIN:
#     litmodel = LitTransformer(args, model, data_loader)
#     logger = WandbLogger(save_dir=args.log_dir, project=args.log_name, name=args.run_name)

#     trainer = pl.Trainer(
#         max_epochs=args.max_epochs,
#         logger=logger,
#         log_every_n_steps=args.log_every_n_steps
#     )
#     trainer.fit(model=litmodel, train_dataloaders=litmodel.data_loader)
#     wandb.finish()

# %%
if MAIN:
    model_cfg = Config()
    model = DemoTransformer(model_cfg).to(device)
    model.load_state_dict(reference_gpt2.state_dict(), strict=False)

    tokenizer = reference_gpt2.tokenizer
# %%
class TransformerSampler:

    def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs):
        '''
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how 
        new tokens are chosen.
        '''
        tokens: List[Int] = self.tokenizer.encode(prompt)
        predicted_tokens = []

        while len(predicted_tokens) < max_tokens_generated:

            input_ids = t.tensor(tokens + predicted_tokens)
            logits: Float[Tensor, "batch=1, pos, d_vocab"] = self.model(input_ids)
            prediction = self.sample_next_token(logits.squeeze(0), logits)
            predicted_tokens.append(prediction)

            if prediction == self.tokenizer.eos_token:
                break

        return tokenizer.decode(tokens + predicted_tokens)


    @t.inference_mode()
    def beam_search(
        self,
        prompt: str, 
        num_return_sequences: int, 
        num_beams: int, 
        max_new_tokens: int, 
        no_repeat_ngram_size: int = 0,
        verbose=False
    ) -> List[Tuple[float, t.Tensor]]:
        '''
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how 
        new tokens are chosen.
        '''
        pass


    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len d_vocab"], 
        logits: Float[Tensor, "seq_len d_vocab"], 
        temperature=1.0, 
        top_k=0, 
        top_p=0.0, 
        frequency_penalty=0.0,
        seed=None
    ) -> int:
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(input_ids, logits, frequency_penalty)
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)


    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        '''
        Returns the most likely token (as an int).
        '''
        out = logits.argmax().item()
        return out


    @staticmethod
    def apply_temperature(logits: Float[Tensor, "d_vocab"], temperature: float) -> Float[Tensor, "d_vocab"]:
        '''
        Applies temperature scaling to the logits.
        '''
        pass

    @staticmethod
    def apply_frequency_penalty(input_ids: Int[Tensor, "seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float) -> Float[Tensor, "d_vocab"]:
        '''
        Applies a frequency penalty to the logits.
        '''
        pass

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        '''
        Samples from the distribution defined by the logits.
        '''
        pass

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        '''
        Samples from the top k most likely tokens.
        '''
        pass

    @staticmethod
    def sample_top_p(logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1) -> int:
        '''
        Samples from the most likely tokens which make up at least p cumulative probability.
        '''
        pass
# %%
if MAIN:
    sampler = TransformerSampler(model, tokenizer)

    prompt = "Jingle bells, jingle bells, jingle all the way"
    print(f"Greedy decoding with prompt: {prompt!r}\n")

    output = sampler.sample(prompt, max_tokens_generated=8, temperature=0.0)
    print(f"Your model said: {output!r}\n")

    expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
    assert output == expected

    print("Tests passed!")

# %%