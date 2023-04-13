# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.chdir("..")
from transformer_from_scratch.solutions import lm_cross_entropy_loss, Config, DemoTransformer
from training_and_sampling import tests, plot_utils
os.chdir("training_and_sampling")
# print(os.getcwd())

# from IPython import get_ipython
# ipython = get_ipython()
# ipython.magic("load_ext autoreload")
# ipython.magic("autoreload 2")

MAIN = __name__ == "__main__"

import re
import sys
import torch as t
from torch import nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Union, Optional, Callable, Dict
import einops
from dataclasses import dataclass
from frozendict import frozendict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from torchtyping import TensorType as TT
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, PreTrainedTokenizer
import time
import numpy as np

if MAIN:
    device = t.device("cuda" if t.cuda.is_available() else "cpu")





# %% SECTION 1: TRAINING

# Load the text data
if MAIN:
    with open("shakespeare-corpus.txt", encoding="utf-8") as file:
        text = file.read()

        # while "  " in text:
        #     text = re.sub("  ", " ", text)
        # while "\n\n\n" in text:
        #     text = re.sub("\n\n\n", "\n\n", text)

        # text = re.split(r"\b", text)

        # tokens = tokenizer.encode(text, return_tensors="pt")
        # words = re.split(r"\b", text)

# %%
class SimpleTokenizer():

    def __init__(self, text: str):
        self.text = text
        self.words = sorted(set(re.split(r"\b", text)))
        self.unk = len(self.words) + 1
        # self.bos_token_id = len(self.words) + 2
        self.word_to_index = {word: index for index, word in enumerate(self.words)}
        self.index_to_word = {index: word for index, word in enumerate(self.words)}

    def encode(self, input_text, return_tensors: Optional[str] = None) -> Union[List, t.Tensor]:
        '''
        Tokenizes and encodes the input text.

        If `return_tensors` is None, should return list of Python integers.
        If `return_tensors` is "pt", should return a PyTorch tensor of shape (1, num_tokens).
        '''
        split_text = re.split(r"\b", input_text)
        encoding = [self.word_to_index.get(word, self.unk) for word in split_text]
        if self.unk in encoding:
            print(f"Warning: Unknown token found in input text")
        if return_tensors == "pt":
            return t.tensor(encoding).unsqueeze(0)
        return encoding

    def decode(self, tokens: Union[List, t.Tensor]):
        '''
        Decodes the tokens into a string of text.
        '''
        if isinstance(tokens, t.Tensor) and tokens.dim() == 2:
            assert tokens.size(0) == 1, "Only batch size 1 is supported"
            tokens = tokens[0]
        return "".join([self.index_to_word[token] for token in tokens])


if MAIN:
    mytokenizer = SimpleTokenizer(text)

    # Some basic testing
    assert isinstance(mytokenizer.encode("Macbeth"), list)
    assert isinstance(mytokenizer.encode("Macbeth", return_tensors="pt"), t.Tensor)
    assert mytokenizer.decode(mytokenizer.encode("Macbeth")) == "Macbeth"
    assert mytokenizer.index_to_word[mytokenizer.encode("Macbeth")[1]]


# %%

def prepare_text(text: str, max_seq_len: int, tokenizer: SimpleTokenizer) -> TT["batch", "max_seq_len"]:
    '''
    Takes a string of text, and returns an array of tokens rearranged into chunks of size max_seq_len.
    '''
    tokens: TT[1, "num_tokens"] = tokenizer.encode(text, return_tensors="pt")

    # We want to rearrange the tokens into chunks of size max_seq_len.
    num_tokens = tokens.size(1) - (tokens.size(1) % max_seq_len)
    tokens = einops.rearrange(
        tokens[0, :num_tokens], "(chunk seq_len) -> chunk seq_len", seq_len=max_seq_len
    )

    # # Append the start token to the beginning of each chunk
    # tokens = t.cat([
    #     t.full((tokens.size(0), 1), tokenizer.bos_token_id, dtype=t.long), 
    #     tokens
    # ], dim=1)

    return tokens


if MAIN:
    max_seq_len=48
    tokens = prepare_text(text[:500], max_seq_len=max_seq_len, tokenizer=mytokenizer)
    print("Does this size look reasonable, as a tokenization of the first 500 characters?\n", tokens.shape)

# %%

class TensorDataset:
    def __init__(self, *tensors: t.Tensor):
        '''Validate the sizes and store the tensors in a field named `tensors`.'''
        batch_sizes = [tensor.shape[0] for tensor in tensors]
        assert len(set(batch_sizes)) == 1, "All tensors must have the same size in the first dimension"
        self.tensors = tensors

    def __getitem__(self, index: Union[int, slice]) -> Tuple[t.Tensor, ...]:
        '''Return a tuple of length len(self.tensors) with the index applied to each.'''
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        '''Return the size in the first dimension, common to all the tensors.'''
        return self.tensors[0].shape[0]


if MAIN:
    tests.test_tensor_dataset(TensorDataset)
    dataset = TensorDataset(tokens)

# %%

from frozendict import frozendict

@dataclass
class TransformerTrainingArgs():
    tokenizer: SimpleTokenizer = mytokenizer
    epochs: int = 3
    batch_size: int = 4
    max_seq_len: int = 48
    optimizer: Callable[..., t.optim.Optimizer] = t.optim.Adam
    optimizer_kwargs: Dict = frozendict(lr=0.001, betas=(0.9, 0.999))
    device: str = "cuda" if t.cuda.is_available() else "cpu"
    filename_save_model: str = "transformer_shakespeare.pt"

# %%

def train_transformer(model: DemoTransformer, text: str, args: TransformerTrainingArgs) -> Tuple[list, list]:
    '''
    Trains an autoregressive transformer on the data in the trainset.

    Returns tuple of (train_loss, test_loss), containing the cross entropy losses for the thing.
    '''
    # Prepare the tokens, take a random train/test split, and create the dataloaders
    tokens = prepare_text(text, max_seq_len=args.max_seq_len, tokenizer=args.tokenizer)
    randperm = t.randperm(tokens.size(0))
    len_trainset = int(0.9 * tokens.size(0))
    trainset = TensorDataset(tokens[randperm[:len_trainset]])
    testset = TensorDataset(tokens[randperm[len_trainset:]])
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    model.to(args.device)
    optimizer = args.optimizer(model.parameters(), **args.optimizer_kwargs)
    train_loss_list = []
    test_loss_list = []

    for epoch in range(args.epochs):

        progress_bar = tqdm(trainloader, desc="Calculating training loss")
        for (tokens,) in progress_bar:

            tokens = tokens.to(args.device)

            logits = model(tokens)
            loss = lm_cross_entropy_loss(logits, tokens)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}, Loss = {loss:.3f}")

        with t.inference_mode():

            test_loss = 0.0
            total = 0

            progress_bar = tqdm(testloader, desc="Calculating test loss")
            for (tokens,) in progress_bar:

                tokens = tokens.to(args.device)

                logits = model(tokens)

                test_loss += lm_cross_entropy_loss(logits, tokens) * tokens.size(0)
                total += tokens.size(0)

            test_loss /= total
            test_loss_list.append(test_loss.item())

        print(f"Train loss = {loss:.4f}, Test loss = {test_loss:.4f}")

    print(f"\nSaving model to: {args.filename_save_model}")
    t.save(model, args.filename_save_model)
    return train_loss_list, test_loss_list

# %%

if MAIN:
    config = Config(
        d_model = 384,
        layer_norm_eps = 1e-5,
        d_vocab = len(mytokenizer.words),
        init_range = 0.02,
        n_ctx = max_seq_len,
        d_head = 64,
        d_mlp = 1536,
        n_heads = 6,
        n_layers = 4
    )

    model = DemoTransformer(config)

    args = TransformerTrainingArgs(
        tokenizer = mytokenizer,
        batch_size = 16,
        epochs = 4,
    )

    train_loss_list, test_loss_list = train_transformer(model, text, args)

    plot_utils.plot_two_lines(
        y1 = train_loss_list,
        y2 = test_loss_list,
        x2 = list(range(
            len(train_loss_list) // len(test_loss_list), 
            len(train_loss_list) + 1,
            len(train_loss_list) // len(test_loss_list)
        )),
        name1 = "Train loss",
        name2 = "Test loss",
        title = "Loss curve for transformer trained on Shakespeare corpus",
        xaxis = "Batches seen",
        yaxis = "Cross entropy loss"
    )
    









# %% SECTION 2: SAMPLING


if MAIN:
    reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
    gpt2 = DemoTransformer(Config())
    gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
    gpt2.to(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %%

def greedy_search(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    '''
    out = logits.argmax().item()
    assert isinstance(out, int)
    return out

def sample_basic(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    distribution = t.distributions.categorical.Categorical(logits=logits)
    out = distribution.sample().item()
    assert isinstance(out, int)
    return out

def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    '''
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    assert temperature > 0
    return logits / temperature

def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    '''
    input_ids: shape (seq, )
    logits: shape (vocab_size, )
    Return: shape (vocab_size, )
    '''
    (vocab_size,) = logits.shape
    id_freqs = t.bincount(input_ids, minlength=vocab_size)
    return logits - freq_penalty * id_freqs

def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    '''
    top_logits, top_idx = t.topk(logits, top_k)
    idx = t.distributions.categorical.Categorical(logits=top_logits).sample()
    return top_idx[idx].item()

def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    Return: a sampled token
    '''
    logits_sorted, indices = logits.sort(descending=True, stable=True)
    cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
    n_keep = t.searchsorted(cumul_probs, top_p, side="left").item() + 1
    n_keep = max(n_keep, min_tokens_to_keep)
    keep_idx = indices[:n_keep]
    keep_logits = logits[keep_idx]
    sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
    return keep_idx[sample].item()

# %%

def apply_sampling_methods(
    input_ids: t.Tensor, 
    logits: t.Tensor, 
    temperature=1.0, 
    freq_penalty=0.0, 
    top_k=0, 
    top_p=0.0,
    seed=0
) -> int:
    '''
    Return the next token, sampled from the model's probability distribution with modifiers.
    input_ids: shape (seq,)
    '''
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0, "Temperature should be non-negative"
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

    # Set random seeds for reproducibility
    t.manual_seed(seed)
    np.random.seed(seed)

    if temperature == 0:
        return greedy_search(logits)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if freq_penalty != 0.0:
        logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)


@t.inference_mode()
def sample_tokens(
    model: DemoTransformer,
    tokenizer: PreTrainedTokenizer,
    initial_text: str,
    max_tokens_generated=30,
    **kwargs # kwargs are for params like temperature, top_k, etc
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    # Note - an alternative to model.eval() is to use the @t.inference_mode() decorator for this whole function.
    model.eval()
    input_ids: list = tokenizer.encode(initial_text)
    generated = []
    for _ in range(max_tokens_generated):
        new_input_ids = t.tensor(input_ids + generated, dtype=t.long, device=device)
        new_input_ids_truncated = new_input_ids[-min(model.cfg.n_ctx, new_input_ids.shape[0]):].unsqueeze(0)
        logits = model(new_input_ids_truncated)[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        generated.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input_ids + generated)

# %%

if MAIN:
    text_output = sample_tokens(
        model, 
        mytokenizer, 
        initial_text="Turn down for what ", 
        max_tokens_generated=100, 
        temperature=0.6
    )
    print(text_output)

# %%

if MAIN:
    initial_text = "The answer to life, the universe and everything is"
    output = sample_tokens(gpt2, tokenizer, initial_text, temperature=0.7, top_p=0.95, max_tokens_generated=64)
    print(f"Your model said: {output}")

# %%

@t.inference_mode()
def beam_search(
    model: DemoTransformer, 
    input_ids: t.Tensor, 
    num_return_sequences: int, 
    num_beams: int, 
    max_new_tokens: int, 
    tokenizer: PreTrainedTokenizer, 
    verbose=False
) -> List[Tuple[float, t.Tensor]]:
    '''
    input_ids: (seq, ) - the prompt

    max_new_tokens: stop after this many new tokens are generated, even if 
    no EOS is generated. In this case, the best incomplete sequences should 
    also be returned.
    verbose: if True, print the current (unfinished) completions after each 
    iteration (for debugging purposes).

    Return list of length num_return_sequences. Each element is a tuple of 
    (logprob, tokens) where the tokens include both prompt and completion, 
    sorted by descending logprob.
    '''
    assert num_return_sequences <= num_beams

    model.eval()
    
    # Create list to store the sequences to return
    # We only add to this when we generate an EOS token, or at the very end
    final_logitsums_and_completions = []

    # Create list to store the current best completions and their logit scores
    best_logitsums_and_completions = [(0, input_ids.tolist())]

    for n in tqdm(range(max_new_tokens)):
        
        # Create a list to store the completions at this stage
        new_best_logitsums_and_completions = []

        # This section loops through all completions so far, and get the next words
        for (logitsum, completion) in best_logitsums_and_completions:

            # Get output (we only care about the vector of logits for the next token)
            output = model(t.tensor(completion).unsqueeze(0).to(device, t.long))
            output = output[0, -1, :].log_softmax(-1)

            # Find the top `num_beams` tokens (because this is the maximum we might need)
            topk_logits, topk_indices = t.topk(output, k=num_beams)

            # Append to the new best completions list
            for logit, idx in zip(topk_logits, topk_indices):
                new_completion_and_logit = (logitsum + logit.item(), completion + [idx.item(),])
                new_best_logitsums_and_completions.append(new_completion_and_logit)

        # This section updates (and sorts) the list of best completions, and also updates `final_logitsums_and_completions` if EOS was produced
        best_logitsums_and_completions = []
        for (logitsum, completion) in sorted(new_best_logitsums_and_completions, key=lambda x: -x[0]):
            
            # If token is eos then add it to final_logitsums_and_completions
            if completion[-1] == tokenizer.eos_token_id:
                final_logitsums_and_completions.append((logitsum, completion))
            
            # Else add it to best_logitsums_and_completions
            # And if that list has length num_beams, we break out of this loop
            else:
                best_logitsums_and_completions.append((logitsum, completion))
                if len(best_logitsums_and_completions) == num_beams:
                    break

        if n == max_new_tokens - 1:
            # If we're at the end, add the best completions to the final completions list, then keep only the best `num_return_sequences` completions
            final_logitsums_and_completions.extend(best_logitsums_and_completions)
            final_logitsums_and_completions = sort_by_logits_and_crop(final_logitsums_and_completions, max_size=num_return_sequences)
            if verbose: print_sequences(f"Returning best {num_return_sequences =} completions:", final_logitsums_and_completions, tokenizer)
        else:
            # If not, then keep only the best `num_beams` completions
            final_logitsums_and_completions = sort_by_logits_and_crop(final_logitsums_and_completions, max_size=num_beams)
            if verbose: print_sequences(f"Printing {num_beams =} best completions:", best_logitsums_and_completions, tokenizer)

    return final_logitsums_and_completions


def print_sequences(name, logitsums_and_completions, tokenizer: PreTrainedTokenizer):
    '''
    Prints out a set of sequences with their corresopnding logitsums.

    prefix: message which is printed out before any of the sequences (to provide context)
    logitsums_and_completions: list of tuples of (logitsum: float, completion: List[int])
    tokenizer: used to decode the completion
    '''
    if len(logitsums_and_completions) == 0:
        return
    print("\n" + name + "\n")
    print("logitsum | completion")
    for logit_sum, completion in logitsums_and_completions:
        text = tokenizer.decode(completion)
        print(f"{logit_sum:>8.3f} | {text}")


def sort_by_logits_and_crop(logitsums_and_completions, max_size):
    '''
    Given a list of tuples of (logitsum: float, completion: List[int]), returns the same
    list sorted in descending order of logitsum (and cropped to size max_size).
    '''
    logitsums_and_completions = sorted(logitsums_and_completions, key=lambda x: x[0], reverse=True)
    logitsums_and_completions = logitsums_and_completions[:min(max_size, len(logitsums_and_completions))]
    return logitsums_and_completions

# %%

if MAIN:
    initial_text = "The answer to life, the universe and everything is"
    input_ids = tokenizer.encode(initial_text, return_tensors="pt").squeeze()
    num_return_sequences = 5
    num_beams = 10
    max_new_tokens = 30

    final_logitsums_and_completions = beam_search(gpt2, input_ids, num_return_sequences, num_beams, max_new_tokens, tokenizer, verbose=False)

    text = tokenizer.decode(final_logitsums_and_completions[0][1])
    print("\n" + text)












# %% SECTION 3: CACHING

if MAIN:
    from transformer_from_scratch.solutions import LayerNorm, MLP, Embed, Unembed

class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: t.Tensor, past_kv_pos_offset: int = 0):
        # tokens: [batch, position]
        batch, seq_len = tokens.shape
        return einops.repeat(
            self.W_pos[past_kv_pos_offset: seq_len+past_kv_pos_offset], 
            "seq d_model -> batch seq d_model", batch=batch
        )

class KeyValueCacheEntry():
    '''
    This holds the old key and value vectors for a single attention block.
    '''
    k: TT["batch", "seq_len", "n_heads", "d_head"]
    v: TT["batch", "seq_len", "n_heads", "d_head"]

    def __init__(self, cfg: Config, batch: int):
        '''
        Initialise k and v as empty tensors (i.e. with seq_len=0)
        '''
        self.cfg = cfg
        self.batch = batch
        self.k = t.empty((batch, 0, cfg.n_heads, cfg.d_head), device=device)
        self.v = t.empty((batch, 0, cfg.n_heads, cfg.d_head), device=device)
    
    def clone(self):
        '''
        Returns a copy of this object.
        '''
        new_entry = KeyValueCacheEntry(self.cfg, self.batch)
        new_entry.k = self.k.clone()
        new_entry.v = self.v.clone()
        return new_entry

class KeyValueCache():
    '''
    This holds a list of KeyValueCacheEntry objects, one for each layer in the model.
    In our forward pass, we iterate through the cache entries stored in this object.
    '''
    entries: List[KeyValueCacheEntry] = []

    def __init__(self, cfg: Config, batch: int = 1):
        self.cfg = cfg
        self.batch = batch
        self.entries = [KeyValueCacheEntry(cfg, batch) for _ in range(cfg.n_layers)]

    def __getitem__(self, idx):
        return self.entries[idx]
    
    def clone(self):
        '''
        Returns a copy of this object.
        '''
        new_cache = KeyValueCache(self.cfg, self.batch)
        new_cache.entries = [entry.clone() for entry in self.entries]
        return new_cache


class Attention(nn.Module):
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
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device="cuda"))

    def forward(self, normalized_resid_pre: t.Tensor, cache_entry: Optional[KeyValueCacheEntry] = None):
        # normalized_resid_pre: [batch, position, d_model]

        # Calculate the new query, key and value vectors
        q = einops.einsum(
            normalized_resid_pre, self.W_Q,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head" 
        ) + self.b_Q
        k = einops.einsum(
            normalized_resid_pre, self.W_K,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head" 
        ) + self.b_K
        v = einops.einsum(
            normalized_resid_pre, self.W_V,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head" 
        ) + self.b_V

        # If cache_entry is not None, this means we use the previous key and value vectors
        # Also we'll need to update the cache with new key and value vectors
        if cache_entry is not None:
            cache_entry.k = k = t.cat([cache_entry.k, k], dim=1)
            cache_entry.v = v = t.cat([cache_entry.v, v], dim=1)

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = einops.einsum(
            q, k, 
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K"
        )
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head ** 0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z = einops.einsum(
            v, attn_pattern, 
            "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head"
        )

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        out = einops.einsum(
            z, self.W_O, 
            "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model"
        ) + self.b_O

        return out

    def apply_causal_mask(self, attn_scores: t.Tensor):
        '''
        Here, attn_scores have shape (batch, n_heads, query_pos, key_pos), where query_pos represents the 
        new (non-cached) positions, and key_pos represent all the positions (cached and non-cached).

        So when we create our mask, the query indices and key indices will both go up to the same value
        (the full sequence length), but the query indices will start at >0.
        '''
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        new_seq_len, full_seq_len = attn_scores.shape[-2:]
        assert new_seq_len <= full_seq_len
        q_posn = einops.repeat(attn_scores.new_tensor(range(full_seq_len-new_seq_len, full_seq_len)), "q -> q k", k=full_seq_len)
        k_posn = einops.repeat(attn_scores.new_tensor(range(full_seq_len)), "k -> q k", q=new_seq_len)
        attn_scores = attn_scores.masked_fill(q_posn < k_posn, self.IGNORE)
        return attn_scores



class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre: t.Tensor, cache_entry: Optional[KeyValueCacheEntry]):
        # resid_pre: [batch, position, d_model]
        # output: [batch, position, d_model]
        resid_mid = self.attn(self.ln1(resid_pre), cache_entry) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post



class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: TT["batch", "seq_pos"], cache: Optional[KeyValueCache] = None):
        # tokens [batch, position]

        # If not using cache, turn it into a list of None's (so we can iterate through it)
        if cache is None:
            cache = [None] * len(self.blocks)
            residual = self.embed(tokens) + self.pos_embed(tokens)
        # If using cache, then we only need to pass forward the newest tokens
        # Remember to add positional offset!
        else:
            n_cached_tokens = cache[0].k.shape[1]
            tokens = tokens[:, n_cached_tokens:]
            residual = self.embed(tokens) + self.pos_embed(tokens, n_cached_tokens)

        for block, cache_entry in zip(self.blocks, cache):
            residual = block(residual, cache_entry)
        
        logits = self.unembed(self.ln_final(residual))
        return logits


@t.inference_mode()
def sample_tokens_with_cache(
    model: DemoTransformer,
    tokenizer: PreTrainedTokenizer,
    initial_text: str,
    max_tokens_generated: int = 30,
    cache: Optional[KeyValueCache] = None,
    **kwargs # kwargs are for params like temperature, top_k, etc
) -> str:
    '''
    Does the exact same thing as sample_tokens, but using cache to be faster.

    If cache is None, it doesn't use cache (same as previous function).
    If cache is an empty cache, it will use cache to generate tokens (assuming none have been cached so far).
    If cache is a non-empty cache, it will assume the cache contains keys/values for the first n tokens.
    '''
    model.eval()
    input_ids: list = tokenizer.encode(initial_text)
    generated = []
    for _ in tqdm(range(max_tokens_generated)):
        new_input_ids = t.tensor(input_ids + generated, dtype=t.long, device=device)
        new_input_ids_truncated = new_input_ids[-min(model.cfg.n_ctx, new_input_ids.shape[0]):].unsqueeze(0)
        logits = model(new_input_ids_truncated, cache)[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        input_ids.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None): 
            break
    return tokenizer.decode(input_ids)


def format_output(output: str, starting_char="> ") -> str:
    return(starting_char + output.replace("\n", "\n" + starting_char))

if MAIN:
    gpt2 = DemoTransformer(Config())
    gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
    gpt2.to(device)

    initial_text = "The answer to life, the universe and everything is"

    t0 = time.time()
    output_without_cache = sample_tokens_with_cache(gpt2, tokenizer, initial_text, temperature=0.7, top_p=0.95, max_tokens_generated=100, cache=None)
    print(f"Time taken (without cache): {time.time() - t0:.2f} seconds")
    print("Model output:\n" + format_output(output_without_cache))

    t0 = time.time()
    output_with_cache = sample_tokens_with_cache(gpt2, tokenizer, initial_text, temperature=0.7, top_p=0.95, max_tokens_generated=100, cache=KeyValueCache(gpt2.cfg))
    print(f"Time taken (with cache): {time.time() - t0:.2f} seconds")
    print("Model output:\n" + format_output(output_with_cache))

    assert output_with_cache == output_without_cache, "Your outputs are different, meaning you've probably made a mistake in your cache implementation."












# %% BONUS: BEAM SEARCH WITH CACHE

# Note, this solution is incomplete and pretty inefficient (especially in terms of memory)
# It uses a different cache for every sequence it keeps track of
# A better datastructure would be a tree of caches:
# > Each node in the tree represents a possible next token for the sequence (root node is initial prompt)
# > Each node has a cache containing the key/value vectors for that token
# > To get a full set of keys/values to use as the cache_entry object, we concatenate along a path in the tree until we get to the root node




    



@t.inference_mode()
def beam_search_with_cache(
    model: DemoTransformer, 
    input_ids: t.Tensor, 
    num_return_sequences: int, 
    num_beams: int, 
    max_new_tokens: int, 
    tokenizer: PreTrainedTokenizer, 
    cache: Optional[KeyValueCache] = None,
    verbose = False,
) -> List[Tuple[float, t.Tensor]]:
    '''
    input_ids: (seq, ) - the prompt

    max_new_tokens: stop after this many new tokens are generated, even if 
    no EOS is generated. In this case, the best incomplete sequences should 
    also be returned.
    verbose: if True, print the current (unfinished) completions after each 
    iteration (for debugging purposes).

    Return list of length num_return_sequences. Each element is a tuple of 
    (logprob, tokens) where the tokens include both prompt and completion, 
    sorted by descending logprob.
    '''
    assert num_return_sequences <= num_beams

    model.eval()
    
    # Create list to store the sequences to return
    # We only add to this when we generate an EOS token, or at the very end
    final_lcc = []

    # Create list to store the current best completions, their logit scores, and caches
    using_cache = cache is not None
    best_lcc = [(0, input_ids.tolist(), cache)]

    for n in tqdm(range(max_new_tokens)):
        
        # Create a list to store the completions at this stage
        new_best_lcc = []

        # This section loops through all completions so far, and get the next words
        for (logitsum, completion, cache) in best_lcc:

            # Get output (we only care about the vector of logits for the next token)
            new_cache = cache.clone() if using_cache else None
            output = model(t.tensor(completion).unsqueeze(0).to(device, t.long), new_cache)
            output = output[0, -1, :].log_softmax(-1)

            # Find the top `num_beams` tokens (because this is the maximum we might need)
            topk_logits, topk_indices = t.topk(output, k=num_beams)

            # Append to the new best completions list
            for logit, idx in zip(topk_logits, topk_indices):
                new_lcc = (logitsum + logit.item(), completion + [idx.item(),], new_cache)
                new_best_lcc.append(new_lcc)
            if using_cache: del new_cache

        # This section updates (and sorts) the list of best completions, and also updates `final_lcc` if EOS was produced
        best_lcc = []
        for (logitsum, completion, cache) in sorted(new_best_lcc, key=lambda x: -x[0]):
            
            # If token is eos then add it to final_lcc
            if completion[-1] == tokenizer.eos_token_id:
                final_lcc.append((logitsum, completion, cache))
            
            # Else add it to best_lcc
            # And if that list has length num_beams, we break out of this loop
            else:
                best_lcc.append((logitsum, completion, cache))
                if len(best_lcc) == num_beams:
                    break

        if n == max_new_tokens - 1:
            # If we're at the end, add the best completions to the final completions list, then keep only the best `num_return_sequences` completions
            final_lcc.extend(best_lcc)
            final_lcc = sort_by_logits_and_crop(final_lcc, max_size=num_return_sequences)
            if verbose: print_sequences(f"Returning best {num_return_sequences =} completions:", final_lcc, tokenizer)
        else:
            # If not, then keep only the best `num_beams` completions
            final_lcc = sort_by_logits_and_crop(final_lcc, max_size=num_beams)
            if verbose: print_sequences(f"Printing {num_beams =} best completions:", best_lcc, tokenizer)

    return final_lcc


def print_sequences(name, lcc, tokenizer: PreTrainedTokenizer):
    '''
    Prints out a set of sequences with their corresopnding logitsums.

    prefix: message which is printed out before any of the sequences (to provide context)
    lcc: list of tuples of (logitsum: float, completion: List[int], cache: KeyValueCache)
    tokenizer: used to decode the completion
    '''
    if len(lcc) == 0:
        return
    print("\n" + name + "\n")
    print("logitsum | completion")
    for logit_sum, completion, cache in lcc:
        text = tokenizer.decode(completion)
        print(f"{logit_sum:>8.3f} | {text}")


def sort_by_logits_and_crop(lcc, max_size):
    '''
    Given a list of tuples of (logitsum: float, completion: List[int], cache: KeyValueCache), 
    returns the same list sorted in descending order of logitsum (and cropped to size max_size).
    '''
    lcc = sorted(lcc, key=lambda x: x[0], reverse=True)
    lcc = lcc[:min(max_size, len(lcc))]
    return lcc


if MAIN:
    initial_text = "The answer to life, the universe and everything is"
    input_ids = tokenizer.encode(initial_text, return_tensors="pt").squeeze()
    num_return_sequences = 5
    num_beams = 10
    max_new_tokens = 50

    t0 = time.time()
    final_logitsums_and_completions = beam_search_with_cache(gpt2, input_ids, num_return_sequences, num_beams, max_new_tokens, tokenizer, cache=None)
    print(f"Time taken (without cache): {time.time() - t0:.2f} seconds")
    text_without_cache = tokenizer.decode(final_logitsums_and_completions[0][1])
    print("Model output:\n" + format_output(text_without_cache))
    
    t0 = time.time()
    final_logitsums_and_completions = beam_search_with_cache(gpt2, input_ids, num_return_sequences, num_beams, max_new_tokens, tokenizer, cache=KeyValueCache(gpt2.cfg))
    print(f"Time taken (with cache): {time.time() - t0:.2f} seconds")
    text_with_cache = tokenizer.decode(final_logitsums_and_completions[0][1])
    print("Model output:\n" + format_output(text_with_cache))

    assert text_with_cache == text_without_cache, "Your outputs are different, meaning you've probably made a mistake in your cache implementation."


# %%
