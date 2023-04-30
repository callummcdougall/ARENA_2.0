import os
# if not os.path.exists("./images"):
#     os.chdir("./ch6")
import re, json
import plotly.io as pio

from st_dependencies import *
styling()

def img_to_html(img_path, width):
    with open("images/page_images/" + img_path, "rb") as file:
        img_bytes = file.read()
    encoded = base64.b64encode(img_bytes).decode()
    return f"<img style='width:{width}px;max-width:100%;st-bottom:25px' src='data:image/png;base64,{encoded}' class='img-fluid'>"
def st_image(name, width):
    st.markdown(img_to_html(name, width=width), unsafe_allow_html=True)

def read_from_html(filename):
    filename = f"images/{filename}.html" if "written_images" in filename else f"images/page_images/{filename}.html"
    with open(filename) as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html)[0]
    call_args = json.loads(f'[{call_arg_str}]')
    try:
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    except:
        del call_args[2]["template"]["data"]["scatter"][0]["fillpattern"]
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    return fig

NAMES = ["training_curve_1", "training_curve_2"]

def complete_fig_dict(fig_dict):
    for name in NAMES:
        if name not in fig_dict:
            fig_dict[name] = read_from_html(name)
    return fig_dict
if "fig_dict" not in st.session_state:
    st.session_state["fig_dict"] = {}
fig_dict_old = st.session_state["fig_dict"]
fig_dict = complete_fig_dict(fig_dict_old)
if len(fig_dict) > len(fig_dict_old):
    st.session_state["fig_dict"] = fig_dict

with open("images/page_images/attn_patterns_demo.html", "rb") as file:
    attn_patterns_demo = file.read()

def section_home():
    st.sidebar.markdown(r"""
## Table of contents

<ul class="contents">
    <li><a class="contents-el" href="#introduction">Introduction</a></li>
    <li><a class="contents-el" href="#imports">Imports</a></li>
    <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
</ul>""", unsafe_allow_html=True)
    st.markdown(r"""
Links to Colab: [**exercises**](https://colab.research.google.com/drive/1LpDxWwL2Fx0xq3lLgDQvHKM5tnqRFeRM?usp=share_link), [**solutions**](https://colab.research.google.com/drive/1ND38oNmvI702tu32M74G26v-mO5lkByM?usp=share_link)
""")
    st_image("transformer-building.png", 350)
    # start
    st.markdown(r"""
# Transformer from scratch

## Introduction

This is a clean, first principles implementation of GPT-2 in PyTorch. The architectural choices closely follow those used by the TransformerLens library (which you'll be using a lot more in later exercises).

If you enjoyed this, I expect you'd enjoy learning more about what's actually going on inside these models and how to reverse engineer them! This is a fascinating young research field, with a lot of low-hanging fruit and open problems! **I recommend starting with my post [Concrete Steps for Getting Started in Mechanistic Interpretability](https://www.neelnanda.io/mechanistic-interpretability/getting-started).**

This notebook was written to accompany my [TransformerLens library](https://github.com/neelnanda-io/TransformerLens) for doing mechanistic interpretability research on GPT-2 style language models, and is a clean implementation of the underlying transformer architecture in the library.

Further Resources:
* [A Comprehensive Mechanistic Interpretability Explainer & Glossary](https://www.neelnanda.io/glossary) - an overview
    * Expecially [the transformers section](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=pndoEIqJ6GPvC1yENQkEfZYR)
* My [200 Concrete Open Problems in Mechanistic Interpretability](https://www.neelnanda.io/concrete-open-problems) sequence - a map
* My walkthrough of [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), for a deeper dive into how to think about transformers:.

Check out these other intros to transformers for another perspective:
* Jay Alammar's [illustrated transformer](https://jalammar.github.io/illustrated-transformer/)
* [Andrej Karpathy's MinGPT](https://github.com/karpathy/minGPT)

**Sharing Guidelines:** This tutorial is still a bit of a work in progress! I think it's usable, but please don't post it anywhere publicly without checking with me first! Sharing with friends is fine. 
""")
    # end
    st.markdown(r"""
## Imports

If you're using Colab, you can just go straight to the page (scroll to the top for the link). If you're using your own IDE such as VSCode, and you've already gone through the setup steps described in **Home**, then you just need to create a file called `answers.py` (or `.ipynb` if you prefer notebooks) in the directory `exercises/transformer_from_scratch`, and run the following code at the top:

```python
import einops
from fancy_einsum import einsum
from dataclasses import dataclass
import torch as t
import torch.nn as nn
import math
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new
from tqdm import tqdm
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
```
""")
    # start
    st.markdown(r"""
## Learning Objectives

Here are the learning objectives for each section of the tutorial. At the end of each section, you should refer back here to check that you've understood everything.
""")

    st.info(r"""
### 1️⃣ Understanding Inputs & Outputs of a Transformer

* Understand what a transformer is used for
* Understand causal attention, and what a transformer's output represents
* Learn what tokenization is, and how models do it
* Understand what logits are, and how to use them to derive a probability distribution over the vocabulary
""")
    st.info(r"""
### 2️⃣ Clean Transformer Implementation

* Understand that a transformer is composed of attention heads and MLPs, with each one performing operations on the residual stream
* Understand that the attention heads in a single layer operate independently, and that they have the role of calculating attention patterns (which determine where information is moved to & from in the residual stream)
* Implement the following transformer modules:
    * LayerNorm (transforming the input to have zero mean and unit variance)
    * Positional embedding (a lookup table from position indices to residual stream vectors)
    * Attention (the method of computing attention patterns for residual stream vectors)
    * MLP (the collection of linear and nonlinear transformations which operate on each residual stream vector in the same way)
    * Embedding (a lookup table from tokens to residual stream vectors)
    * Unembedding (a matrix for converting residual stream vectors into a distribution over tokens)
* Combine these first four modules to form a transformer block, then combine these with an embedding and unembedding to create a full transformer
* Load in weights to your transformer, and demo it on a sample input
""")
    st.info(r"""
### 3️⃣ Training a model

* Use the `Adam` optimizer to train your transformer
* Run a training loop on a very small dataset, and verify that your model's loss is going down
""")
    # end
    
def section_intro():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#what-is-the-point-of-a-transformer">What is the point of a transformer?</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#how-is-the-model-trained">How is the model trained?</a></li>
   </ul></li>
   <li><a class="contents-el" href="#tokens-transformer-inputs">Tokens - Transformer Inputs</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#how-do-we-convert-language-to-vectors">How do we convert language to vectors?</a></li>
       <li><a class="contents-el" href="#idea-integers-to-vectors">Idea: integers to vectors</a></li>
       <li><a class="contents-el" href="#tokens-language-to-sequence-of-integers">Tokens: language to sequence of integers</a></li>
   </ul></li>
   <li><a class="contents-el" href="#logits-transformer-outputs">Logits - Transformer Outputs</a></li>
   <li><a class="contents-el" href="#generation">Generation!</a></li>
</ul>
""", unsafe_allow_html=True)

    # start
    st.markdown(r"""
# Understanding Inputs & Outputs of a Transformer
""")
    st.info(r"""
### Learning Objectives

* Understand what a transformer is used for
* Understand causal attention, and what a transformer's output represents
* Learn what tokenization is, and how models do it
* Understand what logits are, and how to use them to derive a probability distribution over the vocabulary
""")
    st.markdown(r"""
## What is the point of a transformer?

**Transformers exist to model text!**

We're going to focus GPT-2 style transformers. Key feature: They generate text! You feed in language, and the model generates a probability distribution over tokens. And you can repeatedly sample from this to generate text! 

(To explain this in more detail - you feed in a sequence of length $N$, then sample from the probability distribution over the $N+1$-th word, use this to construct a new sequence of length $N+1$, then feed this new sequence into the model to get a probability distribution over the $N+2$-th word, and so on.)

### How is the model trained?

You give it a bunch of text, and train it to predict the next token.

Importantly, if you give a model 100 tokens in a sequence, it predicts the next token for *each* prefix, ie it produces 100 predictions. This is kinda weird but it's much easier to make one that does this. And it also makes training more efficient, because you can 100 bits of feedback rather than just one.

#### Objection: Isn't this trivial for all except the last prediction, since the transformer can just "look at the next one"?

No! We make the transformer have *causal attention*. The core thing is that it can only move information forwards in the sequence. The prediction of what comes after token 50 is only a function of the first 50 tokens, *not* of token 51. We say the transformer is **autoregressive**, because it only predicts future words based on past data.
""")

    st_image("transformer-overview.png", 1000)
    st.markdown("")
    st.success(r"""
#### Key takeaways

Transformers are *sequence modelling engines*. They perform the same processing in parallel at each sequence position, can move information between positions with attention, and conceptually can take a sequence of arbitrary length.\*
""")
    st.markdown(r"""
\* In practice this isn't exactly true - see positional encodings later.

## Tokens - Transformer Inputs

Core point: Input is language (ie a sequence of characters, strings, etc).

### How do we convert language to vectors?

ML models take in vectors, not weird stuff like language. How do we convert between the two?

### Idea: integers $\to$ vectors

We basically make a massive lookup table, which is called an **embedding**. It has one vector for each word in our vocabulary. We label every word in our vocabulary with an integer (this labelling never changes), and we use this integer to index into the embedding.

We sometimes think about **one-hot encodings** of words. These are vectors with zeros everywhere, except for a single one in the position corresponding to the word's index in the vocabulary. This means that indexing into the embedding is equivalent to multiplying the **embedding matrix** by the one-hot encoding (where the embedding matrix is the matrix we get by stacking all the embedding vectors on top of each other).

A key intuition is that one-hot encodings let you think about each integer independently. We don't bake in any relation between words when we perform our embedding, because every word has a completely separate embedding vector.

### Tokens: language $\to$ sequence of integers

Core idea: We need a model that can deal with arbitrary text. We want to convert this into integers, *and* we want these integers to be in a bounded range. 

* **Idea:** Form a vocabulary!
    * **Idea 1:** Get a dictionary! 
        * **Problem:** It can't cope with arbitrary text (e.g. URLs, punctuation, etc), also can't cope with mispellings.
    * **Idea 2:** Vocab = 256 ASCII characters. Fixed vocab size, can do arbitrary text, etc.
        * **Problem:** Loses structure of language - some sequences of characters are more meaningful than others
            * e.g. "language" is a lot more meaningful than "hjksdfiu" - we want the first to be a single token, second to not be. It's a more efficient use of our vocab.

#### What Actually Happens?

The most common strategy is called **Byte-Pair encodings**.

We begin with the 256 ASCII characters as our tokens, and then find the most common pair of tokens, and merge that into a new token. Note that we do have a space character as one of our 256 tokens, and merges using space are very common. For instance, run this code to print the five first merges for the tokenizer used by GPT-2:

*(Note - for all the code on this page, you can delete it or comment it out after you run it.)*

```python
sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n:n[1])
print(sorted_vocab[256:261])
```

Note - you might see the character `Ġ` in front of some tokens. This is a special token that indicates that the token begins with a space. Tokens with a leading space vs not are different.
""")
    # end
    with st.expander("Fun (totally optional) exercise - can you guess what the first-formed 3/4/5/6/7-letter encodings in GPT-2's vocabulary are?"):
        st.markdown(r"""
They are:

```
3 -> "ing"
4 -> " and"
5 -> " that"
6 -> " their"
7 -> " people"
```
""")

    st.markdown(r"""
You can run the code below to see some more of GPT-2's tokenizer's vocabulary:

```python
print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()
```

As you get to the end of the vocabulary, you'll be producing some pretty weird-looking esoteric tokens (because you'll already have exhausted all of the short frequently-occurring ones):

```python
print(sorted_vocab[-20:])
```

Transformers in the `transformer_lens` library have a `to_tokens` method that converts text to numbers. It also prepends them with a special token called `bos` (beginning of sequence) to indicate the start of a sequence (we'll learn more about this later). You can disable this with the `prepend_bos` argument.

Prepends with a special token to give attention a resting position, disable with `prepend_bos=False`
""")
    # start
    st.markdown(r"""
### Some tokenization annoyances

There are a few funky and frustrating things about tokenization, which causes it to behave differently than you might expect. For instance:

##### Whether a word begins with a capital or space matters!

```python
print(reference_gpt2.to_str_tokens("Ralph"))
print(reference_gpt2.to_str_tokens(" Ralph"))
print(reference_gpt2.to_str_tokens(" ralph"))
print(reference_gpt2.to_str_tokens("ralph"))
```

##### Arithmetic is a mess.

Length is inconsistent, common numbers bundle together.

```python
reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000")
```
""")
    st.success(r"""
### Key Takeaways

* We learn a dictionary of vocab of tokens (sub-words).
* We (approx) losslessly convert language to integers via tokenizing it.
* We convert integers to vectors via a lookup table.
* Note: input to the transformer is a sequence of *tokens* (ie integers), not vectors
""")
    st.markdown(r"""
## Logits - Transformer Outputs

**Goal:** Probability distribution over next tokens. (For every *prefix* of the sequence - given n tokens, we make n next token predictions)

**Problem:** How to convert a vector (where some values may be more than one, or negative) to a probability distribution? 

**Answer:** Use a softmax ($x_i \to \frac{e^{x_i}}{\sum e^{x_j}}$). Exponential makes everything positive, normalization makes it add to one.

So the model outputs a tensor of logits, one vector of size $d_{vocab}$ for each input token.

(Note - we call something a logit if it represents a probability distribution, and it is related to the actual probabilities via the softmax function. Logits and probabilities are both equally valid ways to represent a distribution.)
""")
    # end
    st.markdown(r"""
## Text generation

#### **Step 1:** Convert text to tokens

The sequence gets tokenized, so it has shape `[batch, seq_len]`. Here, the batch dimension is just one (because we only have one sequence).

```python
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))
```

#### **Step 2:** Map tokens to logits

From our input of shape `[batch, seq_len]`, we get output of shape `[batch, seq_len, vocab_size]`. The `[i, j, :]`-th element of our output is a vector of logits representing our prediction for the `j+1`-th token in the `i`-th sequence.

```python
tokens = tokens.to(device)
logits, cache = reference_gpt2.run_with_cache(tokens)
print(logits.shape)
```

(`run_with_cache` tells the model to cache all intermediate activations. This isn't important right now; we'll look at it in more detail later.)

#### **Step 3:** Convert the logits to a distribution with a softmax

This doesn't change the shape, it is still `[batch, seq_len, vocab_size]`.

```python
log_probs = logits.log_softmax(dim=-1)
probs = logits.log_softmax(dim=-1)
print(log_probs.shape)
print(probs.shape)
```

#### **Bonus step:** What is the most likely next token at each position?

```python
list(zip(reference_gpt2.to_str_tokens(reference_text), reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])))
```

We can see that, in a few cases (particularly near the end of the sequence), the model accurately predicts the next token in the sequence. We might guess that `"take over the world"` is a common phrase that the model has seen in training, which is why the model can predict it.

#### **Step 4:** Map distribution to a token

```python
next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)
print(repr(next_char))
```

Note that we're indexing `logits[0, -1]`. This is because logits have shape `[1, sequence_length, vocab_size]`, so this indexing returns the vector of length `vocab_size` representing the model's prediction for what token follows the **last** token in the input sequence.

We can see the model predicts the line break character `\n`, since this is common following the end of a sentence.

#### **Step 5:** Add this to the end of the input, re-run

There are more efficient ways to do this (e.g. where we cache some of the values each time we run our input, so we don't have to do as much calculation each time we generate a new value), but this doesn't matter conceptually right now.

```python
for i in range(10):
    # Define new input sequence, by appending the previously generated token
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = reference_gpt2(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = reference_gpt2.to_string(next_token)
    print(f"{tokens.shape[-1]+1}th char = {repr(next_char)}")
```

Note how our model predicts a second line break, followed by a second copy of the input sequence. 
""")
    # start
    st.success(r"""
### Key takeaways

* Transformer takes in language, predicts next token (for *each* token in a causal way)
* We convert language to a sequence of integers with a tokenizer.
* We convert integers to vectors with a lookup table.
* Output is a vector of logits (one for each input token), we convert to a probability distn with a softmax, and can then convert this to a token (eg taking the largest logit, or sampling).
* We append this to the input + run again to generate more text (Jargon: *autoregressive*)
* Meta level point: Transformers are sequence operation models, they take in a sequence, do processing in parallel at each position, and use attention to move information between positions!
""")
    # end

def section_code():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#high-level-architecture">High-Level architecture</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#summary">Summary</a></li>
        <li><a class="contents-el" href="#residual-stream">Residual stream</a></li>
        <li><a class="contents-el" href="#transformer-blocks">Transformer blocks</a></li>
        <li><ul class="contents">
            <li><a class="contents-el" href="#attention">Attention</a></li>
            <li><a class="contents-el" href="#mlps">MLPs</a></li>
        </ul></li>
        <li><a class="contents-el" href="#unembedding">Unembedding</a></li>
        <li><a class="contents-el" href="#bonus-things-less-conceptually-important-but-key-technical-details">Bonus things</a></li>
    </ul></li>
    <li><a class="contents-el" href="#actual-code">Actual Code!</a></li>
    <li><ul class="contents">
    <li><a class="contents-el" href="#parameters-and-activations">Parameters vs Activations</a></li>
    <li><a class="contents-el" href="#config">Config</a></li>
    <li><a class="contents-el" href="#tests">Tests</a></li>
    <li><a class="contents-el" href="#layernorm">LayerNorm</a></li>
    <li><a class="contents-el" href="#embedding">Embedding</a></li>
    <li><a class="contents-el" href="#positional-embedding">Positional Embedding</a></li>
    <li><a class="contents-el" href="#attention-layer">Attention Layer</a></li>
    <li><a class="contents-el" href="#mlp">MLP</a></li>
    <li><a class="contents-el" href="#transformer-block">Transformer Block</a></li>
    <li><a class="contents-el" href="#unembedding">Unembedding</a></li>
    <li><a class="contents-el" href="#full-transformer">Full Transformer</a></li>
    </ul></li>
    <li><a class="contents-el" href="#try-it-out">Try it out!</a></li>
</ul>
""", unsafe_allow_html=True)
    # start
    st.markdown(r"""
# Clean Transformer Implementation
""")
    st.info(r"""
### Learning Objectives

* Understand that a transformer is composed of attention heads and MLPs, with each one performing operations on the residual stream
* Understand that the attention heads in a single layer operate independently, and that they have the role of calculating attention patterns (which determine where information is moved to & from in the residual stream)
* Implement the following transformer modules:
    * LayerNorm (transforming the input to have zero mean and unit variance)
    * Positional embedding (a lookup table from position indices to residual stream vectors)
    * Attention (the method of computing attention patterns for residual stream vectors)
    * MLP (the collection of linear and nonlinear transformations which operate on each residual stream vector in the same way)
    * Embedding (a lookup table from tokens to residual stream vectors)
    * Unembedding (a matrix for converting residual stream vectors into a distribution over tokens)
* Combine these first four modules to form a transformer block, then combine these with an embedding and unembedding to create a full transformer
* Load in weights to your transformer, and demo it on a sample input
""")
    # end
    st.markdown(r"""
This diagram shows the high-level transformer architecture. It can be thought of in terms of a sequence of **attention heads** (denoted $h_1, h_2, ...$) and MLPs (denoted $m$), with each one performing operations on the residual stream (which is the central object of the transformer).
""")
    st.markdown(r"")
    st_image("transformer.png", 900)
    st.markdown(r"")
    # start
    st.markdown(r"""
## High-Level architecture

Go watch my [Transformer Circuits walkthrough](https://www.youtube.com/watch?v=KV5gbOmHbjU) if you want more intuitions!

(Diagram is bottom to top)

### Summary

The input tokens $t$ are integers. We get them from taking a sequence, and tokenizing it (like we saw in the previous section).

The token embedding is a lookup table mapping tokens to vectors, which is implemented as a matrix $W_E$. The matrix consists of a stack of token embedding vectors (one for each token).

### Residual stream

The residual stream is the sum of all previous outputs of layers of the model, is the input to each new layer. It has shape `[batch, seq_len, d_model]` (where `d_model` is the length of a single embedding vector). 

The initial value of the residual stream is denoted $x_0$ in the diagram, and $x_i$ are later values of the residual stream (after more attention and MLP layers have been applied to the residual stream).

The residual stream is *Really* fundamental. It's the central object of the transformer. It's how model remembers things, moves information between layers for composition, and it's the medium used to store the information that attention moves between positions.

### Transformer blocks

Then we have a series of `n_layers` **transformer blocks** (also sometimes called **residual blocks**).

Note - a block contains an attention layer *and* an MLP layer, but we say a transformer has $k$ layers if it has $k$ blocks (i.e. $2k$ total layers).

#### Attention

First we have attention. This moves information from prior positions in the sequence to the current token. 

We do this for *every* token in parallel using the same parameters. The only difference is that we look backwards only (to avoid "cheating"). This means later tokens have more of the sequence that they can look at.

Attention layers are the only bit of a transformer that moves information between positions (i.e. between vectors at different sequence positions in the residual stream).

Attention layers are made up of `n_heads` heads - each with their own parameters, own attention pattern, and own information how to copy things from source to destination. The heads act independently and additively, we just add their outputs together, and back to the stream.

Each head does the following:
* Produces an **attention pattern** for each destination token, a probability distribution of prior source tokens (including the current one) weighting how much information to copy.
* Moves information (via a linear map) in the same way from each source token to each destination token.

A few key points:

* What information we copy depends on the source token's *residual stream*, but this doesn't mean it only depends on the value of that token, because the residual stream can store more information than just the token identity (the purpose of the attention heads is to move information between vectors at different positions in the residual stream!)
* We can think of each attention head as consisting of two different **circuits**:
    * One circuit determines **where to move information to and from** (this is a function of the residual stream for the source and destination tokens)
    * The other circuit determines **what information to move** (this is a function of only the source token's residual stream)
    * For reasons which will become clear later, we refer to the first circuit as the **QK circuit**, and the second circuit as the **OV circuit**

Below is a schematic diagram of the attention layers; don't worry if you don't follow this right now, we'll go into more detail during implementation.
""")
    st_image("transformer-attn.png", 1100)
    st.markdown(r"")
    st.markdown(r"""
### MLPs

The MLP layers are just a standard neural network, with a singular hidden layer and a nonlinear activation function. The exact activation isn't conceptually important ([GELU](https://paperswithcode.com/method/gelu) seems to perform best).

Our hidden dimension is normally `d_mlp = 4 * d_model`. Exactly why the ratios are what they are isn't super important (people basically cargo-cult what GPT did back in the day!).

Importantly, **the MLP operates on positions in the residual stream independently, and in exactly the same way**. It doesn't move information between positions.

Intuition - once attention has moved relevant information to a single position in the residual stream, MLPs can actually do computation, reasoning, lookup information, etc. *What the hell is going on inside MLPs* is a pretty big open problem in transformer mechanistic interpretability - see the [Toy Model of Superposition Paper](https://transformer-circuits.pub/2022/toy_model/index.html) for more on why this is hard.

Another important intuition - `linear map -> non-linearity -> linear map` is pretty much the most powerful force in the universe! It basically [just works](https://xkcd.com/1838/), and can approximate arbitrary functions.
""")
    # end
    st_image("transformer-mlp.png", 720)
    st.markdown(r"""
### Unembedding

Finally, we unembed!

This just consists of applying a linear map $W_U$, going from final residual stream to a vector of logits - this is the output.
""")
    with st.expander("Aside - tied embeddings"):
        st.markdown(r"""
Note - sometimes we use something called a **tied embedding** - this is where we use the same weights for our $W_E$ and $W_U$ matrices. In other words, to get the logit score for a particular token at some sequence position, we just take the vector in the residual stream at that sequence position and take the inner product with the corresponding token embedding vector. This is more training-efficient (because there are fewer parameters in our model), and it might seem pricipled at first. After all, if two words have very similar meanings, shouldn't they have similar embedding vectors because the model will treat them the same, and similar unembedding vectors because they could both be substituted for each other in most output?

However, this is actually not very principled, for the following main reason: **the direct path involving the embedding and unembedding should approximate bigram frequencies**. 

Let's break down this claim. **Bigram frequencies** refers to the frequencies of pairs of words in the english language (e.g. the bigram frequency of "Barack Obama" is much higher than the product of the individual frequencies of the words "Barack" and "Obama"). If our model had no attention heads or MLP layers, then all we have is a linear map from our one-hot encoded token `T` to a probability distribution over the token following `T`. This map is represented by the linear transformation $t \to t^T W_E W_U$ (where $t$ is our one-hot encoded token vector). Since the output of this transformation can only be a function of the token `T` (and no earlier tokens), the best we can do is have this map approximate the true frequency of bigrams starting with `T`, which appear in the training data. Importantly, **this is not a symmetric map**. We want `T = "Barack"` to result in a high probability of the next token being `"Obama"`, but not the other way around!

Even in multi-layer models, a similar principle applies. There will be more paths through the model than just the "direct path" $W_E W_U$, but because of the residual connections there will always exist a direct path, so there will always be some incentive for $W_E W_U$ to approximate bigram frequencies.
""")
    st.markdown(r"""
### Bonus things - less conceptually important but key technical details

#### LayerNorm

* Simple normalization function applied at the start of each layer (i.e. before each MLP, attention layer, and before the unembedding)
* Converts each input vector (independently in parallel for each batch x position residual stream vector) to have mean zero and variance 1.
* Then applies an elementwise scaling and translation
* Cool maths tangent: The scale & translate is just a linear map. LayerNorm is only applied immediately before another linear map. Linear compose linear = linear, so we can just fold this into a single effective linear layer and ignore it.
    * `fold_ln=True` flag in `from_pretrained` does this for you.
* LayerNorm is annoying for interpertability - the scale part is not linear, so you can't think about different bits of the input independently. But it's *almost* linear - if you're changing a small part of the input it's linear, but if you're changing enough to alter the norm substantially it's not linear :(

#### Positional embeddings

* **Problem:** Attention operates over all pairs of positions. This means it's symmetric with regards to position - the attention calculation from token 5 to token 1 and token 5 to token 2 are the same by default
    * This is dumb because nearby tokens are more relevant.
* There's a lot of dumb hacks for this.
* We'll focus on **learned, absolute positional embeddings**. This means we learn a lookup table mapping the index of the position of each token to a residual stream vector, and add this to the embed.
    * Note that we *add* rather than concatenate. This is because the residual stream is shared memory, and likely under significant superposition (the model compresses more features in there than the model has dimensions)
    * We basically never concatenate inside a transformer, unless doing weird shit like generating text efficiently.
* One intuition: 
    * *Attention patterns are like generalised convolutions* (where the transformer learns which words in a sentence are relevant to each other, as opposed to convolutions which just imposes the fixed prior of "pixels close to each other are relevant, pixels far away are not.")
    * Positional information helps the model figure out that words are close to each other, which is helpful because this probably implies they are relevant to each other.

## Actual Code!

Key (for the results you get when running the code immediately below)

```
batch = 1
position = 35
d_model = 768
n_heads = 12
n_layers = 12
d_mlp = 3072 (4 * d_model)
d_head = 64 (d_model / n_heads)
```

### Parameters vs Activations

It's important to distinguish between parameters and activations in the model.

* **Parameters** are the weights and biases that are learned during training.
    * These don't change when the model input changes.
    * They can be accessed direction fromm the model, e.g. `model.W_E` for the token embedding.
* **Activations** are temporary numbers calculated during a forward pass, that are functions of the input.
    * We can think of these values as only existing for the duration of a single forward pass, and disappearing afterwards.
    * We can use hooks to access these values during a forward pass (more on hooks later), but it doesn't make sense to talk about a model's activations outside the context of some particular input.
    * Attention scores and patterns are activations (this is slightly non-intuitve because they're used in a matrix multiplication with another activation).
""")
    st.markdown(r"""
The dropdown below contains a diagram of a single layer (called a `TransformerBlock`) for an attention-only model with no biases. Each box corresponds to an **activation** (and also tells you the name of the corresponding hook point, which we will eventually use to access those activations). The red text below each box tells you the shape of the activation (ignoring the batch dimension). Each arrow corresponds to an operation on an activation; where there are **parameters** involved these are labelled on the arrows.

#### Print All Activation Shapes of Reference Model

Run the following code to print all the activation shapes of the reference model:

```python
if MAIN:
    for activation_name, activation in cache.items():
        # Only print for first layer
        if ".0." in activation_name or "blocks" not in activation_name:
            print(f"{activation_name:30} {tuple(activation.shape)}")
```

#### Print All Parameters Shapes of Reference Model

```python
if MAIN:
    for name, param in reference_gpt2.named_parameters():
        # Only print for first layer
        if ".0." in name or "blocks" not in name:
            print(f"{name:18} {tuple(param.shape)}")
```

The diagram below shows the name of all activations and parameters in a fully general transformer model from transformerlens (except for a few at the start and end, like the embedding and unembedding). Lots of this won't make sense at first, but you can return to this diagram later and check that you understand most/all parts of it.
""")

    with st.expander("Diagram"):
        st.write("""<figure style="max-width:680px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNrdV1FP2zAQ_itWpI1tasSIeApdJaYWJqGNIRA8UBS5sdNadeLUdkJbwn_fOUkJ6ZoCm9R2y4N955yT786fz-cHyxeEWq41lDgeoatuP0LwqGRQDPSttopxhJSecfplLxCRthWbU9c5jKd7nSuJIxUIGVL5lQt_3N431p2-VXzGPD7HSnVpgGgY6xm6Z0SP3M_xtDWibDjSRjxaYW1gQcOFdCUlYFHZSKoY8WJJb_vWk9wmLF2gHAhJqLS1iF0nniIlOCNowLE_PgqxHLIof5U70N6HeZ12_rdydvXTytsDxxh_UHTSQsQLwZp_bO-bWeDrnW3b3Vt057pu7qNtdzJMSFZgCxmB973G97FQunIElG16UgW5kl7LBR4doPc0VPFR2T1v0ZruJev17QrK5ah9zGl9KAKeYg6ASTVOI7KCWAiWCGXguJbY1yikOIJosZRBbAczCADJ8u_DuuX95pbs4NliFShvPA7gBtBmlYMArFL-VUJhrSP0Flqgt3Z_1jYwLq2rk7o6rqvGN0_5AihXf3FSV2MwpDKqD57W1XldhU8mXDdQvGKFyUI33rWhznWWAmHSzfFkRDHxGJkaxhi5nktPl3LlfX5QUIJwOkQiQCnmCUUp9bWQKpsD9PlOQF_sx_OsWIIiq4OwHXTLW9HAg5wWIpFSmVuqFoJzCA0YVsCC8ywnpUgM8IW47WO1mbkXhrkX2QTATnaFuSdLzCVCo1gKksAhgrmIhuWsVnE8IRwRFGI1zp6lg0XwC20jnlVOgY8XeXtWcwx4IwId4mlW5iMAWUq7AdA-bSbKmSHKWTYGzOOdIcrfFVoOevHYW1cWOU11kbO2MIJK9qlQBXmbqeG1BZqzsxWa81-UaCGPX1tfNRByJfnyykcule_mbvRiVeMsQs7ykLMoK-6JG70hHqJPjZwFunoBoCpufZu97zXjgoDBYW8iBl0Gq1qWAaW05a1uo17JzXzVC_HdOwQAfxx_720E3eW345-933bNlkFYLSukQH1GLNd6MJD6lh7RkPYtF0RCA2yuArDlHsE0iQnWtEcY1M2WG2CuaMvCiRaXs8i3XC0TujDqMgz7PyytHn8BSKkJUQ" /></figure>""", unsafe_allow_html=True)

    st.markdown(r"""
### Config

The config object contains all the hyperparameters of the model. We can print the config of the reference model to see what it contains:

```python
# As a reference - note there's a lot of stuff we don't care about in here, to do with library internals or other architectures
print(reference_gpt2.cfg)
```

We define a stripped down config for our model:

```python
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

    
cfg = Config()

if MAIN:
    print(cfg)
```

### Tests

Tests are great, write lightweight ones to use as you go!

**Naive test:** Generate random inputs of the right shape, input to your model, check whether there's an error and print the correct output.

```python
def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randn(shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape, "\n")

def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape, "\n")

def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    print("Output shape:", output.shape)
    try: reference_output = gpt2_layer(input)
    except: reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")
```

### LayerNorm

You should fill in the code below, and then run the tests to verify that your layer is working correctly.

Your LayerNorm should do the following:

* Make mean 0
* Normalize to have variance 1
* Scale with learned weights
* Translate with learned bias

You can use the PyTorch [LayerNorm documentation](https://pyt.org/docs/stable/generated/t.nn.LayerNorm.html) as a reference. A few more notes:

* Your layernorm implementation always has `affine=True`, i.e. you do learn parameters `w` and `b` (which are represented as $\gamma$ and $\beta$ respectively in the PyTorch documentation).
* Remember that, after the centering and normalization, each vector of length `d_model` in your input should have mean 0 and variance 1.
* As the PyTorch documentation page says, your variance should be computed using `unbiased=False`.
* The `layer_norm_eps` argument in your config object corresponds to the $\epsilon$ term in the PyTorch documentation (it is included to avoid division-by-zero errors).
* We've given you a `debug` argument in your config. If `debug=True`, then you can print output like the shape of objects in your `forward` function to help you debug (this is a very useful trick to improve your coding speed).

Fill in the function, where it says `pass` (this will be the basic pattern for most other exercises in this section).

```python
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))
    
    def forward(self, residual):
        # residual: [batch, position, d_model]
        # output: [batch, position, d_model]
        pass
        

if MAIN:
    rand_float_test(LayerNorm, [2, 4, 768])
    load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])
```
""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual):
        # residual: [batch, position, d_model]
        # output: [batch, position, d_model]
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()

        residual = (residual - residual_mean) / residual_std
        return residual * self.w + self.b
```
""")
    st.markdown(r"""
### Embedding

Basically a lookup table from tokens to residual stream vectors.

(Hint - you can implement this in just one line!)

```python
class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        # output: [batch, position, d_model]
        pass

        
if MAIN
    rand_int_test(Embed, [2, 4])
    load_gpt2_test(Embed, reference_gpt2.embed, tokens)
```
""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)
    
    def forward(self, tokens):
        # tokens: [batch, position]
        # output: [batch, position, d_model]
        return self.W_E[tokens]
```
""")
    st.markdown(r"""
### Positional Embedding

```python
class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)
    
    def forward(self, tokens):
        # tokens: [batch, position]
        # output: [batch, position, d_model]
        pass

        
if MAIN:
    rand_int_test(PosEmbed, [2, 4])
    load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)
```
""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        # output: [batch, position, d_model]
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)

        
if MAIN:
    rand_int_test(PosEmbed, [2, 4])
    load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)
```
""")
    # start
    st.markdown(r"""

### Attention Layer

* **Step 1:** Produce an attention pattern - for each destination token, probability distribution over previous tokens (including current token)
    * Linear map from input -> query, key shape `[batch, seq_posn, head_index, d_head]`
    * Dot product every *pair* of queries and keys to get attn_scores `[batch, head_index, query_pos, key_pos]` (query = dest, key = source)
    * Scale and mask `attn_scores` to make it lower triangular, ie causal
    * Softmax along the `key_pos` dimension, to get a probability distribution for each query (destination) token - this is our attention pattern!
* **Step 2:** Move information from source tokens to destination token using attention pattern (move = apply linear map)
    * Linear map from input -> value `[batch, key_pos, head_index, d_head]`
    * Mix along the `key_pos` with attn pattern to get `z`, which is a weighted average of the value vectors `[batch, query_pos, head_index, d_head]`
    * Map to output, `[batch, position, d_model]` (position = query_pos, we've summed over all heads)
""")
    # end
    st.markdown(r"""
Below is a much larger, more detailed version of the attention head diagram from earlier. This should give you an idea of the actual tensor operations involved. A few clarifications on this diagram:

* Whenever there is a third dimension shown in the pictures, this refers to the `head_index` dimension. We can see that all operations within the attention layer are done independently for each head.
* The objects in the box are activations; they have a batch dimension (for simplicity, we assume the batch dimension is 1 in the diagram). The objects to the right of the box are our parameters (weights and biases); they have no batch dimension.
* We arrange the keys, queries and values as `(batch, seq_pos, head_idx, d_head)`, because the biases have shape `(head_idx, d_head)`, so this makes it convenient to add the biases (recall the rules of array broadcasting!).
""")
    st_image("transformer-attn-2.png", 1250)
    with st.expander("A few extra notes on attention (optional)"):
        st.markdown(r"""
Usually we have the relation `e = n * h` (i.e. `d_model = num_heads * d_head`). There are some computational justifications for this, but mostly this is just done out of convention (just like how we usually have `d_mlp = 4 * d_model`!).

---

The names **keys**, **queries** and **values** come from their analogy to retrieval systems. Broadly speaking:

* The **queries** represent some information that a token is **"looking for"**
* The **keys** represent the information that a token **"contains"**
    * So the attention score being high basically means that the source (key) token contains the information which the destination (query) token **is looking for**
* The **values** represent the information that is actually taken from the source token, to be moved to the destination token

---

This diagram can better help us understand the difference between the **QK** and **OV** circuit. We'll discuss this just briefly here, and will go into much more detail later on.

The **QK** circuit consists of the operation of the $W_Q$ and $W_K$ matrices. In other words, it determines the attention pattern, i.e. where information is moved to and from in the residual stream. The functional form of the attention pattern $A$ is:

$$
A = \text{softmax}\left(\frac{x^T W_Q W_K^T x}{\sqrt{d_{head}}}\right)
$$

where $x$ is the residual stream (shape `[seq_len, d_model]`), and $W_Q$, $W_K$ are the weight matrices for a single head (i.e. shape `[d_model, d_head]`).

The **OV** circuit consists of the operation of the $W_V$ and $W_O$ matrices. Once attention patterns are fixed, these matrices operate on the residual stream at the source position, and their output is the thing which gets moved from source to destination position.

The functional form of an entire attention head is:

$$
\begin{aligned}
\text{output} &= \text{softmax}\left(\frac{x W_Q W_K^T x^T}{\sqrt{d_{head}}}\right) (x W_V W_O) \\
    &= Ax W_V W_O
\end{aligned}
$$

where $W_V$ has shape `[d_model, d_head]`, and $W_O$ has shape `[d_head, d_model]`.

Here, we can clearly see that the **QK circuit** and **OV circuit** are doing conceptually different things, and should be thought of as two distinct parts of the attention head.

Again, don't worry if you don't follow all of this right now - we'll go into **much** more detail on all of this in subsequent exercises. The purpose of the discussion here is just to give you a flavour of what's to come!
""")
    st.markdown(r"""
First, it's useful to visualize and play around with attention patterns - what exactly are we looking at here? (Click on a head to lock onto just showing that head's pattern, it'll make it easier to interpret)

```python
import circuitsvis as cv
from IPython.display import display

display(cv.attention.attention_patterns(tokens=reference_gpt2.to_str_tokens(reference_text), attention=cache["pattern", 0][0]))
```
""")
    st.components.v1.html(attn_patterns_demo, height=550)
    st.markdown(r"""
Note - don't worry if you don't get 100% accuracy here; the tests are pretty stringent. Even things like having your `einsum` input arguments in a different order might result in the output being very slightly different. You should be getting at least 99% accuracy though, so if the value is lower then this it probably means you've made a mistake somewhere.

Also, this implementation will probably be the most challenging exercise on this page, so don't worry if it takes you some time! You should look at parts of the solution if you're stuck.

```python
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

    def forward(self, normalized_resid_pre: t.Tensor):
        # normalized_resid_pre: [batch, position, d_model]
        # output: [batch, position, d_model]
        pass

    def apply_causal_mask(self, attn_scores: t.Tensor):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        # output: [batch, n_heads, query_pos, key_pos]
        pass


if MAIN:
    rand_float_test(Attention, [2, 4, 768])
    load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])
```
""")
    with st.expander("Hint (pseudocode for both functions)"):
        st.markdown(r"""
```python
def forward(self, normalized_resid_pre):    
    # Calculate query, key and value vectors

    # Calculate attention scores, then scale and mask, and apply softmax to get probabilities

    # Take weighted sum of value vectors, according to attention probabilities

    # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)


def apply_causal_mask(self, attn_scores):
    # Define a mask that is True for all positions we want to set probabilities to zero for

    # Apply the mask to attention scores, then return the masked scores
```
""")
    with st.expander("Solution"):
        st.markdown(r"""
```python
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

    def forward(self, normalized_resid_pre: t.Tensor):
        # normalized_resid_pre: [batch, position, d_model]
        # output: [batch, position, d_model]

        # Calculate query, key and value vectors
        q = einsum(
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
            normalized_resid_pre, self.W_Q
        ) + self.b_Q
        k = einsum(
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
            normalized_resid_pre, self.W_K
        ) + self.b_K
        v = einsum(
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
            normalized_resid_pre, self.W_V
        ) + self.b_V

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = einsum("batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K", q, k)
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head ** 0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z = einsum("batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head", v, attn_pattern)

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        attn_out = einsum("batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model", z, self.W_O) + self.b_O

        return attn_out

    def apply_causal_mask(self, attn_scores: t.Tensor):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        # output: [batch, n_heads, query_pos, key_pos]

        # Define a mask that is True for all positions we want to set probabilities to zero for
        mask = t.triu(t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        # Apply the mask to attention scores, then return the masked scores
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


if MAIN:
    rand_float_test(Attention, [2, 4, 768])
    load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])
```
""")

    st.markdown(r"""

### MLP

Note, we have the `gelu_new` function imported from transformer lens, which you should use as your activation function.

```python
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
    
    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        # output: [batch, position, d_model]
        pass

        
if MAIN:
    rand_float_test(MLP, [2, 4, 768])
    load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["blocks.0.ln2.hook_normalized"])
```
""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
    
    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        # output: [batch, position, d_model]
        pre = einsum(
            "batch position d_model, d_model d_mlp -> batch position d_mlp", 
            normalized_resid_mid, self.W_in
        ) + self.b_in
        post = gelu_new(pre)
        mlp_out = einsum(
            "batch position d_mlp, d_mlp d_model -> batch position d_model", 
            post, self.W_out
        ) + self.b_out
        return mlp_out
```
""")

    st.markdown(r"""
### Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
    
    def forward(self, resid_pre):
        # resid_pre: [batch, position, d_model]
        # output: [batch, position, d_model]
        pass

        
if MAIN:
    rand_float_test(TransformerBlock, [2, 4, 768])
    load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])
```
""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
    
    def forward(self, resid_pre):
        # resid_pre: [batch, position, d_model]
        # output: [batch, position, d_model]
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post
```
""")

    st.markdown(r"""
### Unembedding

```python
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))
    
    def forward(self, normalized_resid_final):
        # normalized_resid_final: [batch, position, d_model]
        # output: [batch, position, d_vocab]
        pass

        
if MAIN:
    rand_float_test(Unembed, [2, 4, 768])
    load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])
```
""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))
    
    def forward(self, normalized_resid_final):
        # normalized_resid_final: [batch, position, d_model]
        # output: [batch, position, d_vocab]
        logits = einsum("batch position d_model, d_model d_vocab -> batch position d_vocab", normalized_resid_final, self.W_U) + self.b_U
        return logits

        
if MAIN:
    rand_float_test(Unembed, [2, 4, 768])
    load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])
```
""")

    st.markdown(r"""
### Full Transformer

```python
class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
    
    def forward(self, tokens):
        # tokens [batch, position]
        # output: [batch, position, d_vocab]
        pass

        
if MAIN:
    rand_int_test(DemoTransformer, [2, 4])
    load_gpt2_test(DemoTransformer, reference_gpt2, tokens)
```
""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
    
    def forward(self, tokens):
        # tokens [batch, position]
        # output: [batch, position, d_vocab]
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        residual = embed + pos_embed
        for block in self.blocks:
            residual = block(residual)
        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        # logits have shape [batch, position, logits]
        return logits

        
if MAIN:
    rand_int_test(DemoTransformer, [2, 4])
    load_gpt2_test(DemoTransformer, reference_gpt2, tokens)
```
""")

    st.markdown(r"""

## Try it out!

```python
if MAIN:
    demo_gpt2 = DemoTransformer(Config(debug=False))
    demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
    demo_gpt2.to(device)
```

Let's take a test string, and calculate the loss!

We're using the formula for **cross-entropy loss**. The cross entropy loss between a modelled distribution $Q$ and target distribution $P$ is:

$$
-\sum_x P(x) \log Q(x)
$$

In the case where $P$ is just the distribution derived from target classes (i.e. $P(x^*) = 1$ for the correct class $x^*$) then this becomes:

$$
-\log Q(x^*)
$$

in other words, the negative log prob of the true classification. 

```python
if MAIN:
    test_string = '''There is a theory which states that if ever anyone discovers exactly what the Universe is for and why it is here, it will instantly disappear and be replaced by something even more bizarre and inexplicable. There is another theory which states that this has already happened.'''
    test_tokens = reference_gpt2.to_tokens(test_string).to(device)
    demo_logits = demo_gpt2(test_tokens)
```

```python
def lm_cross_entropy_loss(logits: t.Tensor, tokens: t.Tensor):
    # Measure next token loss
    # Logits have shape [batch, position, d_vocab]
    # Tokens have shape [batch, position]
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()

if MAIN:
    loss = lm_cross_entropy_loss(demo_logits, test_tokens)
    print(loss)
    print("Loss as average prob", (-loss).exp())
    print("Loss as 'uniform over this many variables'", (loss).exp())
    print("Uniform loss over the vocab", math.log(demo_gpt2.cfg.d_vocab))
```

We can also greedily generate text:

```python
if MAIN:
    for i in tqdm.tqdm(range(100)):
        test_tokens = reference_gpt2.to_tokens(test_string).to(device)
        demo_logits = demo_gpt2(test_tokens)
        test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())
    print(test_string)
```

In later sections, we'll learn to generate text in slightly more interesting ways than just argmaxing the output.
""")

def section_training():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#config">Config</a></li>
   <li><a class="contents-el" href="#create-data">Create Data</a></li>
   <li><a class="contents-el" href="#create-model">Create Model</a></li>
   <li><a class="contents-el" href="#create-optimizer">Create Optimizer</a></li>
   <li><a class="contents-el" href="#run-training-loop">Run Training Loop</a></li>
   <li><a class="contents-el" href="#a-note-on-this-loss-curve">A note on this loss curve</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# Training a Model!
""")
    st.info(r"""
### Learning Objectives

* Use the `Adam` optimizer to train your transformer
* Run a training loop on a very small dataset, and verify that your model's loss is going down
""")
    st.markdown(r"""
***Note - this section provides a demo rather than exercises. If you've got more time today, you might wnat to to jump to the next section (Training and Sampling) to go through the process of building dataloaders and training loops by yourself, without looking at this section first. It's totally up to you!***
""")
    st.error(r"""
You should use the Colab to run this code, since it will put quite a strain on your GPU otherwise! You can access a complete version of the Colab (with solutions for all the earlier exercises filled in) [here](https://colab.research.google.com/drive/1kP27XsoJsPeCyVtzeolNMlVAsLFEmxZ6?usp=sharing#scrollTo=QfeyG6NZm4SC).

Alternatively, if you just want to read the code and look at the outputs, you can still do it on this page.
""")
    st.markdown(r"""
This is a lightweight demonstration of how you can actually train your own GPT-2 with this code! Here we train a tiny model on a tiny dataset, but it's fundamentally the same code for training a larger/more real model (though you'll need beefier GPUs and data parallelism to do it remotely efficiently, and fancier parallelism for much bigger ones).

For our purposes, we'll train 2L 4 heads per layer model, with context length 256, for 1000 steps of batch size 8, just to show what it looks like.
""")
    st.markdown(r"""
```python
import datasets
import transformers
import plotly.express as px
```

## Config

```python
batch_size = 8
num_epochs = 1
max_steps = 1000
log_every = 10
lr = 1e-3
weight_decay = 1e-2
model_cfg = Config(debug=False, d_model=256, n_heads=4, d_head=64, d_mlp=1024, n_layers=2, n_ctx=256, d_vocab=reference_gpt2.cfg.d_vocab)
```

## Create Data

We load in a tiny dataset I made, with the first 10K entries in the Pile (inspired by Stas' version for OpenWebText!)

```python
if MAIN:
    dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
    print(dataset)
    print(dataset[0]['text'][:100])
    tokens_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model_cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
    data_loader = t.utils.data.DataLoader(tokens_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
```

## Create Model

```python
if MAIN:
    model = DemoTransformer(model_cfg)
    model.to(device)
```

## Create Optimizer

We use AdamW - it's a pretty standard optimizer.

```python
if MAIN:
    optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

## Run Training Loop

This is a pretty standard template for a training loop. There are annotations to explain the different parts of the code.

```python
if MAIN:
    losses = []
    print("Number of batches:", len(data_loader))

    for epoch in range(num_epochs):
        
        for c, batch in tqdm.tqdm(enumerate(data_loader)):
            
            # Get batch of tokens, and run your model on them
            tokens = batch['tokens'].to(device)
            logits = model(tokens)
            # Get the avg cross entropy loss of your predictions
            loss = -get_log_probs(logits, tokens).mean()
            # Backprop on the loss (so our parameters store gradients)
            loss.backward()
            # Update the values of your parameters, w/ Adam optimizer
            optimizer.step()
            # Reset gradients to zero (since grads accumulate with each .backward() call)
            optimizer.zero_grad()

            losses.append(loss.item())
            if c % log_every == 0:
                print(f"Step: {c}, Loss: {loss.item():.4f}")
            if c > max_steps:
                break
```

We can now plot a loss curve!

```python
if MAIN:
    px.line(y=losses, x=np.arange(len(losses))*(model_cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")
```
""")
    st.plotly_chart(fig_dict["training_curve_1"])
    st.markdown(r"""
## A note on this loss curve
""")
    st.info("*This section provides a bit of extra context on why the loss curve looks like this. It's not very important, so if you don't find this interesting then you can skip it and move on to the next section of exercises.*")
    st.markdown(r"""
What's up with the shape of our loss curve? It seems like we start at around 10-11, drop down fast to around 7-8, then level out. It turns out, this is all to do with the kinds of algorithms the model learns during training.

When it starts out, your model will be outputting random noise, which might look a lot like "predict each token with approximately uniform probability", i.e. $Q(x) = 1/d_\text{vocab}$ for all $x$. This gives us a cross entropy loss of $\log (d_\text{vocab})$.

```python
if MAIN:
    d_vocab = model.cfg.d_vocab

    print(f"d_vocab = {d_vocab}")
    print(f"Cross entropy loss on uniform distribution = {math.log(d_vocab)}")
```

The next thing we might expect the model to learn is the frequencies of words in the english language. After all, small common tokens like `" and"` or `" the"` might appear much more frequently than others. This would give us an average cross entropy loss of:

$$
- \sum_x p_x \log p_x
$$

where $p_x$ is the actual frequency of the word in our training data.

We can evaluate this quantity as follows:

```python
if MAIN:
    toks = tokens_dataset[:]["tokens"].flatten()

    freqs = t.bincount(toks, minlength=model.cfg.d_vocab)
    probs = freqs.float() / freqs.sum()

    distn = t.distributions.categorical.Categorical(probs=probs)
    entropy = distn.entropy()

    print(f"Entropy of training data = {entropy}")
```

Now, let's show these values on the graph, and see how they line up with the training pattern!

```python
if MAIN:
    fig = px.line(y=losses, x=np.arange(len(losses))*(model_cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")

    for text, value in zip(["Entropy of uniform distribution", "Entropy of training data"], [math.log(d_vocab), entropy]):
        fig.add_hline(x0=0, x1=1, y=value, line_color="red", line_dash="dash", annotation_text=text, annotation_position="top right")

    fig.show()
```
""")
    st.plotly_chart(fig_dict["training_curve_2"])
    # start
    st.markdown(r"""
We can see this lines up pretty well with our model's observed training patterns.

After unigram frequencies, the next thing our model usually learns is **bigram frequencies** (i.e. the frequency of pairs of adjacent tokens in the training data). For instance, `"I"` and `" am"` are common tokens, but their bigram frequency is much higher than would be suggested if they occurred independently. Bigram frequencies actually take you pretty far, since it also covers:

* Simple grammatical rules (e.g. a full stop being followed by a capitalized word)
* Weird quirks of tokenization (e.g. `" manip"` being followed by `"ulative"`)
* Common names (e.g. `"Barack"` being followed by `" Obama"`)

Even a zero-layer model (i.e. no attention layers or MLPs) can approximate bigram frequencies. The output of a zero-layer transformer for one-hot encoded tokens $t$ is the vector $t^T W_E W_U + b_U$, so we can have the $t$-th row of the matrix $W_E W_U$ be (approxmately) the logit distribution for tokens which follow $t$ in the training data). 

After approximating bigram frequencies, we need to start using smarter techniques, like trigrams (which can only be implemented using attention heads), and **induction heads** (which we'll learn a lot more about in the next set of exercises!).
""")
    # end

# ```python
# %pip install datasets
# %pip install transformers
# ```

# ```python
# import datasets
# import transformers
# import plotly.express as px
# ```

# ## Config

# ```python
# batch_size = 8
# num_epochs = 1
# max_steps = 1000
# log_every = 10
# lr = 1e-3
# weight_decay = 1e-2
# model_cfg = Config(debug=False, d_model=256, n_heads=4, d_head=64, d_mlp=1024, n_layers=2, n_ctx=256, d_vocab=reference_gpt2.cfg.d_vocab)
# ```

# ## Create Data

# We load in a tiny dataset I made, with the first 10K entries in the Pile (inspired by Stas' version for OpenWebText!)

# ```python
# dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
# print(dataset)
# print(dataset[0]['text'][:100])
# tokens_dataset = utils.tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model_cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
# data_loader = t.utils.data.DataLoader(tokens_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
# ```

# ## Create Model

# ```python
# model = DemoTransformer(model_cfg)
# model.to(device)
# ```

# ```
# DemoTransformer(
#   (embed): Embed()
#   (pos_embed): PosEmbed()
#   (blocks): ModuleList(
#     (0): TransformerBlock(
#       (ln1): LayerNorm()
#       (attn): Attention()
#       (ln2): LayerNorm()
#       (mlp): MLP()
#     )
#     (1): TransformerBlock(
#       (ln1): LayerNorm()
#       (attn): Attention()
#       (ln2): LayerNorm()
#       (mlp): MLP()
#     )
#   )
#   (ln_final): LayerNorm()
#   (unembed): Unembed()
# )
# ```

# ## Create Optimizer

# We use AdamW - it's a pretty standard optimizer.

# ```python
# optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# ```

# ## Run Training Loop

# ```python
# losses = []
# print("Number of batches:", len(data_loader))
# for epoch in range(num_epochs):
#     for c, batch in tqdm.tqdm(enumerate(data_loader)):
#         tokens = batch['tokens'].to(device)
#         logits = model(tokens)
#         loss = lm_cross_entropy_loss(logits, tokens)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         losses.append(loss.item())
#         if c % log_every == 0:
#             print(f"Step: {c}, Loss: {loss.item():.4f}")
#         if c > max_steps:
#             break
# ```

# We can now plot a loss curve!

# ```python
# px.line(y=losses, x=np.arange(len(losses))*(model_cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")
# ```
# """)

func_page_list = [
    (section_home, "🏠 Home"), 
    (section_intro, "1️⃣ Understanding Inputs & Outputs of a Transformer"), 
    (section_code, "2️⃣ Clean Transformer Implementation"), 
    (section_training, "3️⃣ Training a model"), 
]

func_list = [func for func, page in func_page_list]
page_list = [page for func, page in func_page_list]

page_dict = {page: idx for idx, (func, page) in enumerate(func_page_list)}


if "current_section" not in st.session_state:
    st.session_state["current_section"] = ["", ""]
if "current_page" not in st.session_state:
    st.session_state["current_page"] = ["", ""]

    
def page():
    # st.session_state["something"] = ""
    # st.session_state["input"] = ""
    with st.sidebar:
        radio = st.radio("Section", page_list) #, on_change=toggle_text_generation)
        st.markdown("---")
        # st.write(st.session_state["current_page"])
    idx = page_dict[radio]
    func = func_list[idx]
    func()
    current_page = r"1_🛠️_Transformer_from_scratch"
    st.session_state["current_section"] = [func.__name__, st.session_state["current_section"][0]]
    st.session_state["current_page"] = [current_page, st.session_state["current_page"][0]]
    prepend = parse_text_from_page(current_page, func.__name__)
    new_section = st.session_state["current_section"][1] != st.session_state["current_section"][0]
    new_page = st.session_state["current_page"][1] != st.session_state["current_page"][0]

    chatbot_setup(prepend=prepend, new_section=new_section, new_page=new_page, debug=False)

    # st.sidebar.write(new_page)
    # st.write(prepend)
    # st.write(len(prepend))
    # st.write(repr(prepend))
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # st.write(len(tokenizer.tokenize(prepend)))


page()