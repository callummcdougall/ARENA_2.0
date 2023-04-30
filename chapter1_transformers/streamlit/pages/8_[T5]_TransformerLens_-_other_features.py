import os
import re, json
import plotly.io as pio

from st_dependencies import *
styling()

def section_other_features():
    st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#available-models">Available Models</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#an-overview-of-the-important-open-source-models-in-the-library">An overview of the important open source models in the library</a></li>
        <li><a class="contents-el" href="#an-overview-of-some-interpretability-friendly-models-i-ve-trained-and-included">An overview of some interpretability-friendly models I've trained and included</a></li>
    </ul></li>
    <li><a class="contents-el" href="#other-resources">Other Resources:</a></li>
    <li><a class="contents-el" href="#transformer-architecture">Transformer architecture</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#parameter-names">Parameter Names</a></li>
        <li><a class="contents-el" href="#activation--hook-names">Activation + Hook Names</a></li>
        <li><a class="contents-el" href="#folding-layernorm-for-the-curious">Folding LayerNorm (For the Curious)</a></li>
    </ul></li>
   <li><a class="contents-el" href="#dealing-with-tokens">Dealing with tokens</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#gotcha-prepend-bos">Gotcha: <code>prepend_bos</code></a></li>
   </ul></li>
   <li><a class="contents-el" href="#factored-matrix-class">Factored Matrix Class</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#basic-examples">Basic Examples</a></li>
       <li><a class="contents-el" href="#medium-example-eigenvalue-copying-scores">Medium Example: Eigenvalue Copying Scores</a></li>
   </ul></li>
   <li><a class="contents-el" href="#generating-text">Generating Text</a></li>
   <li><a class="contents-el" href="#hook-points">Hook Points</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#toy-example">Toy Example</a></li>
   </ul></li>
   <li><a class="contents-el" href="#loading-pre-trained-checkpoints">Loading Pre-Trained Checkpoints</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#example-induction-head-phase-transition">Example: Induction Head Phase Transition</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)
    # start
    st.markdown(r"""
# TransformerLens: Other Features
""")
    st.info(r"""
## Learning Objectives

* Learn about the different models available in the TransformerLens library.
* Read about some transformer implementation details, such as folding layernorm.
* Understand exactly how tokenization works, including subtle details such as the beginning of sequence (BOS) token.
* Use transformers to generate text.
* Build your own custom transformer, with its own hook points.
* Load models at pre-trained checkpoints, and study these models to identify the induction head phase transition.
""")
    # end
    st.error(r"""
Note - some of this might not make sense unless you've gone through some more basic introductions to TransformerLens, like the **TransformerLens & induction circuits** exercises. You can find them from the sidebar on the left.
""")
    # start
    st.markdown(r"""
## Available Models

TransformerLens comes with over 40 open source models available, all of which can be loaded into a consistent(-ish) architecture by just changing the name in `from_pretrained`. You can see [a table of the available models here](https://github.com/neelnanda-io/TransformerLens/blob/main/easy_transformer/model_properties_table.md), including the hyper-parameters and the alias used to load the model in TransformerLens.

**Note:** Though TransformerLens can load in some large transformers (eg OPT-66B), it does not currently support loading a single model across multiple GPUs, so in practice loading a model (eg >7B parameters) will be impractical, depending on your GPU memory. Feel free to reach out if you need to use larger models for your research.

Notably, this means that analysis can be near immediately re-run on a different model by just changing the name.
""")
    # end
    st.markdown(r"""
#### Exercise - re-run some old analysis on a different model

Load in DistilGPT-2 (a distilled version of GPT-2, with half as many layers), with the following code: 

```python
distilgpt2 = HookedTransformer.from_pretrained("distilgpt2")
```

Take some analysis you did for GPT-2 small (e.g. locating induction heads), and re-run the exact same code on DistilGPT-2. Verify that it works without any further modification needed.
""")
    st.markdown(r"""
### An overview of the important open source models in the library

* **GPT-2** - the classic generative pre-trained models from OpenAI
    * Sizes Small (85M), Medium (300M), Large (700M) and XL (1.5B).
    * Trained on ~22B tokens of internet text. ([Open source replication](https://huggingface.co/datasets/openwebtext))
* **GPT-Neo** - Eleuther's replication of GPT-2
    * Sizes 125M, 1.3B, 2.7B
    * Trained on 300B(ish?) tokens of [the Pile](https://pile.eleuther.ai/) a large and diverse dataset including a bunch of code (and weird stuff)
* **[OPT](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/)** - Meta AI's series of open source models
    * Trained on 180B tokens of diverse text.
    * 125M, 1.3B, 2.7B, 6.7B, 13B, 30B, 66B
* **GPT-J** - Eleuther's 6B parameter model, trained on the Pile
* **GPT-NeoX** - Eleuther's 20B parameter model, trained on the Pile
* **Stanford CRFM models** - a replication of GPT-2 Small and GPT-2 Medium, trained on 5 different random seeds.
    * Notably, 600 checkpoints were taken during training per model, and these are available in the library with eg `HookedTransformer.from_pretrained("stanford-gpt2-small-a", checkpoint_index=265)`.

### An overview of some interpretability-friendly models I've trained and included

(Feel free to [reach out](mailto:neelnanda27@gmail.com) if you want more details on any of these models)

Each of these models has about ~200 checkpoints taken during training that can also be loaded from TransformerLens, with the `checkpoint_index` argument to `from_pretrained`.

Note that all models are trained with a Beginning of Sequence token, and will likely break if given inputs without that! 

* **Toy Models**: Inspired by [A Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html), I've trained 12 tiny language models, of 1-4L and each of width 512. I think that interpreting these is likely to be far more tractable than larger models, and both serve as good practice and will likely contain motifs and circuits that generalise to far larger models (like induction heads):
    * Attention-Only models (ie without MLPs): attn-only-1l, attn-only-2l, attn-only-3l, attn-only-4l
    * GELU models (ie with MLP, and the standard GELU activations): gelu-1l, gelu-2l, gelu-3l, gelu-4l
    * SoLU models (ie with MLP, and [Anthropic's SoLU activation](https://transformer-circuits.pub/2022/solu/index.html), designed to make MLP neurons more interpretable): solu-1l, solu-2l, solu-3l, solu-4l
    * All models are trained on 22B tokens of data, 80% from C4 (web text) and 20% from Python Code
    * Models of the same layer size were trained with the same weight initialization and data shuffle, to more directly compare the effect of different activation functions.
* **SoLU** models: A larger scan of models trained with [Anthropic's SoLU activation](https://transformer-circuits.pub/2022/solu/index.html), in the hopes that it makes the MLP neuron interpretability easier. 
    * A scan up to GPT-2 Medium size, trained on 30B tokens of the same data as toy models, 80% from C4 and 20% from Python code. 
        * solu-6l (40M), solu-8l (100M), solu-10l (200M), solu-12l (340M)
    * An older scan up to GPT-2 Medium size, trained on 15B tokens of [the Pile](https://pile.eleuther.ai/)
        * solu-1l-pile (13M), solu-2l-pile (13M), solu-4l-pile (13M), solu-6l-pile (40M), solu-8l-pile (100M), solu-10l-pile (200M), solu-12l-pile (340M)

## Other Resources:

* [Concrete Open Problems in Mechanistic Interpretability](https://docs.google.com/document/d/1WONBzNqfKIxERejrrPlQMyKqg7jSFW92x5UMXNrMdPo/edit), a doc I wrote giving a long list of open problems in mechanistic interpretability, and thoughts on how to get started on trying to work on them. 
    * There's a lot of low-hanging fruit in the field, and I expect that many people reading this could use TransformerLens to usefully make progress on some of these!
* Other demos:
    * **[Exploratory Analysis Demo](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/Exploratory_Analysis_Demo.ipynb)**, a demonstration of my standard toolkit for how to use TransformerLens to explore a mysterious behaviour in a language model.
    * [Interpretability in the Wild](https://github.com/redwoodresearch/Easy-Transformer) a codebase from Arthur Conmy and Alex Variengien at Redwood research using this library to do a detailed and rigorous reverse engineering of the Indirect Object Identification circuit, to accompany their paper
        * Note - this was based on an earlier version of this library, called EasyTransformer. It's pretty similar, but several breaking changes have been made since. 
    * A [recorded walkthrough](https://www.youtube.com/watch?v=yo4QvDn-vsU) of me doing research with TransformerLens on whether a tiny model can re-derive positional information, with [an accompanying Colab](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/No_Position_Experiment.ipynb)
* [Neuroscope](https://neuroscope.io), a website showing the text in the dataset that most activates each neuron in some selected models. Good to explore to get a sense for what kind of features the model tends to represent, and as a "wiki" to get some info
    * A tutorial on how to make an [Interactive Neuroscope](https://github.com/neelnanda-io/TransformerLens/blob/main/Hacky-Interactive-Lexoscope.ipynb), where you type in text and see the neuron activations over the text update live.
""")
    # start
    st.markdown(r"""
### Folding LayerNorm (For the Curious)

(For the curious - this is an important technical detail that's worth understanding, especially if you have preconceptions about how transformers work, but not necessary to use TransformerLens)

LayerNorm is a normalization technique used by transformers, analogous to BatchNorm but more friendly to massive parallelisation. No one *really* knows why it works, but it seems to improve model numerical stability. Unlike BatchNorm, LayerNorm actually changes the functional form of the model, which makes it a massive pain for interpretability! 

Folding LayerNorm is a technique to make it lower overhead to deal with, and the flags `center_writing_weights` and `fold_ln` in `HookedTransformer.from_pretrained` apply this automatically (they default to True). These simplify the internal structure without changing the weights.

Intuitively, LayerNorm acts on each residual stream vector (ie for each batch element and token position) independently, sets their mean to 0 (centering) and standard deviation to 1 (normalizing) (*across* the residual stream dimension - very weird!), and then applies a learned elementwise scaling and translation to each vector.

Mathematically, centering is a linear map, normalizing is *not* a linear map, and scaling and translation are linear maps. 
* **Centering:** LayerNorm is applied every time a layer reads from the residual stream, so the mean of any residual stream vector can never matter - `center_writing_weights` set every weight matrix writing to the residual to have zero mean. 
* **Normalizing:** Normalizing is not a linear map, and cannot be factored out. The `hook_scale` hook point lets you access and control for this.
* **Scaling and Translation:** Scaling and translation are linear maps, and are always followed by another linear map. The composition of two linear maps is another linear map, so we can *fold* the scaling and translation weights into the weights of the subsequent layer, and simplify things without changing the underlying computation. 

[See the docs for more details](https://github.com/neelnanda-io/TransformerLens/blob/main/further_comments.md#what-is-layernorm-folding-fold_ln)
""")
    # end
    st.markdown(r"""
A fun consequence of LayerNorm folding is that it creates a bias across the unembed, a `d_vocab` length vector that is added to the output logits - GPT-2 is not trained with this, but it *is* trained with a final LayerNorm that contains a bias. 

Turns out, this LayerNorm bias learns structure of the data that we can only see after folding! In particular, it essentially learns **unigram statistics** - rare tokens get suppressed, common tokens get boosted, by pretty dramatic degrees! Let's list the top and bottom 20 - at the top we see common punctuation and words like " the" and " and", at the bottom we see weird-ass tokens like " RandomRedditor":

```python
unembed_bias = model.unembed.b_U
bias_values, bias_indices = unembed_bias.sort(descending=True)
```

```python
top_k = 20
print(f"Top {top_k} values")
for i in range(top_k):
    print(f"{bias_values[i].item():.2f} {repr(model.to_string(bias_indices[i]))}")

print("...")
print(f"Bottom {top_k} values")
for i in range(top_k, 0, -1):
    print(f"{bias_values[-i].item():.2f} {repr(model.to_string(bias_indices[-i]))}")
```

This can have real consequences for interpretability - for example, this bias favours " John" over " Mary" by about 1.2, about 1/3 of the effect size of the Indirect Object Identification Circuit! All other things being the same, this makes the John token 3.6x times more likely than the Mary token.

```python
john_bias = model.unembed.b_U[model.to_single_token(' John')]
mary_bias = model.unembed.b_U[model.to_single_token(' Mary')]

print(f"John bias: {john_bias.item():.4f}")
print(f"Mary bias: {mary_bias.item():.4f}")
print(f"Prob ratio bias: {t.exp(john_bias - mary_bias).item():.4f}x")
```
""")
    # start
    st.markdown(r"""

## Dealing with tokens

**Tokenization** is one of the most annoying features of studying language models. We want language models to be able to take in arbitrary text as input, but the transformer architecture needs the inputs to be elements of a fixed, finite vocabulary. The solution to this is **tokens**, a fixed vocabulary of "sub-words", that any natural language can be broken down into with a **tokenizer**. This is invertible, and we can recover the original text, called **de-tokenization**. 

TransformerLens comes with a range of utility functions to deal with tokenization. Different models can have different tokenizers, so these are all methods on the model.

`get_token_position`, `to_tokens`, `to_string`, `to_str_tokens`, `prepend_bos`, `to_single_token`

The first thing you need to figure out is *how* things are tokenized. `model.to_str_tokens` splits a string into the tokens *as a list of substrings*, and so lets you explore what the text looks like. To demonstrate this, let's use it on this paragraph.

Some observations - there are a lot of arbitrary-ish details in here!
* The tokenizer splits on spaces, so no token contains two words.
* Tokens include the preceding space, and whether the first token is a capital letter. `'how'` and `' how'` are different tokens!
* Common words are single tokens, even if fairly long (` paragraph`) while uncommon words are split into multiple tokens (` token|ized`).
* Tokens *mostly* split on punctuation characters (eg `*` and `.`), but eg `'s` is a single token.
""")
    # end
    st.markdown(r"""
```python
example_text = "The first thing you need to figure out is *how* things are tokenized. `model.to_str_tokens` splits a string into the tokens *as a list of substrings*, and so lets you explore what the text looks like. To demonstrate this, let's use it on this paragraph."
example_text_str_tokens = model.to_str_tokens(example_text)
print(example_text_str_tokens)
```

The transformer needs to take in a sequence of integers, not strings, so we need to convert these tokens into integers. `model.to_tokens` does this, and returns a tensor of integers on the model's device (shape `[batch, position]`). It maps a string to a batch of size 1.

```python
example_text_tokens = model.to_tokens(example_text)
print(example_text_tokens)
```

`to_tokens` can also take in a list of strings, and return a batch of size `len(strings)`. If the strings are different numbers of tokens, it adds a PAD token to the end of the shorter strings to make them the same length.

(Note: In GPT-2, 50256 signifies both the beginning of sequence, end of sequence and padding token - see the `prepend_bos` section for details)

```python
example_multi_text = ["The cat sat on the mat.", "The cat sat on the mat really hard."]
example_multi_text_tokens = model.to_tokens(example_multi_text)
print(example_multi_text_tokens)
```

`model.to_single_token` is a convenience function that takes in a string corresponding to a *single* token and returns the corresponding integer. This is useful for eg looking up the logit corresponding to a single token. 

For example, let's input `The cat sat on the mat.` to GPT-2, and look at the log prob predicting that the next token is ` The`. 
""")
    with st.expander("Technical notes"):
        st.markdown(r"""
Note that if we input a string to the model, it's implicitly converted to a string with `to_tokens`. 

Note further that the log probs have shape `[batch, position, d_vocab]==[1, 8, 50257]`, with a vector of log probs predicting the next token for *every* token position. GPT-2 uses causal attention which means heads can only look backwards (equivalently, information can only move forwards in the model.), so the log probs at position k are only a function of the first k tokens, and it can't just cheat and look at the k+1 th token. This structure lets it generate text more efficiently, and lets it treat every *token* as a training example, rather than every *sequence*.
""")
    st.markdown(r"""
```python
cat_text = "The cat sat on the mat."
cat_logits = model(cat_text)
cat_probs = cat_logits.softmax(dim=-1)
print(f"Probability tensor shape [batch, position, d_vocab] == {cat_probs.shape}")

capital_the_token_index = model.to_single_token(" The")
print(f"| The| probability: {cat_probs[0, -1, capital_the_token_index].item():.2%}")
```

`model.to_string` is the inverse of `to_tokens` and maps a tensor of integers to a string or list of strings. It also works on integers and lists of integers.

For example, let's look up token 256 (due to technical details of tokenization, this will be the most common pair of ASCII characters!), and also verify that our tokens above map back to a string.

```python
print(f"Token 256 - the most common pair of ASCII characters: |{model.to_string(256)}|")
# Squeeze means to remove dimensions of length 1. 
# Here, that removes the dummy batch dimension so it's a rank 1 tensor and returns a string
# Rank 2 tensors map to a list of strings
print(f"De-Tokenizing the example tokens: {model.to_string(example_text_tokens.squeeze())}")
```

A related annoyance of tokenization is that it's hard to figure out how many tokens a string will break into. `model.get_token_position(single_token, tokens)` returns the position of `single_token` in `tokens`. `tokens` can be either a string or a tensor of tokens. 

Note that position is zero-indexed, it's two (ie third) because there's a beginning of sequence token automatically prepended (see the next section for details)

```python
print("With BOS:", model.get_token_position(" cat", "The cat sat on the mat"))
print("Without BOS:", model.get_token_position(" cat", "The cat sat on the mat", prepend_bos=False))
```

If there are multiple copies of the token, we can set `mode="first"` to find the first occurence's position and `mode="last"` to find the last

```python
print("First occurence", model.get_token_position(
    " cat", 
    "The cat sat on the mat. The mat sat on the cat.", 
    mode="first"))
print("Final occurence", model.get_token_position(
    " cat", 
    "The cat sat on the mat. The mat sat on the cat.", 
    mode="last"))
```
""")
    # start
    st.markdown(r"""
In general, tokenization is a pain, and full of gotchas. I highly recommend just playing around with different inputs and their tokenization and getting a feel for it. As another "fun" example, let's look at the tokenization of arithmetic expressions - tokens do *not* contain consistent numbers of digits. (This makes it even more impressive that GPT-3 can do arithmetic!)

```python
print(model.to_str_tokens("2342+2017=21445"))
print(model.to_str_tokens("1000+1000000=999999"))
```

I also *highly* recommend investigating prompts with easy tokenization when starting out - ideally key words should form a single token, be in the same position in different prompts, have the same total length, etc. Eg study Indirect Object Identification with common English names like ` Tim` rather than ` Ne|el`. Transformers need to spend some parameters in early layers converting multi-token words to a single feature, and then de-converting this in the late layers, and unless this is what you're explicitly investigating, this will make the behaviour you're investigating be messier.

### Gotcha: `prepend_bos`

##### Key Takeaway: **If you get weird off-by-one errors, check whether there's an unexpected `prepend_bos`!**

A weirdness you may have noticed in the above is that `to_tokens` and `to_str_tokens` added a weird `<|endoftext|>` to the start of each prompt. TransformerLens does this by default, and it can easily trip up new users. Notably, **this includes `model.forward`** (which is what's implicitly used when you do eg `model("Hello World")`). This is called a **Beginning of Sequence (BOS)** token, and it's a special token used to mark the beginning of the sequence. Confusingly, in GPT-2, the End of Sequence (EOS), Beginning of Sequence (BOS) and Padding (PAD) tokens are all the same, `<|endoftext|>` with index `50256`.

You can disable this behaviour by setting the flag `prepend_bos=False` in `to_tokens`, `to_str_tokens`, `model.forward` and any other function that converts strings to multi-token tensors. 
""")
    # end
    st.markdown(r"""

**Gotcha:** You only want to do this at the *start* of a prompt. If you, eg, want to input a question followed by an answer, and want to tokenize these separately, you do *not* want to prepend_bos on the answer.

```python
print("Logits shape by default (with BOS)", model("Hello World").shape)
print("Logits shape with BOS", model("Hello World", prepend_bos=True).shape)
print("Logits shape without BOS - only 2 positions!", model("Hello World", prepend_bos=False).shape)
```

`prepend_bos` is a bit of a hack, and I've gone back and forth on what the correct default here is. The reason I do this is that transformers tend to treat the first token weirdly - this doesn't really matter in training (where all inputs are >1000 tokens), but this can be a big issue when investigating short prompts! The reason for this is that attention patterns are a probability distribution and so need to add up to one, so to simulate being "off" they normally look at the first token. Giving them a BOS token lets the heads rest by looking at that, preserving the information in the first "real" token.

Further, *some* models are trained to need a BOS token (OPT and my interpretability-friendly models are, GPT-2 and GPT-Neo are not). But despite GPT-2 not being trained with this, empirically it seems to make interpretability easier.

For example, the model can get much worse at Indirect Object Identification without a BOS (and with a name as the first token):

```python
ioi_logits_with_bos = model("Claire and Mary went to the shops, then Mary gave a bottle of milk to", prepend_bos=True)
mary_logit_with_bos = ioi_logits_with_bos[0, -1, model.to_single_token(" Mary")].item()
claire_logit_with_bos = ioi_logits_with_bos[0, -1, model.to_single_token(" Claire")].item()
print(f"Logit difference with BOS: {(claire_logit_with_bos - mary_logit_with_bos):.3f}")

ioi_logits_without_bos = model("Claire and Mary went to the shops, then Mary gave a bottle of milk to", prepend_bos=False)
mary_logit_without_bos = ioi_logits_without_bos[0, -1, model.to_single_token(" Mary")].item()
claire_logit_without_bos = ioi_logits_without_bos[0, -1, model.to_single_token(" Claire")].item()
print(f"Logit difference without BOS: {(claire_logit_without_bos - mary_logit_without_bos):.3f}")
```

Though, note that this also illustrates another gotcha - when `Claire` is at the start of a sentence (no preceding space), it's actually *two* tokens, not one, which probably confuses the relevant circuit. (Note - in this test we put `prepend_bos=False`, because we want to analyse the tokenization of a specific string, not to give an input to the model!)

```
print(f"| Claire| -> {model.to_str_tokens(' Claire', prepend_bos=False)}")
print(f"|Claire| -> {model.to_str_tokens('Claire', prepend_bos=False)}")
```
""")
    # start
    st.markdown(r"""
    
## Generating Text

TransformerLens also has basic text generation functionality, which can be useful for generally exploring what the model is capable of (thanks to Ansh Radhakrishnan for adding this!). This is pretty rough functionality, and where possible I recommend using more established libraries like HuggingFace for this.

```python
model.generate("(CNN) President Barack Obama caught in embarrassing new scandal\n", max_new_tokens=50, temperature=0.7, prepend_bos=True)
```
""")
    st.markdown(r"""
## Hook Points

The key part of TransformerLens that lets us access and edit intermediate activations are the HookPoints around every model activation. Importantly, this technique will work for *any* model architecture, not just transformers, so long as you're able to edit the model code to add in HookPoints! This is essentially a lightweight library bundled with TransformerLens that should let you take an arbitrary model and make it easier to study. 
""")
    # end
    st.markdown(r"""
This is implemented by having a HookPoint layer. Each transformer component has a HookPoint for every activation, which wraps around that activation. The HookPoint acts as an identity function, but has a variety of helper functions that allows us to put PyTorch hooks in to edit and access the relevant activation. 

There is also a `HookedRootModule` class - this is a utility class that the root module should inherit from (root module = the model we run) - it has several utility functions for using hooks well, notably `reset_hooks`, `run_with_cache` and `run_with_hooks`. 

The default interface is the `run_with_hooks` function on the root module, which lets us run a forwards pass on the model, and pass on a list of hooks paired with layer names to run on that pass. 

The syntax for a hook is `function(activation, hook)` where `activation` is the activation the hook is wrapped around, and `hook` is the `HookPoint` class the function is attached to. If the function returns a new activation or edits the activation in-place, that replaces the old one, if it returns None then the activation remains as is.
""")
    st.markdown(r"""
### Toy Example

Here's a simple example of defining a small network with HookPoints:

We define a basic network with two layers that each take a scalar input $x$, square it, and add a constant:
$x_0=x$, $x_1=x_0^2+3$, $x_2=x_1^2-4$.

We wrap the input, each layer's output, and the intermediate value of each layer (the square) in a hook point.

```python
from transformer_lens.hook_points import HookedRootModule, HookPoint

class SquareThenAdd(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = nn.Parameter(t.tensor(offset))
        self.hook_square = HookPoint()

    def forward(self, x):
        # The hook_square doesn't change the value, but lets us access it
        square = self.hook_square(x * x)
        return self.offset + square

class TwoLayerModel(HookedRootModule):
    def __init__(self):
        super().__init__()
        self.layer1 = SquareThenAdd(3.0)
        self.layer2 = SquareThenAdd(-4.0)
        self.hook_in = HookPoint()
        self.hook_mid = HookPoint()
        self.hook_out = HookPoint()

        # We need to call the setup function of HookedRootModule to build an
        # internal dictionary of modules and hooks, and to give each hook a name
        super().setup()

    def forward(self, x):
        # We wrap the input and each layer's output in a hook - they leave the
        # value unchanged (unless there's a hook added to explicitly change it),
        # but allow us to access it.
        x_in = self.hook_in(x)
        x_mid = self.hook_mid(self.layer1(x_in))
        x_out = self.hook_out(self.layer2(x_mid))
        return x_out

model = TwoLayerModel()
```

We can add a cache, to save the activation at each hook point

(There's a custom `run_with_cache` function on the root module as a convenience, which is a wrapper around model.forward that return model_out, cache_object - we could also manually add hooks with `run_with_hooks` that store activations in a global caching dictionary. This is often useful if we only want to store, eg, subsets or functions of some activations.)

```python
out, cache = model.run_with_cache(t.tensor(5.0))
print("Model output:", out.item())
for key in cache:
    print(f"Value cached at hook {key}", cache[key].item())
```

We can also use hooks to intervene on activations - eg, we can set the intermediate value in layer 2 to zero to change the output to -5

```python
def set_to_zero_hook(tensor, hook):
    print(hook.name)
    return t.tensor(0.0)

print(
    "Output after intervening on layer2.hook_scaled",
    model.run_with_hooks(
        t.tensor(5.0), fwd_hooks=[("layer2.hook_square", set_to_zero_hook)]
    ).item(),
)
```
""")
    # start
    st.markdown(r"""
## Loading Pre-Trained Checkpoints

There are a lot of interesting questions combining mechanistic interpretability and training dynamics - analysing model capabilities and the underlying circuits that make them possible, and how these change as we train the model. 

TransformerLens supports these by having several model families with checkpoints throughout training. `HookedTransformer.from_pretrained` can load a checkpoint of a model with the `checkpoint_index` (the label 0 to `num_checkpoints-1`) or `checkpoint_value` (the step or token number, depending on how the checkpoints were labelled).
""")
    # end
    st.markdown(r"""
Available models:
* All of my interpretability-friendly models have checkpoints available, including:
    * The toy models - `attn-only`, `solu`, `gelu` 1L to 4L
        * These have ~200 checkpoints, taken on a piecewise linear schedule (more checkpoints near the start of training), up to 22B tokens. Labelled by number of tokens seen.
    * The SoLU models trained on 80% Web Text and 20% Python Code (`solu-6l` to `solu-12l`)
        * Same checkpoint schedule as the toy models, this time up to 30B tokens
    * The SoLU models trained on the pile (`solu-1l-pile` to `solu-12l-pile`)
        * These have ~100 checkpoints, taken on a linear schedule, up to 15B tokens. Labelled by number of steps.
        * The 12L training crashed around 11B tokens, so is truncated.
* The Stanford Centre for Research of Foundation Models trained 5 GPT-2 Small sized and 5 GPT-2 Medium sized models (`stanford-gpt2-small-a` to `e` and `stanford-gpt2-medium-a` to `e`)
    * 600 checkpoints, taken on a piecewise linear schedule, labelled by the number of steps.

The checkpoint structure and labels is somewhat messy and ad-hoc, so I mostly recommend using the `checkpoint_index` syntax (where you can just count from 0 to the number of checkpoints) rather than `checkpoint_value` syntax (where you need to know the checkpoint schedule, and whether it was labelled with the number of tokens or steps). The helper function `get_checkpoint_labels` tells you the checkpoint schedule for a given model - ie what point was each checkpoint taken at, and what type of label was used.

Here are graphs of the schedules for several checkpointed models: (note that the first 3 use a log scale, latter 2 use a linear scale)

```python
from transformer_lens.loading_from_pretrained import get_checkpoint_labels
for model_name in ["attn-only-2l", "solu-12l", "stanford-gpt2-small-a"]:
    checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(model_name)
    line(checkpoint_labels, xaxis="Checkpoint Index", yaxis=f"Checkpoint Value ({checkpoint_label_type})", title=f"Checkpoint Values for {model_name} (Log scale)", log_y=True, markers=True)
for model_name in ["solu-1l-pile", "solu-6l-pile"]:
    checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(model_name)
    line(checkpoint_labels, xaxis="Checkpoint Index", yaxis=f"Checkpoint Value ({checkpoint_label_type})", title=f"Checkpoint Values for {model_name} (Linear scale)", log_y=False, markers=True)
```

### Example: Induction Head Phase Transition

One of the more interesting results analysing circuit formation during training is the [induction head phase transition](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). They find a pretty dramatic shift in models during training - there's a brief period where models go from not having induction heads to having them, which leads to the models suddenly becoming much better at in-context learning (using far back tokens to predict the next token, eg over 500 words back). This is enough of a big deal that it leads to a visible *bump* in the loss curve, where the model's rate of improvement briefly increases. 

As a brief demonstration of the existence of the phase transition, let's load some checkpoints of a two layer model, and see whether they have induction heads. An easy test, as we used above, is to give the model a repeated sequence of random tokens, and to check how good its loss is on the second half. `evals.induction_loss` is a rough util that runs this test on a model.
(Note - this is deliberately a rough, non-rigorous test for the purposes of demonstration, eg `evals.induction_loss` by default just runs it on 4 sequences of 384 tokens repeated twice. These results totally don't do the paper justice - go check it out if you want to see the full results!)

In the interests of time and memory, let's look at a handful of checkpoints (chosen to be around the phase change), indices `[10, 25, 35, 60, -1]`. These are roughly 22M, 200M, 500M, 1.6B and 21.8B tokens through training, respectively. (I generally recommend looking things up based on indices, rather than checkpoint value!). 

```python
from transformer_lens import evals
# We use the two layer model with SoLU activations, chosen fairly arbitrarily as being both small (so fast to download and keep in memory) and pretty good at the induction task.
model_name = "solu-2l"
# We can load a model from a checkpoint by specifying the checkpoint_index, -1 means the final checkpoint
checkpoint_indices = [10, 25, 35, 60, -1]
checkpointed_models = []
tokens_trained_on = []
induction_losses = []
```

We load the models, cache them in a list, and:

```python
for index in checkpoint_indices:
    # Load the model from the relevant checkpoint by index
    model_for_this_checkpoint = HookedTransformer.from_pretrained(model_name, checkpoint_index=index)
    checkpointed_models.append(model_for_this_checkpoint)

    tokens_seen_for_this_checkpoint = model_for_this_checkpoint.cfg.checkpoint_value
    tokens_trained_on.append(tokens_seen_for_this_checkpoint)

    induction_loss_for_this_checkpoint = evals.induction_loss(model_for_this_checkpoint).item()
    induction_losses.append(induction_loss_for_this_checkpoint)
```

We can plot this, and see there's a sharp shift from ~200-500M tokens trained on (note the log scale on the x axis). Interestingly, this is notably earlier than the phase transition in the paper, I'm not sure what's up with that.

(To contextualise the numbers, the tokens in the random sequence are uniformly chosen from the first 20,000 tokens (out of ~48,000 total), so random performance is at least $\ln(20000)\approx 10$. A naive strategy like "randomly choose a token that's already appeared in the first half of the sequence (384 elements)" would get $\ln(384)\approx 5.95$, so the model is doing pretty well here.)

```python
line(induction_losses, x=tokens_trained_on, xaxis="Tokens Trained On", yaxis="Induction Loss", title="Induction Loss over training: solu-2l", markers=True, log_x=True)
```
""")

if "current_section" not in st.session_state:
    st.session_state["current_section"] = ["", ""]
if "current_page" not in st.session_state:
    st.session_state["current_page"] = ["", ""]

# if is_local or check_password():
def page():
    section_other_features()
    current_page = r"7_💽_TransformerLens_-_other_features"
    prepend = parse_text_from_page(current_page, r"section_other_features")
    st.session_state["current_page"] = [current_page, st.session_state["current_page"][0]]
    new_page = st.session_state["current_page"][0] != st.session_state["current_page"][1]
    chatbot_setup(prepend=prepend, new_page=new_page, debug=True)

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # st.sidebar.write(len(prepend))
    # st.sidebar.write(len(tokenizer.tokenize(prepend)))

page()

