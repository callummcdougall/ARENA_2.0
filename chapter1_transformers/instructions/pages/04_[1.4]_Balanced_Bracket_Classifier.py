
import os, sys
from pathlib import Path
chapter = r"chapter1_transformers"
instructions_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/instructions").resolve()
if str(instructions_dir) not in sys.path: sys.path.append(str(instructions_dir))
os.chdir(instructions_dir)

import streamlit as st
import st_dependencies

st_dependencies.styling()

import platform
is_local = (platform.processor() != "")

def section_0():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#motivation'>Motivation</a></li>
    <li class='margtop'><a class='contents-el' href='#have-some-sense-of-proportion'>Have some sense of proportion</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

<img src="https://raw.githubusercontent.com/callummcdougall/TransformerLens-intro/main/images/page_images/gears2.png" width="350">


Colab: [**exercises**](https://colab.research.google.com/drive/1BYarO508z7stRFXZ3T92rI6OtMqP3w7E) | [**solutions**](https://colab.research.google.com/drive/1yILyi5dD3wc4o3vHc1MfbpAlWljyXU-U)

Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.


# [1.4] Balanced Bracket Classifier


## Introduction


When models are trained on synthetic, algorithmic tasks, they often learn to do some clean, interpretable computation inside. Choosing a suitable task and trying to reverse engineer a model can be a rich area of interesting circuits to interpret! In some sense, this is interpretability on easy mode - the model is normally trained on a single task (unlike language models, which need to learn everything about language!), we know the exact ground truth about the data and optimal solution, and the models are tiny. So why care?

Working on algorithmic problems gives us the opportunity to:

* Practice interpretability, and build intuitions and learn techniques.
* Refine our understanding of the right tools and techniques, by trying them out on problems with well-understood ground truth.
* Isolate a particularly interesting kind of behaviour, in order to study it in detail and understand it better (e.g. Anthropic's [Toy Models of Superposition](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=EuO4CLwSIzX7AEZA1ZOsnwwF) paper).
* Take the insights you've learned from reverse-engineering small models, and investigate which results will generalise, or whether any of the techniques you used to identify circuits can be automated and used at scale.

The algorithmic problem we'll work on in these exercises is **bracket classification**, i.e. taking a string of parentheses like `"(())()"` and trying to output a prediction of "balanced" or "unbalanced". We will find an algorithmic solution for solving this problem, and reverse-engineer one of the circuits in our model that is responsible for implementing one part of this algorithm.

This page contains a large number of exercise. Each exercise will have a difficulty and importance rating out of 5, as well as an estimated maximum time you should spend on these exercises and sometimes a short annotation. You should interpret the ratings & time estimates relatively (e.g. if you find yourself spending about 50% longer on the exercises than the time estimates, adjust accordingly). Please do skip exercises / look at solutions if you don't feel like they're important enough to be worth doing, and you'd rather get to the good stuff!


## Motivation

In A [Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), we got a lot of traction interpreting toy language models - that is, transformers trained in exactly the same way as larger models, but with only 1 or 2 layers. It seems likely that there’s a lot of low-hanging fruit left to pluck when studying toy language models!

So, why care about studying toy language models? The obvious reason is that **it’s way easier to get traction**. In particular, the [inputs and outputs](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=UriJZK6E8dnL8NDY-fGl_eFX) of a model are intrinsically interpretable, and in a toy model there’s just not as much space between the inputs and outputs for weird complexity to build up. But the obvious objection to the above is that, ultimately, we care about understanding real models (and ideally extremely large ones like GPT-3), and learning to interpret toy models is not the actual goal. This is a pretty valid objection, but there are two natural ways that studying toy models can be valuable:

The first is by finding fundamental circuits that recur in larger models, and [motifs](https://distill.pub/2020/circuits/zoom-in/#claim-2-motifs) that allow us to easily identify these circuits in larger models. A key underlying question here is that of [universality](https://distill.pub/2020/circuits/zoom-in/#claim-3): does each model learn its own weird way of completing its task, or are there some fundamental principles and algorithms that all models converge on?

The second is by forming a better understanding of how to reverse engineer models - what are the right intuitions and conceptual frameworks, what tooling and techniques do and do not work, and what weird limitations we might be faced with. For instance, the work in A Mathematical Framework presents ideas like [the residual stream as the central object](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=DHp9vZ0h9lA9OCrzG2Y3rrzH), and the significance of the [QK-Circuits](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=n_Lc0Z5N9HMhAYcycDda-UEB) and [OV-Circuits](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=n_Lc0Z5N9HMhAYcycDda-UEB), which seem to generalise to many different models. We'll also see an example later in these exercises which illustrates how MLPs can be thought of as a collection of neurons which activate on different features, just like many seem to in language models. But there’s also ways it can be misleading, and some techniques that work well in toy models seem to generalise less well. 


## Have some sense of proportion

At a surface level, these exercises are designed to guide you through a partial interpretation of the bidirectional model trained on bracket classification. But it's also designed to make you a better interpretability researcher! As a result, most exercises will be doing a combination of:

1. Showing you some new feature/component of the circuit, and
2. Teaching you how to use tools and interpret results in a broader mech interp context.

Here is a rough conceptual graph showing all the different things you should be thinking about when going through these exercises, and how they relate to both of these goals, as well all the `transformerlens` tools which will help you.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/brackets-map-simpler.png" width="750">


## Content & Learning Objectives


#### 1️⃣ Bracket classifier

This section describes how transformers can be used for classification, and the details of how this works in TransformerLens (using permanent hooks). It also takes you through the exercise of hand-writing a solution to the balanced brackets problem.

*This section mainly just lays the groundwork; it is very light on content.*

> ##### Learning objctives
> 
> * Understand how transformers can be used for classification.
> * Understand how to implement specific kinds of transformer behaviour (e.g. masking of padding tokens) via permanent hooks in TransformerLens.
> * Start thinking about the kinds of algorithmic solutions a transformer is likely to find for problems such as these, given its inductive biases.

#### 2️⃣ Moving backwards

Here, you'll perform logit attribution, and learn how to work backwards through particular paths of a model to figure out which components matter most for the final classification probabilities.

This is the first time you'll have to deal with LayerNorm in your models.

*This section should be familiar if you've done logit attribution for induction heads (although these exercises are slightly more challenging from a coding perspective). The LayerNorm-based exercises are a bit fiddly!*

> ##### Learning objctives
>
> * Understand how to perform logit attribution.
> * Understand how to work backwards through a model to identify which components matter most for the final classification probabilities.
> * Understand how LayerNorm works, and look at some ways to deal with it in your models.

#### 3️⃣ Total elevation circuit

*This section is quite challenging both from a coding and conceptual perspective, because you need to link the results of your observations and interventions to concrete hypotheses about how the model works.*

In the largest section of the exercises, you'll examine the attention patterns in different heads, and interpret them as performing some human-understandable algorithm (e.g. copying, or aggregation). You'll use your observations to make  deductions about how a particular type of balanced brackets failure mode (mismatched number of left and right brackets) is detected by your model.

This is the first time you'll have to deal with MLPs in your models.

> ##### Learning objctives
>
> * Practice connecting distinctive attention patterns to human-understandable algorithms, and making deductions about model behaviour. 
> * Understand how MLPs can be viewed as a collection of neurons.
> * Build up to a full picture of the total elevation circuit and how it works.

#### 4️⃣ Bonus exercises

Lastly, there are a few optional bonus exercises which build on the previous content (e.g. having you examine different parts of the model, or use your understanding of how the model works to generate adversarial examples).

*This final section is less guided, although the suggested exercises are similar in flavour to the previous section.*

> ##### Learning objctives
>
> * Use your understanding of how the model works to generate adversarial examples.
> * Take deeper dives into specific anomalous features of the model.


## Setup


```python
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from functools import partial
import json
from typing import List, Tuple, Union, Optional, Callable, Dict
import torch as t
from torch import Tensor
from sklearn.linear_model import LinearRegression
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import einops
from tqdm import tqdm
from jaxtyping import Float, Int, Bool
from pathlib import Path
import pandas as pd
import circuitsvis as cv
import webbrowser
from IPython.display import display
from transformer_lens import utils, ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_interp_on_algorithmic_model"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import plotly_utils
from plotly_utils import hist, bar, imshow
import part4_interp_on_algorithmic_model.tests as tests
from part4_interp_on_algorithmic_model.brackets_datasets import SimpleTokenizer, BracketsDataset

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

```



""", unsafe_allow_html=True)


def section_1():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#today's-toy-model'>Today's Toy Model</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#causal-vs-bidirectional-attention'>Causal vs bidirectional attention</a></li>
        <li><a class='contents-el' href='#using-transformers-for-classification'>Using transformers for classification</a></li>
        <li><a class='contents-el' href='#a-note-on-softmax'>A note on softmax</a></li>
        <li><a class='contents-el' href='#masking-padding-tokens'>Masking padding tokens</a></li>
        <li><a class='contents-el' href='#other-details'>Other details</a></li>
        <li><a class='contents-el' href='#some-useful-diagrams'>Some useful diagrams</a></li>
        <li><a class='contents-el' href='#defining-the-model'>Defining the model</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#tokenizer'>Tokenizer</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#implementing-our-masking'>Implementing our masking</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#dataset'>Dataset</a></li>
    <li class='margtop'><a class='contents-el' href='#algorithmic-solutions'>Algorithmic Solutions</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-handwritten-solution-for-loop'><b>Exercise</b> - handwritten solution (for loop)</a></li>
        <li><a class='contents-el' href='#exercise-handwritten-solution-vectorized'><b>Exercise</b> -  handwritten solution (vectorized)</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#the-model's-solution'>The Model's Solution</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 1️⃣ Bracket classifier


This section describes how transformers can be used for classification, and the details of how this works in TransformerLens (using permanent hooks). It also takes you through the exercise of hand-writing a solution to the balanced brackets problem.

*This section mainly just lays the groundwork; it is very light on content.*

> ##### Learning objctives
> 
> * Understand how transformers can be used for classification.
> * Understand how to implement specific kinds of transformer behaviour (e.g. masking of padding tokens) via permanent hooks in TransformerLens.
> * Start thinking about the kinds of algorithmic solutions a transformer is likely to find for problems such as these, given its inductive biases.


One of the many behaviors that a large language model learns is the ability to tell if a sequence of nested parentheses is balanced. For example, `(())()`, `()()`, and `(()())` are balanced sequences, while `)()`, `())()`, and `((()((())))` are not.

In training, text containing balanced parentheses is much more common than text with imbalanced parentheses - particularly, source code scraped from GitHub is mostly valid syntactically. A pretraining objective like "predict the next token" thus incentivizes the model to learn that a close parenthesis is more likely when the sequence is unbalanced, and very unlikely if the sequence is currently balanced.

Some questions we'd like to be able to answer are:

- How robust is this behavior? On what inputs does it fail and why?
- How does this behavior generalize out of distribution? For example, can it handle nesting depths or sequence lengths not seen in training?

If we treat the model as a black box function and only consider the input/output pairs that it produces, then we're very limited in what we can guarantee about the behavior, even if we use a lot of compute to check many inputs. This motivates interpretibility: by digging into the internals, can we obtain insight into these questions? If the model is not robust, can we directly find adversarial examples that cause it to confidently predict the wrong thing? Let's find out!


## Today's Toy Model

Today we'll study a small transformer that is trained to only classify whether a sequence of parentheses is balanced or not. It's small so we can run experiments quickly, but big enough to perform well on the task. The weights and architecture are provided for you.

### Causal vs bidirectional attention

The key difference between this and the GPT-style models you will have implemented already is the attention mechanism. 

GPT uses **causal attention**, where the attention scores get masked wherever the source token comes after the destination token. This means that information can only flow forwards in a model, never backwards (which is how we can train our model in parallel - our model's output is a series of distributions over the next token, where each distribution is only able to use information from the tokens that came before). This model uses **bidirectional attention**, where the attention scores aren't masked based on the relative positions of the source and destination tokens. This means that information can flow in both directions, and the model can use information from the future to predict the past.

### Using transformers for classification

GPT is trained via gradient descent on the cross-entropy loss between its predictions for the next token and the actual next tokens. Models designed to perform classification are trained in a very similar way, but instead of outputting probability distributions over the next token, they output a distribution over class labels. We do this by having an unembedding matrix of size `[d_model, num_classifications]`, and only using a single sequence position (usually the 0th position) to represent our classification probabilities.

Below is a schematic to compare the model architectures and how they're used:

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/gpt-vs-bert-new.png" width="1250">

Note that, just because the outputs at all other sequence positions are discarded, doesn't mean those sequence positions aren't useful. They will almost certainly be the sites of important intermediate calculations. But it does mean that the model will always have to move the information from those positions to the 0th position in order for the information to be used for classification.

### A note on softmax

For each bracket sequence, our (important) output is a vector of two values: `(l0, l1)`, representing the model's logit distribution over (unbalanced, balanced). Our model was trained by minimizing the cross-entropy loss between these logits and the true labels. Interestingly, since logits are translation invariant, the only value we actually care about is the difference between our logits, `l0 - l1`. This is the model's log likelihood ratio of the sequence being unbalanced vs balanced. Later on, we'll be able to use this `logit_diff` to perform logit attribution in our model.

### Masking padding tokens

The image on the top-right is actually slightly incomplete. It doesn't show how our model handles sequences of differing lengths. After all, during training we need to have all sequences be of the same length so we can batch them together in a single tensor. The model manages this via two new tokens: the end token and the padding token.

The end token goes at the end of every bracket sequence, and then we add padding tokens to the end until the sequence is up to some fixed length. For instance, this model was trained on bracket sequences of up to length 40, so if we wanted to classify the bracket string `(())` then we would pad it to the length-42 sequence:

```
[start] + ( + ( + ) + ) + [end] + [pad] + [pad] + ... + [pad]
```

When we calculate the attention scores, we mask them at all (query, key) positions where the key is a padding token. This makes sure that information doesn't flow from padding tokens to other tokens in the sequence (just like how GPT's causal masking makes sure that information doesn't flow from future tokens to past tokens).

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/gpt-vs-bert-3-new.png" width="850">

Note that the attention scores aren't masked when the query is a padding token and the key isn't. In theory, this means that information can be stored in the padding token positions. However, because the padding token key positions are always masked, this information can't flow back into the rest of the sequence, so it never affects the final output. (Also, note that if we masked query positions as well, we'd get numerical errors, since we'd be taking softmax across a row where every element is minus infinity, which is not well-defined!)

<details>
<summary>Aside on how this relates to <b>BERT</b></summary>

This is all very similar to how the bidirectional transformer **BERT** works:

* BERT has the `[CLS]` (classification) token rather than `[start]`; but it works exactly the same.
* BERT has the `[SEP]` (separation) token rather than `[end]`; this has a similar function but also serves a special purpose when it is used in **NSP** (next sentence prediction).

If you're interested in reading more on this, you can check out [this link](https://albertauyeung.github.io/2020/06/19/bert-tokenization.html/).

</details>

We've implemented this type of masking for you, using TransformerLens's **permanent hooks** feature. We will discuss the details of this below (permanent hooks are a recent addition to TransformerLens which we haven't covered yet, and they're useful to understand).




### Other details

Here is a summary of all the relevant architectural details:

* Positional embeddings are sinusoidal (non-learned).
* It has `hidden_size` (aka `d_model`, aka `embed_dim`) of 56.
* It has bidirectional attention, like BERT.
* It has 3 attention layers and 3 MLPs.
* Each attention layer has two heads, and each head has `headsize` (aka `d_head`) of `hidden_size / num_heads = 28`.
* The MLP hidden layer has 56 neurons (i.e. its linear layers are square matrices).
* The input of each attention layer and each MLP is first layernormed, like in GPT.
* There's a LayerNorm on the residual stream after all the attention layers and MLPs have been added into it (this is also like GPT).
* Our embedding matrix `W_E` has five rows: one for each of the tokens `[start]`, `[pad]`, `[end]`, `(`, and `)` (in that order).
* Our unembedding matrix `W_U` has two columns: one for each of the classes `unbalanced` and `balanced` (in that order). 
    * When running our model, we get output of shape `[batch, seq_len, 2]`, and we then take the `[:, 0, :]` slice to get the output for the `[start]` token (i.e. the classification logits).
    * We can then softmax to get our classification probabilities.
* Activation function is `ReLU`.

To refer to attention heads, we'll again use the shorthand `layer.head` where both layer and head are zero-indexed. So `2.1` is the second attention head (index 1) in the third layer (index 2).


### Some useful diagrams

Here is a high-level diagram of your model's architecture:

<details>
<summary>Diagram</summary>

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/bracket-transformer-entire-model.png" height="1600">

</details>

Here is a [link to a diagram of the archicture of a single model layer](https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/diagram-tl.png) (which includes names of activations, as well as a list of useful methods for indexing into the model).

I'd recommend having both these images open in a different tab.



### Defining the model

Here, we define the model according to the description we gave above. First, you'll need to download the file `brackets_model_state_dict.pt` from [this Google Drive link](https://drive.google.com/drive/folders/18gAF9HuiW9NG0MP2Gq8M7VdhXoKKxymT). You should also download the `brackets_data.json` file.


```python
VOCAB = "()"

cfg = HookedTransformerConfig(
    n_ctx=42,
    d_model=56,
    d_head=28,
    n_heads=2,
    d_mlp=56,
    n_layers=3,
    attention_dir="bidirectional", # defaults to "causal"
    act_fn="relu",
    d_vocab=len(VOCAB)+3, # plus 3 because of end and pad and start token
    d_vocab_out=2, # 2 because we're doing binary classification
    use_attn_result=True, 
    device=device,
    use_hook_tokens=True
)

model = HookedTransformer(cfg).eval()

state_dict = t.load(section_dir / "brackets_model_state_dict.pt")
model.load_state_dict(state_dict)
```

## Tokenizer

There are only five tokens in our vocabulary: `[start]`, `[pad]`, `[end]`, `(`, and `)` in that order. See earlier sections for a reminder of what these tokens represent.

You have been given a tokenizer `SimpleTokenizer("()")` which will give you some basic functions. Try running the following to see what they do:


```python
tokenizer = SimpleTokenizer("()")

# Examples of tokenization
# (the second one applies padding, since the sequences are of different lengths)
print(tokenizer.tokenize("()"))
print(tokenizer.tokenize(["()", "()()"]))

# Dictionaries mapping indices to tokens and vice versa
print(tokenizer.i_to_t)
print(tokenizer.t_to_i)

# Examples of decoding (all padding tokens are removed)
print(tokenizer.decode(t.tensor([[0, 3, 4, 2, 1, 1]])))
```

### Implementing our masking

Now that we have the tokenizer, we can use it to write hooks that mask the padding tokens. If you understand how the padding works, then don't worry if you don't follow all the implementational details of this code.

<details>
<summary>Click to see a diagram explaining how this masking works (should help explain the code below)</summary>

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/masking-padding-tokens.png" width="840">

</details>


```python
def add_perma_hooks_to_mask_pad_tokens(model: HookedTransformer, pad_token: int) -> HookedTransformer:

    # Hook which operates on the tokens, and stores a mask where tokens equal [pad]
    def cache_padding_tokens_mask(tokens: Float[Tensor, "batch seq"], hook: HookPoint) -> None:
        hook.ctx["padding_tokens_mask"] = einops.rearrange(tokens == pad_token, "b sK -> b 1 1 sK")

    # Apply masking, by referencing the mask stored in the `hook_tokens` hook context
    def apply_padding_tokens_mask(
        attn_scores: Float[Tensor, "batch head seq_Q seq_K"],
        hook: HookPoint,
    ) -> None:
        attn_scores.masked_fill_(model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"], -1e5)
        if hook.layer() == model.cfg.n_layers - 1:
            del model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"]

    # Add these hooks as permanent hooks (i.e. they aren't removed after functions like run_with_hooks)
    for name, hook in model.hook_dict.items():
        if name == "hook_tokens":
            hook.add_perma_hook(cache_padding_tokens_mask)
        elif name.endswith("attn_scores"):
            hook.add_perma_hook(apply_padding_tokens_mask)

    return model


model.reset_hooks(including_permanent=True)
model = add_perma_hooks_to_mask_pad_tokens(model, tokenizer.PAD_TOKEN)
```

## Dataset

Each training example consists of `[start]`, up to 40 parens, `[end]`, and then as many `[pad]` as necessary.

In the dataset we're using, half the sequences are balanced, and half are unbalanced. Having an equal distribution is on purpose to make it easier for the model.

Remember to download the `brackets_data.json` file from [this Google Drive link](https://drive.google.com/drive/folders/18gAF9HuiW9NG0MP2Gq8M7VdhXoKKxymT) if you haven't already.


```python
N_SAMPLES = 5000
with open(section_dir / "brackets_data.json") as f:
    data_tuples: List[Tuple[str, bool]] = json.load(f)
    print(f"loaded {len(data_tuples)} examples")
assert isinstance(data_tuples, list)
data_tuples = data_tuples[:N_SAMPLES]
data = BracketsDataset(data_tuples).to(device)
data_mini = BracketsDataset(data_tuples[:100]).to(device)
```

You are encouraged to look at the code for `BracketsDataset` (scroll up to the setup code at the top - but make sure to not look to closely at the solutions!) to see what methods and properties the `data` object has.




#### Data visualisation

As is good practice, let's examine the dataset and plot the distribution of sequence lengths (e.g. as a histogram). What do you notice?


```python
hist(
    [len(x) for x, _ in data_tuples], 
    nbins=data.seq_length,
    title="Sequence lengths of brackets in dataset",
    labels={"x": "Seq len"}
)
```

<details>
<summary>Features of dataset</summary>

The most striking feature is that all bracket strings have even length. We constructed our dataset this way because if we had odd-length strings, the model would presumably have learned the heuristic "if the string is odd-length, it's unbalanced". This isn't hard to learn, and we want to focus on the more interesting question of how the transformer is learning the structure of bracket strings, rather than just their length.

**Bonus exercise (optional) - can you describe an algorithm involving a single attention head which the model could use to distinguish between even and odd-length bracket strings?**

</details>


Now that we have all the pieces in place, we can try running our model on the data and generating some predictions.


```python
# Define and tokenize examples
examples = ["()()", "(())", "))((", "()", "((()()()()))", "(()()()(()(())()", "()(()(((())())()))"]
labels = [True, True, False, True, True, False, True]
toks = tokenizer.tokenize(examples)

# Get output logits for the 0th sequence position (i.e. the [start] token)
logits = model(toks)[:, 0]

# Get the probabilities via softmax, then get the balanced probability (which is the second element)
prob_balanced = logits.softmax(-1)[:, 1]

# Display output
print("Model confidence:\n" + "\n".join([f"{ex:18} : {prob:<8.4%} : label={int(label)}" for ex, prob, label in zip(examples, prob_balanced, labels)]))
```

We can also run our model on the whole dataset, and see how many brackets are correctly classified.


```python
def run_model_on_data(model: HookedTransformer, data: BracketsDataset, batch_size: int = 200) -> Float[Tensor, "batch 2"]:
    '''Return probability that each example is balanced'''
    all_logits = []
    for i in tqdm(range(0, len(data.strs), batch_size)):
        toks = data.toks[i : i + batch_size]
        logits = model(toks)[:, 0]
        all_logits.append(logits)
    all_logits = t.cat(all_logits)
    assert all_logits.shape == (len(data), 2)
    return all_logits


test_set = data
n_correct = (run_model_on_data(model, test_set).argmax(-1).bool() == test_set.isbal).sum()
print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")
```

## Algorithmic Solutions


### Exercise - handwritten solution (for loop)

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠🟠⚪⚪

You shouldn't spend more than ~10 minutes on this exercise.

This exercise and the next one should both be relatively easy (especially if you've already solved this problem on LeetCode before!), and they're very important for the rest of the exercises.
```

A nice property of using such a simple problem is we can write a correct solution by hand. Take a minute to implement this using a for loop and if statements.


```python
def is_balanced_forloop(parens: str) -> bool:
    '''
    Return True if the parens are balanced.

    Parens is just the ( and ) characters, no begin or end tokens.
    '''
    pass


for (parens, expected) in zip(examples, labels):
    actual = is_balanced_forloop(parens)
    assert expected == actual, f"{parens}: expected {expected} got {actual}"
print("is_balanced_forloop ok!")
```

<details>
<summary>Solution</summary>


```python
def is_balanced_forloop(parens: str) -> bool:
    '''
    Return True if the parens are balanced.

    Parens is just the ( and ) characters, no begin or end tokens.
    '''
    # SOLUTION
    cumsum = 0
    for paren in parens:
        cumsum += 1 if paren == "(" else -1
        if cumsum < 0:
            return False
    
    return cumsum == 0
```
</details>


### Exercise -  handwritten solution (vectorized)

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠🟠⚪⚪

You shouldn't spend more than ~10 minutes on this exercise.
```

A transformer has an **inductive bias** towards vectorized operations, because at each sequence position the same weights "execute", just on different data. So if we want to "think like a transformer", we want to get away from procedural for/if statements and think about what sorts of solutions can be represented in a small number of transformer weights.

Being able to represent a solutions in matrix weights is necessary, but not sufficient to show that a transformer could learn that solution through running SGD on some input data. It could be the case that some simple solution exists, but a different solution is an attractor when you start from random initialization and use current optimizer algorithms.


```python
def is_balanced_vectorized(tokens: Float[Tensor, "seq_len"]) -> bool:
    '''
    Return True if the parens are balanced.

    tokens is a vector which has start/pad/end indices (0/1/2) as well as left/right brackets (3/4)
    '''
    pass


for (tokens, expected) in zip(tokenizer.tokenize(examples), labels):
    actual = is_balanced_vectorized(tokens)
    assert expected == actual, f"{tokens}: expected {expected} got {actual}"
print("is_balanced_vectorized ok!")
```

<details>
<summary>Hint</summary>

```python
One solution is to map begin, pad, and end tokens to zero, map open paren to 1 and close paren to -1. Then take the cumulative sum, and check the two conditions which are necessary and sufficient for the bracket string to be balanced.
```

</details>

<details>
<summary>Solution</summary>


```python
def is_balanced_vectorized(tokens: Float[Tensor, "seq_len"]) -> bool:
    '''
    Return True if the parens are balanced.

    tokens is a vector which has start/pad/end indices (0/1/2) as well as left/right brackets (3/4)
    '''
    # SOLUTION
    # Convert start/end/padding tokens to zero, and left/right brackets to +1/-1
    table = t.tensor([0, 0, 0, 1, -1])
    change = table[tokens]
    # Get altitude by taking cumulative sum
    altitude = t.cumsum(change, -1)
    # Check that the total elevation is zero and that there are no negative altitudes
    no_total_elevation_failure = altitude[-1] == 0
    no_negative_failure = altitude.min() >= 0

    return no_total_elevation_failure & no_negative_failure
```
</details>


## The Model's Solution

It turns out that the model solves the problem like this:

At each position `i`, the model looks at the slice starting at the current position and going to the end: `seq[i:]`. It then computes (count of closed parens minus count of open parens) for that slice to generate the output at that position.

We'll refer to this output as the "elevation" at `i`, or equivalently the elevation for each suffix `seq[i:]`.

The sequence is imbalanced if one or both of the following is true:

- `elevation[0]` is non-zero
- `any(elevation < 0)`

For English readers, it's natural to process the sequence from left to right and think about prefix slices `seq[:i]` instead of suffixes, but the model is bidirectional and has no idea what English is. This model happened to learn the equally valid solution of going right-to-left.

We'll spend today inspecting different parts of the network to try to get a first-pass understanding of how various layers implement this algorithm. However, we'll also see that neural networks are complicated, even those trained for simple tasks, and we'll only be able to explore a minority of the pieces of the puzzle.




""", unsafe_allow_html=True)


def section_2():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#moving-back-to-the-residual-stream'>Moving back to the residual stream</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#stage-1:-translating-through-softmax'>Stage 1: Translating through softmax</a></li>
        <li><a class='contents-el' href='#stage-10:-translating-through-linear'>Stage 2: Translating through linear</a></li>
        <li><a class='contents-el' href='#exercise-get-the-post-final-ln-dir'><b>Exercise</b> - get the <code>post_final_ln_dir</code></a></li>
        <li><a class='contents-el' href='#stage-125:-translating-through-layernorm'>Stage 3: Translating through LayerNorm</a></li>
        <li><a class='contents-el' href='#exercise-get-the-pre-final-ln-dir'><b>Exercise</b> - get the <code>pre_final_ln_dir</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#writing-the-residual-stream-as-a-sum-of-terms'>Writing the residual stream as a sum of terms</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-breaking-down-the-residual-stream-by-component'><b>Exercise</b> - breaking down the residual stream by component</a></li>
        <li><a class='contents-el' href='#which-components-matter'>Which components matter?</a></li>
        <li><a class='contents-el' href='#exercise-compute-output-in-unbalanced-direction-for-each-component'><b>Exercise</b> - compute output in unbalanced direction for each component</a></li>
        <li><a class='contents-el' href='#head-influence-by-type-of-failures'>Head influence by type of failures</a></li>
        <li><a class='contents-el' href='#exercise-classify-bracket-strings-by-failure-type'><b>Exercise</b> - classify bracket strings by failure type</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 2️⃣ Moving backwards


Here, you'll perform logit attribution, and learn how to work backwards through particular paths of a model to figure out which components matter most for the final classification probabilities.

This is the first time you'll have to deal with LayerNorm in your models.

*This section should be familiar if you've done logit attribution for induction heads (although these exercises are slightly more challenging from a coding perspective). The LayerNorm-based exercises are a bit fiddly!*

> ##### Learning objctives
>
> * Understand how to perform logit attribution.
> * Understand how to work backwards through a model to identify which components matter most for the final classification probabilities.
> * Understand how LayerNorm works, and look at some ways to deal with it in your models.




Suppose we run the model on some sequence and it outputs the classification probabilities `[0.99, 0.01]`, i.e. highly confident classification as "unbalanced".

We'd like to know _why_ the model had this output, and we'll do so by moving backwards through the network, and figuring out the correspondence between facts about earlier activations and facts about the final output. We want to build a chain of connections through different places in the computational graph of the model, repeatedly reducing our questions about later values to questions about earlier values.

Let's start with an easy one. Notice that the final classification probabilities only depend on the difference between the class logits, as softmax is invariant to constant additions. So rather than asking, "What led to this probability on balanced?", we can equivalently ask, "What led to this difference in logits?". Let's move another step backward. Since the logits are each a linear function of the output of the final LayerNorm, their difference will be some linear function as well. In other words, we can find a vector in the space of LayerNorm outputs such that the logit difference will be the dot product of the LayerNorm's output with that vector.

We now want some way to tell which parts of the model are doing something meaningful. We will do this by identifying a single direction in the embedding space of the start token that we claim to be the "unbalanced direction": the direction that most indicates that the input string is unbalanced. It is important to note that it might be that other directions are important as well (in particular because of layer norm), but for a first approximation this works well.

We'll do this by starting from the model outputs and working backwards, finding the unbalanced direction at each stage.


## Moving back to the residual stream

The final part of the model is the classification head, which has three stages - the final layernorm, the unembedding, and softmax, at the end of which we get our probabilities.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/bracket-transformer-first-attr-0.png" width="500">

Note - for simplicity, we'll ignore the batch dimension in the following discussion.

Some notes on the shapes of the objects in the diagram:

* `x_2` is the vector in the residual stream after layer 2's attention heads and MLPs. It has shape `(seq_len, d_model)`.
* `final_ln_output` has shape `(seq_len, d_model)`.
* `W_U` has shape `(d_model, 2)`, and so `logits` has shape `(seq_len, 2)`.
* We get `P(unbalanced)` by taking the 0th element of the softmaxed logits, for sequence position 0.

### Stage 1: Translating through softmax

Let's get `P(unbalanced)` as a function of the logits. Luckily, this is easy. Since we're doing the softmax over two elements, it simplifies to the sigmoid of the difference of the two logits:

$$
\text{softmax}(\begin{bmatrix} \text{logit}_0 \\ \text{logit}_1 \end{bmatrix})_0 = \frac{e^{\text{logit}_0}}{e^{\text{logit}_0} + e^{\text{logit}_1}} = \frac{1}{1 + e^{\text{logit}_1 - \text{logit}_0}} = \text{sigmoid}(\text{logit}_0 - \text{logit}_1)
$$

Since sigmoid is monotonic, a large value of $\hat{y}_0$ follows from logits with a large $\text{logit}_0 - \text{logit}_1$. From now on, we'll only ask "What leads to a large difference in logits?"

### Stage 2: Translating through linear

The next step we encounter is the decoder: `logits = final_LN_output @ W_U`, where

* `W_U` has shape `(d_model, 2)`
* `final_LN_output` has shape `(seq_len, d_model)`

We can now put the difference in logits as a function of $W$ and $x_{\text{linear}}$ like this:

```
logit_diff = (final_LN_output @ W_U)[0, 0] - (final_LN_output @ W_U)[0, 1]

           = final_LN_output[0, :] @ (W_U[:, 0] - W_U[:, 1])
```

(recall that the `(i, j)`th element of matrix `AB` is `A[i, :] @ B[:, j]`)

So a high difference in the logits follows from a high dot product of the output of the LayerNorm with the corresponding unembedding vector. We'll call this the `post_final_ln_dir`, i.e. the **unbalanced direction** for values in the residual stream *after* the final layernorm.

### Summary

We can use the logit diff as a measure of how strongly our model is classifying a bracket string as unbalanced (higher logit diff = more certain that the string is unbalanced). 

We can approximate logit diff as a linear function of `pre_final_ln_dir` (because the unembedding is linear, and the layernorm is approximately linear). This means we can approximate logit diff as the **dot product** of `post_final_ln_dir` with the residual stream value before the final layernorm. If we could find this `post_final_ln_dir`, then we could start to answer other questions like which components' output had the highest dot product with this value.

The diagram below shows how we can step back through the model to find our **unbalanced direction** `pre_final_ln_dir`. Notation: $x_2$ refers to the residual stream value after layer 2's attention heads and MLPs (i.e. just before the last layernorm), and $L_{final}$ is the linear approximation of the final layernorm.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/brackets-untitled.png" width="1100">


### Exercise - get the `post_final_ln_dir`

```c
Difficulty: 🟠⚪⚪⚪⚪
Importance: 🟠🟠🟠⚪⚪

You shouldn't spend more than ~5 minutes on this exercise.
```

In the function below, you should compute this vector (this should just be a one-line function).


```python
def get_post_final_ln_dir(model: HookedTransformer) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction in which final_ln_output[0, :] should point to maximize P(unbalanced)
    '''
    pass


tests.test_get_post_final_ln_dir(get_post_final_ln_dir, model)
```

<details>
<summary>Solution</summary>


```python
def get_post_final_ln_dir(model: HookedTransformer) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction in which final_ln_output[0, :] should point to maximize P(unbalanced)
    '''
    # SOLUTION
    return model.W_U[:, 0] - model.W_U[:, 1]
```
</details>


### Stage 3: Translating through LayerNorm

We want to find the unbalanced direction before the final layer norm, since this is where we can write the residual stream as a sum of terms. LayerNorm messes with this sort of direction analysis, since it is nonlinear. For today, however, we will approximate it with a linear fit. This is good enough to allow for interesting analysis (see for yourself that the $R^2$ values are very high for the fit)!

With a linear approximation to LayerNorm, which I'll use the matrix `L_final` for, we can translate "What is the dot product of the output of the LayerNorm with the unbalanced-vector?" to a question about the input to the LN. We simply write:

```python
final_ln_output[0, :] = final_ln(x_linear[0, :])

                      = L_final @ x_linear[0, :]
```

<details>
<summary>An aside on layernorm</summary>

Layernorm isn't actually linear. It's a combination of a nonlinear function (subtracting mean and dividing by std dev) with a linear one (a learned affine transformation).

However, in this case it turns out to be a decent approximation to use a linear fit. The reason we've included layernorm in these exercises is to give you an idea of how nonlinear functions can complicate our analysis, and some simple hacky ways that we can deal with them. 

When applying this kind of analysis to LLMs, it's sometimes harder to abstract away layernorm as just a linear transformation. For instance, many large transformers use layernorm to "clear" parts of their residual stream, e.g. they learn a feature 100x as large as everything else and use it with layer norm to clear the residual stream of everything but that element. Clearly, this kind of behaviour is not well-modelled by a linear fit.

</details>

Now, we can ask 'What leads to the _input_ to the LayerNorm having a high dot-product with this new vector?'

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/bracket-transformer-first-attr.png" width="450">


### Exercise - get the `pre_final_ln_dir`

```c
Difficulty: 🟠🟠🟠🟠⚪
Importance: 🟠🟠🟠⚪⚪

You shouldn't spend more than 25-35 minutes on the following set of exercises.

There are several moving parts in the following exercises. The most important thing is to conceptually understand the idea of moving backwards through the model, and finding an "unbalanced direction" in the residual stream.
```

Ideally, we would calculate `pre_final_ln_dir` directly from the model's weights, like we did for `post_final_ln_dir`. Unfortunately, it's not so easy in this case, because in order to get our linear approximation `L_final`, we need to fit a linear regression with actual data that gets passed through the model.

These exercises are split into three parts:

1. Implement `get_activations` to extract activations from the model (in particular, we'll need the inputs and outputs to the final layernorm, but this function will also be useful for later exercises).
2. Implement `get_ln_fit` to fit a linear regression to the inputs and outputs of one of your model's layernorms.
3. Finally, estimate `L_final` using a batch of 5000 input sequences, and use it to calculate `pre_final_ln_dir`.


#### 1. Getting activations

First, we'll deal with how to get activations from our model. For this purpose, we've given you the helper function `get_activations` below. This uses `model.run_with_cache` to get activations for the data given by `toks`. It returns this in the form of a single tensor if `names` is a string, or a cache if `names` is a list.

Note the use of the `names_filter` argument in `model.run_with_cache`. This is a function mapping activation names to booleans, and the cache we get only contains activations where this function returns True.


```python
def get_activations(
    model: HookedTransformer, 
    toks: Int[Tensor, "batch seq"], 
    names: Union[str, List[str]]
) -> Union[t.Tensor, ActivationCache]:
    '''
    Uses hooks to return activations from the model.

    If names is a string, returns the activations for that hook name.
    If names is a list of strings, returns a dictionary mapping hook names to tensors of activations.
    '''
    names_list = [names] if isinstance(names, str) else names
    _, cache = model.run_with_cache(
        toks,
        return_type=None,
        names_filter=lambda name: name in names_list,
    )

    return cache[names] if isinstance(names, str) else cache

```

#### 2. Fitting a linear regression

Now, use these functions and the [sklearn LinearRegression class](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) to find a linear fit to the inputs and outputs of your model's layernorms.

Note - we've provided you with another helper function `LN_hook_names`. This takes one of the layernorms of your model, and outputs the full names of the hooks immediately before and after that layer. Run this code to see how it works:


```python
def LN_hook_names(layernorm: LayerNorm) -> Tuple[str, str]:
    '''
    Returns the names of the hooks immediately before and after a given layernorm.
    e.g. LN_hook_names(model.ln_final) returns ["blocks.2.hook_resid_post", "ln_final.hook_normalized"]
    '''
    if layernorm.name == "ln_final":
        input_hook_name = utils.get_act_name("resid_post", 2)
        output_hook_name = "ln_final.hook_normalized"
    else:
        layer, ln = layernorm.name.split(".")[1:]
        input_hook_name = utils.get_act_name("resid_pre" if ln=="ln1" else "resid_mid", layer)
        output_hook_name = utils.get_act_name('normalized', layer, ln)
    
    return input_hook_name, output_hook_name


pre_final_ln_name, post_final_ln_name = LN_hook_names(model.ln_final)
print(pre_final_ln_name, post_final_ln_name)
```

* The `get_ln_fit` function takes `seq_pos` as an input. If this is an integer, then we are fitting only for that sequence position. If `seq_pos = None`, then we are fitting for all sequence positions (we aggregate the sequence and batch dimensions before performing our regression).
    * The reason for including this parameter is that sometimes we care about how the layernorm operates on a particular sequence position (e.g. for the final layernorm, we only care about the 0th sequence position), but later on we'll also consider the behaviour of layernorm across all sequence positions.
* You should include a fit coefficient in your linear regression (this is the default for `LinearRegression`).


```python
def get_ln_fit(
    model: HookedTransformer, data: BracketsDataset, layernorm: LayerNorm, seq_pos: Optional[int] = None
) -> Tuple[LinearRegression, float]:
    '''
    if seq_pos is None, find best fit aggregated over all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and the r^2 of the fit
    '''
    input_hook_name, output_hook_name = LN_hook_names(layernorm)
    pass


tests.test_get_ln_fit(get_ln_fit, model, data_mini)

(final_ln_fit, r2) = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)
print(f"r^2 for LN_final, at sequence position 0: {r2:.4f}")

(final_ln_fit, r2) = get_ln_fit(model, data, layernorm=model.blocks[1].ln1, seq_pos=None)
print(f"r^2 for LN1, layer 1, over all sequence positions: {r2:.4f}")
```

<details>
<summary>Help - I'm not sure how to fit the linear regression.</summary>

If `inputs` and `outputs` are both tensors of shape `(samples, d_model)`, then `LinearRegression().fit(inputs, outputs)` returns the fit object which should be the first output of your function.

You can get the Rsquared with the `.score` method of the fit object.
</details>

<details>
<summary>Help - I'm not sure how to interpret the seq_pos argument.</summary>

If `seq_pos` is an integer, you should take the vectors corresponding to just that sequence position. In other words, you should take the `[:, seq_pos, :]` slice of your `[batch, seq_pos, d_model]`-size tensors.

If `seq_pos = None`, you should rearrange your tensors into `(batch seq_pos) d_model`, because you want to run the regression on all sequence positions at once.
</details>

<details>
<summary>Solution</summary>


```python
def get_ln_fit(
    model: HookedTransformer, data: BracketsDataset, layernorm: LayerNorm, seq_pos: Optional[int] = None
) -> Tuple[LinearRegression, float]:
    '''
    if seq_pos is None, find best fit aggregated over all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and the r^2 of the fit
    '''
    input_hook_name, output_hook_name = LN_hook_names(layernorm)
    # SOLUTION

    activations_dict = get_activations(model, data.toks, [input_hook_name, output_hook_name])
    inputs = utils.to_numpy(activations_dict[input_hook_name])
    outputs = utils.to_numpy(activations_dict[output_hook_name])

    if seq_pos is None:
        inputs = einops.rearrange(inputs, "batch seq d_model -> (batch seq) d_model")
        outputs = einops.rearrange(outputs, "batch seq d_model -> (batch seq) d_model")
    else:
        inputs = inputs[:, seq_pos, :]
        outputs = outputs[:, seq_pos, :]
    
    final_ln_fit = LinearRegression().fit(inputs, outputs)

    r2 = final_ln_fit.score(inputs, outputs)

    return (final_ln_fit, r2)
```
</details>


#### 3. Calculating `pre_final_ln_dir`

Armed with our linear fit, we can now identify the direction in the residual stream before the final layer norm that most points in the direction of unbalanced evidence.


```python
def get_pre_final_ln_dir(model: HookedTransformer, data: BracketsDataset) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction in residual stream (pre ln_final, at sequence position 0) which
    most points in the direction of making an unbalanced classification.
    '''
    pass


tests.test_get_pre_final_ln_dir(get_pre_final_ln_dir, model, data_mini)
```

<details>
<summary>Solution</summary>


```python
def get_pre_final_ln_dir(model: HookedTransformer, data: BracketsDataset) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction in residual stream (pre ln_final, at sequence position 0) which
    most points in the direction of making an unbalanced classification.
    '''
    # SOLUTION
    post_final_ln_dir = get_post_final_ln_dir(model)

    final_ln_fit = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)[0]
    final_ln_coefs = t.from_numpy(final_ln_fit.coef_).to(device)

    return final_ln_coefs.T @ post_final_ln_dir
```
</details>


## Writing the residual stream as a sum of terms

As we've seen in previous exercises, it's much more natural to think about the residual stream as a sum of terms, each one representing a different path through the model. Here, we have ten components which write to the residual stream: the direct path (i.e. the embeddings), and two attention heads and one MLP on each of the three layers. We can write the residual stream as a sum of these terms.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/attribution.png" width="900">

Once we do this, we can narrow in on the components who are making direct contributions to the classification, i.e. which are writing vectors to the residual stream which have a high dot product with the `pre_final_ln_dir` for unbalanced brackets relative to balanced brackets.

In order to answer this question, we need the following tools:
- A way to break down the input to the LN by component.
- A tool to identify a direction in the embedding space that causes the network to output 'unbalanced' (we already have this)


### Exercise - breaking down the residual stream by component

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠⚪⚪⚪

You shouldn't spend more than 15-20 minutes on this exercise.

It isn't very conceptually important; the hardest part is getting all the right activation names & rearranging / stacking the tensors in the correct way.
```

Use your `get_activations` function to create a tensor of shape `[num_components, dataset_size, seq_pos]`, where the number of components = 10.

This is a termwise representation of the input to the final layer norm from each component (recall that we can see each head as writing something to the residual stream, which is eventually fed into the final layer norm). The order of the components in your function's output should be the same as shown in the diagram above (i.e. in chronological order of how they're added to the residual stream).

(The only term missing from the sum of these is the `W_O`-bias from each of the attention layers).

<details>
<summary>Aside on why this bias term is missing.</summary>

Most other libraries store `W_O` as a 2D tensor of shape `[num_heads * d_head, d_model]`. In this case, the sum over heads is implicit in our calculations when we apply the matrix `W_O`. We then add `b_O`, which is a vector of length `d_model`.

TransformerLens stores `W_O` as a 3D tensor of shape `[num_heads, d_head, d_model]` so that we can easily compute the output of each head separately. Since TransformerLens is designed to be compatible with other libraries, we need the bias to also be shape `d_model`, which means we have to sum over heads before we add the bias term. So none of the output terms for our individual heads will include the bias term. 

In practice this doesn't matter here, since the bias term is the same for balanced and unbalanced brackets. When doing attribution, for each of our components, we only care about the component in the unbalanced direction of the vector they write to the residual stream **for balanced vs unbalanced sequences** - the bias is the same on all inputs.
</details>




```python
def get_out_by_components(model: HookedTransformer, data: BracketsDataset) -> Float[Tensor, "component batch seq_pos emb"]:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2].
    The embeddings are the sum of token and positional embeddings.
    '''
    pass


tests.test_get_out_by_components(get_out_by_components, model, data_mini)
```

Now, you can test your function by confirming that input to the final layer norm is the sum of the output of each component and the output projection biases.


```python
biases = model.b_O.sum(0)
out_by_components = get_out_by_components(model, data)
summed_terms = out_by_components.sum(dim=0) + biases

final_ln_input_name, final_ln_output_name = LN_hook_names(model.ln_final)
final_ln_input = get_activations(model, data.toks, final_ln_input_name)

t.testing.assert_close(summed_terms, final_ln_input)
print("Tests passed!")
```

<details>
<summary>Hint</summary>

Start by getting all the activation names in a list. You will need `utils.get_act_name("result", layer)` to get the activation names for the attention heads' output, and `utils.get_act_name("mlp_out", layer)` to get the activation names for the MLPs' output.

Once you've done this, and run the `get_activations` function, it's just a matter of doing some reshaping and stacking. Your embedding and mlp activations will have shape `(batch, seq_pos, d_model)`, while your attention activations will have shape `(batch, seq_pos, head_idx, d_model)`.
</details>

<details>
<summary>Solution</summary>


```python
def get_out_by_components(model: HookedTransformer, data: BracketsDataset) -> Float[Tensor, "component batch seq_pos emb"]:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2].
    The embeddings are the sum of token and positional embeddings.
    '''
    # SOLUTION
    embedding_hook_names = ["hook_embed", "hook_pos_embed"]
    head_hook_names = [utils.get_act_name("result", layer) for layer in range(model.cfg.n_layers)]
    mlp_hook_names = [utils.get_act_name("mlp_out", layer) for layer in range(model.cfg.n_layers)]

    all_hook_names = embedding_hook_names + head_hook_names + mlp_hook_names
    activations = get_activations(model, data.toks, all_hook_names)

    out = (activations["hook_embed"] + activations["hook_pos_embed"]).unsqueeze(0)

    for head_hook_name, mlp_hook_name in zip(head_hook_names, mlp_hook_names):
        out = t.concat([
            out, 
            einops.rearrange(activations[head_hook_name], "batch seq heads emb -> heads batch seq emb"),
            activations[mlp_hook_name].unsqueeze(0)
        ])

    return out
```
</details>

### Which components matter?

To figure out which components are directly important for the the model's output being "unbalanced", we can see which components tend to output a vector to the position-0 residual stream with higher dot product in the unbalanced direction for actually unbalanced inputs.

The idea is that, if a component is important for correctly classifying unbalanced inputs, then its vector output when fed unbalanced bracket strings will have a higher dot product in the unbalanced direction than when it is fed balanced bracket strings.

In this section, we'll plot histograms of the dot product for each component. This will allow us to observe which components are significant.

For example, suppose that one of our components produced bimodal output like this:

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/exampleplot.png" width="650">

This would be **strong evidence that this component is important for the model's output being unbalanced**, since it's pushing the unbalanced bracket inputs further in the unbalanced direction (i.e. the direction which ends up contributing to the inputs being classified as unbalanced) relative to the balanced inputs.



### Exercise - compute output in unbalanced direction for each component

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠🟠🟠⚪

You shouldn't spend more than 10-15 minutes on this exercise.

It's very important to conceptually understand what object you are computing here. The actual computation is just a few lines of code involving indexing and einsums.
```

In the code block below, you should compute a `(10, batch)`-size tensor called `out_by_component_in_unbalanced_dir`. The `[i, j]`th element of this tensor should be the dot product of the `i`th component's output with the unbalanced direction, for the `j`th sequence in your dataset. 

You should normalize it by **subtracting the mean** of the dot product of this component's output with the unbalanced direction on balanced samples - this will make sure the histogram corresponding to the balanced samples is centered at 0 (like in the figure above), which will make it easier to interpret. Remember, it's only the **difference between the dot product on unbalanced and balanced samples** that we care about (since adding a constant to both logits doesn't change the model's probabilistic output).

We've given you a `hists_per_comp` function which will plot these histograms for you - all you need to do is calculate the `out_by_component_in_unbalanced_dir` object and supply it to that function.


```python
# YOUR CODE HERE - define the object `out_by_component_in_unbalanced_dir`
tests.test_out_by_component_in_unbalanced_dir(out_by_component_in_unbalanced_dir, model, data)

plotly_utils.hists_per_comp(
    out_by_component_in_unbalanced_dir, 
    data, xaxis_range=[-10, 20]
)
```

<details>
<summary>Hint</summary>

Start by defining these two objects:

* The output by components at sequence position zero, i.e. a tensor of shape `(component, batch, d_model)`
* The `pre_final_ln_dir` vector, which has length `d_model`

Then create magnitudes by calculating an appropriate dot product. 

Don't forget to subtract the mean for each component across all the balanced samples (you can use the boolean `data.isbal` as your index).
</details>

<details>
<summary>Solution</summary>


```python
# Get output by components, at sequence position 0 (which is used for classification)
out_by_components_seq0: Float[Tensor, "comp batch d_model"] = get_out_by_components(model, data)[:, :, 0, :]
# Get the unbalanced direction for tensors being fed into the final layernorm
pre_final_ln_dir: Float[Tensor, "d_model"] = get_pre_final_ln_dir(model, data)
# Get the size of the contributions for each component
out_by_component_in_unbalanced_dir = einops.einsum(
    out_by_components_seq0,
    pre_final_ln_dir,
    "comp batch d_model, d_model -> comp batch"
)
# Subtract the mean
out_by_component_in_unbalanced_dir -= out_by_component_in_unbalanced_dir[:, data.isbal].mean(dim=1).unsqueeze(1)

```
</details>


<details>
<summary>Which heads do you think are the most important, and can you guess why that might be?</summary>

The heads in layer 2 (i.e. `2.0` and `2.1`) seem to be the most important, because the unbalanced brackets are being pushed much further to the right than the balanced brackets. 

We might guess that some kind of composition is going on here. The outputs of layer 0 heads can't be involved in composition because they in effect work like a one-layer transformer. But the later layers can participate in composition, because their inputs come from not just the embeddings, but also the outputs of the previous layer. This means they can perform more complex computations.
</details>




### Head influence by type of failures

Those histograms showed us which heads were important, but it doesn't tell us what these heads are doing, however. In order to get some indication of that, let's focus in on the two heads in layer 2 and see how much they write in our chosen direction on different types of inputs. In particular, we can classify inputs by if they pass the 'overall elevation' and 'nowhere negative' tests.

We'll also ignore sentences that start with a close paren, as the behaviour is somewhat different on them (they can be classified as unbalanced immediately, so they don't require more complicated logic).




### Exercise - classify bracket strings by failure type

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠⚪⚪⚪

You shouldn't spend more than 15-20 minutes on this exercise.

These exercises should be pretty straightforward; you'll be able to use much of your code from previous exercises. They are also quite fiddly, so you should look at the solutions if you are stuck.
```

Define, so that the plotting works, the following objects:

* **`negative_failure`**
  * This is an `(N_SAMPLES,)` boolean vector that is true for sequences whose elevation (when reading from right to left) ever dips negative, i.e. there's an open paren that is never closed.                                                         |
* **`total_elevation_failure`**
  * This is an `(N_SAMPLES,)` boolean vector that is true for sequences whose total elevation is not exactly 0. In other words, for sentences with uneven numbers of open and close parens.                                                            |
* **`h20_in_unbalanced_dir`**
  * This is an `(N_SAMPLES,)` float vector equal to head 2.0's contribution to the position-0 residual stream in the unbalanced direction, normalized by subtracting its average unbalancedness contribution to this stream over _balanced sequences_. |
* **`h21_in_unbalanced_dir`**
  * Same as above but head 2.1 |

For the first two of these, you will find it helpful to refer back to your `is_balanced_vectorized` code (although remember you're reading **right to left** here - this _will_ change your results!). 

You can get the last two of these by directly indexing from your `out_by_component_in_unbalanced_dir` tensor.


```python
def is_balanced_vectorized_return_both(
        toks: Float[Tensor, "batch seq"]
) -> Tuple[Bool[Tensor, "batch"], Bool[Tensor, "batch"]]:
    pass


total_elevation_failure, negative_failure = is_balanced_vectorized_return_both(data.toks)

h20_in_unbalanced_dir = out_by_component_in_unbalanced_dir[7]
h21_in_unbalanced_dir = out_by_component_in_unbalanced_dir[8]

tests.test_total_elevation_and_negative_failures(data, total_elevation_failure, negative_failure)
```

<details>
<summary>Solution</summary>


```python
def is_balanced_vectorized_return_both(
        toks: Float[Tensor, "batch seq"]
) -> Tuple[Bool[Tensor, "batch"], Bool[Tensor, "batch"]]:
    # SOLUTION
    table = t.tensor([0, 0, 0, 1, -1]).to(device)
    change = table[toks.to(device)].flip(-1)
    altitude = t.cumsum(change, -1)
    total_elevation_failure = altitude[:, -1] != 0
    negative_failure = altitude.max(-1).values > 0
    return total_elevation_failure, negative_failure
```
</details>


Once you've passed the tests, you can run the code below to generate your plot.


```python
failure_types_dict = {
    "both failures": negative_failure & total_elevation_failure,
    "just neg failure": negative_failure & ~total_elevation_failure,
    "just total elevation failure": ~negative_failure & total_elevation_failure,
    "balanced": ~negative_failure & ~total_elevation_failure
}

plotly_utils.plot_failure_types_scatter(
    h20_in_unbalanced_dir,
    h21_in_unbalanced_dir,
    failure_types_dict,
    data
)
```

Look at the graph and think about what the roles of the different heads are!

<details>
<summary>Read after thinking for yourself</summary>

The primary thing to take away is that 2.0 is responsible for checking the overall counts of open and close parentheses, and that 2.1 is responsible for making sure that the elevation never goes negative.

Aside: the actual story is a bit more complicated than that. Both heads will often pick up on failures that are not their responsibility, and output in the 'unbalanced' direction. This is in fact incentived by log-loss: the loss is slightly lower if both heads unanimously output 'unbalanced' on unbalanced sequences rather than if only the head 'responsible' for it does so. The heads in layer one do some logic that helps with this, although we'll not cover it today.

One way to think of it is that the heads specialized on being very reliable on their class of failures, and then sometimes will sucessfully pick up on the other type.
</details>

In most of the rest of these exercises, we'll focus on the overall elevation circuit as implemented by head 2.0. As an additional way to get intuition about what head 2.0 is doing, let's graph its output against the overall proportion of the sequence that is an open-paren.


```python
plotly_utils.plot_contribution_vs_open_proportion(
    h20_in_unbalanced_dir, 
    "Head 2.0 contribution vs proportion of open brackets '('",
    failure_types_dict, 
    data
)
```

You can also compare this to head 2.1:


```python
plotly_utils.plot_contribution_vs_open_proportion(
    h21_in_unbalanced_dir, 
    "Head 2.1 contribution vs proportion of open brackets '('",
    failure_types_dict,
    data
)
```



""", unsafe_allow_html=True)


def section_3():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#attention-pattern-of-the-responsible-head'>Attention pattern of the responsible head</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-get-attention-probabilities'><b>Exercise</b> - get attention probabilities</a></li>
        <li><a class='contents-el' href='#identifying-meaningful-direction-before-this-head'>Identifying meaningful direction before this head</a></li>
        <li><a class='contents-el' href='#exercise-compute-component-magnitudes'><b>Exercise</b> - compute component magnitudes</a></li>
        <li><a class='contents-el' href='#mlps-as-key-value-pairs'>MLPs as key-value pairs</a></li>
        <li><a class='contents-el' href='#exercise-get-output-by-neuron'><b>Exercise</b> - get output by neuron</a></li>
        <li><a class='contents-el' href='#exercise-implement-the-same-function-using-less-memory'><b>Exercise</b> - implement the same function, using less memory</a></li>
        <li><a class='contents-el' href='#interpreting-the-neurons'>Interpreting the neurons</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#understanding-how-the-open-proportion-is-calculated-head-0-0'>Understanding how the open-proportion is calculated - Head 0.0</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#0-0-attention-pattern'>0.0 Attention Pattern</a></li>
        <li><a class='contents-el' href='#exercise-extracting-queries-and-keys-using-hooks'><b>Exercise</b> - extracting queries and keys using hooks</a></li>
        <li><a class='contents-el' href='#activation-patching'>Activation Patching</a></li>
        <li><a class='contents-el' href='#proposing-a-hypothesis'>Proposing a hypothesis</a></li>
        <li><a class='contents-el' href='#the-0-0-ov-circuit'>The 0.0 OV circuit</a></li>
        <li><a class='contents-el' href='#exercise-validate-the-hypothesis'><b>Exercise</b> - validate the hypothesis</a></li>
        <li><a class='contents-el' href='#exercise-cosine-similarity-of-input-directions-optional'><b>Exercise</b> - cosine similarity of input directions (optional)</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#summary'>Summary</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 3️⃣ Understanding the total elevation circuit




In the largest section of the exercises, you'll examine the attention patterns in different heads, and interpret them as performing some human-understandable algorithm (e.g. copying, or aggregation). You'll use your observations to make  deductions about how a particular type of balanced brackets failure mode (mismatched number of left and right brackets) is detected by your model.

This is the first time you'll have to deal with MLPs in your models.

*This section is quite challenging both from a coding and conceptual perspective, because you need to link the results of your observations and interventions to concrete hypotheses about how the model works.*

> ##### Learning objctives
>
> * Practice connecting distinctive attention patterns to human-understandable algorithms, and making deductions about model behaviour. 
> * Understand how MLPs can be viewed as a collection of neurons.
> * Build up to a full picture of the total elevation circuit and how it works.


## Attention pattern of the responsible head

Which tokens is 2.0 paying attention to when the query is an open paren at token 0? Recall that we focus on sequences that start with an open paren because sequences that don't can be ruled out immediately, so more sophisticated behavior is unnecessary.




### Exercise - get attention probabilities

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠⚪⚪⚪

You shouldn't spend more than 5-10 minutes on this exercise.

This exercise just involves the `get_activations` helper func, and some indexing.
```

Write a function that extracts the attention patterns for a given head when run on a batch of inputs.


```python
def get_attn_probs(model: HookedTransformer, data: BracketsDataset, layer: int, head: int) -> t.Tensor:
    '''
    Returns: (N_SAMPLES, max_seq_len, max_seq_len) tensor that sums to 1 over the last dimension.
    '''
    pass


tests.test_get_attn_probs(get_attn_probs, model, data_mini)
```

<details>
<summary>Solution</summary>


```python
def get_attn_probs(model: HookedTransformer, data: BracketsDataset, layer: int, head: int) -> t.Tensor:
    '''
    Returns: (N_SAMPLES, max_seq_len, max_seq_len) tensor that sums to 1 over the last dimension.
    '''
    # SOLUTION
    return get_activations(model, data.toks, utils.get_act_name("pattern", layer))[:, head, :, :]
```
</details>

Once you've passed the tests, you can plot your results:


```python
attn_probs_20: Float[Tensor, "batch seqQ seqK"] = get_attn_probs(model, data, 2, 0)
attn_probs_20_open_query0 = attn_probs_20[data.starts_open].mean(0)[0]

bar(
    attn_probs_20_open_query0,
    title="Avg Attention Probabilities for query 0, first token '(', head 2.0",
    width=700, template="simple_white"
)
```

You should see an average attention of around 0.5 on position 1, and an average of about 0 for all other tokens. So `2.0` is just moving information from residual stream 1 to residual stream 0. In other words, `2.0` passes residual stream 1 through its `W_OV` circuit (after `LayerNorm`ing, of course), weighted by some amount which we'll pretend is constant. Importantly, this means that **the necessary information for classification must already have been stored in sequence position 1 before this head**. The plot thickens!

### Identifying meaningful direction before this head

If we make the simplification that the vector moved to sequence position 0 by head 2.0 is just `layernorm(x[1]) @ W_OV` (where `x[1]` is the vector in the residual stream before head 2.0, at sequence position 1), then we can do the same kind of logit attribution we did before. Rather than decomposing the input to the final layernorm (at sequence position 0) into the sum of ten components and measuring their contribution in the "pre final layernorm unbalanced direction", we can decompose the input to head 2.0 (at sequence position 1) into the sum of the seven components before head 2.0, and measure their contribution in the "pre head 2.0 unbalanced direction".

Here is an annotated diagram to help better explain exactly what we're doing.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/bracket_transformer-elevation-circuit-2.png" width="900">


#### Exercise - calculate the pre-head 2.0 unbalanced direction

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠🟠⚪⚪

You shouldn't spend more than 15-20 minutes on these exercises.

The second function should be conceptually similar to `get_pre_final_ln_dir` from earlier.
```

Below, you'll be asked to calculate this `pre_20_dir`, which is the unbalanced direction for inputs into head 2.0 at sequence position 1 (based on the fact that vectors at this sequence position are copied to position 0 by head `2.0`, and then used in prediction).

First, you'll implement the function `get_WOV`, to get the OV matrix for a particular layer and head. Recall that this is the product of the `W_O` and `W_V` matrices. Then, you'll use this function to write `get_pre_20_dir`.


```python
def get_WOV(model: HookedTransformer, layer: int, head: int) -> Float[Tensor, "d_model d_model"]:
    '''
    Returns the W_OV matrix for a particular layer and head.
    '''
    pass

def get_pre_20_dir(model, data) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction propagated back through the OV matrix of 2.0 
    and then through the layernorm before the layer 2 attention heads.
    '''
    pass


tests.test_get_pre_20_dir(get_pre_20_dir, model, data_mini)
```

<details>
<summary>Solution</summary>


```python
def get_WOV(model: HookedTransformer, layer: int, head: int) -> Float[Tensor, "d_model d_model"]:
    '''
    Returns the W_OV matrix for a particular layer and head.
    '''
    # SOLUTION
    return model.W_V[layer, head] @ model.W_O[layer, head]

def get_pre_20_dir(model, data) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction propagated back through the OV matrix of 2.0 
    and then through the layernorm before the layer 2 attention heads.
    '''
    # SOLUTION
    W_OV = get_WOV(model, 2, 0)

    layer2_ln_fit, r2 = get_ln_fit(model, data, layernorm=model.blocks[2].ln1, seq_pos=1)
    layer2_ln_coefs = t.from_numpy(layer2_ln_fit.coef_).to(device)

    pre_final_ln_dir = get_pre_final_ln_dir(model, data)

    return layer2_ln_coefs.T @ W_OV @ pre_final_ln_dir
```
</details>


### Exercise - compute component magnitudes

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠🟠⚪⚪

You shouldn't spend more than 10-15 minutes on these exercises.

This exercise should be somewhat similar to the last time you computed component magnitudes.
```

Now that you've got the `pre_20_dir`, you can calculate magnitudes for each of the components that came before. You can refer back to the diagram above if you're confused. **Remember to subtract the mean for each component for balanced inputs.**


```python
# YOUR CODE HERE - define `out_by_component_in_pre_20_unbalanced_dir` (for all components before head 2.0)
tests.test_out_by_component_in_pre_20_unbalanced_dir(out_by_component_in_pre_20_unbalanced_dir, model, data)

plotly_utils.hists_per_comp(
    out_by_component_in_pre_20_unbalanced_dir, 
    data, xaxis_range=(-5, 12)
)
```

<details>
<summary>Solution</summary>


```python
pre_layer2_outputs_seqpos1 = out_by_components[:-3, :, 1, :]
out_by_component_in_pre_20_unbalanced_dir = einops.einsum(
    pre_layer2_outputs_seqpos1,
    get_pre_20_dir(model, data),
    "comp batch emb, emb -> comp batch"
)
out_by_component_in_pre_20_unbalanced_dir -= out_by_component_in_pre_20_unbalanced_dir[:, data.isbal].mean(-1, keepdim=True)

```
</details>


What do you observe?

<details>
<summary>Some things to notice</summary>

One obvious note - the embeddings graph shows an output of zero, in other words no effect on the classification. This is because the input for this path is just the embedding vector in the 0th sequence position - in other words the `[START]` token's embedding, which is the same for all inputs.

---

More interestingly, we can see that `mlp0` and especially `mlp1` are very important. This makes sense -- one thing that mlps are especially capable of doing is turning more continuous features ('what proportion of characters in this input are open parens?') into sharp discontinuous features ('is that proportion exactly 0.5?').

For example, the sum $\operatorname{ReLU}(x-0.5) + \operatorname{ReLU}(0.5-x)$ evaluates to the nonlinear function $|x-0.5|$, which is zero if and only if $x=0.5$. This is one way our model might be able to classify all bracket strings as unbalanced unless they had exactly 50% open parens.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/relu2.png" width="600">

---

Head `1.1` also has some importance, although we will not be able to dig into this today. It turns out that one of the main things it does is incorporate information about when there is a negative elevation failure into this overall elevation branch. This allows the heads to agree the prompt is unbalanced when it is obviously so, even if the overall count of opens and closes would allow it to be balanced.

</details>


In order to get a better look at what `mlp0` and `mlp1` are doing more thoughly, we can look at their output as a function of the overall open-proportion.


```python
plotly_utils.mlp_attribution_scatter(out_by_component_in_pre_20_unbalanced_dir, data, failure_types_dict)
```

### MLPs as key-value pairs

When we implemented transformers from scratch, we observed that MLPs can be thought of as key-value pairs. To recap this briefly:

> We can write the MLP's output as $f(x^T W^{in})W^{out}$, where $W^{in}$ and $W^{out}$ are the different weights of the MLP (ignoring biases), $f$ is the activation function, and $x$ is a vector in the residual stream. This can be rewritten as:
> 
> $$
> f(x^T W^{in}) W^{out} = \sum_{i=1}^{d_{mlp}} f(x^T W^{in}_{[:, i]}) W^{out}_{[i, :]}
> $$
> 
> We can view the vectors $W^{in}_{[:, i]}$ as the **input directions**, and $W^{out}_{[i, :]}$ as the **output directions**. We say the input directions are **activated** by certain textual features, and when they are activated, vectors are written in the corresponding output direction. This is very similar to the concept of keys and values in attention layers, which is why these vectors are also sometimes called keys and values (e.g. see the paper [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/pdf/2012.14913.pdf)).

Including biases, the full version of this formula is:

$$
MLP(x) = \sum_{i=1}^{d_{mlp}}f(x^T W^{in}_{[:, i]} + b^{in}_i) W^{out}_{[i,:]} + b^{out}
$$

Diagram illustrating this (without biases):

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/mlp-neurons.png" width="850">


### Exercise - get output by neuron

```c
Difficulty: 🟠🟠🟠🟠⚪
Importance: 🟠🟠🟠🟠⚪

You shouldn't spend more than 25-35 minutes on these exercises.

It's important to understand exactly what the MLP is doing, and how to work with it.
```

The function `get_out_by_neuron` should return the given MLP's output per neuron. In other words, the output has shape `[batch, seq, neurons, d_model]`, where `out[b, s, i]` is the vector $f(\vec x^T W^{in}_{[:,i]} + b^{in}_i)W^{out}_{[i,:]}$ (and summing over `i` would give you the actual output of the MLP). We ignore $b^{out}$ here, because it isn't attributable to any specific neuron.

When you have this output, you can use `get_out_by_neuron_in_20_dir` to calculate the output of each neuron _in the unbalanced direction_ for the input to head 2.0 at sequence position 1. Note that we're only considering sequence position 1, because we've observed that head 2.0 is mainly just copying info from position 1 to position 0. This is why we've given you the `seq` argument in the `get_out_by_neuron` function, so you don't need to store more information than is necessary.


```python
def get_out_by_neuron(
    model: HookedTransformer, 
    data: BracketsDataset, 
    layer: int, 
    seq: Optional[int] = None
) -> Float[Tensor, "batch *seq neuron d_model"]:
    '''
    If seq is not None, then out[b, s, i, :] = f(x[b, s].T @ W_in[:, i]) @ W_out[i, :],
    i.e. the vector which is written to the residual stream by the ith neuron (where x
    is the input to the residual stream (i.e. shape (batch, seq, d_model)).

    If seq is None, then out[b, i, :] = vector f(x[b].T @ W_in[:, i]) @ W_out[i, :]

    (Note, using * in jaxtyping indicates an optional dimension)
    '''
    pass

def get_out_by_neuron_in_20_dir(model: HookedTransformer, data: BracketsDataset, layer: int) -> Float[Tensor, "batch neurons"]:
    '''
    [b, s, i]th element is the contribution of the vector written by the ith neuron to the residual stream in the 
    unbalanced direction (for the b-th element in the batch, and the s-th sequence position).

    In other words we need to take the vector produced by the `get_out_by_neuron` function, and project it onto the 
    unbalanced direction for head 2.0 (at seq pos = 1).
    '''
    pass


tests.test_get_out_by_neuron(get_out_by_neuron, model, data_mini)
tests.test_get_out_by_neuron_in_20_dir(get_out_by_neuron_in_20_dir, model, data_mini)
```

<details>
<summary>Hint</summary>

For the `get_out_by_neuron` function, define $f(\vec x^T W^{in}_{[:,i]} + b^{in}_i)$ and $W^{out}_{[i,:]}$ separately, then multiply them together. The former is the activation corresponding to the name `"post"`, and you can access it using your `get_activations` function. The latter are just the model weights, and you can access it using `model.W_out`.

Also, remember to keep in mind the distinction between activations and parameters. $f(\vec x^T W^{in}_{[:,i]} + b^{in}_i)$ is an activation; it has a `batch` and `seq_len` dimension. $W^{out}_{[i,:]}$ is a parameter; it has no `batch` or `seq_len` dimension.
</details>

<details>
<summary>Solution</summary>


```python
def get_out_by_neuron(
    model: HookedTransformer, 
    data: BracketsDataset, 
    layer: int, 
    seq: Optional[int] = None
) -> Float[Tensor, "batch *seq neuron d_model"]:
    '''
    If seq is not None, then out[b, s, i, :] = f(x[b, s].T @ W_in[:, i]) @ W_out[i, :],
    i.e. the vector which is written to the residual stream by the ith neuron (where x
    is the input to the residual stream (i.e. shape (batch, seq, d_model)).

    If seq is None, then out[b, i, :] = vector f(x[b].T @ W_in[:, i]) @ W_out[i, :]

    (Note, using * in jaxtyping indicates an optional dimension)
    '''
    # SOLUTION
    # Get the W_out matrix for this MLP
    W_out: Float[Tensor, "neuron d_model"] = model.W_out[layer]

    # Get activations of the layer just after the activation function, i.e. this is f(x.T @ W_in)
    f_x_W_in: Float[Tensor, "batch seq neuron"] = get_activations(model, data.toks, utils.get_act_name('post', layer))

    # f_x_W_in are activations, so they have batch and seq dimensions - this is where we index by seq if necessary
    if seq is not None:
        f_x_W_in: Float[Tensor, "batch neuron"] = f_x_W_in[:, seq, :]

    # Calculate the output by neuron (i.e. so summing over the `neurons` dimension gives the output of the MLP)
    out = einops.einsum(
        f_x_W_in, 
        W_out,
        "... neuron, neuron d_model -> ... neuron d_model",
    )
    return out

def get_out_by_neuron_in_20_dir(model: HookedTransformer, data: BracketsDataset, layer: int) -> Float[Tensor, "batch neurons"]:
    '''
    [b, s, i]th element is the contribution of the vector written by the ith neuron to the residual stream in the 
    unbalanced direction (for the b-th element in the batch, and the s-th sequence position).

    In other words we need to take the vector produced by the `get_out_by_neuron` function, and project it onto the 
    unbalanced direction for head 2.0 (at seq pos = 1).
    '''
    # SOLUTION
    # Get neuron output at sequence position 1
    out_by_neuron_seqpos1 = get_out_by_neuron(model, data, layer, seq=1)
    

    # For each neuron, project the vector it writes to residual stream along the pre-2.0 unbalanced direction
    return einops.einsum(
        out_by_neuron_seqpos1,
        get_pre_20_dir(model, data),
        "batch neuron d_model, d_model -> batch neuron"
    )
```
</details>


### Exercise - implement the same function, using less memory

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠⚪⚪⚪

You shouldn't spend more than 10-15 minutes on this exercise.

Understanding the solution is more important than doing this exercise, so you should look at the solution rather than doing the exercise if you feel like it.
```

*This exercise isn't as important as the previous one, and you can skip it if you don't find this interesting (although you're still recommended to look at the solutions, so you understand what's going on here.)*


If the only thing we want from the MLPs are their contribution in the unbalanced direction, then we can actually do this without having to store the `out_by_neuron_in_20_dir` object. Try and find this method, and implement it below.


```python
def get_out_by_neuron_in_20_dir_less_memory(model: HookedTransformer, data: BracketsDataset, layer: int) -> Float[Tensor, "batch neurons"]:
    '''
    Has the same output as `get_out_by_neuron_in_20_dir`, but uses less memory (because it never stores
    the output vector of each neuron individually).
    '''
    pass


tests.test_get_out_by_neuron_in_20_dir_less_memory(get_out_by_neuron_in_20_dir_less_memory, model, data_mini)
```

<details>
<summary>Hint</summary>

The key is to change the order of operations.

First, project each of the output directions onto the pre-2.0 unbalanced direction in order to get their components (i.e. a vector of length `d_mlp`, where the `i`-th element is the component of the vector $W^{out}_{[i,:]}$ in the unbalanced direction). Then, scale these contributions by the activations $f(\vec x^T W^{in}_{[:,i]} + b^{in}_i)$.**bold text**
</details>

<details>
<summary>Solution</summary>


```python
def get_out_by_neuron_in_20_dir_less_memory(model: HookedTransformer, data: BracketsDataset, layer: int) -> Float[Tensor, "batch neurons"]:
    '''
    Has the same output as `get_out_by_neuron_in_20_dir`, but uses less memory (because it never stores
    the output vector of each neuron individually).
    '''
    # SOLUTION
    W_out: Float[Tensor, "neurons d_model"] = model.W_out[layer]

    f_x_W_in: Float[Tensor, "batch neurons"] = get_activations(model, data.toks, utils.get_act_name('post', layer))[:, 1, :]

    pre_20_dir: Float[Tensor, "d_model"] = get_pre_20_dir(model, data)

    # Multiply along the d_model dimension
    W_out_in_20_dir: Float[Tensor, "neurons"] = W_out @ pre_20_dir
    # Multiply elementwise, over neurons (we're broadcasting along the batch dim)
    out_by_neuron_in_20_dir: Float[Tensor, "batch neurons"] = f_x_W_in * W_out_in_20_dir

    return out_by_neuron_in_20_dir
```
</details>


### Interpreting the neurons

Now, try to identify several individual neurons that are especially important to `2.0`.

For instance, you can do this by seeing which neurons have the largest difference between how much they write in our chosen direction on balanced and unbalanced sequences (especially unbalanced sequences beginning with an open paren).

Use the `plot_neurons` function to get a sense of what an individual neuron does on differen open-proportions.

One note: now that we are deep in the internals of the network, our assumption that a single direction captures most of the meaningful things going on in this overall-elevation circuit is highly questionable. This is especially true for using our `2.0` direction to analyize the output of `mlp0`, as one of the main ways this mlp has influence is through more indirect paths (such as `mlp0 -> mlp1 -> 2.0`) which are not the ones we chose our direction to capture. Thus, it is good to be aware that the intuitions you get about what different layers or neurons are doing are likely to be incomplete.

*Note - we've supplied the default argument `renderer="browser"`, which causes the plots to open in a browser rather than in VSCode. This often works better, with less lag (especially in notebooks), but you can remove this if you prefer.*


```python
for layer in range(2):
    # Get neuron significances for head 2.0, sequence position #1 output
    neurons_in_unbalanced_dir = get_out_by_neuron_in_20_dir_less_memory(model, data, layer)[utils.to_numpy(data.starts_open), :]
    # Plot neurons' activations
    plotly_utils.plot_neurons(neurons_in_unbalanced_dir, model, data, failure_types_dict, layer, renderer="browser")
```

<details>
<summary>Some observations:</summary>

The important neurons in layer 1 can be put into three broad categories:

- Some neurons detect when the open-proportion is greater than 1/2. As a few examples, look at neurons **`1.53`**, **`1.39`**, **`1.8`** in layer 1. There are some in layer 0 as well, such as **`0.33`** or **`0.43`**. Overall these seem more common in Layer 1.

- Some neurons detect when the open-proportion is less than 1/2. For instance, neurons **`0.21`**, and **`0.7`**. These are much more rare in layer 1, but you can see some such as **`1.50`** and **`1.6`**.

- The network could just use these two types of neurons, and compose them to measure if the open-proportion exactly equals 1/2 by adding them together. But we also see in layer 1 that there are many neurons that output this composed property. As a few examples, look at **`1.10`** and **`1.3`**. 
    - It's much harder for a single neuron in layer 0 to do this by themselves, given that ReLU is monotonic and it requires the output to be a non-monotonic function of the open-paren proportion. It is possible, however, to take advantage of the layernorm before **`mlp0`** to approximate this -- **`0.19`** and **`0.34`** are good examples of this.

Note, there are some neurons which appear to work in the opposite direction (e.g. `0.0`). It's unclear exactly what the function of these neurons is (especially since we're only analysing one particular part of one of our model's circuits, so our intuitions about what a particular neuron does might be incomplete). However, what is clear and unambiguous from this plot is that our neurons seem to be detecting the open proportion of brackets, and responding differently if the proportion is strictly more / strictly less than 1/2. And we can see that a large number of these seem to have their main impact via being copied in head `2.0`.

---

Below: plots of neurons **`0.21`** and **`1.53`**. You can observe the patterns described above.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/n21.png" width="550">
<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/n53.png" width="550">
</details>




## Understanding how the open-proportion is calculated - Head 0.0

Up to this point we've been working backwards from the logits and through the internals of the network. We'll now change tactics somewhat, and start working from the input embeddings forwards. In particular, we want to understand how the network calcuates the open-proportion of the sequence in the first place!

The key will end up being head 0.0. Let's start by examining its attention pattern.

### 0.0 Attention Pattern

We want to play around with the attention patterns in our heads. For instance, we'd like to ask questions like "what do the attention patterns look like when the queries are always left-parens?". To do this, we'll write a function that takes in a parens string, and returns the `q` and `k` vectors (i.e. the values which we take the inner product of to get the attention scores).


### Exercise - extracting queries and keys using hooks

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠⚪⚪⚪

You shouldn't spend more than ~10 minutes on this exercise.

Again, this exercise just involves using your `get_activations` function.
```


```python
def get_q_and_k_for_given_input(
    model: HookedTransformer, 
    tokenizer: SimpleTokenizer,
    parens: str, 
    layer: int, 
) -> Tuple[Float[Tensor, "seq_d_model"], Float[Tensor,  "seq_d_model"]]:
    '''
    Returns the queries and keys (both of shape [seq, d_model]) for the given parns input, in the attention head `layer.head`.
    '''
    pass


tests.test_get_q_and_k_for_given_input(get_q_and_k_for_given_input, model, tokenizer)
```

<details>
<summary>Solution</summary>


```python
def get_q_and_k_for_given_input(
    model: HookedTransformer, 
    tokenizer: SimpleTokenizer,
    parens: str, 
    layer: int, 
) -> Tuple[Float[Tensor, "seq_d_model"], Float[Tensor,  "seq_d_model"]]:
    '''
    Returns the queries and keys (both of shape [seq, d_model]) for the given parns input, in the attention head `layer.head`.
    '''
    # SOLUTION
    q_name = utils.get_act_name("q", layer)
    k_name = utils.get_act_name("k", layer)

    activations = get_activations(
        model,
        tokenizer.tokenize(parens),
        [q_name, k_name]
    )

    return activations[q_name][0], activations[k_name][0]
```
</details>


### Activation Patching

Now, we'll introduce the valuable tool of **activation patching**. This was first introduced in [David Bau and Kevin Meng's excellent ROME paper](https://rome.baulab.info/), there called causal tracing.

The setup of activation patching is to take two runs of the model on two different inputs, the clean run and the corrupted run. The clean run outputs the correct answer and the corrupted run does not. The key idea is that we give the model the corrupted input, but then **intervene** on a specific activation and **patch** in the corresponding activation from the clean run (i.e. replace the corrupted activation with the clean activation), and then continue the run. 

One of the common use-cases for activation patching is to compare the model's performance in clean vs patched runs. If the performance degrades with patching, this is a strong signal that the place you patched in is important for the model's computation. The ability to localise is a key move in mechanistic interpretability - if the computation is diffuse and spread across the entire model, it is likely much harder to form a clean mechanistic story for what's going on. But if we can identify precisely which parts of the model matter, we can then zoom in and determine what they represent and how they connect up with each other, and ultimately reverse engineer the underlying circuit that they represent.

However, here our path patching serves a much simpler purpose - we'll be patching at the **query vectors** of head `0.0` with values from a sequence of all left-parens, and at the **key vectors** with the average values from all left and all right parens. This allows us to get a sense for the average attention patterns paid by left-brackets to the rest of the sequence.

We'll write functions to do this for both heads in layer 0, because it will be informative to compare the two.


```python
layer = 0
all_left_parens = "".join(["(" * 40])
all_right_parens = "".join([")" * 40])

model.reset_hooks()
q0_all_left, k0_all_left = get_q_and_k_for_given_input(model, tokenizer, all_left_parens, layer)
q0_all_right, k0_all_right = get_q_and_k_for_given_input(model, tokenizer, all_right_parens, layer)
k0_avg = (k0_all_left + k0_all_right) / 2


# Define hook function to patch in q or k vectors
def hook_fn_patch_qk(
    value: Float[Tensor, "batch seq head d_head"], 
    hook: HookPoint, 
    new_value: Float[Tensor, "... seq d_head"],
    head_idx: Optional[int] = None
) -> None:
    if head_idx is not None:
        value[..., head_idx, :] = new_value[..., head_idx, :]
    else:
        value[...] = new_value[...]


# Define hook function to display attention patterns (using plotly)
def hook_fn_display_attn_patterns(
    pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    head_idx: int = 0
) -> None:
    avg_head_attn_pattern = pattern.mean(0)
    labels = ["[start]", *[f"{i+1}" for i in range(40)], "[end]"]
    display(cv.attention.attention_heads(
        tokens=labels, 
        attention=avg_head_attn_pattern,
        attention_head_names=["0.0", "0.1"],
        max_value=avg_head_attn_pattern.max()
    ))


# Run our model on left parens, but patch in the average key values for left vs right parens
# This is to give us a rough idea how the model behaves on average when the query is a left paren
model.run_with_hooks(
    tokenizer.tokenize(all_left_parens).to(device),
    return_type=None,
    fwd_hooks=[
        (utils.get_act_name("k", layer), partial(hook_fn_patch_qk, new_value=k0_avg)),
        (utils.get_act_name("pattern", layer), hook_fn_display_attn_patterns),
    ]
)
```

<details>
<summary>Help - my <code>attention_heads</code> plots are behaving weirdly (e.g. they continually shrink after I plot them).</summary>

This seems to be a bug in `circuitsvis` - on VSCode, the attention head plots continually shrink in size.

Until this is fixed, one way to get around it is to open the plots in your browser. You can do this inline with the `webbrowser` library:

```python
attn_heads = cv.attention.attention_heads(
    tokens=labels, 
    attention=avg_head_attn_pattern,
    attention_head_names=["0.0"],
)

path = "attn_heads.html"

with open(path, "w") as f:
    f.write(str(attn_heads))

webbrowser.open(path)
```

To check exactly where this is getting saved, you can print your current working directory with `os.getcwd()`.
</details>

<details>
<summary>Question - what are the noteworthy features of head <code>0.0</code> in this plot?</summary>

The most noteworthy feature is the diagonal pattern - most query tokens pay almost zero attention to all the tokens that come before it, but much greater attention to those that come after it. For most query token positions, this attention paid to tokens after itself is roughly uniform. However, there are a few patches (especially for later query positions) where the attention paid to tokens after itself is not uniform. We will see that these patches are important for generating adversarial examples.

We can also observe roughly the same pattern when the query is a right paren (try running the last bit of code above, but using `all_right_parens` instead of `all_left_parens`), although the pattern is less pronounced.
</details>

We are most interested in the attention pattern at query position 1, because this is the position we move information to that is eventually fed into attention head `2.0`, then moved to position 0 and used for prediction.

(Note - we've chosen to focus on the scenario when the first paren is an open paren, because the model actually deals with bracket strings that open with a right paren slightly differently - these are obviously unbalanced, so a complicated mechanism is unnecessary.)

Let's plot a bar chart of the attention probability paid by the the open-paren query at position 1 to all the other positions. Here, rather than patching in both the key and query from artificial sequences, we're running the model on our entire dataset and patching in an artificial value for just the query (all open parens). Both methods are reasonable here, since we're just looking for a general sense of how our query vector at position 1 behaves when it's an open paren.


```python
def hook_fn_display_attn_patterns_for_single_query(
    pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    head_idx: int = 0,
    query_idx: int = 1
):
    bar(
        utils.to_numpy(pattern[:, head_idx, query_idx].mean(0)), 
        title=f"Average attn probabilities on data at posn 1, with query token = '('",
        labels={"index": "Sequence position of key", "value": "Average attn over dataset"}, 
        height=500, width=800, yaxis_range=[0, 0.1], template="simple_white"
    )


data_len_40 = BracketsDataset.with_length(data_tuples, 40).to(device)

model.reset_hooks()
model.run_with_hooks(
    data_len_40.toks[data_len_40.isbal],
    return_type=None,
    fwd_hooks=[
        (utils.get_act_name("q", 0), partial(hook_fn_patch_qk, new_value=q0_all_left)),
        (utils.get_act_name("pattern", 0), hook_fn_display_attn_patterns_for_single_query),
    ]
)
```

<details>
<summary>Question - what is the interpretation of this attention pattern?</summary>

This shows that the attention pattern is almost exactly uniform over all tokens. This means the vector written to sequence position 1 will be approximately some scalar multiple of the vectors at each source position, transformed via the matrix $W_{OV}^{0.0}$.
</details>


### Proposing a hypothesis

Before we connect all the pieces together, let's list the facts that we know about our model so far (going chronologically from our observations):

> * Attention head `2.0` seems to be largely responsible for classifying brackets as unbalanced when they have non-zero net elevation (i.e. have a different number of left and right parens).
>     * Attention head `2.0` attends strongly to the sequence position $i=1$, in other words it's pretty much just moving the residual stream vector from position 1 to position 0 (and applying matrix $W_{OV}$).
>     * So there must be earlier components of the model which write to sequence position 1, in a way which influences the model to make correct classifications (via the path through head `2.0`).
> * There are several neurons in `MLP0` and `MLP1` which seem to calculate a nonlinear function of the open parens proportion - some of them are strongly activating when the proportion is strictly greater than $1/2$, others when it is strictly smaller than $1/2$.
> * If the query token in attention head `0.0` is an open paren, then it attends to all key positions **after** $i$ with roughly equal magnitude.
>     * In particular, this holds for the sequence position $i=1$, which attends approximately uniformly to all sequence positions.

Based on all this, can you formulate a hypothesis for how the elevation circuit works, which ties all three of these observations together?

<details>
<summary>Hypothesis</summary>

The hypothesis might go something like this:

1. **In the attention calculation for head `0.0`, the position-1 query token is doing some kind of aggregation over brackets. It writes to the residual stream information representing the difference between the number of left and right brackets - in other words, the net elevation.**
>   Remember that one-layer attention heads can pretty much only do skip-trigrams, e.g. of the form `keep ... in -> mind`. They can't capture three-way interactions flexibly, in other words they can't compute functions like "whether the number of left and right brackets is equal". (To make this clearer, consider how your model's behaviour would differ on the inputs `()`, `((` and `))` if it was just one-layer). So aggregation over left and right brackets is pretty much all we can do.

2. **Now that sequence position 1 contains information about the elevation, the MLP reads this information, and some of its neurons perform nonlinear operations to give us a vector which contains "boolean" information about whether the number of left and right brackets is equal.**
> Recall that MLPs are great at taking linear functions (like the difference between number of left and right brackets) and converting it to boolean information. We saw something like this was happening in our plots above, since most of the MLPs' neurons' behaviour was markedly different above or below the threshold of 50% left brackets.

3. **Finally, now that the 1st sequence position in the residual stream stores boolean information about whether the net elevation is zero, this information is read by head `2.0`, and the output of this head is used to classify the sequence as balanced or unbalanced.**
> This is based on the fact that we already saw head `2.0` is strongly attending to the 1st sequence position, and that it seems to be implementing the elevation test.
</details>

At this point, we've pretty much empirically verified all the observations above. One thing we haven't really proven yet is that **(1)** is working as we've described above. We want to verify that head `0.0` is calculating some kind of difference between the number of left and right brackets, and writing this information to the residual stream. In the next section, we'll find a way to test this hypothesis.


### The 0.0 OV circuit

**We want to understand what the `0.0` head is writing to the residual stream. In particular, we are looking for evidence that it is writing information about the net elevation.**

We've already seen that query position 1 is attending approximately uniformly to all key positions. This means that (ignoring start and end tokens) the vector written to position 1 is approximately:

$$
\begin{aligned}
h(x) &\approx \frac{1}{n} \sum_{i=1}^n \left(\left(L {\color{orange}x}\right)^T W_{OV}^{0.0}\right)_i \\
&= \frac{1}{n} \sum_{i=1}^n {\color{orange}x}_i^T L^T W_{OV}^{0.0} \\
\end{aligned}
$$

where $L$ is the linear approximation for the layernorm before the first attention layer, and $x$ is the `(seq_len, d_model)`-size residual stream consisting of vectors ${\color{orange}x}_i$ for each sequence position $i$.

We can write ${\color{orange}x}_j = {\color{orange}pos}_j + {\color{orange}tok}_j$, where ${\color{orange}pos}_j$ and ${\color{orange}tok}_j$ stand for the positional and token embeddings respectively. So this gives us:

$$
\begin{aligned}
h(x) &\approx \frac{1}{n} \left( \sum_{i=1}^n {\color{orange}pos}_i^T L^T W_{OV}^{0.0} + \sum_{i=1}^n {\color{orange}tok}_i^T L^T W_{OV}^{0.0}\right) \\
&= \frac{1}{n} \left( \sum_{i=1}^n {\color{orange}pos}_i^T L^T W_{OV}^{0.0} + n_L {\color{orange}\vec v_L} + n_R {\color{orange}\vec v_R}\right)
\end{aligned}
$$

where $n_L$ and $n_R$ are the number of left and right brackets respectively, and ${\color{orange}\vec v_L}, {\color{orange}\vec v_R}$ are the images of the token embeddings for left and right parens respectively under the image of the layernorm and OV circuit:

$$
\begin{aligned}
{\color{orange}\vec v_L} &= {\color{orange}LeftParen}^T L^T W_{OV}^{0.0} \\
{\color{orange}\vec v_R} &= {\color{orange}RightParen}^T L^T W_{OV}^{0.0}
\end{aligned}
$$

where $\color{orange}{LeftParen}$ and $\color{orange}{RightParen}$ are the token embeddings for left and right parens respectively.

Finally, we have an ability to formulate a test for our hypothesis in terms of the expression above:

> If head `0.0` is performing some kind of aggregation, then **we should see that $\color{orange}{\vec v_L}$ and $\color{orange}{\vec v_R}$ are vectors pointing in opposite directions.** In other words, head `0.0` writes some scalar multiple of vector $v$ to the residual stream, and we can extract the information $n_L - n_R$ by projecting in the direction of this vector. The MLP can then take this information and process it in a nonlinear way, writing information about whether the sequence is balanced to the residual stream.


### Exercise - validate the hypothesis

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠⚪⚪⚪

You shouldn't spend more than 10-15 minutes on this exercise.

If you understand what the vectors represent, these exercises should be pretty straightforward.
```

Here, you should show that the two vectors have cosine similarity close to -1, demonstrating that this head is "tallying" the open and close parens that come after it.

You can fill in the function `embedding` (to return the token embedding vector corresponding to a particular character, i.e. the vectors we've called $\color{orange}{LeftParen}$ and $\color{orange}{RightParen}$ above), which will help when computing these vectors.


```python
def embedding(model: HookedTransformer, tokenizer: SimpleTokenizer, char: str) -> Float[Tensor, "d_model"]:
    assert char in ("(", ")")
    idx = tokenizer.t_to_i[char]
    return model.W_E[idx]


# YOUR CODE HERE - define v_L and v_R, as described above.
print("Cosine similarity: ", t.cosine_similarity(v_L, v_R, dim=0).item())
```

<details>
<summary>Solution</summary>


```python
W_OV = model.W_V[0, 0] @ model.W_O[0, 0]
layer0_ln_fit = get_ln_fit(model, data, layernorm=model.blocks[0].ln1, seq_pos=None)[0]
layer0_ln_coefs = t.from_numpy(layer0_ln_fit.coef_).to(device)
v_L = embedding(model, tokenizer, "(") @ layer0_ln_coefs.T @ W_OV
v_R = embedding(model, tokenizer, ")") @ layer0_ln_coefs.T @ W_OV
```
</details>

<details>
<summary>Extra technicality about the two vectors (optional)</summary>

Note - we don't actually require $\color{orange}{\vec v_L}$ and $\color{orange}{\vec v_R}$ to have the same magnitude for this idea to work. This is because, if we have ${\color{orange} \vec v_L} \approx - \alpha {\color{orange} \vec v_R}$ for some $\alpha > 0$, then when projecting along the $\color{orange}{\vec v_L}$ direction we will get $\|{\color{orange} \vec v_L}\| (n_L - \alpha n_R) / n$. This always equals $\|{\color{orange} \vec v_L}\| (1 - \alpha) / 2$ when the number of left and right brackets match, regardless of the sequence length. It doesn't matter that this value isn't zero; the MLPs' neurons can still learn to detect when the vector's component in this direction is more or less than this value by adding a bias term. The important thing is that (1) the two vectors are parallel and pointing in opposite directions, and (2) the projection in this direction *for balanced sequences* is always the same.

</details>


### Exercise - cosine similarity of input directions (optional)

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠⚪⚪⚪⚪

You shouldn't spend more than 10-15 minutes on this exercise.
```

Another way we can get evidence for this hypothesis - recall in our discussion of MLP neurons that $W^{in}_{[:,i]}$ (the $i$th column of matrix $W^{in}$, where $W^{in}$ is the first linear layer of the MLP) is a vector representing the "in-direction" of the neuron. If these neurons are indeed measuring open/closed proportions in the way we think, then we should expect to see the vectors $v_R$, $v_L$ have high dot product with these vectors.

Investigate this by filling in the two functions below. `cos_sim_with_MLP_weights` returns the vector of cosine similarities between a vector and the columns of $W^{in}$ for a given layer, and `avg_squared_cos_sim` returns the average **squared cosine similarity** between a vector $v$ and a randomly chosen vector with the same size as $v$ (we can choose this vector in any sensible way, e.g. sampling it from the iid normal distribution then normalizing it). You should find that the average squared cosine similarity per neuron between $v_R$ and the in-directions for neurons in `MLP0` and `MLP1` is much higher than you would expect by chance.


```python
def cos_sim_with_MLP_weights(model: HookedTransformer, v: Float[Tensor, "d_model"], layer: int) -> Float[Tensor, "d_mlp"]:
    '''
    Returns a vector of length d_mlp, where the ith element is the cosine similarity between v and the 
    ith in-direction of the MLP in layer `layer`.

    Recall that the in-direction of the MLPs are the columns of the W_in matrix.
    '''
    pass

def avg_squared_cos_sim(v: Float[Tensor, "d_model"], n_samples: int = 1000) -> float:
    '''
    Returns the average (over n_samples) cosine similarity between v and another randomly chosen vector.

    We can create random vectors from the standard N(0, I) distribution.
    '''
    pass


print("Avg squared cosine similarity of v_R with ...\n")

cos_sim_mlp0 = cos_sim_with_MLP_weights(model, v_R, 0)
print(f"...MLP input directions in layer 0:  {cos_sim_mlp0.pow(2).mean():.6f}")

cos_sim_mlp1 = cos_sim_with_MLP_weights(model, v_R, 1)
print(f"...MLP input directions in layer 1:  {cos_sim_mlp1.pow(2).mean():.6f}")

cos_sim_rand = avg_squared_cos_sim(v_R)
print(f"...random vectors of len = d_model:  {cos_sim_rand:.6f}")
```

<details>
<summary>Solution</summary>


```python
def cos_sim_with_MLP_weights(model: HookedTransformer, v: Float[Tensor, "d_model"], layer: int) -> Float[Tensor, "d_mlp"]:
    '''
    Returns a vector of length d_mlp, where the ith element is the cosine similarity between v and the 
    ith in-direction of the MLP in layer `layer`.

    Recall that the in-direction of the MLPs are the columns of the W_in matrix.
    '''
    # SOLUTION
    v_unit = v / v.norm()
    W_in_unit = model.W_in[layer] / model.W_in[layer].norm(dim=0)

    return einops.einsum(v_unit, W_in_unit, "d_model, d_model d_mlp -> d_mlp")

def avg_squared_cos_sim(v: Float[Tensor, "d_model"], n_samples: int = 1000) -> float:
    '''
    Returns the average (over n_samples) cosine similarity between v and another randomly chosen vector.

    We can create random vectors from the standard N(0, I) distribution.
    '''
    # SOLUTION
    v2 = t.randn(n_samples, v.shape[0]).to(device)
    v2 /= v2.norm(dim=1, keepdim=True)

    v1 = v / v.norm()

    return (v1 * v2).pow(2).sum(1).mean().item()
```
</details>

As an _extra_-bonus exercise, you can also compare the squared cosine similarities per neuron to your neuron contribution plots you made earlier (the ones with sliders). Do the neurons which have particularly high cosine similarity with $v_R$ correspond to the neurons which write to the unbalanced direction of head `2.0` in a big way whenever the proportion of open parens is not 0.5? (This would provide further evidence that the main source of information about total open proportion of brackets which is used in the net elevation circuit is provided by the multiples of $v_R$ and $v_L$ written to the residual stream by head `0.0`). You can go back to your old plots and check.


## Summary

Great! Let's stop and take stock of what we've learned about this circuit. 

> Head 0.0 pays attention uniformly to the suffix following each token, tallying up the amount of open and close parens that it sees and writing that value to the residual stream. This means that it writes a vector representing the total elevation to residual stream 1. The MLPs in residual stream 1 then operate nonlinearly on this tally, writing vectors to the residual stream that distinguish between the cases of zero and non-zero total elevation. Head 2.0 copies this signal to residual stream 0, where it then goes through the classifier and leads to a classification as unbalanced. Our first-pass understanding of this behavior is complete.

An illustration of this circuit is given below. It's pretty complicated with a lot of moving parts, so don't worry if you don't follow all of it!

Key: the thick black lines and orange dotted lines show the paths through our transformer constituting the elevation circuit. The orange dotted lines indicate the skip connections. Each of the important heads and MLP layers are coloured bold. The three important parts of our circuit (head `0.0`, the MLP layers, and head `2.0`) are all give annotations explaining what they're doing, and the evidence we found for this.


<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/bracket-transformer-attribution-2.png" width="1150">




""", unsafe_allow_html=True)


def section_4():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#detecting-anywhere-negative-failures'>Detecting anywhere-negative failures</a></li>
    <li class='margtop'><a class='contents-el' href='#adversarial-attacks'>Adversarial attacks</a></li>
    <li class='margtop'><a class='contents-el' href='#dealing-with-early-closing-parens'>Dealing with early closing parens</a></li>
    <li class='margtop'><a class='contents-el' href='#try-more-algorithmic-problems'>Try more algorithmic problems</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 4️⃣ Bonus exercises



Lastly, there are a few optional bonus exercises which build on the previous content (e.g. having you examine different parts of the model, or use your understanding of how the model works to generate adversarial examples).

*This final section is less guided, although the suggested exercises are similar in flavour to the previous section.*

> ##### Learning objctives
>
> * Use your understanding of how the model works to generate adversarial examples.
> * Take deeper dives into specific anomalous features of the model.


The main bonus exercise we recommend you try is **adversarial attacks**. You'll need to read the first section of the **detecting anywhere-negative failures** bonus exercise to get an idea for how the other half of the classification circuit works, but once you understand this you can jump ahead to the adversarial attacks section.


## Detecting anywhere-negative failures

When we looked at our grid of attention patterns, we saw that not only did the first query token pay approximately uniform attention to all tokens following it, but so did most of the other tokens (to lesser degrees). This means that we can write the vector written to position $i$ (for general $i\geq 1$) as:

$$
\begin{aligned}
h(x)_i &\approx \frac{1}{n-i+1} \sum_{j=i}^n {\color{orange}x}_j^T L^T W_{OV}^{0.0} \\
&= \frac{1}{n} \left( \sum_{i=1}^n {\color{orange}pos}_i^T L^T W_{OV}^{0.0} + n_L^{(i)} {\color{orange}\vec v_L} + n_R^{(i)} {\color{orange}\vec v_R}\right)
\end{aligned}
$$

where $n_L^{(i)}$ and $n_R^{(i)}$ are the number of left and right brackets respectively in the substring formed from `brackets[i: n]` (i.e. this matches our definition of $n_L$ and $n_R$ when $i=1$).

Given what we've seen so far (that sequence position 1 stores tally information for all the brackets in the sequence), we can guess that each sequence position stores a similar tally, and is used to determine whether the substring consisting of all brackets to the right of this one has any elevation failures (i.e. making sure the total number of ***right*** brackets is at least as great as the total number of ***left*** brackets - recall it's this way around because our model learned the equally valid right-to-left solution).

Recall that the destination token only determines how much to pay attention to the source; the vector that is moved from the source to destination conditional on attention being paid to it is the same for all destination tokens. So the result about left-paren and right-paren vectors having cosine similarity of -1 also holds for all later sequence positions.

**Head 2.1 turns out to be the head for detecting anywhere-negative failures** (i.e. it  detects whether any sequence `brackets[i: n]` has strictly more right than left parentheses, and writes to the residual stream in the unbalanced direction if this is the case). Can you find evidence for this behaviour?

One way you could investigate this is to construct a parens string which "goes negative" at some points, and look at the attention probabilities for head 2.0 at destination position 0. Does it attend most strongly to those source tokens where the bracket goes negative, and is the corresponding vector written to the residual stream one which points in the unbalanced direction?

You could also look at the inputs to head 2.1, just like we did for head 2.0. Which components are most important, and can you guess why?

<details>
<summary>Solution</summary>

You should find that the MLPs are important inputs into head 2.1. This makes sense, because earlier we saw that the MLPs were converting tally information $(n_L - \alpha n_R)$ into the boolean information $(n_L = n_R)$ at sequence position 1. Since MLPs act the same on all sequence positions, it's reasonable to guess that they're storing the boolean information $(n_L^{(i)} > n_R^{(i)})$ at each sequence position $i$, which is what we need to detect anywhere-negative failures.
</details>


## Adversarial attacks

Our model gets around 1 in a ten thousand examples wrong on the dataset we've been using. Armed with our understanding of the model, can we find a misclassified input by hand? I recommend stopping reading now and trying your hand at applying what you've learned so far to find a misclassified sequence. If this doesn't work, look at a few hints.

<details>
<summary>Hint 1</summary>

What's up with those weird patchy bits in the bottom-right corner of the attention patterns? Can we exploit this?

Read the next hint for some more specific directions.
</details>

<details>
<summary>Hint 2</summary>

We observed that each left bracket attended approximately uniformly to each of the tokens to its right, and used this to detect elevation failures at any point. We also know that this approximately uniform pattern breaks down around query positions 27-31. 

With this in mind, what kind of "just barely" unbalanced bracket string could we construct that would get classified as balanced by the model? 

Read the next hint for a suggested type of bracket string.
</details>

<details>
<summary>Hint 3</summary>

We want to construct a string that has a negative elevation at some point, but is balanced everywhere else. We can do this by using a sequence of the form `A)(B`, where `A` and `B` are balanced substrings. The positions of the open paren next to the `B` will thus be the only position in the whole sequence on which the elevation drops below zero, and it will drop just to -1.

Read the next hint to get ideas for what `A` and `B` should be (the clue is in the attention pattern plot!).
</details>

<details>
<summary>Hint 4</summary>

From the attention pattern plot, we can see that left parens in the range 27-31 attend bizarrely strongly to the tokens at position 38-40. This means that, if there is a negative elevation in or after the range 27-31, then the left bracket that should be detecting this negative elevation might miscount. In particular, if `B = ((...))`, this left bracket might heavily count the right brackets at the end, and less heavily weight the left brackets at the start of `B`, thus this left bracket might "think" that the sequence is balanced when it actually isn't.
</details>

<details>
<summary>Solution (for best currently-known advex)</summary>

Choose `A` and `B` to each be a sequence of `(((...)))` terms with length $i$ and $38-i$ respectively (it makes sense to choose `A` like this also, because want the sequence to have maximal positive elevation everywhere except the single position where it's negative). Then, maximize over $i = 2, 4, ...\,$. Unsurprisingly given the observations in the previous hint, we find that the best adversarial examples (all with balanced probability of above 98%) are $i=24, 26, 28, 30, 32$. The best of these is $i=30$, which gets 99.9856% balanced confidence.

```python
def tallest_balanced_bracket(length: int) -> str:
    return "".join(["(" for _ in range(length)] + [")" for _ in range(length)])
    
example = tallest_balanced_bracket(15) + ")(" + tallest_balanced_bracket(4)
```

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/graph.png">

</details>



```python
# YOUR CODE HERE - update the examples list below, to find adversarial examples!
examples = ["()", "(())", "))"]
m = max(len(ex) for ex in examples)
toks = tokenizer.tokenize(examples)
probs = model(toks)[:, 0].softmax(-1)[:, 1]
print("\n".join([f"{ex:{m}} -> {p:.4%} balanced confidence" for (ex, p) in zip(examples, probs)]))
```

<details>
<summary>Solution</summary>


```python
    return "".join(["(" for _ in range(length)] + [")" for _ in range(length)])
example = tallest_balanced_bracket(15) + ")(" + tallest_balanced_bracket(4)
examples.append(example)

```
</details>


## Dealing with early closing parens

We mentioned that our model deals with early closing parens differently. One of our components in particular is responsible for classifying any sequence that starts with a closed paren as unbalnced - can you find the component that does this?

<details>
<summary>Hint</summary>

It'll have to be one of the attention heads, since these are the only things which can move information from sequence position 1 to position 0 (and the failure mode we're trying to detect is when the sequence has a closed paren in position 1).

Which of your attention heads was previously observed to move information from position 1 to position 0?
</details>

Can you plot the outputs of this component when there is a closed paren at first position? Can you prove that this component is responsible for this behavior, and show exactly how it happens?


## Try more algorithmic problems

*Note - this definitely isn't a short bonus exercise, and might serve better as a capstone project than something which can fit into 1 or 2 days.*

Interpreting toy models is a good way to increase your confidence working with TransformerLens and basic interpretability methods. It's maybe not the most exciting category of open problems in mechanistic interpretability, but can still be a useful exercise - and sometimes it can lead to interesting new insights about how interpretability tools can be used.

If you're feeling like it, you can try to hop onto LeetCode and pick a suitable problem (we recommend the "Easy" section) to train a transformer and interpret its output. A few suggestions:

* [Palindrome Number](https://leetcode.com/problems/palindrome-number/)
* [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
* Sorting a list
* [Majority Element](https://leetcode.com/problems/majority-element/)
* [Plus One](https://leetcode.com/problems/plus-one/)

Note that doing this projects will require you to write code for training a transformer. You can recycle code from the training loop we built in the "training a transformer" exercises. You'll also probably be able to reuse some of the code from `SimpleTokenizer` and `BracketsDataset` to suit your particular problem.

You can also get more ideas from Neel Nanda's [200 Concrete Open Problems in MI: The Case for Analysing Toy Language Models](https://www.alignmentforum.org/s/yivyHaCAmMJ3CqSyj/p/GWCgZrzWCZCuzGktv).




""", unsafe_allow_html=True)


func_page_list = [
    (section_0, "🏠 Home"),     (section_1, "1️⃣ Bracket classifier"),     (section_2, "2️⃣ Moving backwards"),     (section_3, "3️⃣ Understanding the total elevation circuit"),     (section_4, "4️⃣ Bonus exercises"), 
]

func_list = [func for func, page in func_page_list]
page_list = [page for func, page in func_page_list]

page_dict = dict(zip(page_list, range(len(page_list))))

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    idx = page_dict[radio]
    func = func_list[idx]
    func()

page()
