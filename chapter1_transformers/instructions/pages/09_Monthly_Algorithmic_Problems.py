
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

st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li><a class='contents-el' href='#prerequisites'>Prerequisites</a></li>
    <li><a class='contents-el' href='#motivation'>Motivation</a></li>
    <li><a class='contents-el' href='#logistics'>Logistics</a></li>
    <li><a class='contents-el' href='#what-counts-as-a-solution'>What counts as a solution?</a></li>
    <li><a class='contents-el' href='#setup'>Setup</a></li>
    <li><a class='contents-el' href='#task-dataset'>Task & Dataset</a></li>
    <li><a class='contents-el' href='#model'>Model</a></li>
</ul></li>""", unsafe_allow_html=True)

st.markdown(
r"""
<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/zoom.png" width="400">

[**Colab**](https://colab.research.google.com/drive/1qTUBj16kp6ZOCEBJefCKdzXvBsU1S-yz)

# Monthly Algorithmic Challenge (June 2023): Palindromes

This marks the first of the (hopefully sequence of) monthly mechanistic interpretability challenges. I designed them in the spirit of [Stephen Casper's challenges](https://www.lesswrong.com/posts/KSHqLzQscwJnv44T8/eis-vii-a-challenge-for-mechanists), but with the more specific aim of working well in the context of the rest of the ARENA material, and helping people put into practice all the things they've learned so far.

## Prerequisites

The following ARENA material should be considered essential:

* **[1.1] Transformer from scratch** (sections 1-3)
* **[1.2] Intro to Mech Interp** (sections 1-3)

The following material isn't essential, but is very strongly recommended:

* **[1.2] Intro to Mech Interp** (section 4)
* **[1.4] Balanced Bracket Classifier** (all sections)

## Motivation

Neel Nanda's post [200 COP in MI: Interpreting Algorithmic Problems](https://www.lesswrong.com/posts/ejtFsvyhRkMofKAFy/200-cop-in-mi-interpreting-algorithmic-problems) does a good job explaining the motivation behind solving algorithmic problems such as these. I'd strongly recommend reading the whole post, because it also gives some high-level advice for approaching such problems.

The main purpose of these challenges isn't to break new ground in mech interp, rather they're designed to help you practice using & develop better understanding for standard MI tools (e.g. interpreting attention, direct logit attribution), and more generally working with libraries like TransformerLens.

Also, they're hopefully pretty fun, because why shouldn't we have some fun while we're learning?

## Logistics

If this first problem is well-received, I'll try to post a new one every month. Because I think this one is on the easier side relatively speaking, I'll leave it open until the end of July (which at time of writing is 16 days). **My solution will be published on 31st July on this page**, at the same time as the next problem in the sequence. Future challenges will also be accompanied by a LessWrong post, but not this one (because it's experimental).

If you try to interpret this model, you can send your attempt in any of the following formats:

* Colab notebook
* GitHub repo (e.g. with ipynb or markdown file explaining results)
* Google Doc (with screenshots and explanations)
* or any other sensible format.

You can send your attempt to me (Callum McDougall) via any of the following methods:

* The [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), via a direct message to me
* My personal email: `cal.s.mcdougall@gmail.com`
* LessWrong message ([here](https://www.lesswrong.com/users/themcdouglas) is my user)

**I'll feature the names of everyone who sends me a solution on this website, and also give a shout out to the best solutions.** It's possible that future challenges will also feature a monetary prize, but this is not guaranteed.


## What counts as a solution?

Going through the exercises **[1.4] Balanced Bracket Classifier** should give you a good idea of what I'm looking for. This model is much less complicated than the one in that exercise, so I'd have a higher standard for what counts as a full solution. In particular, I'd expect you to:

* Describe a mechanism for how the model solves the task, in the form of the QK and OV circuits of various attention heads (and possibly any other mechanisms the model uses, e.g. the direct path, or nonlinear effects from layernorm),
* Provide evidence for your mechanism, e.g. with tools like attention probabilities, targeted ablation / patching, or direct logit attribution.
* (Optional) Include additional detail, e.g. identifying the linear subspaces that the model uses for certain forms of information transmission.

## Setup

```python
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from pathlib import Path

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "monthly_algorithmic_problems" / "july23_palindromes"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from monthly_algorithmic_problems.july23_palindromes.dataset import PalindromeDataset, display_seq
from monthly_algorithmic_problems.july23_palindromes.model import create_model
from plotly_utils import hist, bar, imshow

device = t.device("cuda" if t.cuda.is_available() else "cpu")
```

## Task & Dataset

The directory containing all the relevant files is `chapter1_transformers/exercises/monthly_algorithmic_problems/july23_palindromes`. This contains files `model.py` (for defining the model), `training.py` (for training the model), and `dataset.py` (for the dataset of palindromes and non-palindromes).

Each sequence in the dataset looks like:

```
[start_token, a_1, a_2, ..., a_N, end_token]
```

where `start_token = 31`, `end_token = 32`, and each value `a_i` is a value in the range `[0, 30]` inclusive. 

Each sequence has a corresponding label, which is `1` if the sequence is a palindrome (i.e. `(a_1, a_2, ..., a_N) == (a_N, ..., a_2, a_1)`), and `0` otherwise. The model has been trained to classify each sequence according to this label.

We've given you the class `PalindromeDataset` to store your data. You can slice this object to get batches of tokens and labels. You can also use the function `display_seq` to display a sequence in a more readable format (with any tokens that stop it from being a palindrome highlighted). There's an example later on this page. 

Some other useful methods and attributes of this dataset (you can inspect `dataset.py` to see for yourself) are:

* `dataset.toks`, to get a batch of all the tokens in the dataset, of shape `(size, 2 * half_length + 2)`.
* `dataset.is_palindrome`, to get a tensor of all the labels in the dataset, of shape `(size,)`.
* `dataset.str_toks`, to get a list of lists, with string representations of each sequence, e.g. `["START", "1", "4", ..., "END"]`. This is useful for visualisation, e.g. circuitsvis.

## Model

Our model was trained by minimising cross-entropy loss between its predictions and the true labels. You can inspect the notebook `training_model.ipynb` to see how it was trained.

The model is is a 2-layer transformer with 2 attention heads, and causal attention. It includes layernorm, but no MLP layers. You can load it in as follows:

```python
filename = section_dir / "palindrome_classifier.pt"

model = create_model(
    half_length=10, # this is half the length of the palindrome sequences
    max_value=30, # values in palindrome sequence are between 0 and max_value inclusive
    seed=42,
    d_model=28,
    d_head=14,
    n_heads=2,
    normalization_type="LN",
    d_mlp=None # this is an attn-only model
)

state_dict = t.load(filename)

state_dict = model.center_writing_weights(t.load(filename))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_layer_norm(state_dict)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False);
```

The code to process the state dictionary is a bit messy, but it's necessary to make sure the model is easy to work with. For instance, if you inspect the model's parameters, you'll see that `model.ln_final.w` is a vector of 1s, and `model.ln_final.b` is a vector of 0s (because the weight and bias have been folded into the unembedding).

```python
print("ln_final weight: ", model.ln_final.w)
print("\nln_final, bias: ", model.ln_final.b)
```

<details>
<summary>Aside - the other weight processing parameters</summary>

Here's some more code to verify that our weights processing worked, in other words:

* The unembedding matrix has mean zero over both its input dimension (`d_model`) and output dimension (`d_vocab`)
* All writing weights (i.e. `b_O`, `W_O`, and both embeddings) have mean zero over their output dimension (`d_model`)
* The value biases `b_V` are zero (because these can just be folded into the output biases `b_O`)

```python
W_U_mean_over_input = einops.reduce(model.W_U, "d_model d_vocab -> d_model", "mean")
t.testing.assert_close(W_U_mean_over_input, t.zeros_like(W_U_mean_over_input))

W_U_mean_over_output = einops.reduce(model.W_U, "d_model d_vocab -> d_vocab", "mean")
t.testing.assert_close(W_U_mean_over_output, t.zeros_like(W_U_mean_over_output))

W_O_mean_over_output = einops.reduce(model.W_O, "layer head d_head d_model -> layer head d_head", "mean")
t.testing.assert_close(W_O_mean_over_output, t.zeros_like(W_O_mean_over_output))

b_O_mean_over_output = einops.reduce(model.b_O, "layer d_model -> layer", "mean")
t.testing.assert_close(b_O_mean_over_output, t.zeros_like(b_O_mean_over_output))

W_E_mean_over_output = einops.reduce(model.W_E, "token d_model -> token", "mean")
t.testing.assert_close(W_E_mean_over_output, t.zeros_like(W_E_mean_over_output))

W_pos_mean_over_output = einops.reduce(model.W_pos, "position d_model -> position", "mean")
t.testing.assert_close(W_pos_mean_over_output, t.zeros_like(W_pos_mean_over_output))

b_V = model.b_V
t.testing.assert_close(b_V, t.zeros_like(b_V))
```

</details>

The model was trained to output the correct classification at the `END` token, in other words the value of the residual stream at `END` (post-layernorm) is mapped through `model.W_U` which has shape `(d_model, 2)`, and this gives us our classification logits for `(not palindrome, palindrome)`.

A demonstration of the model working (and of the `display_seq` function):

```python
dataset = PalindromeDataset(size=100, max_value=30, half_length=10)

toks, is_palindrome = dataset[:5]

logits = model(toks)[:, -1]
probs = logits.softmax(-1)
probs_palindrome = probs[:, 1]

for tok, prob in zip(toks, probs_palindrome):
    display_seq(tok, prob)
```

<details>
<summary>Click on this dropdown for a hint on how to start (and some example code).</summary>

The following code will display the attention patterns for each head, on a particular example.

```python
display_seq(dataset.toks[batch_idx], probs_palindrome[batch_idx])

import circuitsvis as cv

cv.attention.attention_patterns(
    attention = t.concat([cache["pattern", layer][batch_idx] for layer in range(model.cfg.n_layers)]),
    tokens = dataset.str_toks[batch_idx],
    attention_head_names = [f"{layer}.{head}" for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)],
)
```

Find (1) a palindromic example, and (2) a non-palindromic example which is close to being palindromic (i.e. only 1 or 2 tokens are different). Then, compare the attention patterns for these two examples. Questions you might want to answer:

* How do the attention patterns for numbers which are palindromic (i.e. they are the same as their mirror image) differ from the numbers which aren't?
* How does information eventually get to the `[END]` token?

</details>

Note - although this model was trained for long enough to get loss close to zero (you can test this for yourself), it's not perfect. There are some weaknesses that the model has which make it vulnerable to adversarial examples, which I've decided to leave in as a fun extra challenge! Note that the model is still very good at its intended task, and the main focus of this challenge is on figuring out how it solves the task, not dissecting the situations where it fails. However, you might find that the adversarial examples help you understand the model better.

Best of luck! 🎈

""", unsafe_allow_html=True)