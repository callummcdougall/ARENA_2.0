import os, sys
from pathlib import Path
chapter = r"chapter1_transformers"
for instructions_dir in [
    Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/instructions").resolve(),
    Path("/app/arena_2.0/chapter1_transformers/instructions").resolve(),
    Path("/mount/src/arena_2.0/chapter1_transformers/instructions").resolve(),
]:
    if instructions_dir.exists():
        break
if str(instructions_dir) not in sys.path: sys.path.append(str(instructions_dir))
os.chdir(instructions_dir)

import streamlit as st
import st_dependencies

st_dependencies.styling()

import platform
is_local = (platform.processor() != "")

import streamlit_analytics
streamlit_analytics.start_tracking()

def section_0_july():

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

    st.markdown(r"""
# Monthly Algorithmic Challenge (July 2023): Palindromes

### Colab: [problem](https://colab.research.google.com/drive/1qTUBj16kp6ZOCEBJefCKdzXvBsU1S-yz) | [solutions](https://colab.research.google.com/drive/1zJepKvgfEHMT1iKY3x_CGGtfSR2EKn40)

This marks the first of the (hopefully sequence of) monthly mechanistic interpretability challenges. I designed them in the spirit of [Stephen Casper's challenges](https://www.lesswrong.com/posts/KSHqLzQscwJnv44T8/eis-vii-a-challenge-for-mechanists), but with the more specific aim of working well in the context of the rest of the ARENA material, and helping people put into practice all the things they've learned so far.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/zoom.png" width="350">

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

Please don't discuss specific things you've found about this model until the challenge is over (although you can discuss general strategies and techniques, and you're also welcome to work in a group if you'd like). The deadline for this problem will be the end of this month, i.e. 31st July. If the challenge is well-received (which I'm arbitrarily defining as there being at least 5 submissions which I judge to be high-quality), then I'll make it a monthly sequence.

## What counts as a solution?

Going through the exercises **[1.4] Balanced Bracket Classifier** should give you a good idea of what I'm looking for. This model is much less complicated than the one in that exercise, so I'd have a higher standard for what counts as a full solution. In particular, I'd expect you to:

* Describe a mechanism for how the model solves the task, in the form of the QK and OV circuits of various attention heads (and possibly any other mechanisms the model uses, e.g. the direct path, or nonlinear effects from layernorm),
* Provide evidence for your mechanism, e.g. with tools like attention plots, targeted ablation / patching, or direct logit attribution.
* (Optional) Include additional detail, e.g. identifying the linear subspaces that the model uses for certain forms of information transmission.

# Setup

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
    
palindromes_dir = instructions_dir / "media/palindromes"
import plotly.graph_objects as go
from streamlit.components.v1 import html as st_html
import json

def section_1_july():
    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li><a class='contents-el' href='#0-hypotheses'>0. Hypotheses</a></li>
    <li><a class='contents-el' href='#1-eyeball-attention-patterns'>1. Eyeball attention patterns</a></li>
    <li><a class='contents-el' href='#2-head-ablations'>2. Head ablations</a></li>
    <li><a class='contents-el' href='#3-full-qk-matrix-of-head-0-0'>3. Full QK matrix of head <code>0.0</code></a></li>
    <li><a class='contents-el' href='#4-investigating-adversarial-examples'>4. Investigating adversarial examples</a></li>
    <li><a class='contents-el' href='#5-composition-of-0-0-and-1-0'>5. Composition of <code>0.0</code> and <code>1.0</code></a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#first-experiment-k-composition'>K-composition</a></li>
        <li><a class='contents-el' href='#second-experiment-v-composition'>V-composition</a></li>
    </ul></li>
    <br>
    <li><a class='contents-el' href='#a-few-more-experiments'>A few more experiments</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#targeted-ablation'>Targeted ablations</a></li>
        <li><a class='contents-el' href='#composition-scores'>Composition scores</a></li>
        <li><a class='contents-el' href='#how-is-the-non-palindromic-information-stored'>How is the "non-palindromic" information stored?</a></li>
    </ul></li>
</ul></li>""", unsafe_allow_html=True)
    
    st.markdown(r"""
# Monthly Algorithmic Challenge (July 2023): Solutions

We assume you've run all the setup code from the previous page "Palindromes Challenge". Here's all the new setup code you'll need:

```python
dataset = PalindromeDataset(size=2500, max_value=30, half_length=10).to(device)

logits, cache = model.run_with_cache(dataset.toks)

logprobs = logits[:, -1].log_softmax(-1)
probs = logprobs.softmax(-1)
probs_palindrome = probs[:, 1]

logprobs_correct = t.where(dataset.is_palindrome.bool(), logprobs[:, 1], logprobs[:, 0])
logprobs_incorrect = t.where(dataset.is_palindrome.bool(), logprobs[:, 0], logprobs[:, 1])
probs_correct = t.where(dataset.is_palindrome.bool(), probs[:, 1], probs[:, 0])

avg_logit_diff = (logprobs_correct - logprobs_incorrect).mean().item()
avg_cross_entropy_loss = -logprobs_correct.mean().item()
print(f"Average cross entropy loss: {avg_cross_entropy_loss:.3f}")
print(f"Average logit diff: {avg_logit_diff:.3f}")
```

<div style='font-family:monospace; font-size:15px;'>
Average cross entropy loss: 0.008<br>
Average logit diff: 7.489
</div><br>

Denote the vectors in the residual stream (other than `START` and `END`) as $\{x_1, x_2, ..., x_{20}\}$. Each $x_i = t_i + p_i$ (the token embedding plus positional embedding). We say that the $i$-th token is palindromic if $t_i = t_{20-i}$ (so the whole sequence is palindromic if and only if all $x_{11}, ..., x_{20}$ are palindromic). We'll sometimes use $x$ to refer to a token in the second half of the sequence, and $x'$ to that token's mirror image.

Rather than going for a "rational reconstruction", I've tried to present the evidence roughly in the order I found it, so this should give you one perspective on what the mech interp process can look like.

# 0. Hypotheses

It's a good idea to spend some time brainstorming hypotheses about how the model might go about solving the task. After thinking about it for a while, I came up with the following two hypotheses:

**1. Reflection**

Each token $x$ in the second half will attend back to $x'$ to get information about whether the two are equal. This information then gets moved to the `END` token.

If this is true, then I expect to see one or both of the layer-0 heads attending in a "lambda pattern" (thanks to Andy Arditi for this terminology), i.e. 20 attends to 1, 19 attends to 2, etc. In layer 1, I expect to see the `END` token attending to the tokens in the second half, where this information is stored. In particular, in the non-palindromic cases I expect to see `END` attending to the tokens in the second half which are non-palindromic (because it only takes one unbalanced pair for the sequence to be non-palindromic). We might expect `END` to attend to the `START` token in palindromic sequences, since it's a useful rest position.

**2. Aggregation**

The `END` token (or the last non-END token) attends uniformly to all tokens, and does some kind of aggregation like the brackets task (i.e. it stores information about whether each token is equal to its reflection). Then, nonlinear operations on the `END` token (self-attention from layer 1 and softmax) turn this aggregated information into a classification.

**Evaluation of these two hypotheses**

Aggregation seems much less likely, because it's not making use of any of the earlier sequence positions to store information, and it's also not making use of the model's QK circuit (i.e. half the model). Maybe aggregation would be more likely if we had MLPs, but not for an attention-only model. Reflection seems like a much more natural hypothesis.

# 1. Eyeball attention patterns

Both the hypotheses above would be associated with very distinctive attention patterns, which is why plotting attention patterns is a good first step here.

I've used my own circuitsvis code which sets up a selection menu to view multiple patterns at once, and I've also used a little HTML hacking to highlight the tokens which aren't palindromic (this is totally unnecessary, but made things a bit visually clearer for me!).

```python
def red_text(s: str):
    return f"<span style='color:red'>{s}</span>"


def create_str_toks_with_html(toks: Int[Tensor, "batch seq"]):
    '''
    Creates HTML which highlights the tokens that don't match their mirror images. Also puts a gap 
    between each token so they're more readable.
    '''
    raw_str_toks = [["START"] + [f"{t:02}" for t in tok[1:-1]] + ["END"] for tok in toks]

    toks_are_palindromes = toks == toks.flip(-1)
    str_toks = []
    for raw_str_tok, palindromes in zip(raw_str_toks, toks_are_palindromes):
        str_toks.append([
            "START - ", 
            *[f"{s} - " if p else f"{red_text(s)} - " for s, p in zip(raw_str_tok[1:-1], palindromes[1:-1])], 
            "END"
        ])
    
    return str_toks


cv.attention.from_cache(
    cache = cache,
    tokens = create_str_toks_with_html(dataset.toks),
    batch_idx = list(range(10)),
    attention_type = "info-weighted",
    radioitems = True,
)
```
""", unsafe_allow_html=True)
    
    with open(palindromes_dir / "fig1.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=625)

    st.markdown(r"""
## Conclusions

* The reflection hypotheses seems straightforwardly correct.
* Head 0.0 is impelenting the "lambda pattern".
    * We can see that $x$ attends back to $x'$ if they're the same, otherwise it usually self-attends.
    * This suggests the quantity $(x - x')^T W_{OV}^{0.0}$ might be important (this is the difference between the vector which is added at $x$ when $x$ is palindromic vs. non-palindromic, ignoring layernorm and assuming attention is always either 0 or 1). I'll return to this later.
* Head 0.1 isn't really doing this, or anything distinctive - maybe this head isn't important?
    * Head 0.1 is actually doing something important at $x = 20$, but I didn't spot this at the time.
* Head 1.0 is implementing the "attend to non-palindromic tokens in the second half" pattern.
    * Although one part of my hypothesis was false - `START` doesn't seem like it's used as an attention placeholder for palindromic sequences.
    * The attention patterns from `END` to other tokens seem pretty random in palindromic sequences.
    * This suggests we might be seeing V-composition between heads 0.0 and 1.0 (otherwise the signal which 1.0 is picking up on in non-palindromic sequences would also be picked up in palindromic sequences, and the model wouldn't work).
* Head 1.1 is attending to $x_{20}$ when it's non-palindromic.
    * Maybe it's doing this to compensate for head 1.0, which never seems to attend to $x_{20}$.

## Other notes

* Using info-weighted attention is a massive win here. In particular, it makes the behaviour of head 1.0 a lot clearer than just using regular attention.

## Next experiments to run

* I think 0.1 and 1.1 are unimportant - to test this I should ablate them and see if loss changes. If not, then I can zoom in on 0.0 and 1.0 for most of the rest of my analysis.
* 0.0 seems to be implementing a very crisp attention pattern - I should look at the full QK circuit to see how this is implemented.
* After these two experiments (assuming the evidence from them doesn't destroy any of my current hypotheses), I should try and investigate how 0.0 and 1.0 are composing.

# 2. Head ablations

I want to show that heads 0.1 and 1.1 don't really matter, so I'm going to write code to ablate them and see how the loss changes.

Note, I'm ablating the head's result vector (because this makes sure we ablate both the QK and OV circuit signals). On larger models we might have to worry about storing `result` in our cache, but this is a very small model so we don't need to worry about that here.
                
```python
def get_loss_from_ablating_head(layer: int, head: int, ablation_type: Literal["zero", "mean"]):

    def hook_patch_result_mean(result: Float[Tensor, "batch seq nheads d_model"], hook: HookPoint):
        '''
        Ablates an attention head (either mean or zero ablation).
        
        Note, when mean-ablating we don't average over sequence positions. Can you see why this is important?
        (You can return here after you understand the full algorithm implemented by the model.)
        '''
        if ablation_type == "mean":
            result_mean: Float[Tensor, "d_model"] = cache["result", layer][:, :, head].mean(0)
            result[:, :, head] = result_mean
        elif ablation_type == "zero":
            result[:, :, head] = 0
        return result

    model.reset_hooks()
    logits = model.run_with_hooks(
        dataset.toks,
        fwd_hooks = [(utils.get_act_name("result", layer), hook_patch_result_mean)],
    )[:, -1]
    logits_correct = t.where(dataset.is_palindrome.bool(), logits[:, 1], logits[:, 0])
    logits_incorrect = t.where(dataset.is_palindrome.bool(), logits[:, 0], logits[:, 1])
    avg_logit_diff = (logits_correct - logits_incorrect).mean().item()
    return avg_logit_diff
    


print(f"Original logit diff = {avg_logit_diff:.3f}")

for ablation_type in ["mean", "zero"]:
    print(f"\nNew logit diff after {ablation_type}-ablating head...")
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            print(f"...{layer}.{head} = {get_loss_from_ablating_head(layer, head, ablation_type):.3f}")
```

<div style='font-family:monospace; font-size:15px;'>
Original logit diff = 7.489<br><br>
New logit diff after mean-ablating head...<br>
...0.0 = 0.614<br>
...0.1 = 6.642<br>
...1.0 = 0.299<br>
...1.1 = 6.815<br><br>
New logit diff after zero-ablating head...<br>
...0.0 = 1.672<br>
...0.1 = 3.477<br>
...1.0 = 2.847<br>
...1.1 = 7.274<br>
</div><br>

Mean ablation shows us that heads 0.1 and 1.1 aren't crucial. Interestingly, zero-ablation would lead us to believe (incorrectly) that head 0.1 is very important. This is a common problem, especially with early heads (because zero-ablating these heads output will be moving the input of later heads off-distribution).

At this point I thought that 1.1 was doing something important at position 20, but decided not to investigate it yet, because looking more into 0.0 and 1.0 seemed like it should tell me most of what I wanted to know about this model.

# 3. Full QK matrix of head 0.0

I wanted to see what the full QK matrices of the heads looked like. I generated them for both heads in layer 0, and also for heads in layer 1 (but I guessed these wouldn't tell me as much, because composition would play a larger role in these heads' input, hence I don't show the layer-1 plots below).

In the attention scores plot, I decided to concatenate the embedding and positional embedding matrices, so I could see all interactions between embeddings and positional embeddings. The main reason I did this wasn't for the cross terms (I didn't expect to learn much from seeing how much token $t_i$ attends to position $p_j$), but just so that I could see all the $(t_i, t_j)$ terms next to the $(p_i, p_j)$ terms in a single plot (and compare them to see if positions or tokens had a larger effect on attention scores).

```python
W_QK: Float[Tensor, "layers heads d_model d_model"] = model.W_Q @ model.W_K.transpose(-1, -2)

W_E_pos = t.concat([model.W_E, model.W_pos], dim=0)

W_QK_full = W_E_pos @ W_QK @ W_E_pos.T

d_vocab = model.cfg.d_vocab
n_ctx = model.cfg.n_ctx
assert W_QK_full.shape == (2, 2, d_vocab + n_ctx, d_vocab + n_ctx)

# More use of HTML to increase readability - plotly supports some basic HTML for titles and axis labels
W_E_labels = [f"W<sub>E</sub>[{i}]" for i in list(range(d_vocab - 2)) + ["START", "END"]]
W_pos_labels = [f"W<sub>pos</sub>[{i}]" for i in ["START"] + list(range(1, n_ctx - 1)) + ["END"]]

imshow(
    W_QK_full.flatten(0, 1),
    title = "Full QK matrix for different heads (showing W<sub>E</sub> and W<sub>pos</sub>)",
    x = W_E_labels + W_pos_labels,
    y = W_E_labels + W_pos_labels,
    labels = {"x": "Source", "y": "Dest"},
    facet_col = 0,
    facet_labels = ["0.0", "0.1", "1.0", "1.1"],
    height = 1000,
    width = 1900,
)
```
""", unsafe_allow_html=True)
    
    fig2 = go.Figure(json.loads(open(palindromes_dir / "fig2.json", 'r').read()))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(r"""
## Conclusions

* As expected, 0.0 had the most distinctive patterns.
    * The strongest pattern was in the token embeddings - each token attends to itself
    * There was a weaker pattern in the positional embeddings - each position in the second half attends to either itself or its mirror (usually with a preference for its mirror)
    * Combining these 2 heuristics, we can see the algorithm that is implemented is the same as the pattern we observed earlier:
        * *If $x$ and $x'$ are the same, then $x$ attends back to $x'$. Otherwise, $x$ self-attends.*
* Head 0.1 has pretty clear patterns too.
    * Oddly, each token anti self-attends.
    * Position 20 (and 18 to a lesser extent) have high attention scores to themselves and their mirror position.
    * Combining these two pieces of evidence, I would guess that 0.1 is doing the same thing as 0.0 is doing, but in reverse, and only for the token pairs $(1, 20)$ and $(3, 18)$:
        * *If $x$ and $x'$ are different, then $x$ attends back to $x'$. Otherwise, $x$ attends to both $x$ and $x'$.*
    * I would guess that this signal is being used in the same way as the signal from 0.0, but in the opposite direction.
    * Going back to the attention patterns at the top, we can see that this is happening for token 20 (not for 18).

To make the plots for head 0.0 clearer, I plotted $W_E$ and $W_{pos}$ for head 0.0 separately, and after softmaxing (note I apply a causal mask for positions, not for tokens). Because I wanted to see if any nonlinear trickery was happening with the layernorm in layer zero, I checked the standard deviation of layernorm at each sequence position - it was very small, meaning this kind of plot is reasonable.

Note that applying softmax can sometimes be misleading, because it extremises patterns and risks making them appear too clean. 

```python
# Check layernorm scale factor mean & std dev, verify that std dev is small
scale = cache["scale", 0, "ln1"][:, :, 0, 0] # shape (batch, seq)
df = pd.DataFrame({
    "std": scale.std(0).cpu().numpy(),
    "mean": scale.mean(0).cpu().numpy(),
})
px.bar(
    df, 
    title="Mean & std of layernorm before first attn layer", 
    template="simple_white", width=600, height=400, barmode="group"
).show()


# Get full matrix for tokens (we take mean over entire LN scale)
W_QK = model.W_Q[0, 0] @ model.W_K[0, 0].T / (model.cfg.d_head ** 0.5)
W_E_scaled = model.W_E[:-2] / scale.mean()
W_QK_full_tokens = W_E_scaled @ W_QK @ W_E_scaled.T

# Get full matrix for tokens (here, we can preserve the seq dim in `scale`)
W_pos_scaled = model.W_pos[1:-1] / scale[:, 1:-1].mean(dim=0).unsqueeze(-1)
# Scale by sqrt(d_head)
W_QK_full_pos = W_pos_scaled @ W_QK @ W_pos_scaled.T
# Apply causal mask 
W_QK_full_pos.masked_fill_(~t.tril(t.ones_like(W_QK_full_pos)).bool(), -1e6)

# Plot both
for (name, matrix) in zip(["tokens", "positions"], [W_QK_full_tokens, W_QK_full_pos]):
    imshow(
        matrix.softmax(-1),
        title = f"Full QK matrix for 0.0 ({name})",
        x = W_pos_labels[1:-1] if name == "positions" else W_E_labels[:-2],
        y = W_pos_labels[1:-1] if name == "positions" else W_E_labels[:-2],
        labels = {"x": "Source", "y": "Dest"},
        height = 800,
        width = 800,
    )
```
""", unsafe_allow_html=True)
    
    fig3 = go.Figure(json.loads(open(palindromes_dir / "fig3.json", 'r').read()))
    fig4_tokens = go.Figure(json.loads(open(palindromes_dir / "fig4_tokens.json", 'r').read()))
    fig4_positions = go.Figure(json.loads(open(palindromes_dir / "fig4_positions.json", 'r').read()))
    st.plotly_chart(fig3, use_container_width=False)
    st.plotly_chart(fig4_tokens, use_container_width=False)
    st.plotly_chart(fig4_positions, use_container_width=False)

    st.markdown(r"""
Result - we can clearly see the pattern that was observed earlier. However, some results aren't as clean as I was expecting (in particular the positional results). The blind spots at positions 17 and 19 are very apparent here.

# 4. Investigating adversarial examples

Before looking at composition between 0.0 and 1.0, I'm going to take a look at the blind spots at 17 and 19, and see if I can generate some adversarial examples. I'll construct strings where only one pair is non-palindromic, and look at the classification probabilities.

```python
# Pick a string to start with, check it's palindromic
batch_idx = 1
assert dataset.is_palindrome[batch_idx].item() == 1

# Create my adversarial examples (with some non-adversarial examples as a baseline)
test_toks = {None: dataset.toks[batch_idx].clone()}
for i in [17, 18, 19, 20]:
    test_toks[i] = dataset.toks[batch_idx].clone()
    test_toks[i][i] += 1
test_toks = t.stack(list(test_toks.values()))

test_logits, test_cache = model.run_with_cache(test_toks)
test_probs = test_logits[:, -1].softmax(-1)
test_probs_balanced = test_probs[:, 1]

for k, v in zip([None, 17, 18, 19, 20], test_probs_balanced):
    print(f"{k} flipped, P(palindrome) = {v:.3f}")

cv.attention.from_cache(
    cache = test_cache,
    tokens = create_str_toks_with_html(test_toks),
    attention_type = "info-weighted",
    radioitems = True,
)
```
""", unsafe_allow_html=True)
    
    with open(palindromes_dir / "fig5.html", 'r') as f: fig5 = f.read()
    st_html(fig5, height=525)

    st.markdown(r"""
## Conclusion

This is exactly what I expected - 17 and 19 are adversarial examples. When only one of these positions is non-palindromic, the model will incorrectly classify the sequence as palindromic with high probability.

We can investigate further by looking at all the advexes in the dataset, and seeing how many of them are of this form. The results show that 2/3 of the "natural advexes" are of this form. Also, every single one of the "type 17/19 sequences" (i.e. the ones which are only non-palindromic at positions 17 or 19) are advexes.

<details>
<summary>A note on why these advexes exist</summary>

The way non-palindromic sequences are generated in the dataset is as follows: a random subset of tokens in the second half are chosen to be non-palindromic, with the size of this subset having a $\operatorname{Binomial}(10, 1/2)$ distribution (i.e. each token was randomly chosen to be palindromic or non-palindromic). This means that, for any small subset, the probability that a sequence is only non-palindromic within that subset is pretty small - hence adversarial examples can easily form.

Two exercises to the reader:

* What is the probability of a sequence generated in this way being non-palindromic only within the subset $\{17, 19\}$?
* How could you change the data generation process to make it harder for adversarial examples like these to form?

</details>

```python
is_advex = (probs_correct < 0.5)

is_palindromic_per_token = (dataset.toks == dataset.toks.flip(-1))
advex_indices = [17, 19]
non_advex_indices = [i for i in range(11, 21) if i not in advex_indices]

is_palindrome_at_non_advex = t.all(is_palindromic_per_token[:, non_advex_indices], dim=-1)
is_17_or_19_type = is_palindrome_at_non_advex & t.any(~is_palindromic_per_token[:, advex_indices], dim=-1)

print(f"Number of advexes which are in the 17/19 category:    {(is_17_or_19_type & is_advex).sum()}")
print(f"Number of advexes which aren't in the 17/19 category: {(~is_17_or_19_type & is_advex).sum()}")
print(f"Number of type-17/19 which aren't advexes:            {(is_17_or_19_type & ~is_advex).sum().item()}")

print("\nAdversarial examples:")
from IPython.display import display, HTML
display(HTML("<br>".join(["".join(x) for x in create_str_toks_with_html(dataset.toks[is_advex])])))
```

Result:

<div style='font-family:times-new-roman;'>
START - 13 - <span style='color:red'>30</span> - 18 - <span style='color:red'>17</span> - 25 - 23 - 11 - 24 - 09 - 01 - 01 - 09 - 24 - 11 - 23 - 25 - <span style='color:red'>20</span> - 18 - <span style='color:red'>11</span> - 13 - END<br>START - <span style='color:red'>27</span> - 12 - 23 - 27 - 21 - 25 - 24 - 24 - 25 - 15 - 15 - 25 - 24 - 24 - 25 - 21 - 27 - 23 - 12 - <span style='color:red'>23</span> - END<br>START - 23 - <span style='color:red'>05</span> - 06 - <span style='color:red'>02</span> - 24 - 18 - 18 - 13 - 19 - 23 - 23 - 19 - 13 - 18 - 18 - 24 - <span style='color:red'>05</span> - 06 - <span style='color:red'>30</span> - 23 - END
</div>

<br>

# 5. Composition of 0.0 and 1.0

This is the final big question that needs to be answered - how are `0.0` and `1.0` composing to give us the actual result?

Here, we return to the quantity $(x - x')^T W_{OV}^{0.0}$ discussed earlier, and I justify my choice of this vector.

Suppose each $x$ attends to $(x, x')$ with probability $(p_1, p_1')$ respectively when $x$ is palindromic, and $(p_2, p_2')$ when $x$ is non-palindromic (so we expect $p_1 + p_1' \approx 1, p_2 + p_2' \approx 1$ in most cases, and $p_2 > p_1$). This means that the vector added to $x$ is $p_2 x^T W_{OV}^{0.0} + p_2' x'^T W_{OV}^{0.0}$ in the non-palindromic case, and $p_1 x^T W_{OV}^{0.0} + p_1' x'^T W_{OV}^{0.0}$ in the palindromic case. The difference between these two vectors is:

$$
((p_2 - p_1) x - (p_1' - p_2') x')^T W_{OV}^{0.0} \approx (p_2 - p_1) (x - x')^T W_{OV}^{0.0}
$$

where I've used the approximations $p_1 + p_1' \approx 1, p_2 + p_2' \approx 1$. This is a positive mulitple of the thing we've defined as our "difference vector". Therefore, it's natural to guess that the "this token is non-palindromic" information is stored in the direction defined by this vector.

First, we should check that both $p_2 - p_1$ and $p_1' - p_2'$ are consistently positive (this definitely looked like the case when we eyeballed attention patterns, but we'd ideally like to be more careful).

Note - the plot that I'm making here is a box plot, which I don't have code for in `plotly_utils`. When there's a plot like this which I find myself wanting to make, I usually defer to using ChatGPT (creating quick and clear visualisations is one of the main ways I use it in my regular workflow).

```python
second_half_indices = list(range(11, 21))
first_half_indices = [21-i for i in second_half_indices]
base_dataset = PalindromeDataset(size=1000, max_value=30, half_length=10).to(device)

# Get a set of palindromic tokens & non-palindromic tokens (with the second half of both tok sequences the same)
palindromic_tokens = base_dataset.toks.clone()
palindromic_tokens[:, 1:11] = palindromic_tokens[:, 11:21].flip(-1)
nonpalindromic_tokens = palindromic_tokens.clone()
# Use some modular arithmetic to make sure the sequence I'm creating is fully non-palindromic
nonpalindromic_tokens[:, 1:11] += t.randint_like(nonpalindromic_tokens[:, 1:11], low=1, high=30)
nonpalindromic_tokens[:, 1:11] %= 31

# Run with cache, and get attention differences
_, cache_palindromic = model.run_with_cache(palindromic_tokens, return_type=None)
_, cache_nonpalindromic = model.run_with_cache(nonpalindromic_tokens, return_type=None)
p1 = cache_palindromic["pattern", 0][:, 0, second_half_indices, second_half_indices] # [batch seqQ]
p1_prime = cache_palindromic["pattern", 0][:, 0, second_half_indices, first_half_indices] # [batch seqQ]
p2 = cache_nonpalindromic["pattern", 0][:, 0, second_half_indices, second_half_indices] # [batch seqQ]
p2_prime = cache_nonpalindromic["pattern", 0][:, 0, second_half_indices, first_half_indices] # [batch seqQ]

fig_names = ["fig6a", "fig6b"]

for diff, title in zip([p2 - p1, p1_prime - p2_prime], ["p<sub>2</sub> - p<sub>1</sub>", "p<sub>1</sub>' - p<sub>2</sub>'"]):
    fig = go.Figure(
        data = [
            go.Box(y=utils.to_numpy(diff[:, i]), name=f"({j1}, {j2})", boxpoints='suspectedoutliers')
            for i, (j1, j2) in enumerate(zip(first_half_indices, second_half_indices))
        ],
        layout = go.Layout(
            title = f"Attn diff: {title}",
            template = "simple_white",
            width = 800,
        )
    ).add_hline(y=0, opacity=1.0, line_color="black", line_width=1)
    fig.show()
    print(f"Avg diff (over non-adversarial tokens) = {diff[:, [i for i in range(10) if i not in [17-11, 19-11]]].mean():.3f}")
```
""", unsafe_allow_html=True)
    
    fig6a = go.Figure(json.loads(open(palindromes_dir / "fig6a.json", 'r').read()))
    fig6b = go.Figure(json.loads(open(palindromes_dir / "fig6b.json", 'r').read()))
    st.plotly_chart(fig6a, use_container_width=False)
    st.markdown(r"""<div style='font-family:monospace; font-size:15px;'>Avg diff (over non-adversarial tokens) = 0.373</div><br>""", unsafe_allow_html=True)
    st.plotly_chart(fig6b, use_container_width=False)
    st.markdown(r"""<div style='font-family:monospace; font-size:15px;'>Avg diff (over non-adversarial tokens) = 0.544</div><br>""", unsafe_allow_html=True)
    
    st.markdown(r"""
## Conclusion

Yep, it looks like this "attn diff" does generally separate palindromic and non-palindromic tokens very well. Also, remember that in most non-palindromic sequences there will be more than one non-palindromic token, so we don't actually need perfect separation most of the time. We'll use the conservative figure of $0.373$ as our coefficient when we perform logit attribution later.

A quick sidenote - when we add back in adversarial positions 17 & 19, the points are no longer cleanly separate. We can verify that in head `1.0`, the `END` token never attends to positions 17 & 19 (which makes sense, if these tokens don't contain useful information). Code showing this is below.

```python
layer_1_head = 0

tokens_are_palindromic = (dataset.toks == dataset.toks.flip(-1)) # (batch, seq)
attn = cache["pattern", 1][:, layer_1_head, -1] # (batch, src token)

attn_palindromes = [attn[tokens_are_palindromic[:, i], i].mean().item() for i in range(attn.shape[1])]
attn_nonpalindromes = [attn[~tokens_are_palindromic[:, i], i].mean().item() for i in range(attn.shape[1])]

bar(
    [attn_palindromes, attn_nonpalindromes], 
    names=["Token is palindromic", "Token is non-palindromic"],
    barmode="group",
    width=800,
    title=f"Average attention from END to other tokens, in head 1.{layer_1_head}",
    labels={"index": "Source position", "variable": "Token type", "value": "Attn"}, 
    template="simple_white",
    x=["START"] + list(map(str, range(1, 21))) + ["END"],
    xaxis_tickangle=-45,
)
```
""", unsafe_allow_html=True)
    
    fig7 = go.Figure(json.loads(open(palindromes_dir / "fig7.json", 'r').read()))
    st.plotly_chart(fig7, use_container_width=False)

    st.markdown(r"""
Another thing which this plot makes obvious is that position 20 is rarely attended to by head 1.0 (explaining the third advex we found above). However, if you look at the attention patterns for head 1.1, you can see that it picks up the slack by attending to position 20 a lot, especially for non-palindromes.

## Next steps

We want to try and formalize this composition between head 0.0 and 1.0. We think that K-composition (and possibly V-composition) is going on.

**Question - do you think this is more likely to involve positional information or token information?**

<details>
<summary>Answer</summary>

It's more likely to involve positional information.

From the model's perspective, if $x$ and $x'$ are different tokens, it doesn't matter if they're $24, 25$ or $25, 24$ - it's all the same. But the positional information which gets moved from $x' \to x$ will always be the same for each $x$, and same for the information which gets moved from $x \to x$. So it's more likely that the model is using that.

This means we should replace our quantity $(x - x')^T W_{OV}^{0.0}$ with $(p - p')^T W_{OV}^{0.0}$ (where $p$ is the positional vector). 

When it comes to layernorm, we can take the mean of the scale factors over the batch dimension, but preserve the seq dimension. We'll denote $\hat{p}$ and $\hat{p}'$ as the positional vectors after applying layernorm. Then we'll call $v_i = (\hat{p}_i - \hat{p}'_i)^T W_{OV}^{0.0}$ the "difference vector" for the $i$th token (where $i$ is a sequence position in the second half).

</details>

Let's use this to set up an experiment. We want to take this "difference vector" $v_i$, and show that (at least for the non-adversarial token positions $i$), this vector is associated with:

* Increasing the attention from `END` to itself (i.e. K-composition)
* Pushing for the "unbalanced" prediction when it's attended to (i.e. V-composition)

## First experiment: K-composition

For each of these difference vectors, we can compute the corresponding keys for head 1.0, and we can also get the query vectors from the `END` token and measure their cosine similarity. For the non-adversarial tokens, we expect a very high cosine similarity, indicating that the model has learned to attend from the `END` token back to any non-palindromic token in the second half.

There are advantages and disadvantages of using cosine similarity. The main disadvantage is that it doesn't tell you anything about magnitudes. The main advantage is that, by normalizing for scale, the information you get from it is more immediately interpretable (because you can use baselines such as "all cosine sims are between 0 and 1" and "the expected value of the cosine sim of two random vectors in N-dimensional space is zero, with a standard deviation of $\sqrt{1/N}$").

```python
def get_keys_and_queries(layer_1_head: int):

    scale0 = cache["scale", 0, "ln1"][:, :, 0].mean(0) # [seq 1]
    W_pos_scaled = model.W_pos / cache["scale", 0, "ln1"][:, :, 0].mean(0) # [seq d_model]

    W_pos_diff_vectors = W_pos_scaled[second_half_indices] - W_pos_scaled[first_half_indices] # [half_seq d_model]
    difference_vectors = W_pos_diff_vectors @ model.W_V[0, 0] @ model.W_O[0, 0] # [half_seq d_model]

    scale1 = cache["scale", 1, "ln1"][:, second_half_indices, layer_1_head].mean(0) # [half_seq 1]
    difference_vectors_scaled = difference_vectors / scale1 # [half_seq d_model]
    all_keys = difference_vectors_scaled @ model.W_K[1, layer_1_head] # [half_seq d_head]

    # Averaging queries over batch dimension (to make sure we're not missing any bias terms)
    END_query = cache["q", 1][:, -1, layer_1_head].mean(0) # [d_head]

    # Get the cosine similarity
    all_keys_normed = all_keys / all_keys.norm(dim=-1, keepdim=True)
    END_query_normed = END_query / END_query.norm()
    cos_sim = all_keys_normed @ END_query_normed

    assert cos_sim.shape == (10,)
    return cos_sim


cos_sim_L1H0 = get_keys_and_queries(0)
cos_sim_L1H1 = get_keys_and_queries(1)

imshow(
    t.stack([cos_sim_L1H0, cos_sim_L1H1]),
    title = "Cosine similarity between difference vector keys and END query",
    width = 850,
    height = 400,
    x = [f"({i}, {j})" for i, j in zip(first_half_indices, second_half_indices)],
    y = ["1.0", "1.1"],
    labels = {"x": "Token pair", "y": "Head"},
    text_auto = ".2f",
)
```
""", unsafe_allow_html=True)
    
    fig8 = go.Figure(json.loads(open(palindromes_dir / "fig8.json", 'r').read()))
    st.plotly_chart(fig8, use_container_width=False)

    st.markdown(r"""
## Conclusion

These results are very striking. We make the following conclusions:

* As expected, for most tokens in the second half, head 1.0 will attend more to any token which attended to itself in head 0.0.
* The exceptions are 17 & 19 (the adversarial tokens we observed earlier) and 20 (which we saw was a blind spot of head 1.0 when we looked at attention patterns earlier).
* Head 1.0 tries to compensate for the blind spots at these sequence positions, it does a particularly good job at position 20.

## Second experiment: V-composition

Let's look at the direct logit attribution we get when we feed this difference vector through the OV matrix of heads in layer 1. We can re-use a lot of our code from the previous function.

```python
def get_DLA(layer_1_head: int):

    W_pos_scaled = model.W_pos / cache["scale", 0, "ln1"][:, :, 0].mean(0)

    W_pos_diff_vectors = W_pos_scaled[second_half_indices] - W_pos_scaled[first_half_indices] # [half_seq d_model]
    difference_vectors = W_pos_diff_vectors @ model.W_V[0, 0] @ model.W_O[0, 0] # [half_seq d_model]

    # This is the average multiple of this vector that gets added to the non-palindromic tokens relative to the
    # palindromic tokens (from the experiment we ran earlier)
    difference_vectors *= 0.373

    scale1 = cache["scale", 1, "ln1"][:, second_half_indices, layer_1_head].mean(0) # [half_seq 1]
    difference_vectors_scaled = difference_vectors / scale1
    all_outputs = difference_vectors_scaled @ model.W_V[1, layer_1_head] @ model.W_O[1, layer_1_head]

    # Scale & get direct logit attribution
    final_ln_scale = cache["scale"][~dataset.is_palindrome.bool(), -1].mean()
    all_outputs_scaled = all_outputs / final_ln_scale
    logit_attribution = all_outputs_scaled @ model.W_U
    # Get logit diff (which is positive for the "non-palindrome" classification)
    logit_diff = logit_attribution[:, 0] - logit_attribution[:, 1]

    return logit_diff


dla_L1H0 = get_DLA(0)
dla_L1H1 = get_DLA(1)
dla_L1 = t.stack([dla_L1H0, dla_L1H1])

imshow(
    dla_L1,
    title = "Direct logit attribution for the path W<sub>pos</sub> 'difference vectors' ➔ 0.0 ➔ (1.0 & 1.1) ➔ logits",
    width = 850,
    height = 400,
    x = [f"({i}, {j})" for i, j in zip(first_half_indices, second_half_indices)],
    y = ["1.0", "1.1"],
    labels = {"x": "Token pair", "y": "Head"},
    text_auto = ".2f",
)
```
""", unsafe_allow_html=True)
    
    fig9 = go.Figure(json.loads(open(palindromes_dir / "fig9.json", 'r').read()))
    st.plotly_chart(fig9, use_container_width=False)

    st.markdown(r"""
## Conclusions

* The results for head 1.0 agree with our expectation. The values in the 3 adversarial cases don't matter because `END` never pays attention to these tokens.
* The results for head 1.1 show us that this head compensates for the blind spot at position 20, but not at positions 17 or 19.
* The sizes of DLAs look about reasonable - in particular, the size of DLA for head 1.0 on all the non-adversarial positions is only a bit larger than the empirically observed logit diff (which is about 7.5 - see code cell below), which makes sense given that head 1.0 will usually pay very large (but not quite 100%) attention to non-palindromic tokens in the second half of the sequence, conditional on some non-palindromic tokens existing.

<br>

# A few more experiments

I consider the main problem to have basically been solved now, but here are a few more experiments we can run that shed more light on the model.

## Targeted ablations

Our previous results suggested that both 0.1 and 1.1 seem to compensate for blind spots at position 20. We should guess that mean ablating them everywhere except at position 20 shouldn't change the loss by much at all.

In the case of head 0.1, we should mean ablate the result everywhere except position 20 (because it's the output at this position that we care about). In the case of head 1.1, we should mean ablate the value vectors everywhere except position 20 (because it's the input at this position that we care about).

Note - in this case we're measuring loss rather than logit diff. This is because the purpose of heads 0.1 and 1.1 is to fix the model's blind spots, not to increase logit diff overall. It's entirely possible for a head to decrease loss and increase logit diff (in fact this is what we see for head 1.1).

```python
def targeted_mean_ablation_loss(
    head: Tuple[int, int],
    ablation_type: Literal["input", "output"],
    ablate_20: bool
):

    # Get values for doing mean ablation everywhere (except possibly position 20)
    layer, head_idx = head
    component = "result" if ablation_type == "output" else "v"
    seq_pos_to_ablate = slice(None) if ablate_20 else [i for i in range(22) if i != 20]
    ablation_values = cache[component, layer][:, seq_pos_to_ablate, head_idx].mean(0) # [seq d_model]

    # Define hook function
    def hook_patch_mean(activation: Float[Tensor, "batch seq nheads d"], hook: HookPoint):
        activation[:, seq_pos_to_ablate, head_idx] = ablation_values
        return activation

    # Run hooked forward pass
    model.reset_hooks()
    logits = model.run_with_hooks(
        dataset.toks,
        fwd_hooks = [(utils.get_act_name(component, layer), hook_patch_mean)],
    )
    logprobs = logits[:, -1].log_softmax(-1)
    logprobs_correct = t.where(dataset.is_palindrome.bool(), logprobs[:, 1], logprobs[:, 0])
    return -logprobs_correct.mean().item()


print(f"Original loss                           = {avg_cross_entropy_loss:.3f}\n")
print(f"0.1 ablated everywhere (incl. posn 20)  = {targeted_mean_ablation_loss((0, 1), 'output', ablate_20=True):.3f}")
print(f"0.1 ablated everywhere (except posn 20) = {targeted_mean_ablation_loss((0, 1), 'output', ablate_20=False):.3f}\n")
print(f"1.1 ablated everywhere (incl. posn 20)  = {targeted_mean_ablation_loss((1, 1), 'input', ablate_20=True):.3f}")
print(f"1.1 ablated everywhere (except posn 20) = {targeted_mean_ablation_loss((1, 1), 'input', ablate_20=False):.3f}")
```

<div style='font-family:monospace; font-size:15px;'>
Original loss                           = 0.008<br>
<br>
0.1 ablated everywhere (incl. posn 20)  = 0.118<br>
0.1 ablated everywhere (except posn 20) = 0.010<br>
<br>
1.1 ablated everywhere (incl. posn 20)  = 0.014<br>
1.1 ablated everywhere (except posn 20) = 0.008
</div><br>

## Composition scores

You can also measure composition scores (see the fourth section of [Intro to Mech Interp](https://arena-ch1-transformers.streamlit.app/[1.2]_Intro_to_Mech_Interp) for more details on what these are). Also, see Andy Arditi's solutions for an implementation of composition scores for this problem. These plots demonstrate strong composition between heads 0.0 and 1.0, and much weaker for all other heads (which is what we expect, since the other heads only compose in a narrow range of situations).

## How is the "non-palindromic" information stored?

We can look at the cosine similarity between the "difference vectors" for each sequence position (code below). The result - cosine similarity is extremely high for all tokens except for the advex positions 17, 19 and 20. This implies that (for these non-advex token positions), the information getting stored in each sequence position in the second half is boolean - i.e. there is a well-defined direction in residual stream space which represents "this token is not palindromic", and this direction is the same for all non-advex positions in the second half of the sequence.

It makes sense that this result doesn't hold for 17 and 19 (because 0.0's attention doesn't work for these positions, so there's no signal that can come from here). Interestingly, the fact that this result doesn't hold for 20 reframes the question of why 20 is adversarial - it's not because it's a blind spot of head 1.0, it's because it's a blind spot of the QK circuit of head 0.0.

```python
W_pos_scaled = model.W_pos / cache["scale", 0, "ln1"][:, :, 0].mean(0)

W_pos_difference_vectors = W_pos_scaled[second_half_indices] - W_pos_scaled[first_half_indices]
difference_vectors = W_pos_difference_vectors @ model.W_V[0, 0] @ model.W_O[0, 0]

difference_vectors_normed = difference_vectors / difference_vectors.norm(dim=-1, keepdim=True)

cos_sim = difference_vectors_normed @ difference_vectors_normed.T

imshow(
    cos_sim,
    x = [f"({i}, {j})" for i, j in zip(first_half_indices, second_half_indices)],
    y = [f"({i}, {j})" for i, j in zip(first_half_indices, second_half_indices)],
    title = "Cosine similarity of 'difference vectors' at different positions",
    width = 700,
    height = 600,
    text_auto = ".2f",
)
```
""", unsafe_allow_html=True)
    
    fig10 = go.Figure(json.loads(open(palindromes_dir / "fig10.json", 'r').read()))
    st.plotly_chart(fig10, use_container_width=False)


def section_0_august():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li><a class='contents-el' href='#prerequisites'>Prerequisites</a></li>
    <li><a class='contents-el' href='#difficulty'>Difficulty</a></li>
    <li><a class='contents-el' href='#motivation'>Motivation</a></li>
    <li><a class='contents-el' href='#logistics'>Logistics</a></li>
    <li><a class='contents-el' href='#what-counts-as-a-solution'>What counts as a solution?</a></li>
    <li><a class='contents-el' href='#setup'>Setup</a></li>
    <li><a class='contents-el' href='#task-dataset'>Task & Dataset</a></li>
    <li><a class='contents-el' href='#model'>Model</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""
# Monthly Algorithmic Challenge (August 2023): First Unique Character

### Colab: [problem](https://colab.research.google.com/drive/15huO8t1io2oYuLdszyjhMhrPF3WiWhf1)

This post is the second in the sequence of monthly mechanistic interpretability challenges. They are designed in the spirit of [Stephen Casper's challenges](https://www.lesswrong.com/posts/KSHqLzQscwJnv44T8/eis-vii-a-challenge-for-mechanists), but with the more specific aim of working well in the context of the rest of the ARENA material, and helping people put into practice all the things they've learned so far.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/writer.png" width="350">

## Prerequisites

The following ARENA material should be considered essential:

* **[1.1] Transformer from scratch** (sections 1-3)
* **[1.2] Intro to Mech Interp** (sections 1-3)

The following material isn't essential, but is recommended:

* **[1.2] Intro to Mech Interp** (section 4)
* **July's Algorithmic Challenge - writeup** (on the sidebar of this page)

## Difficulty

This problem is of roughly comparable difficulty to the July problem. The algorithmic problem is of a similar flavour, and the model architecture is very similar (the main difference is that this model has 3 attention heads per layer, instead of 2). I've done this because this problem is the first I'm also crossposting to LessWrong, and I want it to be reasonably accessible. The next problem in this sequence will probably be a step up in difficulty.

## Motivation

Neel Nanda's post [200 COP in MI: Interpreting Algorithmic Problems](https://www.lesswrong.com/posts/ejtFsvyhRkMofKAFy/200-cop-in-mi-interpreting-algorithmic-problems) does a good job explaining the motivation behind solving algorithmic problems such as these. I'd strongly recommend reading the whole post, because it also gives some high-level advice for approaching such problems.

The main purpose of these challenges isn't to break new ground in mech interp, rather they're designed to help you practice using & develop better understanding for standard MI tools (e.g. interpreting attention, direct logit attribution), and more generally working with libraries like TransformerLens.

Also, they're hopefully pretty fun, because why shouldn't we have some fun while we're learning?

## Logistics

The solution to this problem will be published on this page in the first few days of September, at the same time as the next problem in the sequence. There will also be an associated LessWrong post.

If you try to interpret this model, you can send your attempt in any of the following formats:

* Colab notebook,
* GitHub repo (e.g. with ipynb or markdown file explaining results),
* Google Doc (with screenshots and explanations),
* or any other sensible format.

You can send your attempt to me (Callum McDougall) via any of the following methods:

* The [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), via a direct message to me
* My personal email: `cal.s.mcdougall@gmail.com`
* LessWrong message ([here](https://www.lesswrong.com/users/themcdouglas) is my user)

**I'll feature the names of everyone who sends me a solution on this website, and also give a shout out to the best solutions.** It's possible that future challenges will also feature a monetary prize, but this is not guaranteed.

Please don't discuss specific things you've found about this model until the challenge is over (although you can discuss general strategies and techniques, and you're also welcome to work in a group if you'd like). The deadline for this problem will be the end of this month, i.e. 31st August.

## What counts as a solution?

Going through the solutions for the previous problem in the sequence (July: Palindromes) as well as the exercises in **[1.4] Balanced Bracket Classifier** should give you a good idea of what I'm looking for. In particular, I'd expect you to:

* Describe a mechanism for how the model solves the task, in the form of the QK and OV circuits of various attention heads (and possibly any other mechanisms the model uses, e.g. the direct path, or nonlinear effects from layernorm),
* Provide evidence for your mechanism, e.g. with tools like attention plots, targeted ablation / patching, or direct logit attribution.
* (Optional) Include additional detail, e.g. identifying the subspaces that the model uses for certain forms of information transmission, or using your understanding of the model's behaviour to construct adversarial examples.

# Setup

```python
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from pathlib import Path

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "monthly_algorithmic_problems" / "august23_unique_char"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from monthly_algorithmic_problems.august23_unique_char.dataset import UniqueCharDataset
from monthly_algorithmic_problems.august23_unique_char.model import create_model
from plotly_utils import hist, bar, imshow

device = t.device("cuda" if t.cuda.is_available() else "cpu")
```

## Task & Dataset

The algorithmic task is as follows: the model is presented with a sequence of characters, and for each character it has to correctly identify the first character in the sequence (up to and including the current character) which is unique up to that point.

The null character `"?"` has two purposes:

* In the input, it's used as the start character (because it's often helpful for interp to have a constant start character, to act as a "rest position").
* In the output, it's also used as the start character, **and** to represent the classification "no unique character exists".

Here is an example of what this dataset looks like:

```python
dataset = UniqueCharDataset(size=2, vocab=list("abc"), seq_len=6, seed=42)

for seq, first_unique_char_seq in zip(dataset.str_toks, dataset.str_tok_labels):
    print(f"Seq = {''.join(seq)}, Target = {''.join(first_unique_char_seq)}")
```

<div style='font-family:monospace; font-size:15px;'>
Seq = ?acbba, Target = ?aaaac<br>
Seq = ?cbcbc, Target = ?ccb??
</div><br>

Explanation:

1. In the first sequence, `"a"` is unique in the prefix substring `"acbb"`, but it repeats at the 5th sequence position, meaning the final target character is `"c"` (which appears second in the sequence).
2. In the second sequence, `"c"` is unique in the prefix substring `"cb"`, then it repeats so `"b"` is the new first unique token, and for the last 2 positions there are no unique characters (since both `"b"` and `"c"` have been repeated) so the correct classification is `"?"` (the "null character").

The relevant files can be found at:

```
chapter1_transformers/
└── exercises/
    └── monthly_algorithmic_problems/
        └── august23_unique_char/
            └── august23_unique_char/
                ├── model.py               # code to create the model
                ├── dataset.py             # code to define the dataset
                ├── training.py            # code to training the model
                └── training_model.ipynb   # actual training script
```

We've given you the class `UniqueCharDataset` to store your data, as you can see above. You can slice this object to get batches of tokens and labels (e.g. `dataset[:5]` returns a length-2 tuple, containing the 2D tensors representing the tokens and correct labels respectively). You can also use `dataset.toks` or `dataset.labels` to access these tensors directly, or `dataset.str_toks` and `dataset.str_tok_labels` to get the string representations of the tokens and labels (like we did in the code above).

## Model

Our model was trained by minimising cross-entropy loss between its predictions and the true labels, at every sequence position simultaneously (including the zeroth sequence position, which is trivial because the input and target are both always `"?"`). You can inspect the notebook `training_model.ipynb` to see how it was trained. I used the version of the model which achieved highest accuracy over 50 epochs (accuracy ~99%).

The model is is a 2-layer transformer with 3 attention heads, and causal attention. It includes layernorm, but no MLP layers. You can load it in as follows:

```python
filename = section_dir / "first_unique_char_model.pt"

model = create_model(
    seq_len=20,
    vocab=list("abcdefghij"),
    seed=42,
    d_model=42,
    d_head=14,
    n_layers=2,
    n_heads=3,
    normalization_type="LN",
    d_mlp=None # attn-only model
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

The model's output is a logit tensor, of shape `(batch_size, seq_len, d_vocab+1)`. The `[i, j, :]`-th element of this tensor is the logit distribution for the label at position `j` in the `i`-th sequence in the batch. The first `d_vocab` elements of this tensor correspond to the elements in the vocabulary, and the last element corresponds to the null character `"?"` (which is not in the input vocab).

A demonstration of the model working:

```python
dataset = UniqueCharDataset(size=1000, vocab=list("abcdefghij"), seq_len=20, seed=42)

logits, cache = model.run_with_cache(dataset.toks)

logprobs = logits.log_softmax(-1) # [batch seq_len d_vocab]
probs = logprobs.softmax(-1) # [batch seq_len d_vocab]

batch_size, seq_len = dataset.toks.shape
logprobs_correct = logprobs[t.arange(batch_size)[:, None], t.arange(seq_len)[None, :], dataset.labels] # [batch seq_len]
probs_correct = probs[t.arange(batch_size)[:, None], t.arange(seq_len)[None, :], dataset.labels] # [batch seq_len]

avg_cross_entropy_loss = -logprobs_correct.mean().item()
avg_correct_prob = probs_correct.mean().item()
min_correct_prob = probs_correct.min().item()

print(f"Average cross entropy loss: {avg_cross_entropy_loss:.3f}")
print(f"Average probability on correct label: {avg_correct_prob:.3f}")
print(f"Min probability on correct label: {min_correct_prob:.3f}")
```

<div style='font-family:monospace; font-size:15px;'>
Average cross entropy loss: 0.017<br>
Average probability on correct label: 0.988<br>
Min probability on correct label: 0.001
</div><br>

And a visualisation of its probability output for a single sequence:

```python
imshow(
    probs[0].T,
    y=dataset.vocab,
    x=[f"{dataset.str_toks[0][i]}<br>({i})" for i in range(model.cfg.n_ctx)],
    labels={"x": "Token", "y": "Vocab"},
    xaxis_tickangle=0,
    title="Sample model probabilities (for batch idx = 0), with correct classification highlighted",
    text=[
        ["〇" if str_tok == correct_str_tok else "" for correct_str_tok in dataset.str_tok_labels[0]]
        for str_tok in dataset.vocab
    ], # text can be a 2D list of lists, with the same shape as the data
)
```
""", unsafe_allow_html=True)
    
    first_char_dir = instructions_dir / "media/unique_char"
    fig_demo = go.Figure(json.loads(open(first_char_dir / "fig_demo.json", 'r').read()))
    st.plotly_chart(fig_demo, use_container_width=False)
    
    st.markdown(r"""
If you want some guidance on how to get started, I'd recommend reading the solutions for the July problem - I expect there to be a lot of overlap in the best way to tackle these two problems. You can also reuse some of that code!

Note - although this model was trained for long enough to get loss close to zero (you can test this for yourself), it's not perfect. There are some weaknesses that the model has which might make it vulnerable to adversarial examples, and I've decided to leave these in. The model is still very good at its intended task, and the main focus of this challenge is on figuring out how it solves the task, not dissecting the situations where it fails. However, you might find that the adversarial examples help you understand the model better.

Best of luck! 🎈

""", unsafe_allow_html=True)






func_page_list = [
    (section_0_august, "[August] First Unique Token"),
    (section_0_july, "[July] Palindromes"),
    (section_1_july, "[July] Solutions"),
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


streamlit_analytics.stop_tracking(
    unsafe_password=st.secrets["analytics_password"],
)