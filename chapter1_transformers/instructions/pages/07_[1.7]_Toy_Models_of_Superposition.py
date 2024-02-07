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

st.error("This is no longer the most updated version of these exercises: see [here](https://arena3-chapter1-transformer-interp.streamlit.app/) for the newest page.", icon="🚨")

def section_0():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
    <li class='margtop'><a class='contents-el' href='#reading-material'>Reading Material</a></li>
    <li class='margtop'><a class='contents-el' href='#questions'>Questions</a></li>
    <li class='margtop'><a class='contents-el' href='#toy-model-setup'>Toy Model - setup</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#what's-the-motivation-for-this-setup'>What's the motivation for this setup?</a></li>
        <li><a class='contents-el' href='#exercise-define-your-model'><b>Exercise</b> - define your model</a></li>
        <li><a class='contents-el' href='#exercise-implement-generate-batch'><b>Exercise</b> - implement <code>generate_batch</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#training-our-model'>Training our model</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-calculate-loss'><b>Exercise</b> - implement <code>calculate_loss</code></a></li>
        <li><a class='contents-el' href='#exercise-interpret-these-diagrams'><b>Exercise</b> - interpret these diagrams</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#visualizing-features-across-varying-sparsity'>Visualizing features across varying sparsity</a></li>
    <li class='margtop'><a class='contents-el' href='#correlated-or-anticorrelated-feature-sets'>Correlated or anticorrelated feature sets</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-generate-correlated-batch'><b>Exercise</b> - implement <code>generate_correlated_batch</code></a></li>
        <li><a class='contents-el' href='#exercise-generate-more-feature-correlation-plots'><b>Exercise</b> - generate more feature correlation plots</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#superposition-in-a-privileged-basis'>Superposition in a Privileged Basis</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-plot-w-in-privileged-basis'><b>Exercise</b> - plot W in privileged basis</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#summary-what-have-we-learned'>Summary - what have we learned?</a></li>
    <li class='margtop'><a class='contents-el' href='#feature-geometry'>Feature Geometry</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-compute-dimensionality'><b>Exercise</b> - compute dimensionality</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#further-reading'>Further Reading</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(
r"""
# [1.7] - Toy  Models of Superposition

### Colab: [**exercises**](https://colab.research.google.com/drive/1oJcqxd4CS5zl-RO9fufQJI5lpxTzCYGw) | [**solutions**](https://colab.research.google.com/drive/1ygVrrrJH0DynAj9tkLgwsZ_xOk85p9oV)


Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.
                
Links to other chapters: [**(0) Fundamentals**](https://arena-ch0-fundamentals.streamlit.app/), [**(2) RL**](https://arena-ch2-rl.streamlit.app/).

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/galaxies.jpeg" width="350">


## Introduction

Superposition is a crucially important concept for understanding how transformers work. A definition from Neel Nanda's glossary:

> Superposition is when a model represents more than n features in an $n$-dimensional activation space. That is, features still correspond to directions, but **the set of interpretable directions is larger than the number of dimensions**.

Why should we expect something like this to happen? In general, the world has way more features than the model has dimensions of freedom, and so we can't have a one-to-one mapping between features and values in our model. But the model has to represent these features somehow. Hence, it comes up with techniques for cramming multiple features into fewer dimensions (at the cost of adding noise and interference between features).

This topic is mostly theoretical, rather than exercise-heavy. The exercises that do exist will take you through some of the toy models that have been developed to understand superposition. We'll also suggest some open-ended exploration at the end.


## Content & Learning Objectives

Unlike many other topics in this chapter, there's quite a bit of theory which needs to be understood before we start making inferences from the results of our coding experiments. We start by suggesting a few useful papers / videos / blog posts for you to go through, then we'll move into replication of the main results from the "toy models" paper. We'll conclude with a discussion of future directions for superposition research.

A key point to make here is that, perhaps more so than any other section in this chapter, we really don't understand superposition that well at all! It's hard to point to the seminal work in this field because we don't really know what the critical new insights will look like. That being said, we hope this material gives you enough directions to pursue when you're finished!

> ##### Learning objectives
> 
> - Understand the concept of superposition, and why models need to do it
> - Understand the difference between superposition and polysemanticity
> - Understand the difference between neuron and bottleneck superposition (or computational and representational superposition)
> - Build & train the toy model from Anthropic's paper, replicate the main results
> - Understand the geometric intuitions behind superposition, and how they relate to the more general ideas of superposition in larger models
> - See how superposition varies when you change the following characteristics of the features:
>   - Importance
>   - Sparsity
>   - Correlation
> - Understand the lessons this toy setup carries for real transformers, and read about some of the work being done here


## Setup


```python
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from torch import nn, Tensor
from torch.nn import functional as F
from dataclasses import dataclass
import time
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import einops
import plotly.express as px
from pathlib import Path
from jaxtyping import Float
from typing import Optional, Callable, Union
from tqdm.auto import tqdm
from dataclasses import dataclass

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part7_toy_models_of_superposition', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line

from part7_toy_models_of_superposition.utils import plot_W, plot_Ws_from_model, render_features, plot_feature_geometry
import part7_toy_models_of_superposition.tests as tests
import part7_toy_models_of_superposition.solutions as solutions

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
```

## Reading Material

Here are a few recommended resources to help you get started. Each one is labelled with what you should read, at minimum.

* [200 COP in MI: Exploring Polysemanticity and Superposition](https://www.alignmentforum.org/posts/o6ptPu7arZrqRCxyz/200-cop-in-mi-exploring-polysemanticity-and-superposition)
    * Read the post, up to and including "Tips" (although some parts of it might make more sense after you've read the other things here).
* Neel Nanda's [Dynalist notes on superposition](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=3br1psLRIjQCOv2T4RN3V6F2)
    * These aren't long, you should skim through them, and also use them as a reference during these exercises.
* Appendix of [Finding Neurons in a Haystack: Case Studies with Sparse Probing](https://arxiv.org/abs/2305.01610)
    * Despite this paper not *just* being about superposition, it has some of the best-written explanations of superposition concepts.
    * Sections A.6 - A.9 are particularly good.
* [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (Anthropic paper)
    * You should read up to & including the "Summary: A Hierarchy of Feature Properties" section.
    * The first few sections ("Key Results", "Definitions and Motivation", and "Empirical Phenomena" are particularly important).
    * We'll also be going through other parts of this paper as we work through the exercises.
* Neel Nanda's [video walkthrough of superposition](https://www.youtube.com/watch?v=R3nbXgMnVqQ)
    * This is very long and you don't *have* to watch it, but we weakly recommend it.


## Questions

Here are a set of questions (with some brief answers provided) which you should be able to answer for yourself after reading the above material. Seach for them on Neel's Dynalist notes if you didn't come across them during your reading.

What is a **privileged basis**? Why should we expect neuron activations to be privileged by default? Why *shouldn't* we expect the residual stream to be privileged?

<details>
<summary>Answer</summary>

A privileged basis is one where the standard basis vectors are meaningful, i.e. they represent some human-understandable concepts.

Neuron activations are privileged because of the **nonlinear function that gets applied**. As an example, consider a simple case of 2 neurons (represented as `x` and `y` coordinates), and suppose we want to store two features in this vector space. If we stored them in non-basis directions, then it would be messy to extract each feature individally (we'd get interference between the two). 

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/priv-basis.png" width="750">

The residual stream is not privileged because anything that reads from it and writes to it uses a linear map. As a thought experiment, if we changed all the writing matrices (i.e. $W_{out}$ in the MLP layers and $W_O$ in the attention layers) to $W \to W R$, and all the reading matrices (i.e. $W_{in}$ in the MLP layers and $W_Q$, $W_K$, $W_V$ in the attention layers) to $W \to W R^{-1}$ where $R$ is some arbitrary rotation matrix, then the model's computation would be unchanged. Since the matrix $R$ is arbitrary, it can change the basis in any way it wants, so that basis can't be privileged.

To take this back to the analogy for transformers as [people standing in a line](https://www.lesswrong.com/posts/euam65XjigaCJQkcN/an-analogy-for-understanding-transformers), imagine everyone in the line was speaking and thinking in a different language (which had a 1-1 mapping with English, but which you (an outside observer) didn't have a dictionary for). This wouldn't meaningfully change the way the people in the line were sharing and processing information, it would just change the way the information was stored - and without the dictionary (or any additional context), you can't interpret this information.
</details>

What is the difference between **neuron superposition** and **neuron polysemanticity**?

<details>
<summary>Answer</summary>

Polysemanticity happens when one neuron corresponds to multiple features (see [here](https://distill.pub/2020/circuits/zoom-in/#:~:text=lot%20of%20effort.-,Polysemantic%20Neurons,-This%20essay%20may) for more discussion & examples). If we only had polysemanticity, this wouldn't really be a problem for us (there might exist a basis for features s.t. each basis vector corresponds to a single feature).

Superposition is when there are **more features than neurons**. So it implies polysemanticity (because we must have neurons representing more than one feature), but the converse is not true.
</details>


What are the **importance** and **sparsity** of features? Do you expect more or less polysemantic neurons if sparsity is larger?

<details>
<summary>Answer</summary>

**Importance** = how useful is this feature for achieving lower loss?

**Sparsity** = how frequently is it in the input data?

If sparsity is larger, then we expect more polysemantic neurons. This is because a single neuron can afford to represent several different sparse features (usually it'll only be representing one of them at any given time, so there won't be interference).
</details>

How would you define a **feature**?

<details>
<summary>Answer</summary>

There's no single correct answer to this. Many of the definitions are unfortunately circular (e.g. "a feature is a thing which could be represented by a neuron"). A few possible definitions are this one from Neel's [Dynalist notes](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#q=feature):

> A feature is a property of an input to the model, or some subset of that input (eg a token in the prompt given to a language model, or a patch of an image).

or this similar one from Chris Olah's [Distil Circuits Thread](https://distill.pub/2020/circuits/zoom-in/):

> A feature is a a scalar function of the input. In this essay, neural network features are directions, and often simply individual neurons. We claim such features in neural networks are typically meaningful features which can be rigorously studied. A **meaningful feature** is one that genuinely responds to an articulable property of the input, such as the presence of a curve or a floppy ear. 
</details>


## Toy Model - setup

In this section, we'll be examining & running experiments on the toy model studied in [Anthropic's paper](https://transformer-circuits.pub/2022/toy_model/index.html).

You can follow along with the paper from the [Demonstrating Superposition](https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating) section onwards; it will approximately follow the order of the sections in this notebook.

This paper presented a very rudimentary model for **bottleneck superposition** - when you try and represent more than $n$ features in a vector space of dimension $n$. The model is as follows:

* We take a 5-dimensional input $x$
* We map it down into 2D space
* We map it back up into 5D space (using the transpose of the first matrix)
* We add a bias and ReLU


### What's the motivation for this setup?

The input $x$ represents our five features (they're uniformly sampled between 0 and 1).

Each feature can have **importance** and **sparsity**. Recall our earlier definitions:

* **Importance** = how useful is this feature for achieving lower loss?
* **Sparsity** = how frequently is it in the input data?

This is realised in our toy model as follows:

* **Importance** = the coefficient on the weighted mean squared error between the input and output, which we use for training the model
    * In other words, our loss function is $L = \sum_x \sum_i I_i (x_i - x_i^\prime)^2$, where $I_i$ is the importance of feature $i$.
* **Sparsity** = the probability of the corresponding element in $x$ being zero
    * In other words, this affects the way our training data is generated (see the method `generate_batch` in the `Module` class below)

The justification for using $W^T W$ is as follows: we can think of $W$ (which is a matrix of shape `(2, 5)`) as a grid of "overlap values" between the features and bottleneck dimensions. The values of the 5x5 matrix $W^T W$ are the dot products between the 2D representations of each pair of features. To make this intuition clearer, imagine each of the columns of $W$ were unit vectors, then $W^T W$ would be a matrix of cosine similarities between the features (with diagonal elements equal to 1, because the similarity of a feature with itself is 1). To see this for yourself:


```python
W = t.randn(2, 5)
W_normed = W / W.norm(dim=0, keepdim=True)

imshow(W_normed.T @ W_normed, title="Cosine similarities of each pair of 2D feature embeddings", width=600)
```

To put it another way - if the columns of $W$ were orthogonal, then $W^T W$ would be the identity (i.e. $W^{-1} = W^T$). This can't actually be the case because $W$ is a 2x5 matrix, but its columns can be "nearly orthgonal" in the sense of having pairwise cosine similarities close to 0.


Another nice thing about using two bottleneck dimensions is that we get to visualise our output! We've got a few helper functions for this purpose.


```python
plot_W(W_normed)
```

Compare this plot to the `imshow` plot above, and make sure you understand what's going on here (and how the two plots relate to each other). A lot of the subsequent exercises run with this idea of a geometric interpretation of the model's features and bottleneck dimensions.

<details>
<summary>Help - I'm confused about how these plots work.</summary>

As mentioned, you can view $W$ as being a set of five 2D vectors, one for each of our five features. The heatmap shows us the cosine similarities between each pair of these vectors, and the second plot shows us these five vectors in 2D space.

For example, run the following code@

```python
t.manual_seed(2)

W = t.randn(2, 5)
W_normed = W / W.norm(dim=0, keepdim=True)

imshow(W_normed.T @ W_normed, title="Cosine similarities of each pair of 2D feature embeddings", width=600)

plot_W(W_normed)
```

In the heatmap, we can see two pairs of vectors (the 1st & 2nd, and the 0th & 4th) have very high cosine similarity. This is reflected in the 2D plot, where these features are very close to each other (the 0th feature is the darkest color, the 4th feature is the lightest).

</details>


### Exercise - define your model

```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 15-25 minutes on this exercise.
```

Below is some code for your model. It shouldn't be unfamiliar if you've already built simple neural networks earlier in this course.

A few things to note:

* The `Config` class has an `n_instances` class. This is so we can optimize multiple models at once in a single training loop (this'll be useful later on)
    * You should treat this as basically like a batch dimension for your weights
    * i.e. your weights and biases will have an `n_instances` dimension at the start, and for every input in your batch *and* every instance in your weights, you should be doing a separate computation.
* The `feature_probability` and `importance` arguments correspond to sparsity and importance of features.
    * Note that feature probability is used in the `generate_batch` function, to get our training data.


You should fill in the `__init__` function, which defines `self.W` and `self.b_final` (see the type annotations). Make sure that `W` is initialized with the [Xavier normal method](https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1a86191a828a085e1c720dbce185d6c307.html).  `b_final` can be initialized with zeros.

You should also fill in the `forward` function, to calculate the output (again, the type annotations should be helpful here).

You will fill out the `generate_batch` function in the exercise after this one.

```python
@dataclass
class Config:
    # We optimize n_instances models in a single training loop to let us sweep over sparsity or 
    # importance curves  efficiently. You should treat `n_instances` as kinda like a batch dim,
    # but one which is built into our training setup.
    n_instances: int
    n_features: int = 5
    n_hidden: int = 2
    # Ignore the correlation arguments for now.
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0

     
class Model(nn.Module):

    W: Float[Tensor, "n_instances n_hidden n_features"]
    b_final: Float[Tensor, "n_instances n_features"]
    # Our linear map (ignoring n_instances) is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self, 
        config: Config, 
        feature_probability: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,               
        device=device
    ):
        super().__init__()
        self.config = config

        if feature_probability is None: feature_probability = t.ones(())
        self.feature_probability = feature_probability.to(device)
        if importance is None: importance = t.ones(())
        self.importance = importance.to(device)
        
        pass
        

    def forward(
        self, 
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        pass

    def generate_batch(self, n_batch) -> Float[Tensor, "n_batch instances features"]:
        '''
        Generates a batch of data. We'll return to this function later when we apply correlations.
        '''    
        feat = t.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device)
        feat_seeds = t.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device)
        feat_is_present = feat_seeds <= self.feature_probability
        batch = t.where(
            feat_is_present,
            feat,
            t.zeros((), device=self.W.device),
        )
        return batch


tests.test_model(Model)
```

<details>
<summary>Solution</summary>


```python
class Model(nn.Module):

    W: Float[Tensor, "n_instances n_hidden n_features"]
    b_final: Float[Tensor, "n_instances n_features"]
    # Our linear map (ignoring n_instances) is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self, 
        config: Config, 
        feature_probability: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,               
        device=device
    ):
        super().__init__()
        self.config = config

        if feature_probability is None: feature_probability = t.ones(())
        self.feature_probability = feature_probability.to(device)
        if importance is None: importance = t.ones(())
        self.importance = importance.to(device)
        
        # SOLUTION
        self.W = nn.Parameter(t.empty((config.n_instances, config.n_hidden, config.n_features), device=device))
        nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(t.zeros((config.n_instances, config.n_features), device=device))

    
    def forward(
        self, 
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        # SOLUTION
        hidden = einops.einsum(
           features, self.W,
           "... instances features, instances hidden features -> ... instances hidden"
        )
        out = einops.einsum(
            hidden, self.W,
            "... instances hidden, instances hidden features -> ... instances features"
        )
        out = out + self.b_final
        out = F.relu(out)
        return out
```
</details>

### Exercise - implement `generate_batch`

```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 10-15 minutes on this exercise.
```

Next, you should implement the function `generate_batch` above. This should return a tensor of shape `(n_batch, instances, features)`, where:

* The `instances` and `features` values are taken from the model config,
* Each feature is present with probability `self.feature_probability` (note that `self.feature_probability` is guaranteed to broadcast with the `(n_batch, instances, features)` shape),
* Each present feature is sampled from a uniform distribution between 0 and 1.

Note - after you've implemented this, we recommend you read the solutions, because later exercises will have you make more complicated versions of this function (when we add correlation).

```python
tests.test_generate_batch(Model)
```

<details>
<summary>Solution</summary>

```python
def generate_batch(self, n_batch) -> Float[Tensor, "n_batch instances features"]:
    '''
    Generates a batch of data. We'll return to this function later when we apply correlations.
    '''    
    feat = t.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device)
    feat_seeds = t.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device)
    feat_is_present = feat_seeds <= self.feature_probability
    batch = t.where(
        feat_is_present,
        feat,
        t.zeros((), device=self.W.device),
    )
    return batch
```

</details>

## Training our model


The details of training aren't very conceptually important, so we've given you most of the code to train the model below. We use **learning rate schedulers** to control the learning rate as the model trains - you'll use this later on during the RL chapter.

### Exercise - implement `calculate_loss`

```yaml
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 5-10 minutes on this exercise.
```

You should fill in the `calculate_loss` function below. The loss function **for a single instance** is given by:

$$
L=\frac{1}{BF}\sum_x \sum_i I_i\left(x_i-x_i^{\prime}\right)^2
$$

where:

* $B$ is the batch size,
* $F$ is the number of features,
* $x_i$ are the inputs and $x_i'$ are the model's outputs,
* $I_i$ is the importance of feature $i$,
* $\sum_i$ is a sum over features,
* $\sum_x$ is a sum over the elements in the batch.

For the general version (i.e. with multiple instances), we sum the loss over instances as well (since we're effectively training `n_instances` different copies of our weights at once, one for each instance).


```python
def linear_lr(step, steps):
    return (1 - (step / steps))

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))

def optimize(
    model: Model, 
    n_batch: int = 1024,
    steps: int = 10_000,
    print_freq: int = 100,
    lr: float = 1e-3,
    lr_scale: Callable = constant_lr,
):
    '''Optimizes the model using the given hyperparameters.'''
    cfg = model.config

    optimizer = t.optim.AdamW(list(model.parameters()), lr=lr)

    progress_bar = tqdm(range(steps))
    for step in progress_bar:
        step_lr = lr * lr_scale(step, steps)
        for group in optimizer.param_groups:
            group['lr'] = step_lr
            optimizer.zero_grad()
            batch = model.generate_batch(n_batch)
            out = model(batch)
            loss = calculate_loss(out, batch, model)
            loss.backward()
            optimizer.step()

            if step % print_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(loss=loss.item()/cfg.n_instances, lr=step_lr)

def calculate_loss(
    out: Float[Tensor, "batch instances features"],
    batch: Float[Tensor, "batch instances features"],
    model: Model
) -> Float[Tensor, ""]:
    '''
    Calculates the loss for a given batch, using this loss described in the Toy Models paper:
    
        https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

    Note, `model.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
    '''
    pass


tests.test_calculate_loss(calculate_loss)
```

<details>
<summary>Solution</summary>

```python
def calculate_loss(
    out: Float[Tensor, "batch instances features"],
    batch: Float[Tensor, "batch instances features"],
    model: Model
) -> Float[Tensor, ""]:
    '''
    Calculates the loss for a given batch, using this loss described in the Toy Models paper:
    
        https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

    Note, `model.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
    '''
    # SOLUTION
    error = model.importance * ((batch - out) ** 2)
    loss = einops.reduce(error, 'batch instances features -> instances', 'mean').sum()
    return loss
```

</details>

Now, we'll reproduce a version of the figure from the introduction, although with a slightly different version of the code.

A few notes:

* The `importance` argument is the same for all instances. It takes values between 1 and ~0.66 for each feature (so for every instance, there will be some features which are more important than others).
* The `feature_probability` is the same for all features, but it varies across instances. In other words, we're runnning several different experiments at once, and we can compare the effect of having larger feature sparsity in these experiments.


```python
config = Config(
    n_instances = 10,
    n_features = 5,
    n_hidden = 2,
)
    
importance = (0.9**t.arange(config.n_features))

feature_probability = (20 ** -t.linspace(0, 1, config.n_instances))
    
line(importance, width=600, height=400, title="Importance of each feature (same over all instances)", labels={"y": "Feature importance", "x": "Feature"})

line(feature_probability, width=600, height=400, title="Feature probability (varied over instances)", labels={"y": "Probability", "x": "Instance"})
```

```python
model = Model(
    config=config,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None]
)


optimize(model)

plot_Ws_from_model(model, config)
```

### Exercise - interpret these diagrams

```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 10-20 minutes on this exercise.
```

Remember that for all these diagrams, the darker colors have lower importance and the lighter colors have higher importance. Also, the sparsity of all features is increasing as we move from left to right (at the far left there is no sparsity, at the far right feature probability is 5% for all features).

<details>
<summary>Hint</summary>

For low sparsity, think about what the model would learn to do if all 5 features were present all the time. What's the best our model could do in this case, and how does that relate to the **importance** values?

For high sparsity, think about what the model would learn to do if there was always exactly one feature present. Does this make interference between features less of a problem?
</details>

<details>
<summary>Answer (intuitive)</summary>

When there is no sparsity, the model can never represent more than 2 features faithfully, so it makes sense for it to only represent the two most important features. It stores them orthogonally in 2D space, and sets the other 3 features to zero. This way, it can reconstruct these two features perfectly, and ignores all the rest.

When there is high sparsity, we get a pentagon structure. Most of the time at most one of these five features will be active, which helps avoid **interference** between features. When we try to recover our initial features by projecting our point in 2D space onto these five directions, most of the time when feature $i$ is present, we can be confident that our projection onto the $i$-th feature direction only captures this feature, rather than being affected by the presence of other features. We omit the mathematical details here.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/download (7).png" width="900">

The key idea here is that two forces are competing in our model: **feature benefit** (representing more thing is good!), and **interference** (representing things non-orthogonally is bad). The higher the sparsity, the more we can reduce the negative impact of interference, and so the trade-off skews towards "represent more features, non-orthogonally".

</details>


## Visualizing features across varying sparsity


Now that we've got our pentagon plots and started to get geometric intuition for what's going on, let's scale things up! We're now operating in dimensions too large to visualise, but hopefully our intuitions will carry over.


```python
config = Config(
    n_instances = 20,
    n_features = 100,
    n_hidden = 20,
)
    
importance = (100 ** -t.linspace(0, 1, config.n_features))

feature_probability = (20 ** -t.linspace(0, 1, config.n_instances))
    
line(importance, width=600, height=400, title="Importance of each feature (same over all instances)", labels={"y": "Feature importance", "x": "Feature"})

line(feature_probability, width=600, height=400, title="Feature probability (varied over instances)", labels={"y": "Probability", "x": "Instance"})
```

Here, we're going to use a different kind of visualisation:

* The left hand plots a bar graph of all the features and their corresponding embedding norms $||W_i||$.
    * As we increase sparsity, the model is able to represent more features (i.e. we have more features with embedding norms close to 1).
    * We also color the bars according to whether they're orthogonal to other features (blue) or not (red). So we can see that for low sparsity most features are represented orthogonally (like our left-most plots above) but as we increase sparsity we transition to all features being represented non-orthogonally (like our right-most pentagon plots above).
* The right hand plots show us the dot products between all pairs of feature vectors (kinda like the heatmaps we plotted at the start of this section).
    * This is another way of visualising the increasing interference between features as we increase sparsity.
    * Note that all these right hand plots represent **matrices with rank at most `n_hidden=20`**. The first few are approximately submatrices of the identity (because we perfectly reconstruct 20 features and delete the rest), but the later plots start to display inference as we plot more than 20 values (the diagonals of these matrices have more than 20 non-zero elements).

See the section [Basic Results](https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-basic-results) for more of an explanation of this graph and what you should interpret from it.


```python
model = Model(
    config=config,
    device=device,
    importance = importance[None, :],
    feature_probability = feature_probability[:, None]
)

optimize(model)

fig = render_features(model, np.s_[::2]) # Plot every second instance
fig.update_layout(width=1200, height=1600)
```

## Correlated or anticorrelated feature sets

One major thing we haven't considered in our experiments is **correlation**. We could guess that superposition is even more common when features are **anticorrelated** (for a similar reason as why it's more common when features are sparse). Most real-world features are anticorrelated (e.g. the feature "this is a sorted Python list" and "this is some text in an edgy teen vampire romance novel" are probably anticorrelated - that is, unless you've been reading some pretty weird fanfics).

In this section, you'll define a new data-generating function for correlated features, and run the same experiments as in the first section.

### Exercise - implement `generate_correlated_batch`

```yaml
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 20-40 minutes on this exercise.

It's not conceptually important, and it's very fiddly / delicate, so you should definitely look at the solutions if you get stuck. Understanding the results and why they occur is more important than the implementation!
```

You should implement the function `generate_correlated_batch`, which will replace the `generate_batch` function you were given at the start of these exercises. You can start by copying the code from `generate_batch`.

Before implementing the function, you should read the [experimental details in Anthropic's paper](https://transformer-circuits.pub/2022/toy_model/index.html#geometry-correlated-setup), where they describe how they setup correlated and anticorrelated sets. TLDR, we have:

* Correlated feature pairs, where features co-occur
    * i.e. they always all occur or none of them occur
    * We can simulate this by using the same random seed for each feature in a pair
* Anticorrelated feature pairs, where features never co-occur
    * i.e. if one feature occurs, the other must not occur
    * We can simulate this by having a random seed for "is a feature pair all zero", and a random seed for "which feature in the pair is active" (used if the feature pair isn't all zero)

**Important note** - we're using a different convention to the Anthropic paper. They have both features in an anticorrelated pair set to zero with probability $1-p$, and with probability $p$ we choose one of the features in the pair to set to zero. The problem with this is that the "true feature probability" is $p/2$, not $p$. You should make sure that each feature actually occurs with probability $p$, which means setting the pair to zero with probability $1-2p$. You can assume $p$ will always be less than $1/2$.

```python
def generate_correlated_batch(self: Model, n_batch: int) -> Float[Tensor, "n_batch instances fetures"]:
    '''
    Generates a batch of data.

    There are `n_correlated_pairs` pairs of correlated features (i.e. they always co-occur), and 
    `n_anticorrelated` pairs of anticorrelated features (i.e. they never co-occur; they're
    always opposite).

    So the total number of features defined this way is `2 * n_correlated_pairs + 2 * n_anticorrelated`.

    You should stack the features in the order (correlated, anticorrelated, uncorrelated), where
    the uncorrelated ones are all the remaining features.

    Note, we assume the feature probability varies across instances but not features, i.e. all features
    in each instance have the same probability of being present.
    '''
    n_correlated_pairs = self.config.n_correlated_pairs
    n_anticorrelated_pairs = self.config.n_anticorrelated_pairs

    n_uncorrelated = self.config.n_features - 2 * (n_correlated_pairs + n_anticorrelated_pairs)
    assert n_uncorrelated >= 0, "Need to have number of paired correlated + anticorrelated features <= total features"
    assert self.feature_probability.shape == (self.config.n_instances, 1), "Feature probability should not vary across features in a single instance."

    # Define uncorrelated features, the standard way
    feat = t.rand((n_batch, self.config.n_instances, n_uncorrelated), device=self.W.device)
    feat_seeds = t.rand((n_batch, self.config.n_instances, n_uncorrelated), device=self.W.device)
    feat_is_present = feat_seeds <= self.feature_probability
    batch_uncorrelated = t.where(
        feat_is_present,
        feat,
        t.zeros((), device=self.W.device),
    )

    # YOUR CODE HERE - compute batch_correlated and batch_anticorrelated, and stack all three batches together
    pass


Model.generate_batch = generate_correlated_batch
```

<details>
<summary>Solution</summary>

```python
def generate_correlated_batch(self: Model, n_batch: int) -> Float[Tensor, "n_batch instances fetures"]:
    '''
    Generates a batch of data.

    There are `n_correlated_pairs` pairs of correlated features (i.e. they always co-occur), and 
    `n_anticorrelated` pairs of anticorrelated features (i.e. they never co-occur; they're
    always opposite).

    So the total number of features defined this way is `2 * n_correlated_pairs + 2 * n_anticorrelated`.

    You should stack the features in the order (correlated, anticorrelated, uncorrelated), where
    the uncorrelated ones are all the remaining features.

    Note, we assume the feature probability varies across instances but not features, i.e. all features
    in each instance have the same probability of being present.
    '''
    n_correlated_pairs = self.config.n_correlated_pairs
    n_anticorrelated_pairs = self.config.n_anticorrelated_pairs

    n_uncorrelated = self.config.n_features - 2 * (n_correlated_pairs + n_anticorrelated_pairs)
    assert n_uncorrelated >= 0, "Need to have number of paired correlated + anticorrelated features <= total features"
    assert self.feature_probability.shape == (self.config.n_instances, 1), "Feature probability should not vary across features in a single instance."

    # Define uncorrelated features, the standard way
    feat = t.rand((n_batch, self.config.n_instances, n_uncorrelated), device=self.W.device)
    feat_seeds = t.rand((n_batch, self.config.n_instances, n_uncorrelated), device=self.W.device)
    feat_is_present = feat_seeds <= self.feature_probability
    batch_uncorrelated = t.where(
        feat_is_present,
        feat,
        t.zeros((), device=self.W.device),
    )

    # SOLUTION
    # Define correlated features: have the same sample determine if they're zero or not
    feat = t.rand((n_batch, self.config.n_instances, 2 * n_correlated_pairs), device=self.W.device)
    feat_set_seeds = t.rand((n_batch, self.config.n_instances, n_correlated_pairs), device=self.W.device)
    feat_set_is_present = feat_set_seeds <= self.feature_probability
    feat_is_present = einops.repeat(
        feat_set_is_present,
        "batch instances features -> batch instances (features pair)", pair=2
    )
    batch_correlated = t.where(
        feat_is_present, 
        feat,
        t.zeros((), device=self.W.device),
    )

    # Define anticorrelated features: have them all be zero with probability `feature_probability`, and
    # have a single feature randomly chosen if they aren't all zero
    feat = t.rand((n_batch, self.config.n_instances, 2 * n_anticorrelated_pairs), device=self.W.device)
    # First, generate seeds (both for entire feature set, and for features within the set)
    feat_set_seeds = t.rand((n_batch, self.config.n_instances, n_anticorrelated_pairs), device=self.W.device)
    first_feat_seeds = t.rand((n_batch, self.config.n_instances, n_anticorrelated_pairs), device=self.W.device)
    # Create boolean mask for whether the entire set is zero
    # Note: the *2 here didn't seem to be used by the paper, but it makes more sense imo! You can leave it out and still get good results.
    feat_set_is_present = feat_set_seeds <= 2 * self.feature_probability
    # Where it's not zero, create boolean mask for whether the first element is zero
    first_feat_is_present = first_feat_seeds <= 0.5
    # Now construct our actual features and stack them together, then rearrange
    first_feats = t.where(
        feat_set_is_present & first_feat_is_present, 
        feat[:, :, :n_anticorrelated_pairs],
        t.zeros((), device=self.W.device)
    )
    second_feats = t.where(
        feat_set_is_present & (~first_feat_is_present), 
        feat[:, :, n_anticorrelated_pairs:],
        t.zeros((), device=self.W.device)
    )
    batch_anticorrelated = einops.rearrange(
        t.concat([first_feats, second_feats], dim=-1),
        "batch instances (pair features) -> batch instances (features pair)", pair=2
    )

    return t.concat([batch_correlated, batch_anticorrelated, batch_uncorrelated], dim=-1)
```

</details>

Once you've completed this function, try running the code below. You should see that the correlated features (the two columns on the left) always co-occur, and the two anticorrelated features (on the right) never do.

<details>
<summary>Example output you should be getting</summary>

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/example_output_2.png" width="700">
</details>

```python
config = Config(
    n_instances = 10,
    n_features = 4,
    n_hidden = 2,
    n_correlated_pairs = 1,
    n_anticorrelated_pairs = 1,
)

importance = t.ones(config.n_features, dtype=t.float, device=device)
feature_probability = (20 ** -t.linspace(0, 1, config.n_instances))

model = Model(
    config=config,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None]
)

batch = model.generate_batch(n_batch = 1)

imshow(
    batch.squeeze(),
    labels={"x": "Feature", "y": "Instance"}, 
    title="Feature heatmap (first two features correlated, last two anticorrelated)"
)
```

The code below tests your function, by generating a large number of batches and measuring them statistically.

```python
feature_probability = (20 ** -t.linspace(0.5, 1, config.n_instances))
model.feature_probability = feature_probability[:, None].to(device)

batch = model.generate_batch(n_batch = 10000)

corr0, corr1, anticorr0, anticorr1 = batch.unbind(dim=-1)
corr0_is_active = corr0 != 0
corr1_is_active = corr1 != 0
anticorr0_is_active = anticorr0 != 0
anticorr1_is_active = anticorr1 != 0

assert (corr0_is_active == corr1_is_active).all(), "Correlated features should be active together"
assert (corr0_is_active.float().mean(0).cpu() - feature_probability).abs().mean() < 0.01, "Each correlated feature should be active with probability `feature_probability`"

assert (anticorr0_is_active & anticorr1_is_active).int().sum().item() == 0, "Anticorrelated features should never be active together"
assert (anticorr0_is_active.float().mean(0).cpu() - feature_probability).abs().mean() < 0.01, "Each anticorrelated feature should be active with probability `feature_probability`"
```

Now, let's try making some plots with two pairs of correlated features (matching the [first figure](https://transformer-circuits.pub/2022/toy_model/index.html#geometry-organization) in the Anthropic paper):

```python
config = Config(
    n_instances = 5,
    n_features = 4,
    n_hidden = 2,
    n_correlated_pairs = 2,
    n_anticorrelated_pairs = 0,
)

# All same importance
importance = t.ones(config.n_features, dtype=t.float, device=device)
# We use very low feature probabilities, from 5% down to 0.25%
feature_probability = (400 ** -t.linspace(0.5, 1, 5))

model = Model(
    config=config,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None]
)

optimize(model)

plot_Ws_from_model(model, config)
```

### Exercise - generate more feature correlation plots

```yaml
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 10-15 minutes on this exercise.

It should just involve changing the parameters in your code above.
```

You should now plot the second and third figures in the [set of feature correlation plots](https://transformer-circuits.pub/2022/toy_model/index.html#geometry-organization) (keeping the same importance and feature probability). You may not get exactly the same results as the paper, but they should still roughly match (e.g. you should see no antipodal pairs in the code above, but you should see at least some when you test the anticorrelated sets).

<details>
<summary>Solution</summary>

For the first plot:

```python
config = Config(
    n_instances = 5,
    n_features = 4,
    n_hidden = 2,
    n_correlated_pairs = 0,
    n_anticorrelated_pairs = 2,
)

# All same importance
importance = t.ones(config.n_features, dtype=t.float, device=device)
# We use low feature probabilities, from 5% down to 0.25%
feature_probability = (400 ** -t.linspace(0.5, 1, 5))

model = Model(
    config=config,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None]
)

optimize(model)

plot_Ws_from_model(model, config)
```

For the second plot:

```python
config = Config(
    n_instances = 5,
    n_features = 6,
    n_hidden = 2,
    n_correlated_pairs = 3,
    n_anticorrelated_pairs = 0,
)

# All same importance
importance = t.ones(config.n_features, dtype=t.float, device=device)
# We use low feature probabilities, from 5% down to 0.25%
feature_probability = (400 ** -t.linspace(0.5, 1, 5))

model = Model(
    config=config,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None]
)

optimize(model)

plot_Ws_from_model(model, config)
```

</details>

## Superposition in a Privileged Basis

So far, we've explored superposition in a model without a privileged basis. We can rotate the hidden activations arbitrarily and, as long as we rotate all the weights, have the exact same model behavior. That is, for any ReLU output model with weights 
$W$, we could take an arbitrary orthogonal matrix $O$ and consider the model $W' = OW$. Since $(OW)^T(OW) = W^T W$, the result would be an identical model!

Models without a privileged basis are elegant, and can be an interesting analogue for certain neural network representations which don't have a privileged basis – word embeddings, or the transformer residual stream. But we'd also (and perhaps primarily) like to understand neural network representations where there are neurons which do impose a privileged basis, such as transformer MLP layers or conv net neurons.

Our goal in this section is to explore the simplest toy model which gives us a privileged basis. There are at least two ways we could do this: we could add an activation function or apply L1 regularization to the hidden layer. We'll focus on adding an activation function, since the representation we are most interested in understanding is hidden layers with neurons, such as the transformer MLP layer.

This gives us the following "ReLU hidden layer" model:

$$
\begin{aligned}
h & =\operatorname{ReLU}(W x) \\
x^{\prime} & =\operatorname{ReLU}\left(W^T h+b\right)
\end{aligned}
$$

We'll train this model on the same data as before.

Adding a ReLU to the hidden layer radically changes the model from an interpretability perspective. The key thing is that while $W$ in our previous model was challenging to interpret - recall that we visualized $W^T W$ rather than $W$:

```python
W = t.randn(2, 5)
W_normed = W / W.norm(dim=0, keepdim=True)

# Using arguments to match the color scheme of the paper
imshow(W_normed.T @ W_normed, title="Cosine similarities of each pair of 2D feature embeddings", width=600, color_continuous_scale="RdBu_r", zmin=-1.4, zmax=1.4)
```

while on the other hand, $W$ in the ReLU hidden layer model can be directly interpreted, since it connects features to basis-aligned neurons.

### Exercise - plot $W$ in privileged basis

```yaml
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 20-40 minutes on this exercise.
```

Replicate the [first set of results](https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss:~:text=model%20and%20a-,ReLU%20hidden%20layer%20model,-%3A) in the Anthropic paper on studying superposition in a privileged basis. That is, you should:

* Define a new model with a ReLU hidden layer, as described above & in the paper
    * We recommend you define a new class which inherits from `Model`, because the `__init__` and attributes will be the same as before.
    * You'll have to rewrite the `forward` function.
    * You'll also have to rewrite the `generate_batch` function, since the experimental setup has changed - see the [Experiment Setup](https://transformer-circuits.pub/2022/toy_model/index.html#computation-setup) section.
* Train the model in the same way as before
    * You'll be able to re-use the same `optimize` function code, but you'll need a different `calculate_loss` function (again, see the [Experiment Setup](https://transformer-circuits.pub/2022/toy_model/index.html#computation-setup) section).
    * You should use just one instance, with zero sparsity and uniform importance (i.e. no need to supply these arguments into your `init`)
* Plot the matrix $W$
    * You can use the code from above (but you should plot a normed version of $W$ rather than $W^T W$).

Note - if you implement this correctly, you might find that your results are a permutation of the paper's results (since they stack them in an intuitive order to make their plots).

You might also find some other small deviations from the paper's results. But the most important thing to pay attention to is how **there's a shift from monosemantic to polysemantic neurons as sparsity increases**. Monosemantic neurons do exist in some regimes! Polysemantic neurons exist in others. And they can both exist in the same model! Moreover, while it's not quite clear how to formalize this, it looks a great deal like there's a neuron-level phase change, mirroring the feature phase changes we saw earlier.

In the plots you make below, you should see:

* Total monosemanticity at 5 features & 5 neurons
* With more features than neurons, some of the neurons become polysemantic (but some remain monosemantic)

```python
# YOUR CODE HERE - replicate the paper's results
```

<details>
<summary>Solution</summary>

```python
class NeuronModel(Model):
    def __init__(
        self, 
        config: Config, 
        feature_probability: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,               
        device=device
    ):
        super().__init__(config, feature_probability, importance, device)

    def forward(
        self, 
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        activations = F.relu(einops.einsum(
           features, self.W,
           "... instances features, instances hidden features -> ... instances hidden"
        ))
        out = F.relu(einops.einsum(
            activations, self.W,
            "... instances hidden, instances hidden features -> ... instances features"
        ) + self.b_final)
        return out

    def generate_batch(self, n_batch) -> Tensor:
        feat = 2 * t.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device) - 1
        feat_seeds = t.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device)
        feat_is_present = feat_seeds <= self.feature_probability
        batch = t.where(
            feat_is_present,
            feat,
            t.zeros((), device=self.W.device),
        )
        return batch
    

def calculate_neuron_loss(
    out: Float[Tensor, "batch instances features"],
    batch: Float[Tensor, "batch instances features"],
    model: Model
) -> Float[Tensor, ""]:
    error = model.importance * ((batch.abs() - out) ** 2)
    loss = einops.reduce(error, 'batch instances features -> instances', 'mean').sum()
    return loss


def optimize(
    model: Union[Model, NeuronModel], 
    n_batch: int = 1024,
    steps: int = 10_000,
    print_freq: int = 100,
    lr: float = 1e-3,
    lr_scale: Callable = constant_lr,
):
    '''
    Optimizes the model using the given hyperparameters.
    
    This version can accept either a Model or NeuronModel instance.
    '''
    cfg = model.config

    optimizer = t.optim.AdamW(list(model.parameters()), lr=lr)

    progress_bar = tqdm(range(steps))
    for step in progress_bar:
        step_lr = lr * lr_scale(step, steps)
        for group in optimizer.param_groups:
            group['lr'] = step_lr
            optimizer.zero_grad()
            batch = model.generate_batch(n_batch)
            out = model(batch)
            if isinstance(model, NeuronModel):
                loss = calculate_neuron_loss(out, batch, model)
            else:
                loss = calculate_loss(out, batch, model)
            loss.backward()
            optimizer.step()

            if step % print_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(loss=loss.item()/cfg.n_instances, lr=step_lr)

                

for n_features in [5, 6, 8]:

    config = Config(
        n_instances = 1,
        n_features = n_features,
        n_hidden = 5,
    )

    model = NeuronModel(
        config=config,
        device=device,
        feature_probability=t.ones(model.config.n_instances, device=device)[:, None],
    )

    optimize(model, steps=1000)

    W = model.W[0]
    W_normed = W / W.norm(dim=0, keepdim=True)
    imshow(W_normed.T, width=600, color_continuous_scale="RdBu_r", zmin=-1.4, zmax=1.4)
```

</details>

Try playing around with different settings (sparsity, importance). What kind of results do you get?

You can also try and go further, replicating results later in the paper (e.g. the neuron weight bar plots further on in the paper).

<details>
<summary>Code for neuron weight plots</summary>

Note - this currently fails to replicate the [paper's plots](https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss:~:text=The%20solutions%20are%20visualized%20below), because the `render_features` function plots by feature rather than by neuron. If I have time, I'll come back and write a new function to reproduce these plots. If you're reading this and are interested in doing this, please send me a message at `cal.s.mcdougall@gmail.com`.

```python
for n_features in [5, 6, 8]:

    config = Config(
        n_instances = 10,
        n_features = 8,
        n_hidden = 5,
    )

    model = NeuronModel(
        config=config,
        device=device,
        feature_probability = (20 ** -t.linspace(0, 1, config.n_instances))[:, None],
    )

    optimize(model)

    W = model.W[0]
    W_normed = W / W.norm(dim=0, keepdim=True)
    imshow(W_normed.T, width=400, height=400, color_continuous_scale="RdBu_r", zmin=-1.4, zmax=1.4)

    fig = render_features(model)
    fig.update_layout(width=400, height=800)
    fig.show()
```
</details>

</details>

## Summary - what have we learned?

With toy models like this, it's important to make sure we take away generalizable lessons, rather than just details of the training setup. 

The core things to take away form this paper are:

* What superposition is
* How it responds to feature importance and sparsity
* How it responds to correlated and uncorrelated features


## Feature Geometry

> Note - this section is optional, since it goes into quite extreme detail about the specific problem setup we're using here. If you want, you can jump to the next section.


We've seen that superposition can allow a model to represent extra features, and that the number of extra features increases as we increase sparsity. In this section, we'll investigate this relationship in more detail, discovering an unexpected geometric story: features seem to organize themselves into geometric structures such as pentagons and tetrahedrons!

The code below runs a third experiment, with all importances the same. We're first interested in the number of features the model has learned to represent. This is well represented with the squared **Frobenius norm** of the weight matrix $W$, i.e. $||W||_F^2 = \sum_{ij}W_{ij}^2$.

<details>
<summary>Question - can you see why this is a good metric for the number of features represented?</summary>

By reordering the sums, we can show that the squared Frobenius norm is the sum of the squared norms of each of the 2D embedding vectors:

$$
\big\|W\big\|_F^2 = \sum_{j}\big\|W_{[:, j]}\big\|^2 = \sum_j \left(\sum_i W_{ij}^2\right)
$$

Each of these embedding vectors has squared norm approximately $1$ if a feature is represented, and $0$ if it isn't. So this is roughly the total number of represented features.
</details>

If you run the code below, you'll also plot the total number of "dimensions per feature", $m/\big\|W\big\|_F^2$.


```python
config = Config(
    n_features = 200,
    n_hidden = 20,
    n_instances = 20,
)

feature_probability = (20 ** -t.linspace(0, 1, config.n_instances))

model = Model(
    config=config,
    device=device,
    # For this experiment, use constant importance.
    feature_probability = feature_probability[:, None]
)

optimize(model)

plot_feature_geometry(model)
```

Surprisingly, we find that this graph is "sticky" at $1$ and $1/2$. On inspection, the $1/2$ "sticky point" seems to correspond to a precise geometric arrangement where features come in "antipodal pairs", each being exactly the negative of the other, allowing two features to be packed into each hidden dimension. It appears that antipodal pairs are so effective that the model preferentially uses them over a wide range of the sparsity regime.

It turns out that antipodal pairs are just the tip of the iceberg. Hiding underneath this curve are a number of extremely specific geometric configurations of features.

How can we discover these geometric configurations? Consider the following metric, which the authors named the **dimensionality** of a feature:

$$
D_i = \frac{\big\|W_i\big\|^2}{\sum_{j} \big( \hat{W_i} \cdot W_j \big)^2}
$$

Intuitively, this is a measure of what "fraction of a dimension" a specific feature gets. Let's try and get a few intuitions for this metric:

* It's never less than zero.
    * It's equal to zero if and only if the vector is the zero vector, i.e. the feature isn't represented.
* It's never greater than one (because when $j = i$, the term in the denominator sum is equal to the numerator).
    * It's equal to one if and only if the $i$-th feature vector $W_i$ is orthogonal to all other features (because then $j=i$ is the only term in the denominator sum).
    * Intuitively, in this case the feature has an entire dimension to itself.
* If there are $k$ features which are all parallel to each other, and orthogonal to all others, then they "share" the dimensionality equally, i.e. $D_i = 1/k$ for each of them.
* The sum of all $D_i$ can't be greater than the total number of features $m$, with equality if and only if all the vectors are orthogonal.

### Exercise - compute dimensionality

```yaml
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 10-15 minutes on this exercise.
```

Remember, $W$ has shape `(n_instances, n_hidden, n_features)`. The vectors $W_i$ refer to the feature vectors (i.e. they have length `n_hidden`), and you should broadcast your calculations over the `n_instances` dimension.




```python
@t.inference_mode()
def compute_dimensionality(
    W: Float[Tensor, "n_instances n_hidden n_features"]
) -> Float[Tensor, "n_instances n_features"]:
    pass
        

tests.test_compute_dimensionality(compute_dimensionality)
```

<details>
<summary>Solution</summary>

```python
@t.inference_mode()
def compute_dimensionality(
    W: Float[Tensor, "n_instances n_hidden n_features"]
) -> Float[Tensor, "n_instances n_features"]:
    # SOLUTION
    # Compute numerator terms
    W_norms = W.norm(dim=1, keepdim=True)
    numerator = W_norms.squeeze() ** 2

    # Compute denominator terms
    W_normalized = W / W_norms
    # t.clamp(W_norms, 1e-6, float("inf"))
    denominator = einops.einsum(W_normalized, W, "i h f1, i h f2 -> i f1 f2").pow(2).sum(-1)

    return numerator / denominator
```
</details>

The code below plots the fractions of dimensions, as a function of sparsity.

```python
W = model.W.detach()
dim_fracs = compute_dimensionality(W)

plot_feature_geometry(model, dim_fracs=dim_fracs)
```

What's going on here? It turns out that the model likes to create specific weight geometries and kind of jumps between the different configurations.

The moral? Superposition is very hard to pin down! There are many points between a dimensionality of 0 (not learning a feature) and 1 (dedicating a dimension to a feature). As an analogy, we often think of water as only having three phases: ice, water and steam. But this is a simplification: there are actually many phases of ice, often corresponding to different crystal structures (eg. hexagonal vs cubic ice). In a vaguely similar way, neural network features seem to also have many other phases within the general category of "superposition."

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/grid_all.png" width="900">

Note that we should take care not to read too much significance into these results. A lot of it depends delicately on the details of our experimental setup (e.g. we used $W^T W$, a positive semidefinite matrix, and there's a correspondence between low-dimensional symmetric pos-semidef matrices like these and the kinds of polytopes that we've seen in the plots above). But hopefully this has given you a sense of the relevant considerations when it comes to packing features into fewer dimensions.


## Further Reading

Here are some other papers or blog posts you might want to read, which build on the ideas we discussed in this section.

There are also a number of papers here which study individual neurons. So far, we've mainly discussed what Neel refers to as **bottleneck superposition**, when a low-dimensional space is forced to act as a kind of "storage" for a higher-dimensional space. This happens in transformers, e.g. with the residual stream as the lower-dimensional space, and the space of all possible features as the (much) higher-dimensional space. We've not considered **neuron superposition**, which is what happens when there are more features represented in neuron activation space than there are neurons.

* [Superposition, Memorization, and Double Descent](https://www.alignmentforum.org/posts/6Ks6p33LQyfFkNtYE/paper-superposition-memorization-and-double-descent)
    * Anthropic's follow up to theis superposition paper.
    * They generalise the training setup, and find that the hidden vectors in the training set often show clean structures.
* [Polysemanticity and Capacity in Neural Networks](https://arxiv.org/pdf/2210.01892.pdf)
    * Paper by Redwood, which builds on the ideas we discussed in this section. 
    * They define a measure called **capacity**, which is the same as what we called **dimensionality** above.
    * They study the effect that sparsity and kurtosis of the input distribution has on optimal capacity allocation.
* [Softmax Linear Units](https://transformer-circuits.pub/2022/solu/index.html)
    * This is a proposed architectural change which appears to increase the number of interpretable MLPs with low performance cost. In particular, it may reduce the instance of superposition.
    * TL;DR: SOLU is an activation function $\vec{x} \to \vec{x} * \operatorname{softmax}(\vec{x})$ which encourages sparsity in activations in the same way that softmax encourages sparsity (often softmax'ed probability distributions have one probability close to one and the others close to zero). Encouraging activation sparsity might make it harder for neurons to be polysemantic.
    * Note that several transformers in the TransformerLens library have been trained with SOLU - see the [model page](https://neelnanda-io.github.io/TransformerLens/model_properties_table.html) for more details.
* [Finding Neurons in a Haystack: Case Studies with Sparse Probing](https://arxiv.org/abs/2305.01610)
    * The authors train a set of sparse linear probes on neuron activations to predict the presence of certain input features.
    * They manage to find **sparse combinations of neurons which represent many features in superposition**, e.g. a neuron which activates on the bigram phrase "social security" but not either word individually (see image below).
    * This could make an excellent capstone project! If you're interested in this, we would also recommend the [OthelloGPT exercises](https://arena-ch1-transformers.streamlit.app/[1.6]_OthelloGPT) (particularly the section containing exercises on training a linear probe).

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/socialsecurity.png" width="750">

* [Open questions](https://transformer-circuits.pub/2022/toy_model/index.html#open-questions) from the original Anthropic paper. What do you think about them? Do any seem tractible to you?
""", unsafe_allow_html=True)

section_0()

streamlit_analytics.stop_tracking(
    unsafe_password=st.secrets["analytics_password"],
)