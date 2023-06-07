
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
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
    <li class='margtop'><a class='contents-el' href='#reading-material'>Reading Material</a></li>
    <li class='margtop'><a class='contents-el' href='#questions'>Questions</a></li>
    <li class='margtop'><a class='contents-el' href='#toy-model-setup'>Toy Model - setup</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#what's-the-motivation-for-this-setup'>What's the motivation for this setup?</a></li>
        <li><a class='contents-el' href='#exercise-define-your-model'><b>Exercise</b> - define your model</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#training-our-model'>Training our model</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-interpret-these-diagrams'><b>Exercise</b> - interpret these diagrams</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#visualizing-features-across-varying-sparsity'>Visualizing features across varying sparsity</a></li>
    <li class='margtop'><a class='contents-el' href='#other-concepts'>Other concepts</a></li>
    <li class='margtop'><a class='contents-el' href='#summary-what-have-we-learned'>Summary - what have we learned?</a></li>
    <li class='margtop'><a class='contents-el' href='#further-reading'>Further Reading</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/galaxies.jpeg" width="350">


Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.


# [1.7] - Toy Models of Superposition


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
from typing import Optional
from tqdm.auto import tqdm
from dataclasses import dataclass

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part7_toy_models_of_superposition"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line

from part7_toy_models_of_superposition.utils import plot_W, plot_Ws_from_model, render_features
import part7_toy_models_of_superposition.tests as tests
# import part7_toy_models_of_superposition.solutions as solutions

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

```

## Reading Material

Here are a few recommended resources to help you get started. Each one is labelled with what you should read, at minimum.

* [200 COP in MI: Exploring Polysemanticity and Superposition](https://www.alignmentforum.org/posts/o6ptPu7arZrqRCxyz/200-cop-in-mi-exploring-polysemanticity-and-superposition)
    * Read the post, up to and including "Tips" (although some parts of it might make more sense after you've read the other things here).
* Neel Nanda's [Dynalist notes on superposition](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=3br1psLRIjQCOv2T4RN3V6F2)
    * These aren't long, you should skim through them, and also use them as a reference during these exercises.
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
    * In other words, our loss function is $L = \sum_x \sum_i I_i (x_i - x_i')^2$, where $I_i$ is the importance of feature $i$.
* **Sparsity** = the probability of the corresponding element in $x$ being zero
    * In other words, this affects the way our training data is generated (see the method `generate_batch` in the `Module` class below)

The justification for using $W^T W$ is as follows: we can think of $W$ (which is a matrix of shape `(2, 5)`) as a grid of "overlap values" between the features and bottleneck dimensions. The values of the 5x5 matrix $W^T W$ are the dot products between the 2D representations of each pair of features. To make this intuition clearer, imagine each of the columns of $W$ were unit vectors, then $W^T W$ would be a matrix of cosine similarities between the features (with diagonal elements equal to 1, because the similarity of a feature with itself is 1). To see this for yourself:


```python
W = t.randn(2, 5)
W_normed = W / W.norm(dim=0, keepdim=True)

imshow(W_normed.T @ W_normed, title="Cosine similarities of each pair of 2D feature embeddings", width=600)
```

To put it another way - if the columns of $W$ were orthogonal, then $W^T W$ would be the identity (i.e. $W^{-1} = W^T$). This can't actually be the case because $W$ is a 2x5 matrix, but its columns can be "nearly orthgonal" in the sense of having pairwise cosine similarities close to -1.


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

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠🟠⚪⚪

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


```python
@dataclass
class Config:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as 
    # kinda like a batch dimension, but one which is built into our training setup.
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


The details of training aren't very conceptually important, so we've given you code to train the model below. A few notes:

* We use **learning rate schedulers** to control the learning rate as the model trains - you'll use this later on during the RL chapter.
* The code uses vanilla PyTorch rather than Lightning like you might be used to at this point.


```python
def linear_lr(step, steps):
    return (1 - (step / steps))
    
def constant_lr(*_):
    return 1.0
    
def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))
    
def optimize(
    model: Model, 
    n_batch=1024,
    steps=10_000,
    print_freq=100,
    lr=1e-3,
    lr_scale=constant_lr,
    hooks=[]
):
    cfg = model.config

    opt = t.optim.AdamW(list(model.parameters()), lr=lr)

    start = time.time()
    progress_bar = tqdm(range(steps))
    for step in progress_bar:
        step_lr = lr * lr_scale(step, steps)
        for group in opt.param_groups:
            group['lr'] = step_lr
            opt.zero_grad(set_to_none=True)
            batch = model.generate_batch(n_batch)
            out = model(batch)
            error = (model.importance*(batch.abs() - out)**2)
            loss = einops.reduce(error, 'b i f -> i', 'mean').sum()
            loss.backward()
            opt.step()

            if hooks:
                hook_data = dict(model=model, step=step, opt=opt, error=error, loss=loss, lr=step_lr)
                for h in hooks: h(hook_data)
            if step % print_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(
                    loss=loss.item() / cfg.n_instances,
                    lr=step_lr,
                )

```

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

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠🟠🟠⚪

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

fig = render_features(model, np.s_[::2])
fig.update_layout(width=1200, height=2000)
```

## Correlated or anticorrelated feature sets

One major thing we haven't considered in our experiments is **correlation**. We could guess that superposition is even more common when features are **anticorrelated** (for a similar reason as why it's more common when features are sparse). Most real-world features are anticorrelated (e.g. the feature "this is a sorted Python list" and "this is some text in an edgy teen vampire romance novel" are probably anticorrelated - that is, unless you've been reading some pretty weird fanfics).

In this section, you'll define a new data-generating function for correlated features, and run the same experiments as in the first section.

### Exercise - implement `generate_correlated_batch`

```c
Difficulty: 🟠🟠🟠🟠⚪
Importance: 🟠🟠⚪⚪⚪

You should spend up to 20-30 minutes on this exercise.

It's not conceptually important, and it's quite fiddly / delicate, so you should look at the solutions if you get stuck.
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

### Exercise - generate the other feature correlation plots

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠🟠⚪⚪

You should spend up to ~10 minutes on this exercise.

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

## Summary - what have we learned?

With toy models like this, it's important to make sure we take away generalizable lessons, rather than just details of the training setup. 

The core things to take away form this paper are:

* What superposition is
* How it responds to feature importance and sparsity
* How it responds to correlated and uncorrelated features


# Feature Geometry

> Note - this section is optional, since it goes into quite extreme detail about the specific problem setup we're using here. If you want, you can jump to the next section.


We've seen that superposition can allow a model to represent extra features, and that the number of extra features increases as we increase sparsity. In this section, we'll investigate this relationship in more detail, discovering an unexpected geometric story: features seem to organize themselves into geometric structures such as pentagons and tetrahedrons!

The code below runs a third experiment, with all importances the same. We're first interested in the number of features the model has learned to represent. This is well represented with the squard **Frobenius norm** of the weight matrix $W$, i.e. $||W||_F^2 = \sum_{ij}W_{ij}^2$.

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

fig = px.line(
    x=1/model.feature_probability[:, 0].cpu(),
    y=(model.config.n_hidden/(t.linalg.matrix_norm(model.W.detach(), 'fro')**2)).cpu(),
    log_x=True,
    markers=True,
    template="ggplot2",
    height=600,
    width=1000,
    title=""
)
fig.update_xaxes(title="1/(1-S), <-- dense | sparse -->")
fig.update_yaxes(title=f"m/||W||_F^2")
fig.show()
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

The code below pltos th


```python
@t.no_grad()
def compute_dimensionality(W):
    norms = t.linalg.norm(W, 2, dim=-1) 
    W_unit = W / t.clamp(norms[:, :, None], 1e-6, float('inf'))

    interferences = (t.einsum('eah,ebh->eab', W_unit, W)**2).sum(-1)

    dim_fracs = (norms**2/interferences)
    return dim_fracs.cpu()


dim_fracs = compute_dimensionality(model.W.transpose(-1, -2))


density = model.feature_probability[:, 0].cpu()
W = model.W.detach()

for a,b in [(1,2), (2,3), (2,5), (2,6), (2,7)]:
    val = a/b
    fig.add_hline(val, line_color="purple", opacity=0.2, annotation=dict(text=f"{a}/{b}"))

for a,b in [(5,6), (4,5), (3,4), (3,8), (3,12), (3,20)]:
    val = a/b
    fig.add_hline(val, line_color="blue", opacity=0.2, annotation=dict(text=f"{a}/{b}", x=0.05))

for i in range(len(W)):
    fracs_ = dim_fracs[i]
    N = fracs_.shape[0]
    xs = 1/density
    if i!= len(W)-1:
        dx = xs[i+1]-xs[i]
    fig.add_trace(
        go.Scatter(
            x=1/density[i]*np.ones(N)+dx*np.random.uniform(-0.1,0.1,N),
            y=fracs_,
            marker=dict(
                color='black',
                size=1,
                opacity=0.5,
            ),
            mode='markers',
        )
    )
fig.update_xaxes(showgrid=False).update_yaxes(showgrid=False).update_layout(showlegend=False)
fig.show()
```

What's going on here? It turns out that the model likes to create specific weight geometries and kind of jumps between the different configurations.

The moral? Superposition is very hard to pin down! There are many points between a dimensionality of 0 (not learning a feature) and 1 (dedicating a dimension to a feature). As an analogy, we often think of water as only having three phases: ice, water and steam. But this is a simplification: there are actually many phases of ice, often corresponding to different crystal structures (eg. hexagonal vs cubic ice). In a vaguely similar way, neural network features seem to also have many other phases within the general category of "superposition."

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/grid_all.png" width="900">

Note that we should take care not to read too much significance into these results. A lot of it depends delicately on the details of our experimental setup (e.g. we used $W^T W$, a positive semidefinite matrix, and there's a correspondence between low-dimensional symmetric pos-semidef matrices like these and the kinds of polytopes that we've seen in the plots above). There are also many real-world dynamics which our analysis hasn't even considered (e.g. the fact that superposition is much more likely when features are not just sparse but **anticorrelated**). But hopefully this has given you a sense of the relevant considerations when it comes to packing features into fewer dimensions.


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