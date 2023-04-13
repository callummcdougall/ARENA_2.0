import os
from st_dependencies import *
st.set_page_config(layout="wide")

st.markdown(r"""
<style>
div[data-testid="column"] {
    background-color: #f9f5ff;
    padding: 15px;
}
.st-ae h2 {
    margin-top: -15px;
}
p {
    line-height:1.48em;
}
.streamlit-expanderHeader {
    font-size: 1em;
    color: darkblue;
}
.css-ffhzg2 .streamlit-expanderHeader {
    color: lightblue;
}
header {
    background: rgba(255, 255, 255, 0) !important;
}
code {
    color:red;
    white-space: pre-wrap !important;
}
.css-ffhzg2 code:not(pre code) {
    color: orange;
}
.css-ffhzg2 .contents-el {
    color: white !important;
}
pre code {
    font-size:13px !important;
}
.katex {
    font-size:17px;
}
h2 .katex, h3 .katex, h4 .katex {
    font-size: unset;
}
ul.contents {
    line-height:1.3em; 
    list-style:none;
    color-black;
    margin-left: -10px;
}
ul.contents a, ul.contents a:link, ul.contents a:visited, ul.contents a:active {
    color: black;
    text-decoration: none;
}
ul.contents a:hover {
    color: black;
    text-decoration: underline;
}
</style>""", unsafe_allow_html=True)

st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#about-transformerlens">About TransformerLens</a></li>
    <li><a class="contents-el" href="#about-these-pages">About these pages</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#transformer-from-scratch">Transformer from scratch</a></li>
        <li><a class="contents-el" href="#transformerlens-induction-circuits">TransformerLens & induction circuits</a></li>
        <li><a class="contents-el" href="#interpretability-on-an-algorithmic-model">Interpretability on an algorithmic model</a></li>
        <li><a class="contents-el" href="#grokking">Grokking</a></li>
        <li><a class="contents-el" href="#indirect-obbjct-identification">Indirect Objct Identification</a></li>
    </ul></li>
    <li><a class="contents-el" href="#how-you-should-use-this-material">How you should use this material</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#option-1-colab">Option 1: Colab</a></li>
        <li><a class="contents-el" href="#option-2-your-own-ide">Option 2: Your own IDE</a></li>
        <li><a class="contents-el" href="#chatbot-assistant">Chatbot assistant</a></li>
    </ul></li>
    <li><a class="contents-el" href="#prerequisites">Prerequisites</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#linear-algebra">Linear algebra</a></li>
        <li><a class="contents-el" href="#neural-networks">Neural Networks</a></li>
        <li><a class="contents-el" href="#basic-python">Basic Python</a></li>
        <li><a class="contents-el" href="#transformer-circuits">Transformer Circuits</a></li>
        <li><a class="contents-el" href="#other-topics">Other topics</a></li>
    </ul></li>
    <li><a class="contents-el" href="#feedback">Feedback</a></li>
</ul>
""", unsafe_allow_html=True)

def section_home():
    st_image("magnifying-glass-2.png", width=600)
    # start
    st.markdown(r"""
# Mechanistic Interpretability & TransformerLens
    
This page contains a collation of resources and exercises on interpretability. The focus is on [`TransformerLens`](https://github.com/neelnanda-io/TransformerLens), a library maintained by Neel Nanda.

## About TransformerLens

From the description in Neel Nanda's repo:

> TransformerLens is a library for doing [mechanistic interpretability](https://distill.pub/2020/circuits/zoom-in/) of GPT-2 Style language models. The goal of mechanistic interpretability is to take a trained model and reverse engineer the algorithms the model learned during training from its weights. It is a fact about the world today that we have computer programs that can essentially speak English at a human level (GPT-3, PaLM, etc), yet we have no idea how they work nor how to write one ourselves. This offends me greatly, and I would like to solve this!
> 
> TransformerLens lets you load in an open source language model, like GPT-2, and exposes the internal activations of the model to you. You can cache any internal activation in the model, and add in functions to edit, remove or replace these activations as the model runs. The core design principle I've followed is to enable exploratory analysis. One of the most fun parts of mechanistic interpretability compared to normal ML is the extremely short feedback loops! The point of this library is to keep the gap between having an experiment idea and seeing the results as small as possible, to make it easy for **research to feel like play** and to enter a flow state. Part of what I aimed for is to make my experience of doing research easier and more fun, hopefully this transfers to you!

## About these pages

Here is a rundown of all the pages in the Streamlit app, and what to expect from each of them: 

### Transformer from scratch

This shows you how to build a transformer from scratch, which mirrors the prototypical transformers used in the `transformer_lens` library. Even if you're familiar with transformers I'd recommend at least skimming this page, to get an idea of how this library works. This page is a lot shorter than most of the other pages.

### TransformerLens & induction circuits

* Sections 1️⃣ and 2️⃣ demonstrate the core features of TransformerLens, and walk you through identifying induction circuits in models. There is a stronger focus on introducing concepts and laying groundwork here, rather than coding and exercises.
* Section 3️⃣ shows you how to use hooks to access a model's activations (and intervene on them). It has a stronger focus on coding.
* Section 4️⃣ shows you how to reverse-engineer induction circuits directly from the model's weights. It is much more conceptual rather than coding-focused, and is based heavily on Anthropic's [Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) paper.

### Interpretability on an algorithmic model

These exercises take you through the application of TransformerLens features to an algorithmic task - interpreting a model trained to classify bracket strings as balanced or unbalanced. There are fewer new concepts here than in the previous section; instead there's more of a focus on forming and testing hypotheses about a model's behaviour in a small, relatively easy-to-understand domain.

### Grokking

Coming soon!

### Indirect Objct Identification

Coming soon!

## How you should use this material

### Option 1: Colab

The simplest way to get these exercises up and running is to use Colab. This guarantees good GPU support, and means you'll spend less time messing around with environments and IDEs. Each set of exercises will have a link to the accompanying Colab notebook, which you should make a copy of and work through. The Colabs have essentially the same structure as the Streamlit pages.

[Here](https://drive.google.com/drive/folders/1Yi8dCqfhm8L9g7QLVAYptGsPnaQnAYgT?usp=sharing) is the link to the folder containing all the Colabs, and the data you'll need. You can find the individual Colabs below (not all the exercises have been converted into Colab form yet):

* Transformer from scratch: [**exercises**](https://colab.research.google.com/drive/1LpDxWwL2Fx0xq3lLgDQvHKM5tnqRFeRM?usp=share_link), [**solutions**](https://colab.research.google.com/drive/1ND38oNmvI702tu32M74G26v-mO5lkByM?usp=share_link)
* TransformerLens & induction circuits: [**exercises**](https://colab.research.google.com/drive/17i8LctAgVLTJ883Nyo8VIEcCNeKNCYnr?usp=share_link), [**solutions**](https://colab.research.google.com/drive/15p2TgU7RLaVjLVJFpwoMhxOWoAGmTlI3?usp=share_link)
* Interpretability on an algorithmic model: [**exercises**](https://colab.research.google.com/drive/1puoiNww84IAEgkUMI0PnWp_aL1vai707?usp=sharing), [**solutions**](https://colab.research.google.com/drive/1Xm2AlQtonkvSQ1tLyBJx31AYmVjopdcf?usp=sharing)

You can make a copy of the **exercises** notebooks in your own drive, and fill in the code cells whenever indicated. The solutions will be available in dropdowns next to each of the code cells. You can also look at the **solutions** notebooks if you're just interested in the output (since they have all the correct code filled in, and all the output on display within the notebook).

### Option 2: Your own IDE

An alternative way to use this material is to run it on an IDE of your own choice (we strongly recommend VSCode). The vast majority of the exercises will not require a particularly good GPU, and where there are exceptions we will give some advice for how to get the most out of the exercises regardless.

Full instructions for running the exercises in this way:

* Clone the [GitHub repo](https://github.com/callummcdougall/TransformerLens-intro) into your local directory.
* Open in your choice of IDE (we recommend VSCode).
* Make & activate a virtual environment
    * We strongly recommend using `conda` for this. You can install `conda` [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), and find basic instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
* Install requirements.
    * First, install PyTorch using the following command: `conda install pytorch=1.11.0 torchdata torchvision -c pytorch -y`.
    * Then install the rest of the requirements by navigating to the directory and running `pip install -r requirements.txt`.
* While in the directory, run `streamlit run Home.py` in your terminal (this should work since Streamlit is one of the libraries in `requirements.txt`).
    * This should open up a local copy of the page you're reading right now, and you're good to go!

To complete one of the exercise pages, you should:

* Navigate to the appropriate directory in the repo (e.g. `exercises/transformer_from_scratch`)
* Create a file called `answers.py` (or `answers.ipynb` if you prefer using notebooks)
* Work your way through the Streamlit page, copying over the code blocks into your own IDE and editing / running them there.
    * For each page, this will always start with some basic imports (including libraries like `transformer_lens`, as well as local files like `tests.py` and `solutions.py`).
""")
    # end
    with st.expander("Help - I get error `ImportError: DLL load failed while importing lib` when I try and import things."):
        st.markdown(r"""
To fix this problem, run the following code in your terminal:

```
conda install libboost boost-cpp -c conda-forge
```
 
then restart your IDE. Hopefully this fixes the problem.
""")
    # start
    st.markdown(r"""
### Chatbot assistant

In the sidebar of this page, below the contents page, you will (at first) see an error message saying "Please set the OpenAI key...". This is space for a chatbot assistant, which can help answer your questions about the material. Take the following steps to set it up:

* Go to the [OpenAI API](https://openai.com/blog/openai-api) and sign up for an account.
* Create a secret key from [this page](https://platform.openai.com/account/api-keys). Copy this key.
* Create a file `.streamlit/secrets.toml` in this repo, and have the first line read `api_secret = "<your key>"`.
* Refresh the page, and you should now be able to use the chatbot.

This interface was built using the `openai` library, and it exists to help answer questions you might have about the material. All prompts from this chatbot are prepended with most\* of the material on the page and section you're currently reading. For instance, try passing in the question ***What are 2 ways to use this material?*** to the chatbot, and it should describe the two options given above (i.e. colab, or your own IDE). This feature is very experimental, so please [let me know](mailto:cal.s.mcdougall@gmail.com) if you have any feedback!

\**Because of the context window, the entire page isn't always included in the prompt (e.g. generally code blocks aren't included). When in doubt, you can copy sections of the page into the prompt and run it! If you get an error message saying that the prompt is too long, then you can use the **clear chat** button and start again.*

Here are some suggestions for the kinds of questions you can ask the chatbot (in the appropriate sections of the course):

* *(copying in a function to the start of your prompt)* What does this function do?
* Why are skip connections important in deep learning?
* Give a visual explanation of ray tracing.
* What are the different methods of hyperparameter tuning provided by Weights and Biases?

## Prerequisites

This material starts with a guided implementation of transformers, so you don't need to understand how they work before starting. However, there are a few things we do recommend:

### Linear algebra

This is probably the most important prerequisite. You should be comfortable with the following concepts:

- [Linear transformations](https://www.youtube.com/watch?v=kYB8IZa5AuE) - what they are, and why they matter
- How [matrix multiplication](http://mlwiki.org/index.php/Matrix-Matrix_Multiplication) works
- Basic matrix properties: rank, trace, determinant, transpose, inverse
- Bases, and basis transformations

[This video series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) by 3B1B provides a good overview of these core topics (although you can probably skip it if you already have a reasonably strong mathematical background).
""")
    st.markdown(r"""
<img src="https://imgs.xkcd.com/comics/machine_learning_2x.png" alt="xkcd" width="300"/>
""", unsafe_allow_html=True)
    st.markdown("")
    st.markdown(r"""

### Neural Networks

It would be very helpful to understand the basics of what neural networks are, and how they work. The best introductory resources here are 3B1B's videos on neural networks:

* [But what is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk)
* [Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w)
* [What is backpropagation really doing? | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

### Basic Python

It's important to be able to code at a reasonably proficient level in Python. As a rough guide, you should:

* Understand most (e.g. >75%) of the material [here](https://book.pythontips.com/en/latest/), up to and including chapter 21. Not all of this will be directly useful for these exercises, but reading through this should give you a rough idea of the kind of level that is expcted of you.
* Be comfortable with easy or medium [LeetCode problems](https://leetcode.com/).
* Know what vectorisation is, and how to use languages like NumPy or PyTorch to perform vectorised array operations.
    * In particular, these exercises are all based in PyTorch, so going through a tutorial like [this one](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) might be a good idea (the tensors section is very important; most of the following sections would have diminishing returns from studying but might still be useful).

### Transformer Circuits

This isn't a prerequisite for the first set of exercises (and in fact, if you haven't come across transformers before, you're recommended to do the exercises **Transformers from scratch** before attempting to read this paper). However, after you've done those exercises, you will need to understand the material in the paper [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) before moving on to the later exercises. 

The paper is highly technical, so don't worry if it takes you a few passes to understand it. Here are a few things you can do to help build up to a full understanding of the core parts of the paper:

* Watch Neel's [video walkthrough](https://www.youtube.com/watch?v=KV5gbOmHbjU) instead, which does a great job highlighting which parts of the paper are the most conceptually important.
* Read Neel's [Mechanistic Interpretability Glossary](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J), which is a great resource for understanding the terminology and concepts that are used in the following pages.
* Read [this LessWrong post](https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated), which uses a series of diagrams to explain induction heads and how they work (these are at the core of the second set of exercises).
* When it comes to reading the actual paper, the set of tips below should help you get the most out of it (these were written by Neel, and many of them overlap with the points he makes in his video walkthrough).
""")
    # end
    with st.expander("Tips & Insights for the Paper"):
        st.markdown(r"""

* The eigenvalue stuff is very cool, but doesn't generalise that much, it's not a priority to get your head around
* It's really useful to keep clear in your head the difference between parameters (learned numbers that are intrinsic to the network and independent of the inputs) and activations (temporary numbers calculated during a forward pass, that are functions of the input).
    * Attention is a slightly weird thing - it's an activation, but is also used in a matrix multiplication with another activation (z), which makes it parameter-y.
        * The idea of freezing attention patterns disentangles this, and lets us treat it as parameters.
* The residual stream is the fundamental object in a transformer - each layer just applies incremental updates to it - this is really useful to keep in mind throughout!
    * This is in contrast to a classic neural network, where each layer's output is the central object
    * To underscore this, a funky result about transformers is that the aspect ratio isn't *that* important - if you increase d_model/n_layer by a factor of 10 from optimal for a 1.5B transformer (ie controlling for the number of parameters), then loss decreases by <1%.
* The calculation of attention is a bilinear form (ie via the QK circuit) - for any pair of positions it takes an input vector from each and returns a scalar (so a ctx x ctx tensor for the entire sequence), while the calculation of the output of a head pre weighting by attention (ie via the OV circuit) is a linear map from the residual stream in to the residual stream out - the weights have the same shape, but are doing functions of completely different type signatures!
* How to think about attention: A framing I find surprisingly useful is that attention is the "wiring" of the neural network. If we hold the attention patterns fixed, they tell the model how to move information from place to place, and thus help it be effective at sequence prediction. But the key interesting thing about a transformer is that attention is *not* fixed - attention is computed and takes a substantial fraction of the network's parameters, allowing it to dynamically set the wiring. This can do pretty meaningful computation, as we see with induction heads, but is in some ways pretty limited. In particular, if the wiring is fixed, an attention only transformer is a purely linear map! Without the ability to intelligently compute attention, an attention-only transformer would be incredibly limited, and even with it it's highly limited in the functional forms it can represent.
    * Another angle - attention as generalised convolution. A naive transformer would use 1D convolutions on the sequence. This is basically just attention patterns that are hard coded to be uniform over the last few tokens - since information is often local, this is a decent enough default wiring. Attention allows the model to devote some parameters to compute more intelligent wiring, and thus for a big enough and good enough model will significantly outperform convolutions.
* One of the key insights of the framework is that there are only a few activations of the network that are intrinsically meaningful and interpretable - the input tokens, the output logits and attention patterns (and neuron activations in non-attention-only models). Everything else (the residual stream, queries, keys, values, etc) are just intermediate states on a calculation between two intrinsically meaningful things, and you should instead try to understand the start and the end. Our main goal is to decompose the network into many paths between interpretable start and end states
    * We can get away with this because transformers are really linear! The composition of many linear components is just one enormous matrix
* A really key thing to grok about attention heads is that the QK and OV circuits act semi-independently. The QK circuit determines which previous tokens to attend to, and the OV circuit determines what to do to tokens *if* they are attended to. In particular, the residual stream at the destination token *only* determines the query and thus what tokens to attend to - what the head does *if* it attends to a position is independent of the destination token residual stream (other than being scaled by the attention pattern).
* Skip trigram bugs are a great illustration of this - it's worth making sure you really understand them. The key idea is that the destination token can *only* choose what tokens to pay attention to, and otherwise not mediate what happens *if* they are attended to. So if multiple destination tokens want to attend to the same source token but do different things, this is impossible - the ability to choose the attention pattern is insufficient to mediate this.
    * Eg, keep...in -> mind is a legit skip trigram, as is keep...at -> bay, but keep...in -> bay is an inherent bug from this pair of skip trigrams
* The tensor product notation looks a lot more complicated than it is. $A \otimes W$ is shorthand for "the function $f_{A,W}$ st $f_{A,W}(x)=AxW$" - I recommend mentally substituting this in in your head everytime you read it.
* K, Q and V composition are really important and fairly different concepts! I think of each attention head as a circuit component with 3 input wires (Q,K,V) and a single output wire (O). Composition looks like connecting up wires, but each possible connection is a choice! The key, query and value do different things and so composition does pretty different things.
    * Q-Composition, intuitively, says that we want to be more intelligent in choosing our destination token - this looks like us wanting to move information to a token based on *that* token's context. A natural example might be the final token in a several token word or phrase, where earlier tokens are needed to disambiguate it, eg `E|iff|el| Tower|`
    * K-composition, intuitively, says that we want to be more intelligent in choosing our source token - this looks like us moving information *from* a token based on its context (or otherwise some computation at that token).
        * Induction heads are a clear example of this - the source token only matters because of what comes before it!
    * V-Composition, intuitively, says that we want to *route* information from an earlier source token *other than that token's value* via the current destination token. It's less obvious to me when this is relevant, but could imagine eg a network wanting to move information through several different places and collate and process it along the way
        * One example: In the ROME paper, we see that when models recall that "The Eiffel Tower is in" -> " Paris", it stores knowledge about the Eiffel Tower on the " Tower" token. When that information is routed to `| in|`, it must then map to the output logit for `| Paris|`, which seems likely due to V-Composition
* A surprisingly unintuitive concept is the notion of heads (or other layers) reading and writing from the residual stream. These operations are *not* inverses! A better phrasing might be projecting vs embedding.
    * Reading takes a vector from a high-dimensional space and *projects* it to a smaller one - (almost) any two pair of random vectors will have non-zero dot product, and so every read operation can pick up *somewhat* on everything in the residual stream. But the more a vector is aligned to the read subspace, the most that vector's norm (and intuitively, its information) is preserved, while other things are lower fidelity
        * A common reaction to these questions is to start reasoning about null spaces, but I think this is misleading - rank and nullity are discrete concepts, while neural networks are fuzzy, continuous objects - nothing ever actually lies in the null space or has non-full rank (unless it's explicitly factored). I recommend thinking in terms of "what fraction of information is lost". The null space is the special region with fraction lost = 1
    * Writing *embeds* a vector into a small dimensional subspace of a larger vector space. The overall residual stream is the sum of many vectors from many different small subspaces.
        * Every read operation can see into every writing subspace, but will see some with higher fidelity, while others are noise it would rather ignore.
    * It can be useful to reason about this by imagining that d_head=1, and that every vector is a random Gaussian vector - projecting a random Gaussian onto another in $\mathbb{R}^n$ will preserve $\frac{1}{n}$ of the variance, on average.
* A key framing of transformers (and neural networks in general) is that they engage in **lossy compression** - they have a limited number of dimensions and want to fit in more directions than they have dimensions. Each extra dimension introduces some interference, but has the benefit of having more expressibility. Neural networks will learn an optimal-ish solution, and so will push the compression as far as it can until the costs of interference dominate.
    * This is clearest in the case of QK and OV circuits - $W_QK=W_Q^TW_K$ is a d_model x d_model matrix with rank d_head. And to understand the attention circuit, it's normally best to understand $W_QK$ on its own. Often, the right mental move is to forget that $W_QK$ is low rank, to understand what the ideal matrix to learn here would be, and then to assume that the model learns the best low rank factorisation of that.
        * This is another reason to not try to interpret the keys and queries - the intermediate state of a low rank factorisations are often a bit of a mess because everything is so compressed (though if you can do SVD on $W_QK$ that may get you a meaningful basis?)
        * Rough heuristic for thinking about low rank factorisations and how good they can get - a good way to produce one is to take the SVD and zero out all but the first d_head singular values.
    * This is the key insight behind why polysemanticity (back from w1d5) is a thing and is a big deal - naturally the network would want to learn one feature per neuron, but it in fact can learn to compress more features than total neurons. It has some error introduced from interference, but this is likely worth the cost of more compression.
        * Just as we saw there, the sparsity of features is a big deal for the model deciding to compress things! Inteference cost goes down the more features are sparse (because unrelated features are unlikely to co-occur) while expressibility benefits don't really change that much.
    * The residual stream is the central example of this - every time two parts of the network compose, they will be communicating intermediate states via the residual stream. Bandwidth is limited, so these will likely try to each be low rank. And the directions within that intermediate product will *only* make sense in the context of what the writing and reading components care about. So interpreting the residual stream seems likely fucked - it's just specific lower-dimensional parts of the residual stream that we care about at each point, corresponding to the bits that get preserved by our $W_Q$ / $W_K$ / $W_V$ projection matrices, or embedded by our $W_O$ projection matrix. The entire residual stream will be a mix of a fuckton of different signals, only some of which will matter for each operation on it.
* The *'the residual stream is fundamentally uninterpretable'* claim is somewhat overblown - most models do dropout on the residual stream which somewhat privileges that basis
    * And there are [*weird*](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/) results about funky directions in the residual stream.
* Getting your head around the idea of a privileged basis is very worthwhile! The key mental move is to flip between "a vector is a direction in a geometric space" and "a vector is a series of numbers in some meaningful basis, where each number is intrinsically meaningful". By default, it's easy to spend too much time in the second mode, because every vector is represented as a series of numbers within the GPU, but this is often less helpful!

### An aside on why we need the tensor product notation at all

Neural networks are functions, and are built up of several subcomponents (like attention heads) that are also functions - they are defined by how they take in an input and return an output. But when doing interpretability we want the ability to talk about the network as a function intrinsically and analyse the structure of this function, *not* in the context of taking in a specific input and getting a specific output. And this means we need a language that allows us to naturally talk about functions that are the sum (components acting in parallel) or composition (components acting in series) of other functions.

A simple case of this: We're analysing a network with several linear components acting in parallel - component $C_i$ is the function $x \rightarrow W_ix$, and can be represented intrinsically as $W_i$ (matrices are equivalent to linear maps). We can represent the layer with all acting in parallel as $x \rightarrow \sum_i W_ix=(\sum_i W_i)x$, and so intrinsically as $\sum_i W_i$ - this is easy because matrix notation is designed to make addition of.

Attention heads are harder because they map the input tensor $x$ (shape: `[position x d_model]`) to an output $Ax(W_OW_V)^T$ - this is a linear function, but now on a *tensor*, so we can't trivially represent addition and composition with matrix notation. The paper uses the notation $A\otimes W_OW_V$, but this is just different notation for the same underlying function. The output of the layer is the sum over the 12 heads: $\sum_i A^{(i)}x(W_O^{(i)}W_V^{(i)})^T$. And so we could represent the function of the entire layer as $\sum_i A^{(i)} x (W_O^{(i)}W_V^{(i)})$. There are natural extensions of this notation for composition, etc, though things get much more complicated when reasoning about attention patterns - this is now a bilinear function of a pair of inputs: the query and key residual streams. (Note that $A$ is also a function of $x$ here, in a way that isn't obvious from the notation.)

The key point to remember is that if you ever get confused about what a tensor product means, explicitly represent it as a function of some input and see if things feel clearer.
""")
    # start
    st.markdown(r"""
### Other topics

Here are a few other topics that would probably be useful to have some familiarity with. They are listed in approximately descending order of importance (and none of them are as important as the three sections above):

* Basic probability & statistics (e.g. normal and uniform distributions, independent random variables, estimators)
* Calculus (and how it relates to backpropagation and gradient descent)
* Information theory (e.g. what is cross entropy, and what does it mean for a predictive model to minimise cross entropy loss between its predictions and the true labels)
* Familiarity with other useful Python libraries (e.g. `einops` for rearranging tensors, `typing` for typechecking, `plotly` for interactive visualisations)
* Working with VSCode, and basic Git (this will be useful if you're doing these exercises from VSCode rather than from Colab)
""")
    # end
    st.markdown(r"""
## Feedback

If you have any feedback on this course (e.g. bugs, confusing explanations, parts that you feel could be structured better), please let me know using [this Google Form](https://forms.gle/2ZhdHa87wWsrATjh9).
""")

if "current_page" not in st.session_state:
    st.session_state["current_page"] = ["", ""]

def page():
    section_home()
    prepend = parse_text_from_page(r"home", r"section_home")
    current_page = r"Home"
    st.session_state["current_page"] = [current_page, st.session_state["current_page"][0]]
    new_page = st.session_state["current_page"][0] != st.session_state["current_page"][1]
    chatbot_setup(prepend=prepend, new_page=new_page, debug=False)


page()