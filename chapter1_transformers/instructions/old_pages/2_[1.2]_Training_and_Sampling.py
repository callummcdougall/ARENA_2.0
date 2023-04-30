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

NAMES = []

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

def section_home():
    st.sidebar.markdown(r"""
## Table of contents

<ul class="contents">
    <li><a class="contents-el" href="#introduction">Introduction</a></li>
    <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
</ul>""", unsafe_allow_html=True)
    st.markdown(r"""
Links to Colab: [**exercises**](https://colab.research.google.com/drive/1CzqjUwFyvjvodKq5s41mBPrvhuCtE7ny?usp=share_link), [**solutions**](https://colab.research.google.com/drive/1y006XhAjInACWhne66zlHoUHx9ncNv3p?usp=share_link)
""")
    st_image("sampling.png", 350)
    # start
    st.markdown(r"""
# Training and Sampling

## Introduction

In the previous set of exercises, we built a transformer from scratch. Here, we're going to look closer at how a transformer works in practice. We'll cover three topics: how to train transformers, how to sample from their output to autoregressively generate text, and how to use caching to run them more efficiently.

These exercises mainly focus on building up your understanding of transformers, and the important considerations that go into using them. Subsequent exercises will focus more on interpretability, so you can skip to them if you want (this material generally won't be very important for future exercises).

## Learning Objectives

Here are the learning objectives for each section of the tutorial. At the end of each section, you should refer back here to check that you've understood everything.
""")
    st.info(r"""
## 1️⃣ Training

* Review the interpretation of a transformer's output, and learn how it's trained by minimizing cross-entropy loss between predicted and actual next tokens
* Construct datasets and dataloaders for the corpus of Shakespeare text
* Implement a transformer training loop
""")
    st.info(r"""
## 2️⃣ Sampling

* Learn how to sample from a transformer
    * This includes basic methods like greedy search or top-k, and more advanced methods like beam search
""")
    st.info(r"""
## 3️⃣ Caching

* Learn how to cache the output of a transformer, so that it can be used to generate text more efficiently
* Update your sampling functions to make use of your caching methods
""")
    # end

    
def section_training():
    st.sidebar.markdown(r"""
## Table of contents

<ul class="contents">
    <li><a class="contents-el" href="#imports">Imports</a></li>
    <li><a class="contents-el" href="#cross-entropy-loss">Cross entropy loss</a></li>
    <li><a class="contents-el" href="#tokenizers">Tokenizers</a></li>
    <li><a class="contents-el" href="#preparing-text">Preparing text</a></li>
    <li><a class="contents-el" href="#datasets-and-dataloaders">Datasets and Dataloaders</a></li>
    <li><a class="contents-el" href="#training-loop">Training loop</a></li>

</ul>""", unsafe_allow_html=True)
    st.markdown(r"""
# Training
""")
    # start
    st.info(r"""
### Learning Objectives

* Review the interpretation of a transformer's output, and learn how it's trained by minimizing cross-entropy loss between predicted and actual next tokens
* Construct datasets and dataloaders for the corpus of Shakespeare text
* Implement a transformer training loop
""")
    st.markdown(r"""
Hopefully, you've now successfully implemented a transformer, and seen how to use it to generate output autoregressively. You might also have seen the example training loop at the end of the last section. Here, you'll train your transformer in a more hands-on way, using the [complete works of William Shakespeare](https://www.gutenberg.org/files/100/100-0.txt).

This is the task recommended by Jacob Hilton in his [curriculum](https://github.com/jacobhilton/deep_learning_curriculum).
""")
    # end
    st.markdown(r"""
## Imports

```python
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.chdir("..")
from transformer_from_scratch.solutions import Config, DemoTransformer
from training_and_sampling import tests, plot_utils
os.chdir("training_and_sampling")

import re
import torch as t
from torch import nn
from torch.utils.data import DataLoader
import transformers
from typing import List, Tuple, Union, Optional, Callable, Dict
import numpy as np
import einops
from dataclasses import dataclass
from frozendict import frozendict
import plotly.express as px
from tqdm import tqdm
from torchtyping import TensorType as TT
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, PreTrainedTokenizer
import time
import numpy as np

MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")
```
""")
    # start
    st.markdown(r"""
## Cross entropy loss

Your transformer's input has shape `(batch, seq_len)`, where the `[i, j]`-th element is the token id of the `j`-th token in the `i`-th sequence. Your transformer's output has shape `(batch, seq_len, vocab_size)`, where the `[i, j, :]`-th element is a vector of logits, representing a probability distribution over the token that **follows** the `j`-th token in the `i`-th sequence.

When training our model, we use cross-entropy loss between the model's predictions and the actual next tokens. In other words, we can take the `[:, :-1, :]`-th slice of our output (which is a tensor of probability distributions for the **last** `seq_len - 1` tokens in each sequence), and compare this to the `[:, 1:, :]`-th slice (which represents the actual tokens we're trying to predict).

In the last section, we saw the function `lm_cross_entropy_loss` which calculated this for us. Let's take another look at this function, so we understand how it works:

```python
def lm_cross_entropy_loss(logits: t.Tensor, tokens: t.Tensor):
    # Measure next token loss
    # Logits have shape [batch, position, d_vocab]
    # Tokens have shape [batch, position]
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()
```

First, we get `log_probs`, which are the log probabilities of each token in the vocab. Log probs are (as you'd probably guess!) the log of the probabilities implied by the logit distribution. We get them from logits by taking softmax, then taking log again (so they're equal to logits, up to a constant difference). If you examine the formula for [cross entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), you'll notice that it's just the negative of the log probability of the correct token.

In the second line, we use the `gather` method to take the log probabilities corresponding to the correct token. This is a bit confusing, and you don't need to understand the exact syntax of `gather`. This line of code does the following:
* Indexes `log_probs`, taking the `[:, :-1]`-th slice (so we have the logits corresponding to the **last** `seq_len - 1` tokens in each sequence)
* Indexes `tokens`, taking the `[:, 1:]`-th slice (so we have the actual tokens we're trying to predict)
* Indexes into the reduced `log_probs` tensor using `gather`, so we get the log probabilities of the correct tokens

Finally, we take the mean of the negative log probabilities, and return this as our loss. Remember that log probs are always negative (because log of a number less than 1 is negative), so our loss will always be non-negative. It will tend to zero only if our model tends towards assigning 100% probability to the correct token, and 0% to all others.

## Tokenizers

Now that we've got cross entropy loss out of the way, let's start working with our dataset. We'll be using the Shakespeare corpus for this exercises; you can get the text as follows:

```python
with open("shakespeare-corpus.txt", encoding="utf-8") as file:
    text = file.read()
```

You should print out the first few lines of this text, and get a feel for what it looks like.

Rather than using a fancy tokenizer, we'll just split the text into tokens using a regular expression. This is a bit crude, but it's good enough for our purposes.
""")
    # end
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `SimpleTokenizer`

Below, you should fill in the `SimpleTokenizer` class. Some guidance for this exercise:

#### __init__

The `text` argument is meant to be a string (this will be the same as the `text` object you defined above). Here, you should define `self.words` as a list of all the different tokens that appear in the text, sorted in some reasonable way (you can split the text with `re.split(r"\b", text))`). You should then define `self.word_to_index` and `self.index_to_word`, which are dictionaries that map tokens to their token ids, and vice-versa (with the token ids being the positions of the tokens in `self.words`).

Also, it's good practice to include an unknown token `unk` in your vocabulary, just in case you feed the model a token that it hasn't seen before (you can give it the index one larger than the largest in your words list). We won't bother using a start token here (although you might want to think about doing this, as a bonus exercise).

#### `encode`

This takes in some text, and returns tokens. If `return_tensors` is None (the default), this should return a simple list of integers. If `return_tensors == "pt"`, this should return a PyTorch tensor of shape `(1, seq_len)` (it's good practice to always add a batch dimension, even if there's only one sequence in the batch).

If the input text contains an unknown token, then you can print an error message (or raise an exception).

#### `decode`

Finally, this should take in a list or tensor of tokens (you can assume that the batch dimension will be 1 if it's a tensor), and returns a string of the decoded text.
        
```python
class SimpleTokenizer():

    def __init__(self, text: str):
        pass

    def encode(self, input_text, return_tensors: Optional[str] = None) -> Union[List, t.Tensor]:
        '''
        Tokenizes and encodes the input text.

        If `return_tensors` is None, should return list of Python integers.
        If `return_tensors` is "pt", should return a PyTorch tensor of shape (1, num_tokens).
        '''
        pass

    def decode(self, tokens: Union[List, t.Tensor]):
        '''
        Decodes the tokens into a string of text.
        '''
        pass


if MAIN:
    mytokenizer = SimpleTokenizer(text)

    # Some basic testing
    assert isinstance(mytokenizer.encode("Macbeth"), list)
    assert isinstance(mytokenizer.encode("Macbeth", return_tensors="pt"), t.Tensor)
    assert mytokenizer.decode(mytokenizer.encode("Macbeth")) == "Macbeth"
    assert mytokenizer.index_to_word[mytokenizer.encode("Macbeth")[1]]
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
class SimpleTokenizer():

    def __init__(self, text: str):
        self.text = text
        self.words = sorted(set(re.split(r"\b", text)))
        self.unk = len(self.words) + 1
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
```
""")
    # start
    st.markdown(r"""
## Preparing text

We have our tokenizer, but we still need to be able to take in our `text` object and turn it into a tensor of token ids, without any of them overlapping. This is important because overlapping sequences might cause use to double-count certain sequences during training, and will make it seem like our model is learning faster than it really is.
""")
    # end
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `prepare_text`

Below, you should fill in the `prepare_text` function.

```python
def prepare_text(text: str, max_seq_len: int, tokenizer: SimpleTokenizer) -> TT["batch", "max_seq_len"]:
    '''
    Takes a string of text, and returns an array of tokens rearranged into chunks of size max_seq_len.
    '''
    pass


if MAIN:
    max_seq_len=48
    tokens = prepare_text(text[:500], max_seq_len=max_seq_len, tokenizer=mytokenizer)
    print("Does this size look reasonable, as a tokenization of the first 500 characters?\n", tokens.shape)
```
""")
        with st.expander("Hint"):
            st.markdown(r"""
This exercise just involves encoding the text, then rearranging the size `(1, num_tokens)` tensor into a 2D tensor of shape `(batch, max_seq_len)`. You'll have to crop some tokens off the end, if the number of tokens doesn't exactly divide by `max_seq_len`.
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def prepare_text(text: str, max_seq_len: int, tokenizer: SimpleTokenizer):
    '''
    Takes a string of text, and returns an array of tokens rearranged into chunks of size max_seq_len.
    '''
    tokens: TT[1, "num_tokens"] = tokenizer.encode(text, return_tensors="pt")

    # We want to rearrange the tokens into chunks of size max_seq_len.
    num_tokens = tokens.size(1) - (tokens.size(1) % max_seq_len)
    tokens = einops.rearrange(
        tokens[0, :num_tokens], "(chunk seq_len) -> chunk seq_len", seq_len=max_seq_len
    )

    return tokens
```
""")
    # start
    st.markdown(r"""
## Datasets and Dataloaders

### Build Your Own TensorDataset

The class `torch.utils.data.dataset.TensorDataset` is a convenient wrapper for passing around multiple tensors that have the same size in the first dimension. The most common example of this is in supervised learning, where you have one tensor of inputs and a second tensor with corresponding labels. Often these tensors will have different `dtype`s, so it doesn't make sense to `torch.stack` them into one big tensor, and it be cumbersome to pass them around as separate variables or as a tuple.

`TensorDataset` accepts and stores any number of tensors in the constructor along with implementing `__getitem__` so that `my_dataset[n]` returns a tuple containing element `n` from each stored `Tensor`. Similarly, `my_dataset[:5]` returns a tuple containing the first five elements from each stored `Tensor`.

### Slice Objects in Python

`slice` is a built-in type containing `start`, `stop`, and `step` fields which can be integers or `None`. Given `x=[1,2,3,4,5,6,7]`, writing `x[1:5:2]` is syntactic sugar for `x[slice(1, 5, 2)]`.
""")
    # end    
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `TensorDataset`
""")
        st.error(r"""
This should be a relatively unchallenging exercise, and you can skip it if it doesn't seem interesting to you.
""")
        st.markdown(r"""
You should fill in the methods below, and verify that the tests pass.

Note that we're only passing in one tensor to this class (the `tokens` tensor), but this class should also be able to accept multiple tensors (this will be useful when we get to some later examples, like training models to solve algorithmic tasks).

```python
class TensorDataset:
    def __init__(self, *tensors: t.Tensor):
        '''Validate the sizes and store the tensors in a field named `tensors`.'''
        pass

    def __getitem__(self, index: Union[int, slice]) -> tuple[t.Tensor, ...]:
        '''Return a tuple of length len(self.tensors) with the index applied to each.'''
        pass

    def __len__(self):
        '''Return the size in the first dimension, common to all the tensors.'''
        pass


if MAIN:
    tests.test_tensor_dataset(TensorDataset)
    dataset = TensorDataset(tokens)
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
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
```
""")
    # start
    st.markdown(r"""
## Training loop

Now, it's time for our training loop! We've left this exercise very open-ended, like our implementation of the ResNet training loop in last week's exercises. The principles are exactly the same, and we've provided you with a skeleton of the function to help get you started. 

Again, we use a `dataclass` object to store the training parameters, because this is a useful way of keeping your code organised. Note one extra feature here - rather than defining our `optimizer_kwargs` object as a dictionary, we define it as a `frozendict` (which is a special dataclass that works just like regular dicts, except that it isn't mutable). This is a helpful way to get around the fact that you aren't allowed to set dataclass fields to mutable object like dictionaries or lists.
""")
    # end
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - write a training loop

You should read and understand the code below, and fill in the section marked `YOUR CODE HERE`.

```python
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

    
    # YOUR CODE HERE - implement training and testing loops

    
    print(f"\nSaving model to: {args.filename_save_model}")
    t.save(model, args.filename_save_model)
    return train_loss_list, test_loss_list
```

You can take a look at the solutions for an example implementation (although it's totally fine to have something which looks different to this).
""")
        with st.expander("Example implementation"):
           st.markdown(r"""
```python

```
""")
        st.markdown(r"""
Once you've written a training loop, you can run it (and plot your output) with the following code:

```python
if MAIN:
    config = Config(
        d_model = 384,
        layer_norm_eps = 1e-5,
        d_vocab = 50257,
        init_range = 0.02,
        n_ctx = 1024,
        d_head = 64,
        d_mlp = 1536,
        n_heads = 6,
        n_layers = 4
    )

    model = DemoTransformer(config)

    args = TransformerTrainingArgs(
        tokenizer = mytokenizer,
        batch_size = 8,
        epochs = 3,
    )

    train_loss_list, test_loss_list = train_transformer(model, text, args)

    px.line(train_loss_list).show()
    px.line(test_loss_list).show()
```

You can try playing around with some of the hyperparameters, and see how they affect the training process. You might also want to try out using different datasets (there are many online you can use!).
""")
    st.markdown(r"""
Finally, now that you've trained the model, let's try sampling from it! (We'll learn more about how these sampling methods work in the next section.)

```python
if MAIN:
    from solutions import sample_tokens

    text_output = sample_tokens(
        model, 
        mytokenizer, 
        initial_text="Turn down for what ", 
        max_tokens_generated=100, 
        temperature=0.6
    )
    print(text_output)
```
""")
    
def section_sampling():
    st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#sampling-boilerplate">Sampling Boilerplate</a></li>
    <li><a class="contents-el" href="#greedy-search">Greedy Search</a></li>
    <li><a class="contents-el" href="#sampling-with-categorical">Sampling with Categorical</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#basic-sampling">Basic Sampling</a></li>
        <li><a class="contents-el" href="#temperature">Temperature</a></li>
        <li><a class="contents-el" href="#frequency-penalty">Frequency Penality</a></li>
        <li><a class="contents-el" href="#sampling-manual-testing">Sampling - Manual Testing</a></li>
    </ul></li>
    <li><a class="contents-el" href="#top-k-sampling">Top-K Sampling</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#top-k-sampling-example">Top-K Sampling - Example</a></li>
    </ul></li>
    <li><a class="contents-el" href="#top-p-aka-nucleus-sampling">Top-p aka Nucleus Sampling</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#top-p-sampling-example">Top-p Sampling - Example</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Sampling
""")
    # start
    st.info(r"""
#### Learning Objectives

* Learn how to sample from a transformer
    * This includes basic methods like greedy search or top-k, and more advanced methods like beam search
""")
    st.markdown(r"""
One obvious method to sample tokens from a distribution would be to always take the token assigned the highest probability. But this can lead to some boring and repetitive outcomes, and at worst it can lock our transformer's output into a loop.

First, you should read HuggingFace's blog post [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate).

Once you've done that, we've included some exercises below that will allow you to write your own methods for sampling from a transformer. You'll be working with a pretrained model rather than the Shakespeare model in the previous set of exercises (because sampling can behave quite unpredictably unless tokenization and training are done very carefully), although you might want to try substituting in your Shakespeare model to these exercises if you have extra time at the end, and see how it behaves.
""")
    # end
    st.markdown(r"""
```python
if MAIN:
    reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
    gpt2 = DemoTransformer(Config())
    gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
    gpt2.to(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
```
""")
    # start
    st.markdown(r"""
## Sampling Boilerplate

The provided functions `apply_sampling_methods` and `sample_tokens` include the boilerplate for sampling from the model. Note that there is a special token `tokenizer.eos_token`, which during training was added to the end of a each article. GPT-2 will generate this token when it feels like the continuation is at a reasonable stopping point, which is our cue to stop generation.
""")
    # end
    st.markdown(r"""
The functions called in `apply_sampling_methods` are not defined yet - you are going to implement them below.

```python
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
    model.eval()
    input_ids: list = tokenizer.encode(initial_text)
    generated = []
    for _ in range(max_tokens_generated):
        new_input_ids = t.tensor(input_ids + generated, dtype=t.long, device=device)
        new_input_ids_window = new_input_ids[-min(model.cfg.n_ctx, new_input_ids.shape[0]):].unsqueeze(0)
        logits = model(new_input_ids_window)[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        generated.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input_ids + generated)
```

A few notes on the `sample_tokens` function:

* We use `tokenizer.encode` to convert the initial text string into a list of logits. You can also pass the argument `return_tensors="pt"` in order to return the output as a tensor.
* `new_input_ids` is a concatenation of the original input ids, and the ones that have been autoregressively generated.
* `new_input_ids_truncated` truncates `new_input_ids` at `max_seq_len` (because you might get an error at the positional embedding stage if your input sequence length is too large).
* The line `all_logits = ...` is necessary because HuggingFace's GPT doesn't just output logits, it outputs an object which contains `logits` and `past_key_values`. In contrast, your model will probably just output logits, so we can directly define logits as the model's output.
""")
    with st.expander("Question - why do you think we take logits[0, -1] ?"):
        st.markdown(r"""
Our model input has shape `(batch, seq_len)`, and each element is a token id. Our output has dimension `(batch, seq_len, vocab_size)`, where the `[i, j, :]`th element is a vector of logits representing a prediction for the `j+1`th token.

In this case, our batch dimension is 1, and we want to predict the token that follows after all the tokens in the sequence, hence we want to take `logits[0, -1, :]`.
""")

    with st.columns(1)[0]:
        # start
        st.markdown(r"""
### Greedy Search

Implement `greedy_search`, which just returns the most likely next token. If multiple tokens are equally likely, break the tie by returning the smallest token.

Why not break ties randomly? It's nice that greedy search is deterministic, and also nice to not have special code for a case that rarely occurs (floats are rarely exactly equal).

Tip: the type checker doesn't know the return type of `item()` is int, but you can assert that it really is an int and this will make the type checker happy.
""")
        # end
        st.markdown(r"""
```python
def greedy_search(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    '''
    pass


if MAIN:
    prompt = "Jingle bells, jingle bells, jingle all the way"
    print("Greedy decoding with prompt: ", prompt)
    output = sample_tokens(gpt2, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
    print(f"Your model said: {output}")
    expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
    assert output == expected

    print("Greedy decoding a second time (should be deterministic): ")
    output = sample_tokens(gpt2, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
    print(f"Your model said: {output}")
    expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
    assert output == expected

    print("Tests passed!")
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def greedy_search(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    '''
    out = logits.argmax().item()
    assert isinstance(out, int)
    return out
```
""")
    # start
    st.markdown(r"""
## Sampling with Categorical

PyTorch provides a [`distributions` package](https://pytorch.org/docs/stable/distributions.html#distribution) with a number of convenient methods for sampling from various distributions.

For now, we just need [`t.distributions.categorical.Categorical`](https://pytorch.org/docs/stable/distributions.html#categorical). Use this to implement `sample_basic`, which just samples from the provided logits (which may have already been modified by the temperature and frequency penalties).
""")
    # end
    with st.columns(1)[0]:
        st.markdown(r"""
### Basic Sampling

Note that this will be slow since we aren't batching the samples, but don't worry about speed for now.
        
```python
def sample_basic(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    pass


if MAIN:
    N = 20000
    probs = t.linspace(0, 0.4, 5)
    unnormalized_logits = probs.log() + 1.2345
    samples = t.tensor([sample_basic(unnormalized_logits) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
    t.testing.assert_close(counts, probs, atol=0.01, rtol=0)
    print("Tests passed!")
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def sample_basic(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    distribution = t.distributions.categorical.Categorical(logits=logits)
    out = distribution.sample().item()
    assert isinstance(out, int)
    return out
```
""")
        
    with st.columns(1)[0]:
        # start
        st.markdown(r"""
### Temperature

Temperature sounds fancy, but it's literally just dividing the logits by the temperature.
""")
        # end
        st.markdown(r"""
```python
def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    '''
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    assert temperature > 0
    pass

    
if MAIN:
    logits = t.tensor([1, 2]).log()
    cold_logits = apply_temperature(logits, 0.001)
    print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
    t.testing.assert_close(cold_logits, 1000.0 * logits)
    hot_logits = apply_temperature(logits, 1000.0)
    print("A high temperature flattens the distribution: ", hot_logits)
    t.testing.assert_close(hot_logits, 0.001 * logits)
    print("Tests passed!")
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    '''
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    assert temperature > 0
    return logits / temperature
```
""")
        # start
        with st.expander(r"""
Question - what is the limit of applying 'sample_basic' after adjusting with temperature, when temperature goes to zero? How about when temperature goes to infinity?
"""):
            st.markdown(r"""
The limit when temperature goes to zero is greedy search (because dividing by a small number makes the logits very big, in other words the difference between the maximum logit one and all the others will grow). 

The limit when temperature goes to infinity is uniform random sampling over all words (because all logits will be pushed towards zero).
""")
    with st.columns(1)[0]:
        st.markdown(r"""
### Frequency Penalty

The frequency penalty is simple as well: count the number of occurrences of each token, then subtract `freq_penalty` for each occurrence. Hint: use `t.bincount` (documentation [here](https://pytorch.org/docs/stable/generated/torch.bincount.html)) to do this in a vectorized way.
""")
        # end
        with st.expander(r"""Help - I'm getting a RuntimeError; my tensor sizes don't match."""):
            st.markdown(r"""
Look at the documentation page for `t.bincount`. You might need to use the `minlength` argument - why?
""")

        st.markdown(r"""
```python
def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    '''
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    pass

    
if MAIN:
    bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
    input_ids = tokenizer.encode(bieber_prompt, return_tensors="pt").squeeze()
    logits = t.ones(tokenizer.vocab_size)
    penalized_logits = apply_freq_penalty(input_ids, logits, 2.0)
    assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space"
    assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space"
    print("Tests passed!")
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    '''
    input_ids: shape (seq, )
    logits: shape (vocab_size, )
    Return: shape (vocab_size, )
    '''
    (vocab_size,) = logits.shape
    id_freqs = t.bincount(input_ids, minlength=vocab_size)
    return logits - freq_penalty * id_freqs
```
""")
    st.markdown(r"""
### Sampling - Manual Testing

Run the below cell to get a sense for the `temperature` and `freq_penalty` arguments. Play with your own prompt and try other values.

Note: your model can generate newlines or non-printing characters, so calling `print` on generated text sometimes looks awkward on screen. You can call `repr` on the string before printing to have the string escaped nicely.

```python
if MAIN:
    N_RUNS = 1
    your_prompt = "Jingle bells, jingle bells, jingle all the way"
    cases = [
        ("High freq penalty", dict(freq_penalty=100.0)),
        ("Negative freq penalty", dict(freq_penalty=-3.0)),
        ("Too hot!", dict(temperature=2.0)),
        ("Pleasantly cool", dict(temperature=0.7)),
        ("Pleasantly warm", dict(temperature=0.9)),
        ("Too cold!", dict(temperature=0.01)),
    ]
    for (name, kwargs) in cases:
        for i in range(N_RUNS):
            output = sample_tokens(gpt2, tokenizer, your_prompt, max_tokens_generated=24, **kwargs)
            print(f"Sample {i} with: {name} ({kwargs}):")
            print(f"Your model said: {repr(output)}\n")
```
""")
    # start
    st.markdown(r"""
## Top-K Sampling

Conceptually, the steps in top-k sampling are:
- Find the `top_k` largest probabilities
- Set all other probabilities to zero
- Normalize and sample
""")
    # end
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `sample_top_k`
        
Your implementation should stay in log-space throughout (don't exponentiate to obtain probabilities). This means you don't actually need to worry about normalizing, because `Categorical` accepts unnormalised logits.
""")

        with st.expander(r"Help - I don't know what function I should use for finding the top k."):
            st.markdown(r"Use [`t.topk`](https://pytorch.org/docs/stable/generated/torch.topk.html).")

        st.markdown(r"""
```python
def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    '''
    pass


if MAIN:
    k = 3
    probs = t.linspace(0, 0.4, 5)
    unnormalized_logits = probs.log() + 1.2345
    samples = t.tensor([sample_top_k(unnormalized_logits, k) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    expected = probs.clone()
    expected[:-k] = 0
    expected /= expected.sum()
    print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
    t.testing.assert_close(counts, expected, atol=0.01, rtol=0)
    print("Tests passed!")
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    '''
    top_logits, top_idx = t.topk(logits, top_k)
    idx = t.distributions.categorical.Categorical(logits=top_logits).sample()
    return top_idx[idx].item()
```
""")
    st.markdown(r"""
### Top-K Sampling - Example

The [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) famously included an example prompt about unicorns. Now it's your turn to see just how cherry picked this example was.

The paper claims they used `top_k=40` and best of 10 samples.

```python
if MAIN:
    your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
    output = sample_tokens(gpt2, tokenizer, your_prompt, temperature=0.7, top_k=40, max_tokens_generated=64)
    print(f"Your model said: {repr(output)}")
```
""")
    # start
    st.markdown(r"""
## Top-p aka Nucleus Sampling

The basic idea is that we choose the most likely words, up until the total probability of words we've chosen crosses some threshold. Then we sample from those chosen words based on their logits.

The steps are:

- Sort the probabilities from largest to smallest
- Find the cutoff point where the cumulative probability first equals or exceeds `top_p`. We do the cutoff inclusively, keeping the first probability above the threshold.
- If the number of kept probabilities is less than `min_tokens_to_keep`, keep that many tokens instead.
- Set all other probabilities to zero
- Normalize and sample

Optionally, refer to the paper [The Curious Case of Neural Text Degeneration](https://arxiv.org/pdf/1904.09751.pdf) for some comparison of different methods.
""")
    with st.columns(1)[0]:
        # end
        st.markdown(r"""
#### Exercise - implement `sample_top_p`
""")
        with st.expander(r"Example of top-p sampling (if you're confused)."):
            st.markdown(r"""
If our probabilities were `(0.4, 0.3, 0.2, 0.1)` and our cutoff was `top_p=0.8`, then we'd sample from the first three elements (because their total probability is `0.9` which is over the threshold, but the first two only have a total prob of `0.7` which is under the threshold). Once we've chosen to sample from those three, we would renormalise them by dividing by their sum (so the probabilities we use when sampling are `(4/9, 3/9, 2/9)`.
""")
        with st.expander(r"Help - I'm stuck on how to implement this function."):
            st.markdown(r"""
First, sort the logits using the `sort(descending=True)` method (this returns values and indices). Then you can get `cumulative_probs` by applying softmax to these logits and taking the cumsum. Then, you can decide how many probabilities to keep by using the `t.searchsorted` function.
    
Once you've decided which probabilities to keep, it's easiest to sample from them using the original logits (you should have preserved the indices when you called `logits.sort`). This way, you don't need to worry about renormalising like you would if you were using probabilities.
""")

        st.markdown(r"""
```python
def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    pass

    
if MAIN:
    N = 2000
    unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
    samples = t.tensor([sample_top_p(unnormalized_logits, 0.5) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
    print("top_p of 0.5 or lower should only return token 2: ", counts)
    assert counts[0] == 0 and counts[1] == 0

    N = 2000
    unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
    samples = t.tensor([sample_top_p(unnormalized_logits, 0.50001) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
    print("top_p in (0.5, 0.8] should return tokens 1 and 2: ", counts)
    assert counts[0] == 0

    N = 5000
    top_p = 0.71
    probs = t.linspace(0, 0.4, 5)
    unnormalized_logits = probs.log() + 1.2345
    samples = t.tensor([sample_top_p(unnormalized_logits, top_p) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    expected = probs.clone()
    expected[0:2] = 0
    expected /= expected.sum()
    print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
    t.testing.assert_close(counts, expected, atol=0.01, rtol=0.0)

    print("All tests passed!")
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
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
```
""")
    st.markdown(r"""
### Top-p Sampling - Example

```python
if MAIN:
    your_prompt = "Eliezer Shlomo Yudkowsky (born September 11, 1979) is an American decision and artificial intelligence (AI) theorist and writer, best known for"
    output = sample_tokens(gpt2, tokenizer, your_prompt, temperature=0.7, top_p=0.95, max_tokens_generated=64)
    print(f"Your model said: {repr(output)}")
```
""")
    # start
    st.markdown(r"""
## Beam search

Finally, we'll implement a more advanced way of searching over output: **beam search**. You should read the [HuggingFace page](https://huggingface.co/blog/how-to-generate#beam-search) on beam search before moving on.

In beam search, we maintain a list of size `num_beams` completions which are the most likely completions so far as measured by the product of their probabilities. Since this product can become very small, we use the sum of log probabilities instead. Note - log probabilities are *not* the same as your model's output. We get log probabilities by first taking softmax of our output and then taking log. You can do this with the [`log_softmax`](https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html) function (or use the tensor method).
""")
    # end
    with st.expander(r"Log probabilities are equal to the logit output after being translated by some amount X (where X is a function of the original logit output). Can you prove this?"):
        st.markdown(r"""
Suppose our vector of logits is $x$, and we take softmax to get a vector of probabilities $p$, then log again to get a vector of log probabilities $l$. Then the $i$-th element of this vector of logprobs is:

$$
\begin{align}
l_i &= \log p_i \\
&= \log \frac{\exp(x_i)}{\sum_j \exp(x_j)} \\
&= x_i - \log \sum_j \exp(x_j) \\
&= x_i - C
\end{align}
$$

where $C = \log \sum_j \exp(x_j)$ is the same for all elements. So we can see that $l_i$ is equal to the logit output $x_i$ after being translated by $C$.

It's important not to mix up logits and logprobs!
""")

    with st.expander(r"Why do you think we use log softmax rather than logit output?"):
        st.markdown(r"""
It allows us to compare sequences of different lengths. For instance, if a sequence terminates after 5 tokens (i.e. we generate an EOS token), we want to be able to compare this to a sequence of length 10. We wouldn't be able to do this if the logit output was used, since there's an arbitrary constant factor of difference between the logits we'd get for the two sequences.
""")
    # start
    st.markdown(r"""
At each iteration, we run the batch of completions through the model and take the log-softmax to obtain `vocab_size` log-probs for each completion, or `num_beams * vocab_size` possible next completions in total.

If we kept all of these, then we would have `num_beams * vocab_size * vocab_size` completions after the next iteration which is way too many, so instead we sort them by their score and loop through from best (highest) log probability to worst (lowest).

For each next completion, if it ends in the end of sequence (EOS) token then we add it to a list of finished completions along with its score. Otherwise, we add it to the list of "to be continued" completions. The iteration is complete when the "to be continued" list has `num_beams` entries in it.

If our finished list now contains at least `num_return_sequences` completions, then we are done. If the length of the completion is now `len(prompt) + max_new_tokens`, then we are also done. Otherwise, we go to the next iteration.

A few clarifications about beam search, before you implement it below:

* GPT's tokenizer stores its EOS token as `tokenizer.eos_token_id`. If this token is generated by the model, you should terminate your sequence here.
* Remember that your model should be in eval mode (this affects dropout), and you should be in inference mode (this affects gradients). For the latter, we've given you the `@t.inference_mode()` decorator.
""")
    # end
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `beam_search`

This exercise is meant to be very challenging, so we've also included a dropdown that gives you a suggested structure of the function (in the form of comments, which you can add code beneath). You are encouraged to use this as a reference if you get stuck.

```python
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

    pass
```

We've also provided you with two helper functions:

* `print_sequences`, which is useful for printing your output if `verbose == True`.
* `sort_by_logits_and_crop`, which will sort your outputs by logprob and crop the list down to as many as you need (e.g. because you'll only need to keep hold of `num_beams` entries at each iteration of beam search).

```python
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
```
""")
        with st.expander("Suggested function structure"):
            st.markdown(r"""
```python
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

        # This section loops through all completions so far, and get the next words
        for (logitsum, completion) in best_logitsums_and_completions:

            # Get output (we only care about the vector of logits for the next token)

            # Find the top `num_beams` tokens (because this is the maximum we might need)

            # Append to the new best completions list

        # This section updates (and sorts) the list of best completions, and also updates `final_logitsums_and_completions` if EOS was produced
        best_logitsums_and_completions = []
        for (logitsum, completion) in sorted(new_best_logitsums_and_completions, key=lambda x: -x[0]):
            
            # If token is eos then add it to final_logitsums_and_completions
            
            # Else add it to best_logitsums_and_completions
            # And if that list has length num_beams, we break out of this loop

        if n == max_new_tokens - 1:
            # If we're at the end, add the best completions to the final completions list, then keep only the best `num_return_sequences` completions
        else:
            # If not, then keep only the best `num_beams` completions

    return final_logitsums_and_completions
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
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
```
""")
        st.markdown(r"""
You can test your function by trying to generate output, with `verbose=True`. 

```python
initial_text = "The answer to life, the universe and everything is"
input_ids = tokenizer.encode(initial_text, return_tensors="pt").squeeze()
num_return_sequences = 5
num_beams = 10
max_new_tokens = 30

final_logitsums_and_completions = beam_search(gpt2, input_ids, num_return_sequences, num_beams, max_new_tokens, tokenizer, verbose=True)

text = tokenizer.decode(final_logitsums_and_completions[0][1])
print("\n" + "=" * 60 + "\n\nFinal output:\n\n" + text)
```
""")

# turn down for what you do you think,
# That take the last, of many, which is so much I
# As this blows along than my life thou say’st, which makes thy hand,
# Thou wilt be given, or more
# Entitled in thy great world’s fresh blood will,
# To answer th’ alluring countenance, beauty

def section_caching():
    st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#efficient-text-generation-via-caching">Efficient text generation via caching</a></li>
    <li><a class="contents-el" href="#bonus-cached-beam-search">Bonus - cached beam search</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# Caching
""")
    st.error(r"""
*Note, both the main exercise in this section and the bonus exercise are harder than anything you'll have done before (with the possible exception of implementing beam search in the last set of exercises), and is intended to take quite some time. There are many different ways to solve it, and you're expected to try and find your own way (you should think about this for a while before looking at the suggestions in the dropdowns).*
""")
    # start
    st.markdown(r"""
## Learning Objectives

* Learn how to cache the output of a transformer, so that it can be used to generate text more efficiently
* Update your sampling functions to make use of your caching methods
    
## Efficient text generation via caching
    
The text generation we've done so far is needlessly re-computing certain values, which is very noticeable when you try to generate longer sequences.

Suppose you're generating text, and you've already run GPT on the sentence "My life motto:". Now you want to run the model on the sentence "My life motto: Always". Which computations from the first sentence can you reuse?
""")
    with st.expander("Answer"):
        st.markdown(r"""
At each attention layer, the only things the attention layer needs from the previous sequence positions are the key and value vectors. This is explained in the following diagram:
""")
        st_image("tl-cache.png", 600)
        st.markdown(r"")
        st.markdown(r"""
As long as we have the key and value vectors from the previous sequence positions (labelled **old keys** and **old values** in the diagram), we can combine this with the new key, query and value vectors for our new sequence position and produce the output **z** which gets added to the residual stream at the last sequence position. This will in turn allow us to compute the new key, query and value vectors for the next attention layer, and so on, until we reach the final output which becomes our vector of logits corresponding to the predicted next token.

Note, this diagram represents what the attention mechanism would look like if the cache had size $s_k - 1$, and the number of new tokens was $1$. It could easily be generalised to an arbitrary number of new tokens.
""")
    st.markdown(r"""
Modify your GPT-2 to optionally use a cache. When you run your GPT on `"My life motto:"`, it should store the necessary values in the cache. Then in the next forward pass with just `" Always"` as input, it should load the cached values instead of recomputing them (and update the cache). This only needs to work with a single input sequence (batch size of 1), and you can assume that after the first forward pass, the input will be just one token.

The design of the cache is completely up to you - discuss possible designs with your partner before writing code. It should be possible to have only one GPT2 instance and many different cache instances at one time. Imagine that you want to use one instance to serve multiple users submitting requests for text generation like in [AI Dungeon](https://aidungeon.io/).

You'll also need to rewrite parts of your `DemoTransformer` code, in order to get this to work.

Some example considerations:

* Which GPT-2 classes need to interact with the cache?
    * Will you need to change the positional embedding, and if so then how?
* Should the cache be mutable and be updated in place, or does updating actually just create a separate instance?
* Is it possible for other programmers to incorrectly use your cache? Is there a way to prevent this failure mode or at least detect this and complain loudly?

```python
# Your code here - define your cache object, and modify your DemoTransformer code to use it
```

Once you've done this, you should verify your code works by sampling tokens from your model with / without using a cache. You should get the same results (because the `apply_sampling_methods` function sets a random seed), with a noticeable speedup when using a cache.
""")
    
    with st.expander("Help - I'm stuck on how to start!"):
        st.markdown(r"""
Here is a template for the cache object. There are two classes defined: one is a cache for a single layer (i.e. it contains one tensor of keys and one tensor of values), and one is a cache for the whole model (i.e. it contains a list of cache entries, one for each layer of the model).

```python
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
        pass


class KeyValueCache():
    '''
    This holds a list of KeyValueCacheEntry objects, one for each layer in the model.
    In our forward pass, we iterate through the cache entries stored in this object.
    '''
    entries: List[KeyValueCacheEntry] = []

    def __init__(self, cfg: Config, batch: int = 1):
        pass

    def __getitem__(self, idx):
        pass
```

When you run a forward pass on your model, you can zip through the layers and cache entries:

```python
for block, cache_entry in zip(self.blocks, cache):
    residual = block(residual, cache_entry)
```

(You'll also need to rewrite the attention block and transformer block modules, to use your cache object.)
""")
    # end
    with st.expander(r"Example implementation (don't read until you've thought about this for a while)."):
        st.markdown(r"""
First, we import the layers which we won't be modifying from the previous set of exercises:
        
```python
from transformer_from_scratch.solutions import LayerNorm, MLP, Embed, Unembed
```

Next, we define our cache objects (see the dropdown above for a description of these):

```python
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
```

Lastly, we'll redefine some of our modules, to use the cache. Below each code block, I've added a comment explaining what I've changed.

```python
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
```

This change is easy to miss, but very important. If we've cached the key and value vectors corresponding to some tokens, and we only need to pass the second half of the sequence through the model, then we need to make sure that the positional embeddings are offset by the length of the cached tokens. This is what the `past_kv_pos_offset` parameter does in the code above. The rest of the module is unchanged.

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
```

This is the module which has been changed the most. The `forward` function has been changed to use the cache:

* If the `cache_entry` object is not None, this indicates we are to use it and the values it stores. We concatenate the new keys and values with the ones stored in the cache, to get "full keys" and "full values".
* We then go through the same attention calculations as normal. The only subtlety comes from applying the causal mask, where we need to remember that the queries come from the **most recent** tokens, and the keys come from all the tokens (i.e. they extend further back in time than the queries).
    * For instance, if we just have one new token, then the mask `q_posn < k_posn` should just be a single row containing `False` values (because our only query is the most recent token, and we don't want to mask anywhere).

```python
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
```

The only change here is the `attn` function; we also take in the `cache_entry` object.

```python
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
```

Lastly, we rewrite the `forward` method of the entire transformer, to add an optional `cache` argument. If this is provided, then we assume this object stores a list of `KeyValueCacheEntry` objects, one for each block in the transformer. We need to do the following in the forward function:

* Take a slice of the tokens (because we're only passing through the newest tokens, not those which have already had their keys and values cached). We make sure the offset in the positional embedding is correct.
* Iterate through the blocks of the model **and** the cache entries.

---

The final code below is a rewrite of the `sample_tokens` function (which has become `sample_tokens_with_cache`). This looks basically the same as the old function; the only difference is we also pass in the cache object during model evaluation. The code below that shows an example of how to use this sampling function, and verifies that the result is the same as it is when we don't use caching.

```python
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
    initial_text = "The answer to life, the universe and everything is"

    t0 = time.time()
    output_without_cache = sample_tokens_with_cache(gpt2, tokenizer, initial_text, temperature=0.7, top_p=0.95,max_tokens_generated=100, cache=None)
    print(f"Time taken (without cache): {time.time() - t0:.2f} seconds")
    print("Model output:\n\n" + format_output(output_without_cache) + "\n\n")

    t0 = time.time()
    output_with_cache = sample_tokens_with_cache(gpt2, tokenizer, initial_text, temperature=0.7, top_p=0.95, max_tokens_generated=100, cache=KeyValueCache(gpt2.cfg))
    print(f"Time taken (with cache): {time.time() - t0:.2f} seconds")
    print("Model output:\n\n" + format_output(output_with_cache))

    assert output_with_cache == output_without_cache, "Your outputs are different, meaning you've probably made a mistake in your cache implementation."
```
""")
    st.markdown(r"""
## Bonus - cached beam search

Can you modify your beam search function to use caching?

This is a bit more complicated than adapting your other sampling functions, because you'll need to keep track of a cache for each beam. You might want to consider using a different datastructure (on top of the ones you've already defined) to store the cached keys and values for all your different beams.
""")
    with st.expander("Hint (for datastructure to store caches for your beam search)"):
        st.markdown(r"""
One option would be to use a tree, structured as follows:

* The root node contains the cache for the initial input sequence.
* All other nodes represent a single token (so an edge between two nodes indicates that the second token was generated after the first token, in one of the sequences being tracked by the beam search).
* To generate the full keys and values used in a cache, we concatenate along the tree path from the root node to the current node.
""")
    with st.expander("Solution (incomplete)"):
        st.markdown(r"""
This solution shows how you could directly use the code already written above to implement beam search with caching. It doesn't use the tree method described in the dropdown above, but instead it keeps a record of the cache for each sequence in the current list of best sequences, which makes it very memory-inefficient (I haven't had time to go back and improve it). However, this should at least give you an idea of what a final version of this code would look like.

```python
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
    print("Model output:\n\n" + format_output(text_without_cache) + "\n\n")
    
    t0 = time.time()
    final_logitsums_and_completions = beam_search_with_cache(gpt2, input_ids, num_return_sequences, num_beams, max_new_tokens, tokenizer, cache=KeyValueCache(gpt2.cfg))
    print(f"Time taken (with cache): {time.time() - t0:.2f} seconds")
    text_with_cache = tokenizer.decode(final_logitsums_and_completions[0][1])
    print("Model output:\n\n" + format_output(text_with_cache))

    assert text_with_cache == text_without_cache, "Your outputs are different, meaning you've probably made a mistake in your cache implementation."
```
""")
#     with st.expander("Help, my cache is silently computing the wrong thing!"):
#         st.markdown(r"""
# A good debugging strategy is to use `nn.Module.register_forward_hook` to store the inputs and outputs to each module where you suspect a bug. Run your module in cached and uncached modes and check that the values you expect to be identical actually are.

# Check your positional encoding and your attention mask. Try adding more assert statements to verify assumptions in your code.
# """)

func_page_list = [
    (section_home, "🏠 Home"), 
    (section_training, "1️⃣ Training"),
    (section_sampling, "2️⃣ Sampling"),
    (section_caching, "3️⃣ Caching"),
]

func_list = [func for func, page in func_page_list]
page_list = [page for func, page in func_page_list]

page_dict = {page: idx for idx, (func, page) in enumerate(func_page_list)}

# if "text_generation" not in st.session_state:
#     st.session_state["text_generation"] = True
# def toggle_text_generation():
#     st.session_state["text_generation"] = not st.session_state["text_generation"]

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
    current_page = r"2_🎲_Training_and_Sampling"
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


# %%



# idea - walk back previous change; have an old page and a new page (update the values in st.session_state)