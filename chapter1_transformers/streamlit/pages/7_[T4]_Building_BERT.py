import os
if not os.path.exists("./images"):
    os.chdir("./ch1")
from st_dependencies import *
styling()

st.markdown("""
<style>
label.effi0qh3 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 15px;
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
    color: red;
    white-space: pre-wrap !important;
}
code:not(h1 code):not(h2 code):not(h3 code):not(h4 code) {
    font-size: 13px;
}
a.contents-el > code {
    color: black;
    background-color: rgb(248, 249, 251);
}
.css-ffhzg2 a.contents-el > code {
    color: orange;
    background-color: rgb(26, 28, 36);
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

def section_home():
    st.markdown(r"""
# Further investigations

Hopefully, last week you were able to successfully implement a transformer last week (or get pretty close!). If you haven't done that yet, then this should be your first priority going forwards with this week. **If you are struggling with getting your transformer to work, please send me (Callum) a link to your GitHub repo and I will be able to help troubleshoot.**

The rest of this week will involve continuing to iterate on your transformer architecture, as well as doing some more experiments with transformers. In the following pages, we'll provide a few suggested exercises. These range from highly open-ended (with no testing functions or code template provided) to highly structured (in the style of last week's exercises). 

All of the material here is optional, so you can feel free to do whichever exercises you want - or just go back over the transformers material that we've covered so far. **You should only implement them once you've done last week's tasks** (in particular, building a transformer and training it on the Shakespeare corpus). 

Below, you can find a description of each of the set of exercises on offer. You can do them in any order, as long as you make sure to do exercises 1 and 2 at some point. Note that you can do e.g. 3B before 3A, but this is not advised since you'd have to import the solution from 3A and work with it, possibly without fully understanding the architecture.

---

## 1️⃣ Build and sample from GPT-2

As was mentioned in yesterday's exercises, you've already built something that was very close to GPT-2. In this task, you'll be required to implement an exact copy of GPT-2, and load in the weights just like you did last week for ResNet. Just like last week, this might get quite fiddly!

We will also extend last week's work by looking at some more advanced sampling methods, such as **beam search**.

## 2️⃣ Build BERT

BERT is an encoder-only transformer, which has a different kind of architecture (and a different purpose) than GPT-2. In this task, you'll build a copy of BERT and load in weights, then fine-tune it on a sentiment analysis task.

## 3️⃣ Finetune BERT

Once you've built BERT, you'll be able to train it to perform well on tasks like classification and sentiment analysis. This finetuning task requires you to get hands-on with a bit of data cleaning and wrangling!

## 4️⃣ Other bonus exercises

Visit this page for a series of fun exercises to attempt! These are much more open-ended than the relatively well-defined, structured exercises above.
""")

def section1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#introduction">Introduction</a></li>
    <li><a class="contents-el" href="#gpt-architecture-an-overview">GPT architecture: an overview</a></li>
    <li><a class="contents-el" href="#notes-on-copying-weights">Copying weights</a></li>
    <li><a class="contents-el" href="#testing-gpt">Testing GPT</a></li>
    <li><a class="contents-el" href="#testing-gpt">Beam search</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""

## Introduction

Here, you have an opportunity to rewrite your transformer implementations so that they exactly match the architecture of GPT-2, then load in GPT's weights and biases to your own model just like you did in the first week with ResNet. You can re-use some of your ResNet code, e.g. the function to copy over weights (although you will have to rewrite some of it; more on this below).

We will be using the GPT implementation from [HuggingFace](https://huggingface.co/docs/transformers/model_doc/auto). They provide a repository of pretrained models (often, transformer models) as well as other valuable documentation.

```python
transformers.AutoModelForCausalLM.from_pretrained("gpt2")
```

## GPT architecture: an overview

First, we will start by going through the differences between GPT and your implementation (i.e. the diagram from [W1D3](https://arena-w1d3.streamlitapp.com/Putting_it_all_together#decoderblock-and-decoderonlytransformer)).

* The order of the LayerNorms in the decoder block have changed: they now come *before* the attention and MLP blocks, rather than after.
* The attention block has two dropout layers: one immediately after the softmax (i.e. before multiplying by `V`), and one immediately after multiplying with `W_O` at the very end of the attention block. Note that the dropout layers won't actually affect weight-loading or performance in eval mode (and you should still be able to train your model without them), but all the same it's nice to be able to exactly match GPT's architecture!
* All your linear layers should have biases - even though some of them are called projections (which would seem to suggest not having a bias), this is often how they are implemented.
* GPT-2 uses a learned positional embedding (i.e. `nn.Embedding`) rather than a sinusoidal one.""")

    with st.expander("Question - how do you think you would use a positional encoding during a forward pass, if it was an 'nn.Embedding' object?"):
        st.markdown("""
When we used a sinusoidal encoding, we simply took a slice of the first `seq_len` rows. When using `nn.Embedding`, the equivalent is to pass in `t.arange(seq_len)`. So our first step in `forward` should look something like:

```python
pos = t.arange(x.shape[1], device=x.device)
x = self.token_embedding(x) + self.positional_encoding(pos)
```
""")

    st.markdown("""
We've provided you with the function `utils.print_param_count(*models)`, which can be passed a list of models (ideally two: yours and the pretrained GPT), and displays a color-coded dataframe making it easy to see which parameters match and which don't:

```python
my_gpt = GPT(config).train()
gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2").train()

utils.print_param_count(my_gpt, gpt)
```

Ideally, this will produce output that looks something like this (up to possibly having different layer names):
""")

    st_image('gpt-compared.png', width=770)

    st.markdown("""

Note - the `utils.print_param_count` function works slightly differently than it did in the ResNet utils file, `w0d3/utils`. This is because it iterates through `model.parameters()` rather than `model.state_dict()`. For why it does this, read the next section.

## Copying weights

Here is the template for a function which you should use to copy over weights from pretrained GPT to your implementation. 

```python
def copy_weights_from_gpt(my_gpt: GPT, gpt) -> GPT:
    '''
    Copy over the weights from gpt to your implementation of gpt.

    gpt should be imported using: 
        gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

    Returns your gpt model, with weights loaded in.

    You might find the function `copy_weights` from w0d3 helpful as a template.
    '''
    
    # FILL IN CODE: define a state dict from my_gpt.named_parameters() and gpt.named_parameters()

    my_gpt.load_state_dict(state_dict)
    return my_gpt
```

A few notes here, regarding how this function will be different from the copying weights function you were given in w0d3:

* The linear layer weights are actually transposed between GPT and your implementation (you can see this from row 4 of the table above). This applies to the linear layers with square weight matrices too, so take care to transpose when copying these weights over! (Note that the embeddings aren't transposed; only the linear layers.)
* It's easier to iterate through using `model.named_parameters()` rather than `model.state_dict()` in this case. 
    * Optional exercise: inspect `gpt.state_dict()` and `gpt.named_parameters()`, and see what objects are in one but not the other. Why are these objects in one but not the other, and why do you think it's better to use `named_parameters` than `state_dict` when copying over? (or you can skip this exercise, and just take my word for it!)
""")

    with st.expander("Answer to optional exercise"):
        st.markdown("""
Upon inspection, we find:

```python
state_dict_names = set(gpt.state_dict().keys())
param_names = set(dict(gpt.named_parameters()).keys())

print(len(state_dict_names))            # 173
print(len(param_names))                 # 148
print(param_names < state_dict_names)   # True
```

So `state_dict` is a superset of `parameters`, and it contains a lot more objects. What are these objects? When we print out the elements of the set difference, we see that they are all biases or masked biases of the transformer layer. If we inspect the corresponding objects in `gpt.state_dict().values()`, we see that most of these are nothing more than the objects we've been using as attention masks (i.e. a triangular array of 1s and 0s, with 1s in the positions which aren't masked). We clearly don't need to copy these over into our model!

The only exception is `lm_head` at the end of the model, and upon inspection we see that the weights of this bias-free linear layer match up exactly with the token embedding matrix. Clearly we only need to copy in the token embedding values once!

The moral of the story - `state_dict` sometimes contains things which we don't need, or duplicates of paramters which are used twice, and it depends heavily on the exact implementation details of the model's architecture.

Note that we're in the fortunate position of not having any batch norm layers in this architecture, because if we did then we couldn't get away with using `parameters` (batch norm layers contain objects like moving averages, which are buffers, not parameters).
""")

    st.markdown("""
## Testing GPT

Once you've copied over your weights, you can test your GPT implementation.

GPT2's tokenizer uses a special kind of encoding scheme called [byte-level byte-pair encoding](https://docs.google.com/document/d/1XJQT8PJYzvL0CLacctWcT0T5NfL7dwlCiIqRtdTcIqA/edit#heading=h.dgmfzuyi6796). You should import a tokenizer, and test your model, using the following code:

```python
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
utils.test_load_pretrained_weights(my_gpt, tokenizer)
```

The testing function above gives your GPT the prompt `"Former President of the United States of America, George"`, and tests whether the next predicted tokens contain `" Washington"` and `" Bush"` (as they are expected to). You are encouraged to look at this function in `utils`, and understand how it works.

If you get this working, you can try fine-tuning your GPT on text like your Shakespeare corpus. See if you can use this to produce better output than your original train-from-scratch, tokenize-by-words model. However, if you want to get the best possible output, you might want to try the following task first...

## Beam search

Finally, we'll implement a more advanced way of searching over output: **beam search**. You should read the [HuggingFace page](https://huggingface.co/blog/how-to-generate#beam-search) on beam search before moving on.

In beam search, we maintain a list of size `num_beams` completions which are the most likely completions so far as measured by the product of their probabilities. Since this product can become very small, we use the sum of log probabilities instead. Note - log probabilities are *not* the same as your model's output. We get log probabilities by first taking softmax of our output and then taking log. You can do this with the [`log_softmax`](https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html) function (or use the tensor method).""")

    with st.expander("Log probabilities are equal to the logit output after being translated by some amount X (where X is a function of the original logit output). Can you prove this?"):
        st.markdown("""
Proof coming soon. Basic idea: the amount by which they're translated is the log of the denominator expression in softmax.
""")

    with st.expander("Why do you think we use log softmax rather than logit output?"):
        st.markdown("""
It makes the output more meaninfgul, since the sum of log probabilities is precisely the log probability of the model producing this exact sequence from scratch. If we used the raw logit output then we wouldn't get this property.

Importantly, we also wouldn't be able to compare sequences of different lengths if we just used the raw logit output.
""")

    st.markdown("""
At each iteration, we run the batch of completions through the model and take the log-softmax to obtain `vocab_size` log-probs for each completion, or `num_beams * vocab_size` possible next completions in total.

If we kept all of these, then we would have `num_beams * vocab_size * vocab_size` completions after the next iteration which is way too many, so instead we sort them by their score and loop through from best (highest) log probability to worst (lowest).

For each next completion, if it ends in the end of sequence (EOS) token then we add it to a list of finished completions along with its score. Otherwise, we add it to the list of "to be continued" completions. The iteration is complete when the "to be continued" list has `num_beams` entries in it.""")

    st.markdown("""
If our finished list now contains at least `num_return_sequences` completions, then we are done. If the length of the completion is now `len(prompt) + max_new_tokens`, then we are also done. Otherwise, we go to the next iteration.

A few clarifications about beam search, before you implement it below:

* GPT's tokenizer stores its EOS token in `tokenizer.eos_token_id`. You'll have to either define this attribute in your own tokenizer (you can set it to `None` if you never want to terminate early), or handle this case in the beam search function (e.g. by using `getattr(tokenizer, "eos_token_id", None)`, which returns `tokenizer.eos_token_id` if it exists and None if not, without returning an error).
* Another note on the difference between using this function on your model and on GPT - your model outputs logits, whereas HuggingFace's GPT implementation outputs an object with a logits attribute. Again, you can look at the `sample_tokens` function from yesterday to see how to handle this special case.
* Remember that your model should be in eval mode (this affects dropout), and you should be in inference mode (this affects gradients).
""")
    st.markdown("""
```python
import transformers
import torch as t

def beam_search(
    model, input_ids: t.Tensor, num_return_sequences: int, num_beams: int, max_new_tokens: int, tokenizer, verbose=False
) -> list[tuple[float, t.Tensor]]:
    '''
    input_ids: (seq, ) - the prompt
    max_new_tokens: stop after this many new tokens are generated, even if no EOS is generated. In this case, the best incomplete sequences should also be returned.
    verbose: if True, print the current (unfinished) completions after each iteration for debugging purposes
    
    Return list of length num_return_sequences. Each element is a tuple of (logprob, tokens) where the tokens include both prompt and completion, sorted by descending logprob.
    '''
    assert num_return_sequences <= num_beams
    pass

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2").to(device).train()

your_prompt = "I don't want to rule the universe. I just think"
input_ids = tokenizer(your_prompt, return_tensors="pt", return_attention_mask=False)["input_ids"][0]

num_return_sequences = 3
num_beams = 6
max_new_tokens = 10

final_logitsums_and_completions = beam_search(gpt, input_ids, num_return_sequences, num_beams, max_new_tokens, tokenizer, verbose=True)
```

As a guide, here is some of the verbose output from the solution, using the code above:

```
Printing num_beams=6 best completions:

logitsum | completion
  -1.284 | I don't want to rule the universe. I just think it
  -1.473 | I don't want to rule the universe. I just think that
  -2.682 | I don't want to rule the universe. I just think we
  -2.987 | I don't want to rule the universe. I just think I
  -3.010 | I don't want to rule the universe. I just think there
  -3.052 | I don't want to rule the universe. I just think the

Printing num_beams=6 best completions:

logitsum | completion
  -1.687 | I don't want to rule the universe. I just think it's
  -3.738 | I don't want to rule the universe. I just think that it
  -3.739 | I don't want to rule the universe. I just think there's
  -3.754 | I don't want to rule the universe. I just think that if
  -4.030 | I don't want to rule the universe. I just think it is
  -4.171 | I don't want to rule the universe. I just think that the

...

Returning best num_return_sequences=3 completions:

logitsum | completion
 -12.560 | I don't want to rule the universe. I just think there's a lot of things that need to be
 -13.238 | I don't want to rule the universe. I just think it's too much of a stretch to say that
 -13.411 | I don't want to rule the universe. I just think there's a lot of things we can do to
```

""")

def section2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#introduction">Introduction</a></li>
    <li><a class="contents-el" href="#bert-architecture-an-overview">BERT architecture: an overview</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#mlm-and-special-tokens">MLM and special tokens</a></li>
    </li></ul>
    <li><a class="contents-el" href="#other-architectural-differences-for-bert">Other architectural differences for BERT</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#attention-masking-for-pad">Attention masking for [PAD]</a></li>
        <li><a class="contents-el" href="#tokentype-embedding">TokenType embedding</a></li>
        <li><a class="contents-el" href="#unembedding">Unembedding</a></li>
    </li></ul>
    <li><a class="contents-el" href="#bert-config">BERT config</a></li>
    <li><a class="contents-el" href="#copying-weights">Copying weights</a></li>
    <li><a class="contents-el" href="#testing-bert">Testing BERT</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""

## Introduction

So far, we've only looked at decoder-only transformers (your own implementations, and GPT-2). Today, we'll take a look at encoder-only transformers.

BERT (Bidirectional Encoder Representations from Transformers) is the most famous in a line of Muppet-themed language research, originating with [ELMo](https://arxiv.org/pdf/1802.05365v2.pdf) (Embeddings from Language Models) and continuing with a series of increasingly strained acronyms:

* [Big BIRD](https://arxiv.org/pdf/1910.13034.pdf) - Big Bidirectional Insertion Representations for Documents
* [Ernie](https://arxiv.org/pdf/1904.09223.pdf) - Enhanced Representation through kNowledge IntEgration
* [Grover](https://arxiv.org/pdf/1905.12616.pdf) - Generating aRticles by Only Viewing mEtadata Records
* [Kermit](https://arxiv.org/pdf/1906.01604.pdf) - Kontextuell Encoder Representations Made by Insertion Transformations

Today you'll implement your own BERT model such that it can load the weights from a full size pretrained BERT, and use it to predict some masked tokens.

You can import (and inspect) BERT using:

```python
bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
```

Note, there might be some error warnings that appear when you import `bert`, don't worry about these.

## BERT architecture: an overview

The diagram below shows an overview of BERT's architecture. This differs from the [W1D3](https://arena-w1d3.streamlitapp.com/Putting_it_all_together#decoderblock-and-decoderonlytransformer) architecture in quite a few ways, which we will discuss below.
""")

    st.write("""<figure style="max-width:1000px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNqNVNlq4zAU_RWhZzclYZ7CEGgmaQm400LTJ3kocnTjiFiSkaWhadN_rxbbqRPPUGPQ1d10ztHyjjeKAZ7iQtNqh9aLTCL31TaPjgy7P5PR24vMQZuUysLSAu5dj_KU47-VrKy5JWH4g66uZqHglxJCyTk52TF2BJEDY1wWqOZvcMxJyiVQ_TPXs7tl-uzHlB5Ao99KCz9bc2DoWYa6psdftaF5rH8gqSq4QQ_W-PVP0ECyr2wuGEVQ_YS12oOMS3BpoAB9DK6lX5sE0yG6ni1bDhHPDWMxwxl9_DG-0KpS1nTazEu12dfkZPqicYImCRqNRgmSVryUvkc9rFnk2of-qGpuuDpD77wRWRum5f8IDKixPlQwoIh3u5KoCfKz7_U925XejgQtBs5Wp9sTlNuVlKDJjTEgPZ9unUvt0TgG79PHNmkykDVpWhgZZSUXJ6kPxHXo2n2PlwPQb-Ycod_XOxNvwdjbxx-vaMcZc8KG7fbXIm7BWSDWTFqWzTEjzdix_xevgLkxcIIFaEE5cy_Eu3dn2OxAQIanzmSwpbY0_oH4cKm2YtTAknGjNJ5uaVlDgqk16ukgN3hqtIU2acGpU0I0WR-fvPhwKw" /></figure>""", unsafe_allow_html=True)

    # graph TD
    #     subgraph " "

    #         subgraph BertLanguageModel
    #             InputF[Input] --> BertCommonB[BertCommon] --> |embedding size|b[Linear<br>GELU<br>Layer Norm<br>Tied Unembed] --> |vocab size|O[Logit Output]
    #         end

    #             subgraph BertCommon
    #             Token --> |integer|TokenEmbed[Token<br/>Embedding] --> AddEmbed[Add<br>Layer Norm] --> Dropout --> BertBlocks[BertBlocks<br>1, 2, ..., num_layers] --> |embedding size|Output
    #             Position --> |integer|PosEmbed[Positional<br/>Embedding] --> AddEmbed
    #             TokenType --> |integer|TokenTypeEmb[Token Type<br/>Embedding] --> AddEmbed
    #         end

    #         subgraph BertBlock
    #             Input --> BertSelfInner[Attention] --> Add[Add<br>Layer Norm 1] --> MLP --> Add2[Add<br>Layer Norm 2] --> AtnOutput[Output]
    #             Input --> Add --> Add2
    #         end

    #         subgraph BertMLP
    #             MLPInput[Input] --> Linear1 -->|4x hidden size|GELU --> |4x hidden size|Linear2 --> MLPDropout[Dropout] --> MLPOutput[Output]
    #         end
    #     end

    st.markdown("""
Conceptually, the most important way in which BERT and GPT differ is that BERT has bidirectional rather than unidirectional masking in its attention blocks. This is related to the tasks BERT is pretrained on. Rather than being trained to predict the next token in a sequence like GPT, BERT is pretrained on a combination of **masked language modelling** and **next sentence prediction**, neither of which require forward masking.

Note - this diagram specifically shows the version of BERT which performs masked language modelling. Many different versions of BERT share similar architectures. In fact, this language model and the classifier model which we'll work with in subsequent exercises both share the section we've called `BertCommon` - they only differ in the few layers which come at the very end.

""")

    st.write("""<figure style="max-width:500px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNqVUktvwjAM_itRznDpsZo4AGOaVOikwandwW1MsdQkVR6TGPDfl6RMiAObZkWy4_dnfSfeaoE8552B4cC2y1qxINY3o6Pm4dVq9N5F5mhcAarz0OE69OhvOVEEGWwdacW28_vIqxq8W1VJfbDpdJZaLbSUWs2rmz3GzigbFIJUxyx94bmpClII5qkxs5fnYhd1AUc0bKONjL8toWA7lequPT51C81YX1aF7six0rs4_7YaKvEQ56IHa2lPaP4HMnuAMvsbZlatyFjH3rSlNKFU_THCWxo9aO8S7nSJawPlJWvjnmjPZVb9rNzCWP0L3GDyCZdoJJAIVDhFd83dASXWPA-mwD343kUmXEIqeKffj6rluTMeJ9wPAhwuCcLFJM_30NvgRUFOm_VIr8SyyzcIHsrZ" /></figure>""", unsafe_allow_html=True)

    # graph TD
    # subgraph " "

    #     subgraph BertLanguageModel
    #         direction TB
    #         InputF[Input] --> BertCommonB[BertCommon] --> |embedding size|b[Linear<br>GELU<br>Layer Norm<br>Tied Unembed] --> |vocab size|O[Logit Output]
    #     end

    #     subgraph BertClassifier
    #         direction TB
    #         InputF2[Input] --> BertCommonB2[BertCommon] --> |embedding size|b2[First Position Only<br>Dropout<br>Linear] --> |num classes|O2[Classification Output]
    #     end

    # end


    st.markdown("""
### MLM and special tokens

You can import BERT's tokenizer using the following code:

```python
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
```

You should read [this page](https://albertauyeung.github.io/2020/06/19/bert-tokenization.html/) to get an idea of how the tokenizer works, then answer the following questions to check your understanding. You might also find [this paper](https://arxiv.org/abs/1810.04805) on BERT helpful.

""")

    with st.expander("Run 'tokenizer.tokenize' on some string. What do you notice about the output, whenever it has length greater than 1?"):
        st.markdown("""
When the length is greater than one, all tokens other than the first are prefixed with `##`. This is to indicate that the token is a suffix following some other subwords, rather than being the start of a word.
""")

    st.markdown("""If you run `tokenizer.special_tokens_map`, you will produce the following output:

```
{'unk_token': '[UNK]',
 'sep_token': '[SEP]',
 'pad_token': '[PAD]',
 'cls_token': '[CLS]',
 'mask_token': '[MASK]'}
```

Question - what do each of these tokens mean?""")

    with st.expander("[UNK]"):
        st.markdown("""`[UNK]` means unknown token. You can run something like `tokenizer.tokenize("🤗")` to produce this token.""")

    with st.expander("[CLS]"):
        st.markdown("""`[CLS]` stands for classification. Recall that the output of a transformer is a set of logits representing distributions over the vocabulary - one logit vector for each token in the input. The logit vector corresponding to `[CLS]` is the one we use to calculate predictive loss during fine-tuning on classification tasks (as well as pre-training on next-sentence prediction). """)

    with st.expander("[SEP]"):
        st.markdown("""`[SEP]` (separator) is appended to the end of sentences during NSP, in order to indicate where one sentence finishes and the next starts. It is also used in classification (here it is appended to the end of the sentence), but this is less important.""")

    with st.expander("[PAD]"):
        st.markdown("""`[PAD]` tokens are appended to the end of a sequence to bring its length up to the maximum allowed sequence length. This is important because we need all data to be the same length when batching them.""")

    with st.expander("[MASK]"):
        st.markdown("""`[MASK]` is used for masked language modelling. For instance, if our data contained the sentence `"The lemon was yellow and tasted sour"`, we might convert this into `"The [MASK] was yellow and tasted sour"`, and then measure the cross entropy between the model's predicted token distribution at `[MASK]` and the true answer `"lemon"`.""")

    st.markdown("""
A few more specific questions about tokenization, before we move on to talk about BERT architecture. Note that some of these are quite difficult and subtle, so don't spend too much time thinking about them before you reveal the answer.

In appendix **A.1** of [the BERT paper](https://arxiv.org/abs/1810.04805), the masking procedure is described. 15% of the time the tokens are masked (meaning we process these tokens in some way, and perform gradient descent on the model's predictive loss on this token). However, masking doesn't always mean we replace the token with `[MASK]`. Of these 15% of cases, 80% of the time the word is replaced with `[MASK]`, 10% of the time it is replaced with a random word, and 10% of the time it is kept unchanged. This is sometimes referred to as the 80-10-10 rule.
""")

    with st.expander("What might go wrong if you used 100-0-0 (i.e. only ever replaced the token with [MASK]) ?"):
        st.markdown("""
In this case, the model isn't incentivised to learn meaningful encodings of any tokens other than the masked ones. It might focus only on the `[MASK]` token and ignore all the others. 

As an extreme example, your model could hypothetically reach zero loss during pretraining by  perfectly predicting the masked tokens, and just **outputting a vector of zero logits for all non-masked tokens**. This means you haven't taught the model to do anything useful. When you try and fine-tune it on a task which doesn't involve `[MASK]` tokens, your model will only output zeros!

In practice, this extreme case is unlikely to happen, because for meaningful computation to be done at the later transformer layers, the residual stream for the non-`[MASK]` tokens must also contain meaningful information. See the comment below these dropdowns.
""")

    with st.expander("What might go wrong if you used 80-0-20 (i.e. only [MASK] and unchanged tokens) ?"):
        st.markdown("""
In the above case, the model isn't incentivised to store any information about non-masked tokens. In this case, the model **isn't incentivised to do anything other than copy those tokens**. This is because it knows that any non-masked token is already correct, so it can "cheat" by copying it. This is bad because the model's output for non-masked tokens won't capture anything about its relationship with other tokens.

The "extreme failure mode" analogous to the one in the 100-0-0 case is that the model learns to perfectly predict the masked tokens, and copy every other token. This model is also useless, because it hasn't learned any contextual relationships between different tokens; it's only learned to copy.
""")

    with st.expander("What might go wrong if you used 80-20-0 (i.e. only [MASK] and random tokens) ?"):
        st.markdown("""
In this case, the model will treat `[MASK]` tokens and the random tokens it has to predict as exactly the same. This is because it has no training incentive to treat them differently; both tokens will be different to the true token, but carry no information about the true token.

So the model would just treat all tokens the same as `[MASK]`, and we get the same extreme failure mode as in the 100-0-0 case.
""")

    st.markdown("""
It's worth noting that all of these points are pretty speculative. The authors used trial and error to arrive at this masking method and these proportions, and there is [some evidence](https://arxiv.org/pdf/2202.08005.pdf) that other methods / proportions work better. Basically, we still don't understand exactly why and how masking works, and to a certain extent we have to accept that!""")


    st.markdown("""
## Other architectural differences for BERT

Here are another few differences, on top of the bidirectional attention.

### Attention masking for `[PAD]`

Although we don't apply forward masking, we do still need to apply a form of masking in our attention blocks.

Above, we discussed the `[PAD]` token, which is used to extend sequences so that they have the maximum length. We don't want to read and write any information from/to the padding tokens, and so we should apply a mask to set these attention scores to a very large negative number before taking softmax. Here is some code to implement this:

```python
class MultiheadAttention(nn.Module):

    def __init__(self, config: TransformerConfig):
        pass

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor]) -> t.Tensor:
        pass 


class BERTBlock(nn.Module):

    def __init__(self, config):
        pass

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        additive_attention_mask: shape (batch, nheads=1, seqQ=1, seqK)
        '''
        pass


def make_additive_attention_mask(one_zero_attention_mask: t.Tensor, big_negative_number: float = -10000) -> t.Tensor:
    '''
    one_zero_attention_mask: 
        shape (batch, seq)
        Contains 1 if this is a valid token and 0 if it is a padding token.

    big_negative_number:
        Any negative number large enough in magnitude that exp(big_negative_number) is 0.0 for the floating point precision used.

    Out: 
        shape (batch, nheads=1, seqQ=1, seqK)
        Contains 0 if attention is allowed, big_negative_number if not.
    '''
    pass

utils.test_make_additive_attention_mask(make_additive_attention_mask)
```

An explanation of this code: `additive_attention_mask` is the tensor which gets added to the attention scores before taking softmax (unless it is `None`, in which case we don't do any masking). It has the purpose of zeroing all the attention probabilities corresponding to padding tokens. You should implement the function `make_additive_attention_mask`, which creates `additive_attention_mask` from your input data.

A few notes on the implementation of `make_additive_attention_mask`:

* You should make a mask to zero the attention probabilities at position `(q_idx, k_idx)` if and only if `k_idx` corresponds to a padding token.
    * Mathematically, this means that each row of your attention probabilities will only have non-zero values in the positions corresponding to non-`[PAD]` tokens.
    * Conceptually, this means that your model won't be paying attention to any of the padding tokens.
* Your `additive_attention_mask` can be of shape `(batch, 1, 1, seqK)`, because it will get broadcasted when added to the attention scores. 
""")

    # with st.expander("Question - what would be the problem with adding the attention mask at position (q_idx, k_idx) if EITHER q_idx OR k_idx corresponded to a padding token, rather than only adding at if k_idx is a padding token?"):
    #     st.markdown("""
    # If you did this, then in the softmax stage you'd be taking softmax over a row of all `-t.inf` values (or extremely small values). This would produce `nan` outputs (or highly numerically unstable outputs).
    # """)

    st.markdown("""
Also, a note on using `nn.Sequential`. If this is how you define your BertBlocks, then you might run into a problem when you try and call `self.bertblocks(x, additive_attention_mask)`. This is because `nn.Sequential` can only take one input which is sequentially fed into all its blocks. The easiest solution is to manually iterate over all the blocks in `nn.Sequential`, like this:

```python
for block in self.bertblocks:
    x = block(x, additive_attention_mask)
```

You can also use a `nn.ModuleList` rather than a `nn.Sequential`. You can think of this as an `nn.Sequential` minus the ability to run the entire thing on a single input (and plus the existence of an `append` method, which Sequential doesn't have). You can also think of `nn.ModuleList` as a Python list, but with the extra ability to register its contents as modules. For instance, using `self.layers = [layer1, layer2, ...]` won't work because the list contents won't be registered as modules and so won't appear in `model.parameters()` or `model.state_dict()`. But if you use `self.layers = nn.ModuleList([layer1, layer2, ...])`, you don't have this problem.

### TokenType embedding

Rather than just having token and positional embeddings, we actually take a sum of three embeddings: token, positional, and **TokenType embedding**. This embedding takes the value 0 or 1, and has the same output size as the other embeddings. It's only used in **Next Sentence Prediction** to indicate that a sequence position belongs to the first or second sentence, respectively. Here is what your `BertCommon` should look like:

```python
class BertCommon(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        pass
            
    def forward(
        self,
        x: t.Tensor,
        one_zero_attention_mask: Optional[t.Tensor] = None,
        token_type_ids: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        '''
        input_ids: (batch, seq) - the token ids
        one_zero_attention_mask: (batch, seq) - only used in training, passed to `make_additive_attention_mask` and used in the attention blocks.
        token_type_ids: (batch, seq) - only used for NSP, passed to token type embedding.
        '''
        pass
```

You can see how this is applied in the diagram in the [BERT paper](https://arxiv.org/abs/1810.04805), at the top of page 5. This also illustrates how the `[CLS]` and `[SEP]` tokens are used.
""")

    with st.expander("Question - what size should the TokenType embedding matrix have?"):
        st.markdown("""It should be initialised as `nn.Embedding(num_embeddings=2, embedding_dim=hidden_size)`, because the input will be an array of indices with value either zero or one.
    
    See below for the value of `hidden_size`, and the other architectural parameters.""")

    st.markdown("""

### Unembedding

Rather than simply ending with `LayerNorm -> Tied Unembed`, the Bert Language Model ends with a sequence of `Linear -> GELU -> LayerNorm -> Tied Unembed`. Additionally, the tied unembedding in BERT has a bias (which isn't tied to anything, i.e. it's just another learned parameter). The best way to handle this is to define the bias as an `nn.Parameter` object with size `(vocab_size,)`. Unfortunately, this seems to make copying weights a bit messier. I think `nn.Parameter` objects are registered first even if they're defined last, so you might find the output of `utils.print_param_count(my_bert, bert)` is shifted by 1 (see image below for what my output looked like), and you'll need to slightly rewrite the function you used to copy weights from GPT (more on this below).""")

    st_image('bert-compared.png', width=770)

    st.markdown(r"""
## BERT config

Referring to the [BERT paper](https://arxiv.org/abs/1810.04805), and try and find the values of all the `TransformerConfig` parameters. Note that we are using $\text{BERT}_\text{BASE}$ rather than $\text{BERT}_\text{LARGE}$. The `layer_norm_epsilon` isn't mentioned in the paper, but can be found by examining the BERT model. Also, the `vocab_size` figure given in the paper is just approximate - you should inspect your tokenizer to find the actual vocab size.
""")

    with st.expander("Answer"):
        st.markdown("""

* `num_layers`, `num_heads` and `vocab_size` can be found at the end of page 3
* `max_seq_len` and `dropout` are mentioned on page 13
* `vocab_size` can be found via `tokenizer.vocab_size`
* `layer_norm_epsilon` can be found by just printing `bert` and inspecting the output

We find that our parameters are:

```python
config = TransformerConfig(
    num_layers = 12,
    num_heads = 12,
    vocab_size = 28996,
    hidden_size = 768,
    max_seq_len = 512,
    dropout = 0.1,
    layer_norm_epsilon = 1e-12
)
```
""")

    st.markdown(r"""
## Copying weights

Again, the process of copying over weights from BERT to your model is a bit messy! A quick summary of some of the points made above, as well as a few more notes:

* It's easier to iterate through using `model.named_parameters()` rather than `model.state_dict()` in this case. 
    * Optional exercise: inspect `bert.state_dict()` and `bert.named_parameters()`, and see what objects are in one but not the other. Why are these objects in one but not the other, and based on this can you see why it's better to use `named_parameters` than `state_dict` when copying over? (or you can skip this exercise, and just take my word for it!)
""")

    with st.expander("Answer to optional exercise"):
        st.markdown("""
Upon inspection, we find:

```python
state_dict_names = set(bert.state_dict().keys())
param_names = set(dict(bert.named_parameters()).keys())

print(len(state_dict_names))  # 205
print(len(param_names))       # 202

print(state_dict_names - param_names)
# {'bert.embeddings.position_ids', 'cls.predictions.decoder.weight', 'cls.predictions.decoder.bias'}
```

From this output, we conclude that `state_dict` is a superset of `parameters`, and contains three items not in `parameters`.

Inspecting `bert.embeddings.position_ids`, we see that it is just an unsqueezed `t.arange(512)` array, which we can guess gets passed into the positional encoding.

Inspecting the other two, we see that they belong to the linear layer at the very end of BERT, i.e. the unembedding with bias. `cls.predictions.decoder.weight` is a duplicate of the token embedding matrix at the start (but this object is only counted once in `bert.parameters()`, because it's a tied embedding). `cls.predictions.decoder.bias` is a duplicate of `cls.predictions.bias`, and we don't need to count this twice! Again, there is only one underlying parameter.

The moral of the story - not only can `state_dict` sometimes contain things we don't need, but it can also sometimes contain multiple objects which refer to the same underlying parameters.
""")

    st.markdown("""
* Unlike for GPT, you shouldn't have to transpose any of your weights.
* BERT's attention layer weights are stored as `W_Q`, `W_K`, `W_V`, `W_O` separately (in that order) rather than as `W_QKV`, `W_O`. You should do the same in your model, because it makes iterating over and copying weights easier.

Here is a code template which you can fill in to write a weight copying function. Referring to `copy_weights` from W0D3 might be helpful.

```python
def copy_weights_from_bert(my_bert: BertLanguageModel, bert: transformers.models.bert.modeling_bert.BertForMaskedLM) -> BertLanguageModel:
    '''
    Copy over the weights from bert to your implementation of bert.

    bert should be imported using: 
        bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")

    Returns your bert model, with weights loaded in.
    '''
    
    # FILL IN CODE: define a state dict from my_bert.named_parameters() and bert.named_parameters()

    my_bert.load_state_dict(state_dict)
    return my_bert
```

## Testing BERT

Once you've built your BERT architecture and copied over weights, you can test it by looking at its prediction on a masked sentence. Below is a `predict` function which you should implement. You might find it useful to inspect the `utils.test_load_pretrained_weights` function (which is used for the GPT exercises) - although if you can get by without looking at this function, you should try and do so.

Note that you might have to be careful with your model's output if you want `predict` to work on your BERT and the imported BERT. This is because (just like GPT) the imported BERT will output an object which has a `logits` attribute, rather than just a tensor of logits.""")

    st.markdown(r"""
```python
def predict(model, tokenizer, text: str, k=15) -> List[List[str]]:
    '''
    Return a list of k strings for each [MASK] in the input.
    '''
    pass

def test_bert_prediction(predict, model, tokenizer):
    '''Your Bert should know some names of American presidents.'''
    text = "Former President of the United States of America, George[MASK][MASK]"
    predictions = predict(model, tokenizer, text)
    print(f"Prompt: {text}")
    print("Model predicted: \n", "\n".join(map(str, predictions)))
    assert "Washington" in predictions[0]
    assert "Bush" in predictions[0]

test_bert_prediction(predict, my_bert, tokenizer)
```
""")

def section3():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#imports">Imports</a></li>
   <li><a class="contents-el" href="#fine-tuning-bert">Fine-Tuning BERT</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#imdb-dataset">IMDB Dataset</a></li>
   </ul></li>
   <li><a class="contents-el" href="#data-visualization">Data Visualization</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#basic-inspection">Basic Inspection</a></li>
       <li><a class="contents-el" href="#detailed-inspection">Detailed Inspection</a></li>
   </ul></li>
   <li><a class="contents-el" href="#training-loop">Training Loop</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#training-all-parameters">Training All Parameters</a></li>
       <li><a class="contents-el" href="#learning-rate">Learning Rate</a></li>
       <li><a class="contents-el" href="#loss-functions">Loss Functions</a></li>
       <li><a class="contents-el" href="#gradient-clipping">Gradient Clipping</a></li>
       <li><a class="contents-el" href="#batch-size">Batch Size</a></li>
       <li><a class="contents-el" href="#optimizer">Optimizer</a></li>
       <li><a class="contents-el" href="#when-all-else-fails">When all else fails...</a></li>
   </ul></li>
   <li><a class="contents-el" href="#inspecting-the-errors">Inspecting the Errors</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown("""
## Imports

```python
import os
import re
import tarfile
from dataclasses import dataclass
import requests
import torch as t
import transformers
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm
import plotly.express as px
import pandas as pd
from typing import Callable, Optional, List
import time

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_FOLDER = "./data/bert-imdb/"
IMDB_PATH = os.path.join(DATA_FOLDER, "acllmdb_v1.tar.gz")
SAVED_TOKENS_PATH = os.path.join(DATA_FOLDER, "tokens.pt")

device = t.device("cuda" if t.cuda.is_available() else "cpu")
```

## Fine-Tuning BERT

Fine-tuning a pretrained model is awesome - it typically is much faster and more accurate than directly training on the task you care about, especially if your task has relatively few labels available. In our case we will be using the [IMDB Sentiment Classification Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

It's traditional to treat this as a binary classification task where each review is positive or negative. Today we're also going to predict the star rating review from 1 to 10, inclusive.

It's a bit redundant to train with both the star rating and the positive/negative labels as targets (you could just use the star rating), but we'll do it anyway to practice having multiple terms in the loss function.

There are a few ways to treat the star rating. One way is to have each rating be a category and use the regular cross entropy loss.
""")

    with st.expander("Question - what are the disadvantages of doing this?"):
        st.markdown("Cross entropy doesn't capture the intuition that the classes are ordered. Intuitively, we should penalize our model more for predicting 10 stars for a 1-star movie than predicting 2 stars.")

    st.markdown("""
Another way would be to treat the star rating as a continuous value, and use mean squared or mean absolute error. We'll do this today because it's simple and works well, but note that a more sophisticated approach like [ordinal regression](https://en.wikipedia.org/wiki/Ordinal_regression) could also be used.

### IMDB Dataset

Previously, we've used the `torchvision` package to download CIFAR10 for us. Today we'll load and process the training data ourselves to get an idea of what's involved.

Use [`requests.get`](https://requests.readthedocs.io/en/latest/user/quickstart/) to fetch the data and then write the `content` field of the response to disk. It's 82MB, so may take a few seconds depending on your connection. On future calls to the function, if the file already exists, your function should just read the local file instead of downloading the data again.

```python
def maybe_download(url: str, path: str) -> None:
    '''
    Download the file from url and save it to path. 
    If path already exists, do nothing.
    '''
    pass

os.makedirs(DATA_FOLDER, exist_ok=True)
maybe_download(IMDB_URL, IMDB_PATH)
```

Now we have a tar archive, which we can read using the standard library module [tarfile](https://docs.python.org/3/library/tarfile.html). You'll need to refer to this page as you write the function below. (Note, this is quite hard to do if you're not used to using `tarfile`, so if you're struggling then you can look at the solutions.)

Your code below should:
* Open the archive with `tarfile.open`
    * You should open in reading mode. The filename of `IMDB_URL` and the table at the top of the tarfile documentation page should tell you what `mode` argument to use in this function.
* Use `tar.getmembers()` to loop through the members of the tarfile, in order to get all the data you need to construct your dataset. 
    * The `member.name` method will return the filename.
        * A filename like `aclImdb/test/neg/127_3.txt` means it belongs to the test set, has a negative sentiment, has an id of 127 (we will ignore this), and was rated 3/10 stars.
    * The object returned from `tarfile.open` has an `extractfile` method which takes `member` as input and returns the file object.
        * You can then read the text using the `read()` file method.

Your final output should be a list of `Review` objects. This is a dataclass we've given you, which contains all the information about a particular imdb review, and which we'll use to construct our dataset. 

You should have 25000 train and 25000 test entries.

This should take less than 10 seconds, but it's good practice to use `tqdm` to monitor your progress as most datasets will be much larger than this one.

```python
@dataclass(frozen=True)
class Review:
    split: str          # "train" or "test"
    is_positive: bool   # sentiment classification
    stars: int          # num stars classification
    text: str           # text content of review

def load_reviews(path: str) -> list[Review]:
    pass

reviews = load_reviews(IMDB_PATH)
assert sum((r.split == "train" for r in reviews)) == 25000
assert sum((r.split == "test" for r in reviews)) == 25000
```

## Data Visualization

Charles Babbage, the inventor of the first mechanical computer, was famously asked "Pray, Mr. Babbage, if you put into the machine wrong figures, will the right answers come out?"

200 years later, if you put wrong figures into the machine, the right answers still do not come out.

Inspecting the data before training can be tedious, but will catch many errors, either in your code or upstream, and allow you to validate or refute assumptions you've made about the data. Remember: "Garbage In, Garbage Out".

### Basic Inspection

Take some time now to do a basic inspection of your data. This should at minimum include:

- Plot the distribution of review lengths in characters.
    - Our BERT was only trained to handle 512 tokens maximum, so if we assume that a token is roughly 4 characters, we will have to truncate reviews that are longer than around 2048 characters.
    - Are positive and negative reviews different in length on average? If so, truncating would differentially affect the longer reviews.
- Plot the distribution of star ratings. Is it what you expected?

You might find the following useful - you can call `pd.DataFrame(reviews)` to generate a dataframe where the columns are the attributes of the `Review` dataclass.

### Detailed Inspection

Either now, or later while your model is training, it's a worthwhile and underrated activity to do a more in-depth inspection. For a language dataset, some things I would want to know are:

- What is the distribution over languages? Many purportedly English datasets in fact have some of the data in other natural languages like Spanish, and computer languages like HTML tags.
    - This can cause bias in the results. Suppose that a small fraction of reviews were in Spanish, and purely by chance they have more positive/negative sentiment than the base rate. Our classifier would then incorrectly learn that Spanish words inherently have positive/negative sentiment.
    - Libraries like [Lingua](https://github.com/pemistahl/lingua-py) can (imperfectly) check for this.
- How are non-ASCII characters handled?
    - The answer is often "poorly". A large number of things can go wrong around quoting, escaping, and various text encodings. Spending a lot of time trying to figure out why there are way too many backslashes in front of your quotation marks is an Authentic ML Experience. Libraries like [`ftfy`](https://pypi.org/project/ftfy/) can be useful here.
- What data can you imagine existing that is NOT part of the dataset? Your neural network is not likely to generalize outside the specific distribution it was trained on. You need to understand the limitations of your trained classifier, and notice if you in fact need to collect different data to do the job properly:
    - What specific geographical area, time period, and demographic was the data sampled from? How does this compare to the deployment use case?
    - What filters were applied upstream that could leave "holes" in the distribution?
- What fraction of labels are objectively wrong?
    - Creating accurate labels is a laborious process and humans inevitably make mistakes. It's expensive to check and re-check labels, so most published datasets do contain incorrect labels.
    - Errors in training set labels can be mitigated through stronger regularization to prevent the model from memorizing the errors, or other techniques.
    - Most pernicious are errors in **test set** labels. Even a small percentage of these can cause us to select a model that outputs the (objectively mistaken) label over one that does the objectively right thing. The paper [Pervasive Label Errors in Test Sets
Destabilize Machine Learning Benchmarks](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/f2217062e9a397a1dca429e7d70bc6ca-Paper-round1.pdf) shows that these are more common than you might expect, and describes implications of this in more detail.

```python
"TODO: YOUR CODE HERE, TO VISUALISE DATASET"
```
""")

    with st.expander("Click to see a few example visualisations and observations:"):
        st.markdown("""
```python
df = pd.DataFrame(reviews)
df["length"] = [len(text) for text in df["text"]]

px.histogram(x=df["stars"]).update_layout(bargap=0.1)
```""")
        st_image('data_pic_1.png', width=650)
        st.markdown("""
There are no five or six star reviews.

---

```python
px.histogram(x=df["length"])
```
""")

        st_image('data_pic_2.png', width=650)
        st.markdown("""
The distribution is very heavy-tailed, peaks around 1000 characters.

---

```python
px.histogram(df, x="length", color="is_positive", barmode="overlay")
```
""")
        st_image('data_pic_3.png', width=650)
        st.markdown("""
Slightly more of the shirt 200-500 word reviews for positive reviews, but apart from that the distributions are very similar.

---

Now we'll look at some languages.

```python
# !pip install lingua-language-detector

from lingua import Language, LanguageDetectorBuilder
detector = LanguageDetectorBuilder.from_languages(*Language).build()
# Note, detector takes much longer to run when it is detecting all languages

# Sample 500 datapoints, because it takes a while to run
languages_detected = df.sample(500)["text"].apply(detector.detect_language_of).value_counts()
display(languages_detected)
```

Result:

```
Language.ENGLISH    500
Name: text, dtype: int64
```

We can guess that all, or virtually all, of the reviews are in English.
""")

    st.markdown("""
Now, you should implement `to_dataset`. Calling this function could take a minute, as tokenization requires a lot of CPU even with the efficient Rust implementation provided by HuggingFace. We aren't writing our own tokenizer because it would be extremely slow to do it in pure Python.

Note that you really don't want to have to do long-running tasks like this repeatedly. It's always a good idea to store the preprocessed data on disk and load that on future runs. Then you only have to re-run it if you want to preprocess the data in some different way.

Previously, you created custom datasets which inherited from `torch.utils.data.Dataset`. Now, you'll do it in a different way: by creating a `TensorDataset`. This is an object which takes in a series of tensors, and constructs a dataset by interpreting the 0th dimension of each tensor as its batch dimension. From the `TensorDataset` docstring:

```
Init signature: TensorDataset(*tensors: torch.Tensor) -> None
Docstring:     
Dataset wrapping tensors.

Each sample will be retrieved by indexing tensors along the first dimension.
```

There are times when this will be a more efficient way to create datasets then inheriting from `Dataset`.

One last note - there are 25000 reviews in each of the test and train datasets, and so you might want to only measure a subset of them (e.g. the first 1000) if you want training to be relatively quick. You can still get decent results from this many.

```python
def to_dataset(tokenizer, reviews: list[Review]) -> TensorDataset:
    '''Tokenize the reviews (which should all belong to the same split) and bundle into a TensorDataset.

    The tensors in the TensorDataset should be (in this exact order):

    input_ids: shape (batch, sequence length), dtype int64
    attention_mask: shape (batch, sequence_length), dtype int
    sentiment_labels: shape (batch, ), dtype int
    star_labels: shape (batch, ), dtype int
    '''
    pass

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
train_data = to_dataset(tokenizer, [r for r in reviews if r.split == "train"])
test_data = to_dataset(tokenizer, [r for r in reviews if r.split == "test"])
t.save((train_data, test_data), SAVED_TOKENS_PATH)
```

## Bonus (optional)

You can go on to Step 2, but if you have time at the end, you can come back and try the bonus exercise for this part.

### Better Truncation

We arbitrarily kept the first `max_length` tokens and truncated the rest. Is this strategy optimal? If you read some of the reviews, a common pattern is to sum up and conclude the review in the final 1-2 sentences.

This suggests that we might do better by keeping some number of tokens at the end and truncating the middle instead. Implement this strategy and see if you can measure a difference in accuracy.

### Better Data Cleaning

You may have noticed that paragraph breaks are denoted by the string "< br / > < br / >". We might suppose that these are not very informative, and it would be better to strip them out. Particularly, if we are truncating our reviews then we would rather have 10 tokens of text than this in the truncated sequence. Replace these with the empty string (in all splits of the data) and see if you can measure a difference in accuracy.

## BertClassifier

Now we'll set up our BERT to do classification, starting with the `BertCommon` from before and adding a few layers on the end:

- Use only the output logits at the first sequence position (index 0).
- Add a dropout layer with the same dropout probability as before.
- Add a `Linear` layer from `hidden_size` to `2` for the classification as positive/negative.
- Add a `Linear` layer from `hidden_size` to `1` for the star rating.
- By default, our star rating Linear layer is initialized to give inputs of roughly mean 0 and std 1. Multiply the output of this layer by 5 and add 5 to bring these closer to the 1-10 output we want; this isn't strictly necessary but helps speed training.

You may find it helpful to refer back to this diagram, comparing BertLanguageModel and BertClassifier:
""")

    st.write("""<figure style="max-width:500px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNqVUktvwjAM_itRznDpsZo4AGOaVOikwandwW1MsdQkVR6TGPDfl6RMiAObZkWy4_dnfSfeaoE8552B4cC2y1qxINY3o6Pm4dVq9N5F5mhcAarz0OE69OhvOVEEGWwdacW28_vIqxq8W1VJfbDpdJZaLbSUWs2rmz3GzigbFIJUxyx94bmpClII5qkxs5fnYhd1AUc0bKONjL8toWA7lequPT51C81YX1aF7six0rs4_7YaKvEQ56IHa2lPaP4HMnuAMvsbZlatyFjH3rSlNKFU_THCWxo9aO8S7nSJawPlJWvjnmjPZVb9rNzCWP0L3GDyCZdoJJAIVDhFd83dASXWPA-mwD343kUmXEIqeKffj6rluTMeJ9wPAhwuCcLFJM_30NvgRUFOm_VIr8SyyzcIHsrZ" /></figure>""", unsafe_allow_html=True)

    # graph TD
    # subgraph " "

    #     subgraph BertLanguageModel
    #         direction TB
    #         InputF[Input] --> BertCommonB[BertCommon] --> |embedding size|b[Linear<br>GELU<br>Layer Norm<br>Tied Unembed] --> |vocab size|O[Logit Output]
    #     end

    #     subgraph BertClassifier
    #         direction TB
    #         InputF2[Input] --> BertCommonB2[BertCommon] --> |embedding size|b2[First Position Only<br>Dropout<br>Linear] --> |num classes|O2[Classification Output]
    #     end

    # end

    st.markdown("You should spend some time thinking about the easiest way to copy over pretrained weights from BERT. Hint - you shouldn't need to use a new BERT implementation; the one you used with your BertLanguageModel should suffice.")

    with st.expander("Help - I'm not sure how to copy over my weights."):

        st.markdown("""
Although your `BertClassifier` now has some extra layers that you'll have to train from scratch, remember that the core of it is a `BertCommon` object. The BertLanguageModel implementation by HuggingFace (which you used when copying over the weights in the last exercise) also has a subnetwork with the same architecture as `BertCommon` (you can inspect HuggingFace's implementation to check what it calls its equivalent of `BertCommon`). 

You could rewrite your `copy_weights_from_bert` function to copy directly between the `BertCommon` modules (this copying function won't work immediately because of the problem of offset layers). You'll probably need to rewrite it because you no longer have a parameter for the unembedding bias, so no messy reordering step will be needed like last time! Once you've done this, you can write your `BertClassifier` by having it define `self.bert_common = BertCommon(config)` *and* copying over weights.

Alternatively, you have a BertCommon module inside your BertLanguageModel, so you can just strip that out and use it in the intialisation of your BertClassifier.
""")

    st.markdown("""
Now, you should implement your BERT Classifier, with pretrained weights loaded in:

```python
class BertClassifier(nn.Module):
    pass
```

## Training Loop

You should copy over a training loop from before, and modify it in the ways described below.

```python
"TODO: YOUR TRAINING LOOP GOES HERE"
```

### Training All Parameters

When fine-tuning a language model, ensure all the parameters have `requires_grad=True`. This is different from fine-tuning an image model, where you typically "freeze" (`requires_grad=False`) the existing layers and just train your new layers.

### Learning Rate

The learning rate for fine-tuning should be much lower than when training from scratch. In Appendix A.3 of the [BERT Paper](https://arxiv.org/pdf/1810.04805.pdf), they suggest a learning rate for Adam between 2e-5 and 5e-5.

### Loss Functions

Use `torch.nn.CrossEntropyLoss` for the classification loss. For the star loss, empirically `F.l1_loss` works well. When you have multiple loss terms, you usually take a weighted sum of them, with weights chosen so that their scales aren't too different. Empirically, 0.02 seems to work well here, but you could also try something cleverer (e.g. a moving weight which adjusts based on the sizes of the respective loss functions).

### Gradient Clipping

Especially early in training, some batches can have very large gradients, like more than 1.0. The resulting large parameter updates can break training. To work around this, you can manually limit the size of gradients using `t.nn.utils.clip_grad_norm_`. Generally, a limit of 1.0 works decently.

### Batch Size

For a model the size of BERT, you typically want the largest batch size that fits in GPU memory. The BERT paper suggests a batch size of 16, so if your GPU doesn't have enough memory you could use a smaller size, accumulate your gradients, and call `optimizer.step` every second batch. Next week, we'll learn how to use multiple GPUs instead.

### Optimizer

`t.optim.AdamW` works well here. Empirically it seems to outperform `t.optim.Adam` often (although the two are the same unless you use non-zero weight decay - here we recommend a weight decay of 0.01).

### When all else fails...

If your loss is behaving strangely and you don't seem to be getting convergence, here are a few things it might be worth taking a look at:

- Double check that your BertClassifier is actually using the pretrained weights and not random ones.
- The classification loss for positive/negative should be around `log(2)` before any optimizer steps are taken, because the model is predicting randomly. If this isn't the case, there might be a bug in your loss calculation.
- Try decoding a batch from your DataLoader and verify that the labels match up and the tokens and padding are right. It should be `[CLS]`, the review, `[SEP]`, and then `[PAD]` up to the end of the sequence.
- Try using an even smaller learning rate to see if this affects the loss curve. It's usually better to have a learning rate that is too low and spend more iterations reaching a good solution than to use one that is too high, which can cause training to not converge at all.
- If your model is predicting all 1 or all 0, this can be a helpful thing to investigate.
- It may just be a bad seed. The paper [On the Stability of Fine-Tuning BERT: Misconceptions, Explanations, and Strong Baselines](https://arxiv.org/pdf/2006.04884.pdf) notes that random seed can make a large difference to the results.

## Inspecting the Errors

Print out an example that your model got egregiously wrong - for example, the predicted star rating is very different from the actual, or the model placed a very high probability on the incorrect class.

Decode the text and make your own prediction, then check the true label. How good was your own prediction? Do you agree more with the "true" label or with your model?

If the model was in fact wrong, speculate on why it got that example wrong.
""")

def section4():
    st.markdown("""# LeetCode""")
    st.markdown("")
    st_image('balanced_brackets.png', width=400)
    st.markdown("""
Pick some of your favourite easy LeetCode problems (e.g. detecting whether a bracket string is balanced), and train a transformer to solve it. Some questions you might like to think about:

* How can you formulate the problem in a way that can be solved by your transformer?
    * What should your tokens be? 
    * How can you construct a custom dataset for this function? (hint - you'll need to solve the LeetCode problem yourself in order to generate labels to go with your input data!)
    * How should you interpret your output in a way that allows your transformer to train?
* What architectural features does your transformer need to solve this problem?
    * How many layers does it need, at minimum?
    * Is the problem one where you can use unidirectional attention masking (or even one where unidirectional attention is more useful, because you can discard information you don't need?) 
""")
    with st.expander("Example for how you might formulate the balanced brackets problem (don't read until you've thought about this!)"):
        st.markdown(r"""
Earlier this week, you read about the BERT tokens `[CLS]`, `[SEP]` and `[PAD]`. You can set up something similar for this task: append a `[CLS]` token to the start of your sequence, `[SEP]` to the end, andthen  `[PAD]` tokens to the end to get it up to a certain length. You can then use your transformer's output at sequence position 0 (the position corresponding to the classification token) to predict whether the sequence is balanced or not. Since your transformer's output will be a set of logits over your vocabulary, the easiest way to convert this into a prediction would be to affix an extra linear layer which takes you down to 2 logits, then softmax those to get your classification probabilities (i.e. treat these two values as representing $\mathbb{P}(\text{balanced})$ and $\mathbb{P}(\text{not balanced})$ respectively).

Your transformer should have bidirectional attention in this case, i.e. no masking. This is because the output corresponding to your `[CLS]` token needs to be able to read the brackets ahead of it in the sequence.

As you can see, there's a lot of subtlety that goes into formulating a task like this in a way that a transformer can solve!""")
        st.info("""
Note - you might be wondering why we need the `[SEP]` token. After all, the `[CLS]` token is clearly necessary because our model uses it for training, and we need padding tokens to get it to the right length. But why do we need the `[SEP]` token, if we aren't separating two different sentences like in NSP?

I'm not actually certain what the answer is. At best guess, it's something to do with the transformer needing to know where the start and the end of the bracket string is. One of the conditions for a bracket string to be balanced is whether its "altitude" at the end of the bracket string is zero. The presence of the `[SEP]` token indicates to the transformer where the end of the string is, so it can check the altitude is zero at this point. Having a `[PAD]` token at the end of the string instead wouldn't work, because these sequence positions are masked so can't store information.

(This could be wrong though - it might be possible to train without `[SEQ]` tokens and I just didn't find the right hyperparameters!)

Hopefully, stuff like this will become clearer in the interpretability week, when we take a closer look at transformers trained on tasks like this one, and try and reverse-engineer how they're solving the problem!
""")

    st.markdown("""
---

# Semantle""")

    st_image('semantle.png', width=150)
    st.markdown("""
    
Design your own game of [Semantle](https://semantle.com/), using your transformer's learned token embeddings. How easy is this version of the game to play, relative to the official version? 
    
Why do you expect the cosine similarity between vectors in your transformer's learned embedding to carry meaninfgul information about the word similarities, in the same way that Word2vec does? Or if not, then why not?""")

func_list = [section_home, section1, section2, section3, section4]

page_list = ["🏠 Home", "1️⃣ Build and sample from GPT-2", "2️⃣ Build BERT", "3️⃣ Finetune BERT", "4️⃣ Other bonus exercises"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:

        radio = st.radio("Section", page_list)

        st.markdown("---")

    func_list[page_dict[radio]]()

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if is_local or check_password():
    page()
