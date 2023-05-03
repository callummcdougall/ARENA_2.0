import os
import streamlit as st
# st.set_page_config(layout="wide")
import platform
is_local = (platform.processor() != "")
from pathlib import Path
import sys

import st_dependencies
st_dependencies.styling()

# Navigate to the root directory, i.e. ARENA_2 for me, or the working directory for people locally
while "chapter" in os.getcwd():
    os.chdir("..")
# Now with this reference point, we can add things to sys.path
root_suffix = r"/chapter1_transformers/instructions"
root_dir = os.getcwd() + root_suffix
root_path = Path(root_dir)
if root_dir not in sys.path: sys.path.append(root_dir)

if os.getcwd().endswith("chapter1_transformers") and "./instructions" not in sys.path:
    sys.path.append("./instructions")
if os.getcwd().endswith("pages") and "../" not in sys.path:
    sys.path.append("../")

st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
    <li class="margtop"><a class="contents-el" href="#about-this-page">About this page</a></li>
    <li class="margtop"><a class="contents-el" href="#how-you-should-use-this-material">How you should use this material</a></li>
    <li class="margtop"><ul class="contents">
        <li><a class="contents-el" href="#option-1-colab">Option 1: Colab</a></li>
        <li><a class="contents-el" href="#option-2-your-own-ide">Option 2: Your own IDE</a></li>
        <li><a class="contents-el" href="#chatbot-assistant">Chatbot assistant</a></li>
    </ul></li>
    <li class="margtop"><a class="contents-el" href="#hints">Hints</a></li>
    <li class="margtop"><a class="contents-el" href="#test-functions">Test functions</a></li>
    <li class="margtop"><a class="contents-el" href="#tips">Tips</a></li>
    <li class="margtop"><a class="contents-el" href="#feedback">Feedback</a></li>
</ul>
""", unsafe_allow_html=True)

def section_home():
    # start
    st.markdown(r"""
<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/magnifying-glass-2.png" width="600">

# Chapter 1: Transformers & Mechanistic Interpretability 

The material on this page covers the next 8 days of the curriculum. It will cover transformers (what they are, how they are trained, how they are used to generate output) as well as mechanistic interpretability (what it is, what are some of the most important results in the field so far, why it might be important for alignment).

Some highlights from this chapter include:

* Building your own transformer from scratch, and using it to sample autoregressive output
* Using the [TransformerLens](https://github.com/neelnanda-io/TransformerLens) library developed by Neel Nanda to locate induction heads in a 2-layer model
* Finding a circuit for [indirect object identification](https://arxiv.org/abs/2211.00593) in GPT-2 small
* Intepreting model trained on toy tasks, e.g. classification of bracket strings, or modular arithmetic
* Replicating Anthropic's results on [superposition](https://transformer-circuits.pub/2022/toy_model/index.html)

Unlike the first chapter (where all the material was compulsory), this chapter has 4 days of compulsory content and 4 days of bonus content. During the compulsory days you will build and train transformers, and get a basic understanding of mechanistic interpretability of transformer models which includes induction heads & use of TransformerLens. The next 4 days, you have the option to continue with whatever material interests you out of the remaining sets of exercises. There will also be bonus material if you want to leave the beaten track of exercises all together!

---

## About this page

This page was made using an app called Streamlit. It's hosted from the ARENA 2.0 [GitHub repo](https://github.com/callummcdougall/ARENA_2.0). It provides a very simple way to display markdown, as well as more advanced features like interactive plots and animations.

There is a navigation bar on the left menu, which should show a page called `Home` (the current page) as well as a page for each set of exercises.

If you want to change to dark mode, you can do this by clicking the three horizontal lines in the top-right, then navigating to Settings → Theme.

## How you should use this material

### Option 1: VSCode

This is the option we strongly recommend for all participants of the in-person ARENA program.

<details>
<summary>Click this dropdown for setup instructions.</summary>

* Clone the [GitHub repo](https://github.com/callummcdougall/ARENA_2.0) into your local directory.
* Open in your choice of IDE (we recommend VSCode).
* Make & activate a virtual environment
    * We strongly recommend using `conda` for this. You can install `conda` [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), and find basic instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
* Install requirements.
    * First, install PyTorch using the following command: `conda install pytorch=1.11.0 torchdata torchvision -c pytorch -y`.
    * Then install the rest of the requirements by navigating to the directory and running `pip install -r requirements.txt`.
* While in the directory, run `streamlit run Home.py` in your terminal (this should work since Streamlit is one of the libraries in `requirements.txt`).
    * This should open up a local copy of the page you're reading right now, and you're good to go!

To complete one of the exercise pages, you should:

* Navigate to `exercises` in the repo, then to the appropriate sub-folder (e.g. `transformer_from_scratch` for the first set of exercises).
* Create a file called `answers.py`.
* Go through the Streamlit page, and copy over / fill in then run the appropriate code as you go through the exercises.
</details>

### Option 2: Colab

This option is recommended if either of the following is true:

* You have limited access to GPU support
* You want to avoid the hassle of setting up your own environment

<details>
<summary>Click this dropdown for links to each of the colab exercises.</summary>

<div style='text-align: center'>
<img src="https://raw.githubusercontent.com/callummcdougall/ARENA_2.0/main/media/transformers/transformer-building.png" width="160" style="margin-bottom:3px;margin-top:15px">

Transformers from scratch<br>[**exercises**]() | [**solutions**]()

<img src="https://raw.githubusercontent.com/callummcdougall/ARENA_2.0/main/media/transformers/sampling.png" width="160" style="margin-bottom:3px;margin-top:15px">

Training, Sampling, Caching<br>[**exercises**]() | [**solutions**]()

<img src="https://raw.githubusercontent.com/callummcdougall/ARENA_2.0/main/media/transformers/circuit.png" width="160" style="margin-bottom:3px;margin-top:15px">

Intro to Mechanistic Interpretability<br>[**exercises**]() | [**solutions**]()

<img src="https://raw.githubusercontent.com/callummcdougall/ARENA_2.0/main/media/transformers/leaves.png" width="160" style="margin-bottom:3px;margin-top:15px">

Indirct Object Identification<br>[**exercises**]() | [**solutions**]()

<img src="https://raw.githubusercontent.com/callummcdougall/ARENA_2.0/main/media/transformers/gears2.png" width="160" style="margin-bottom:3px;margin-top:15px">

Interpretability on an Algorithmic Model<br>[**exercises**]() | [**solutions**]()
</div>

</details>

For each of these sections, you can make a copy of the **exercises** notebooks in your own drive, and fill in the code cells whenever indicated. The solutions will be available in dropdowns next to each of the code cells (or you can look at the **solutions** notebooks, which have all code pre-run and output displayed).

### Chatbot assistant

In the sidebar of this page, below the contents page, you will (at first) see an error message saying "Please set the OpenAI key...". This is space for a chatbot assistant, which can help answer your questions about the material. Take the following steps to set it up:

* Go to the [OpenAI API](https://openai.com/blog/openai-api) and sign up for an account.
* Create a secret key from [this page](https://platform.openai.com/account/api-keys). Copy this key.
* Create a file `.streamlit/secrets.toml` in this repo, and have the first line read `openai_api_key = "<your key>"`.
* Refresh the page, and you should now be able to use the chatbot.

This interface was built using the `openai` library, and it exists to help answer questions you might have about the material. All prompts from this chatbot are prepended with most\* of the material on the page and section you're currently reading. For instance, try passing in the question ***What are 2 ways to use this material?*** to the chatbot, and it should describe the two options given above (i.e. colab, or your own IDE). This feature is very experimental, so please [let me know](mailto:cal.s.mcdougall@gmail.com) if you have any feedback!

\**Because of the context window, the entire page isn't always included in the prompt (e.g. generally code blocks aren't included). When in doubt, you can copy sections of the page into the prompt and run it! If you get an error message saying that the prompt is too long, then you can use the **clear chat** button and start again.*

Here are some suggestions for the kinds of questions you can ask the chatbot (in the appropriate sections of the course):

* *(copying in a function to the start of your prompt)* What does this function do?
* What is an intuitive explanation of induction heads?
* What is the difference between top-k and top-p sampling?

## Hints

There will be occasional hints throughout the document, for when you're having trouble with a certain task but you don't want to read the solutions. Click on the expander to reveal the solution in these cases. Below is an example of what they'll look like:

<details>
<summary>Help - I'm stuck on a particular problem.</summary>

Here is the answer!
</details>

Always try to solve the problem without using hints first, if you can.

## Test functions

Most of the blocks of code will also come with test functions. These are imported from python files with names such as `exercises/part1_raytracing_tests.py`. You should make sure these files are in your working directory while you're writing solutions. One way to do this is to clone the [main GitHub repo](https://github.com/callummcdougall/arena-v1) into your working directory, and run it there. When we decide exactly how to give participants access to GPUs, we might use a different workflow, but this should suffice for now. Make sure that you're getting the most updated version of utils at the start of every day (because changes might have been made), and keep an eye out in the `#errata` channel for mistakes which might require you to change parts of the test functions.

## Tips

* To get the most out of these exercises, make sure you understand why all of the assertions should be true, and feel free to add more assertions.
* If you're having trouble writing a batched computation, try doing the unbatched version first.
* If you find these exercises challenging, it would be beneficial to go through them a second time so they feel more natural.

## Feedback

If you have any feedback on this course (e.g. bugs, confusing explanations, parts that you feel could be structured better), please let me know using [this Google Form](https://forms.gle/2ZhdHa87wWsrATjh9).
""", unsafe_allow_html=True)
    # end

# ## Support

# If you ever need help, you can send a message on the ARENA Slack channel `#technical-questions`. You can also reach out to a TA (e.g. Callum) if you'd like a quick videocall to talk through a concept or a problem that you've been having, although there might not always be someone available.

# You can also read the solutions by downloading them from the [GitHub](https://github.com/callummcdougall/arena-v1). However, ***this should be a last resort***. Really try and complete the exercises as a pair before resorting to the solutions. Even if this involves asking a TA for help, this is preferable to reading the solutions. If you do have to read the solutions, then make sure you understand why they work rather than just copying and pasting. 

# At the end of each day, it can be beneficial to look at the solutions. However, these don't always represent the optimal way of completing the exercises; they are just how the author chose to solve them. If you think you have a better solution, we'd be really grateful if you could send it in, so that it can be used to improve the set of exercises for future ARENA iterations.

# Happy coding!

# if is_local or check_password():

section_home()
