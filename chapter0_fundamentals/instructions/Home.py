import os, sys
import streamlit as st
import platform
from pathlib import Path
is_local = (platform.processor() != "")

# Get to the right directory: the streamlit one (not pages)
# Get to chapter0_fundamentals directory (or whatever the chapter dir is)

# Navigate to the root directory, i.e. ARENA_2 for me, or the working directory for people locally
while "chapter" in os.getcwd():
    os.chdir("..")
# Now with this reference point, we can add things to sys.path
root_suffix = r"/chapter0_fundamentals/instructions"
root_dir = os.getcwd() + root_suffix
root_path = Path(root_dir)
if root_dir not in sys.path: sys.path.append(root_dir)

if os.getcwd().endswith("chapter0_fundamentals") and "./instructions" not in sys.path:
    sys.path.append("./instructions")
if os.getcwd().endswith("pages") and "../" not in sys.path:
    sys.path.append("../")

import st_dependencies
st_dependencies.styling()

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
<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/prereqs.png" width="600">

# Chapter 0: Fundamentals

The material on this page covers the first five days of the curriculum. It can be seen as a grounding in all the fundamentals necessary to complete the more advanced sections of this course (such as RL, transformers, mechanistic interpretability, and generative models).

Some highlights from this chapter include:
* Building your own 1D and 2D convolution functions
* Building and loading weights into a Residual Neural Network, and finetuning it on a classification task
* Working with [weights and biases](https://wandb.ai/site) to optimise hyperparameters
* Implementing your own backpropagation mechanism

---

## About this page

This page was made using an app called Streamlit. It's hosted from the prerequisite materials [GitHub repo](https://github.com/callummcdougall/Prerequisite-materials). It provides a very simple way to display markdown, as well as more advanced features like interactive plots and animations. This is how the instructions for each day will be presented.

On the left, you can see a sidebar (or if it's collapsed, you will be able to see if you click on the small arrow in the top-left to expand it). This sidebar should show a page called `Home` (which is the page you're currently reading), as well as one for each of the different parts of today's exercises.

If you want to change to dark mode, you can do this by clicking the three horizontal lines in the top-right, then navigating to Settings → Theme.

## How you should use this material

### Option 1: VSCode

This is the option we strongly recommend for all participants of the in-person ARENA program.

<details>
<summary>Click this dropdown for setup instructions.</summary>

First, clone the [GitHub repo](https://github.com/callummcdougall/ARENA_2.0) into your local directory. The repo has the following structure (omitting the unimportant parts):

```
.
├── chapter0_fundamentals
│   ├── exercises
│   │   ├── part1_ray_tracing
│   │   │   ├── solutions.py
│   │   │   ├── tests.py
│   │   │   └── answers.py*
│   │   ├── part2_cnns
│   │   ⋮    ⋮
│   └── instructions
│       └── Home.py
├── chapter1_transformers
├── chapter2_rl
├── chapter3_training_at_scale
└── requirements.txt
```

There is a directory for each chapter of the course (e.g. `chapter0_fundamentals`). Each of these directories has an `instructions` folder (which contain the files used to generate the pages you're reading right now) `exercises` folder (where you'll be doing the actual exercises). The latter will contain a subfolder for each day of exercises, and that folder will contain files such as `solutions.py` and `tests.py` (as well as other data sometimes, which gets used as part of the exercises). You'll be completing the exercises in an `answers.py` file in this subfolder (which you'll need to create).

Once you've cloned the repo and navigated into it (at the root directory), you should do the following:

* Make & activate a virtual environment.
    * We strongly recommend using `conda` for this. You can install `conda` [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), and find basic instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
    * The command for creating a new env is `conda create -–name arena python=3.10`.
* Install requirements.
    * First, install PyTorch.
        * If you're on Windows, the command is `conda install pytorch=1.13.1 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia`.
        * If you're on MAC, the command is `conda install pytorch=1.13.1 torchvision`.
    * Then install the rest of the requirements by navigating running `pip install -r requirements.txt`.
* To run a set of exercises, navigate to the appropriate `instructions` directory (e.g. `chapter0_fundamentals/instructions`) and run `streamlit run Home.py` in your terminal.
    * This should open up a local copy of the page you're reading right now, and you're good to go!

</details>

### Option 2: Colab

This option is recommended if either of the following is true:

* You have limited access to GPU support
* You want to avoid the hassle of setting up your own environment

You can see all files in [this Google Drive folder](https://drive.google.com/drive/folders/1uq1pV6-9aQ5fO5ZhTfcL_GL2GXU8ANcy?usp=share_link). Also, you can get individual links in the dropdown below.

<details>
<summary>Click this dropdown for links to each of the colab exercises.</summary>

<div style='text-align: center'>
<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/raytracing.png" width="160" style="margin-bottom:3px;margin-top:15px">

Raytracing<br>[**exercises**](https://colab.research.google.com/drive/1T3yXhK9CgK49HfN_x2WwD2CUv_bcPjA5) | [**solutions**](https://colab.research.google.com/drive/17qAsbvGChdA1zCjJ3QU8bv-4-rXpdppZ)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/cnn.png" width="160" style="margin-bottom:3px;margin-top:15px">

as_strided, convolutions and CNNs<br>[**exercises**](https://colab.research.google.com/drive/1HFsebBH7SJ7wqVCmTAt097FkDbCC6AQf) | [**solutions**](https://colab.research.google.com/drive/1ttKR6WOCKDBXmbwvKd-gpI2AUXp1OzBa)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/resnet.png" width="160" style="margin-bottom:3px;margin-top:15px">

ResNets & Model Training: Links to Colab<br>[**exercises**](https://colab.research.google.com/drive/1GRAtbOHmy6MHWSoz9AdAam3CUjczB1mo) | [**solutions**](https://colab.research.google.com/drive/1Th-j4FcYWgVTNEGzWjFlSQdPwm4-GbiD)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/stats.png" width="160" style="margin-bottom:3px;margin-top:15px">

Optimization & Hyperparameters<br>[**exercises**](https://colab.research.google.com/drive/12PiedkLJH9GkIu5F_QC7iFf8gJb3k8mp) | [**solutions**](https://colab.research.google.com/drive/1yKhqfOJUTcdu0zhuRONjXLCmoLx7frxP)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/backprop.png" width="160" style="margin-bottom:3px;margin-top:15px">

Build Your Own Backprop Framework<br>[**exercises**](https://colab.research.google.com/drive/1_aeNgUU8H7psOH8jttByO_8lv9Wp7O0o) | [**solutions**](https://colab.research.google.com/drive/1Fs7nvNbeDirDi2KEtN5rxWAzLba_tvbu)
</div>
</details>

For each of these sections, you can make a copy of the **exercises** notebooks in your own drive, and fill in the code cells whenever indicated. The solutions will be available in dropdowns next to each of the code cells (or you can look at the **solutions** notebooks, which have all code pre-run and output displayed).

### Chatbot assistant

We've created an experimental chatbot assistant to help you answer questions about the material. It performs a low-dimensional embedding of any questions that it is asked, then assembles context from the curriculum by choosing blocks of content with an embedding that has high cosine similarity of the question's embedding. This was inspired by [AlignmentSearch](https://www.lesswrong.com/posts/bGn9ZjeuJCg7HkKBj/introducing-alignmentsearch-an-ai-alignment-informed), and has similar benefits and drawbacks relative to the alternative of using GPT directly.

You'll be able to access the assistant just fine using the public link, but if you want to use the chatbot while running the page locally, you'll need to do the following:

* Go to the [OpenAI API](https://openai.com/blog/openai-api) and sign up for an account.
* Create a secret key from [this page](https://platform.openai.com/account/api-keys). Copy this key.
* Create a file `.streamlit/secrets.toml` in the appropriate `instructions` directory, and have the first line be `openai_api_key = "<your key>"`.
* Refresh the Streamlit page, and you should now be able to use the chatbot.

You can see example questions to ask the chatbot if you navigate to the chatbot page.

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
