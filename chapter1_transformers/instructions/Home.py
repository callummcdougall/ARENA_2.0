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
        <li><a class="contents-el" href="#option-1-vscode">Option 1: VSCode</a></li>
        <li><a class="contents-el" href="#option-2-colab">Option 2: Colab</a></li>
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

There is a directory for each chapter of the course (e.g. `chapter0_fundamentals`). Each of these directories has an `instructions` folder (which contain the files used to generate the pages you're reading right now) `exercises` folder (where you'll be doing the actual exercises). The latter will contain a subfolder for each day of exercises, and that folder will contain files such as `solutions.py` and `tests.py` (as well as other data sometimes, which gets used as part of the exercises). To do the exercises, you'll be creating a file `answers.py` (or `.ipynb` if you prefer) in the same folder as the solutions and tests files. You'll then go through the corresponding streamlit page, copying over & running code (filling in the blanks as you go).

You'll be completing the exercises in an `answers.py` (or file in this subfolder (which you'll need to create).

Once you've cloned the repo and navigated into it (at the root directory), there are two possible ways you can proceed (use the tabs to see both options).

### Option 1A: Conda

* Make & activate a virtual environment.
    * We strongly recommend using `conda` for this. You can install `conda` [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), and find basic instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
    * The command for creating a new env is `conda create --name arena python=3.8`.
* Install requirements.
    * First, install PyTorch.
        * If you're on Windows, the command is `conda install pytorch=1.13.1 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia`.
        * If you're on MAC/Linux, the command is `conda install pytorch=1.13.1 torchvision`.
    * Then install the rest of the requirements by navigating running `pip install -r requirements.txt`.

### Option 1B: Docker

If you are using linux or WSL, this is the best way to ensure that your environment contains all the relevant dependencies.

1. Install docker using [this guide](https://docs.docker.com/engine/install/), depending on the platform you are using.
2. Install the [nvidia container runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
3. Follow the instructions [here](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry) (i.e. under the heading "Authenticating to the container registry") to authenticate with the GitHub container registry.
4. Launch the container with `docker run --rm -d --runtime=nvidia --gpus '"device='"0"'"' -p 2222:22 ghcr.io/pranavgade20/arena:latest`.
        
</details>

Note - if you choose this option, then you may require more compute than your laptop can provide. If you're following this material virtually, you may want to consider a cloud provider such as Lambda Labs or Paperspace.

<details>
<summary>Setup instructions for Lambda Labs</summary>

Here is a set of instructions for spinning up a Lambda Labs instance, and accessing it via VSCode. We may shortly add more sets of instructions for other cloud providers (since Lambda Labs has finite capacity, and it isn't always possible to find a free instance).



## Instructions for signing up

Sign up for an account [here](https://lambdalabs.com/service/gpu-cloud).

Add an **SSH key**. Give it a name like `<Firstname><Lastname>` (we will refer to this as `<keyname>` from now on).

When you create it, it will automatically be downloaded. The file may have a `.pem` extension - this is a common container format for keys or certificates.

## VSCode remote-ssh extension

The [**remote ssh extension**](https://code.visualstudio.com/docs/remote/ssh) is very useful for abstracting away some of the messy command-line based details of SSH. You should install this extension now.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/architecture-ssh.png" width="600">


At this point, the instructions differ between Windows and Linux/MacOS.

### Windows

Having installed the SSH extension, Windows may have automatically created a .ssh file for you, and it will be placed in `C:\Users\<user>` by default. If it hasn't done this, then you should create one yourself (you can do this from the Windows command prompt via `md C:\Users\<user>\.ssh`).

Move your downloaded SSH key into this folder. Then, set permissions on the SSH key:
		
* Right click on file, press “Properties”, then go to the “Security” tab.
* Click “Advanced”, then “Disable inheritance” in the window that pops up.
    
<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/instruction1.png" width="500">

* Choose the first option “Convert inherited permissions…”

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/instruction2.png" width="500">

* Go back to the “Security” tab, click "Edit" to change permissions, and remove every user except the owner.
    * You can check who the owner is by going back to "Security -> Advanced" and looking for the "Owner" field at the top of the window).

#### Linux / MacOS

* Make your `.ssh` directory using the commands `mkdir -p ~/.ssh` then `chmod 700 ~/.ssh`.
* Set permissions on the key: `chmod 600 ~/.ssh/<keyname>`

## Launch your instance

Go back to the Lambda Labs page, go to "instances", and click "Launch instance".

You'll see several options, some of them might be greyed out if unavailable. Pick a cheap one (we're only interested in testing this at the moment, and at any rate even a relatively cheap one will probably be more powerful than the one you're currently using in your laptop). 

Enter your SSH key name. Choose a region (your choice here doesn't really matter for our purposes).

Once you finish this process, you should see your GPU instance is running:

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/gpu_instance.png" width="700">

You should also see an SSH LOGIN field, which will look something like: `ssh ubuntu@<ip-address>`.

Once you've got to this stage, you can proceed with either option α or β below. After choosing one of them, you can skip to the section "Connect to your instance".

## Option α: Docker

1. Go to https://ssheasy.com/ (or a similar website, or a ssh client like openssh or putty) and enter the IP of your instance in the `Host to connect` field, `ubuntu` in the `User` field, and select the private key for the key you uploaded to lambda labs. We recommend you upload [this file](https://github.com/callummcdougall/ARENA_2.0/blob/main/infrastructure/arena_ssh.pub) to lambda labs, and upload [this file](https://github.com/callummcdougall/ARENA_2.0/blob/main/infrastructure/arena_ssh) to ssheasy.
2. Clone the repository with `git clone --depth 1 -b main https://github.com/callummcdougall/ARENA_2.0/`
3. Go to the infrastructure folder with `cd ARENA_2.0/infrastructure`
4. Run the first part of the setup script: `bash docker-provision-on-instance-part-1.sh`. This will download docker and set up drivers for your GPUs
5. Reboot your instance with `sudo reboot`
6. Reconnect with step 1; navigate to the infrastructure folder with `cd ARENA_2.0/infrastructure`; and run the second part of the setup script: `bash docker-provision-on-instance-part-1.sh`
7. Set up your config file:

```c
Host arena2
    HostName <ip-address>
    IdentityFile C:\Users\<user>\.ssh\<keyname>
    User root
    Port 2222
```

**Make sure the IdentityFile points to your SSH key, and that you've swapped `<ip-address>`, `<user>` and `<keyname>` for your own values.**

## Option β: Pip

Setting up a **config file** remove the need to use long command line arguments, e.g. `ssh -i ~/.ssh/<keyname> ubuntu@instance-ip-address`.

Click on the <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/vscode-ssh.png" width="35"> button in the bottom left, choose "Open SSH Configuration File...", then click <code>C:\\Users\\<user>\\.ssh\\config</code>.

An empty config file will open. You should copy in the following instructions:

```c
Host <ip-address>
    IdentityFile C:\Users\<user>\.ssh\<keyname>
    User <user>
```

where the IP address and user come from the **SSH LOGIN** field in the table, and the identity file is the path of your SSH key. For instance, the file I would use (corresponding to the table posted above) looks like:

```c
Host <ip-address>
    IdentityFile C:\Users\<user>\.ssh\<keyname>
    User <user>
```

---

### Connect to your instance

Click the green button <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/vscode-ssh.png" width="35"> again, and choose "Connect to Host...". Your IP address should appear as one of the hosts. Choose this option.

A new VSCode window will open up. If you're asked if you want to install the recommended extensions for Python, click yes. If you're asked to choose an OS (Windows, Mac or Linux), choose Linux.

Click on the file explorer icon in the top-left, and open the directory `ubuntu` (or whichever directory you want to use as your working directory in this machine). 

And there you go - you're all set! 

To check your GPU is working, you can open a Python or Notebook file and run `!nvidia-smi`. You should see GPU information which matches the machine you chose from the Lambda Labs website, and is different from the result you get from running this command on your local machine. 

Another way to check out your GPU For instance is to run the PyTorch code `torch.cuda.get_device_name()`. For example, this is what I see after SSHing in:

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/gpu_type.png" width="500">
<br>
<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/gpu_type_2.png" width="450">

You can also use `torch.cuda.get_device_properties` (which takes your device as an argument).

Once you've verified this is working, you can start running code on GPUs. The easiest way to do this is just to drag and drop your files into the file explorer window on the left hand side.

You'll also need to choose a Python interpreter. Choose the conda or miniconda one if it's available, if not then choose the top-listed version.

Lastly, ***if you chose option β above rather than α***, then you'll probably need to `%pip install` some libraries.

For convenience, here is a list of libraries I've had to pip-install when spinning up a Lambda Labs VM, during the indirect object identification exercises (which forms a superset of all the libraries you'll need for the first two chapters). If you're working on material from the first chapter, you can leave out the circuitsvis and transformer_lens libraries (as well as protobuf, which fixes an error I sometimes get from the transformer_lens library). Lastly, if my circuitsvis fork (which allows you to label the attention heads you plot) isn't working, you can replace it with `pip install circuitsvis`.

```python
%pip install plotly
%pip install wandb
%pip install pytorch-lightning
%pip install jaxtyping
%pip install einops
%pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python
%pip install transformer_lens
%pip install protobuf==3.20.*
```

</details>

### Option 2: Colab

This option is recommended if either of the following is true:

* You have limited access to GPU support
* You want to avoid the hassle of setting up your own environment

You can see all files in [this Google Drive folder](https://drive.google.com/drive/folders/1N5BbZVh5_pZ3sH1lv4krp-2_wJrB-Ahg?usp=share_link). Also, you can get individual links in the dropdown below.

<details>
<summary>Click this dropdown for links to each of the colab exercises.</summary>

<div style='text-align: center'>
<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/transformer-building.png" width="160" style="margin-bottom:3px;margin-top:15px">

Transformer from scratch<br>[**exercises**](https://drive.google.com/file/d/1Zl3zSdli_epSfaoQ_HeBCuE6dkGWTowd/view?usp=sharing) | [**solutions**](https://drive.google.com/file/d/1neFAal6woQ7p-u0LpU7ZqvHeEaJ6j2DG/view?usp=share_link)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/circuit.png" width="160" style="margin-bottom:3px;margin-top:15px">

Intro to Mechanistic Interpretability<br>[**exercises**](https://drive.google.com/file/d/1w9zCWpE7xd1sDuMT_rsjARfFozeWiKF4/view?usp=share_link) | [**solutions**](https://drive.google.com/file/d/10tzGmOCQb3LoDB69vPFw71DV2d395hJl/view?usp=share_link)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/leaves.png" width="160" style="margin-bottom:3px;margin-top:15px">

Indirct Object Identification<br>[**exercises**](https://drive.google.com/file/d/1M4F9SU_vHUUCQkhmtWnmY2eomOJu5B5s/view?usp=share_link) | [**solutions**](https://drive.google.com/file/d/1AA0wj2sHoZwtmy82WXORcZzk9urL1lVA/view?usp=share_link)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/gears2.png" width="160" style="margin-bottom:3px;margin-top:15px">

Interpretability on an Algorithmic Model<br>[**exercises**](https://drive.google.com/file/d/1BYarO508z7stRFXZ3T92rI6OtMqP3w7E/view?usp=share_link) | [**solutions**](https://drive.google.com/file/d/1yILyi5dD3wc4o3vHc1MfbpAlWljyXU-U/view?usp=share_link)
</div>

</details>

For each of these sections, you can make a copy of the **exercises** notebooks in your own drive, and fill in the code cells whenever indicated. The solutions will be available in dropdowns next to each of the code cells (or you can look at the **solutions** notebooks, which have all code pre-run and output displayed).

### Chatbot assistant

We've created an experimental chatbot assistant to help you answer questions about the material. It performs a low-dimensional embedding of any questions that it is asked, then assembles context from the curriculum by choosing blocks of content with an embedding that has high cosine similarity of the question's embedding. This was inspired by [AlignmentSearch](https://www.lesswrong.com/posts/bGn9ZjeuJCg7HkKBj/introducing-alignmentsearch-an-ai-alignment-informed), and has similar benefits and drawbacks relative to the alternative of using GPT directly.

You can see example questions to ask the chatbot if you navigate to the chatbot page.

This feature is very experimental, so please [let me know](mailto:cal.s.mcdougall@gmail.com) if you have any feedback!

## Hints

There will be occasional hints throughout the document, for when you're having trouble with a certain task but you don't want to read the solutions. Click on the expander to reveal the solution in these cases. Below is an example of what they'll look like:

<details>
<summary>Help - I'm stuck on a particular problem.</summary>

Here is the answer!
</details>

Always try to solve the problem without using hints first, if you can.

## Test functions

Most of the blocks of code will also come with test functions. These are imported from python files with names such as `exercises/part1_raytracing_tests.py`. You should make sure these files are in your working directory while you're writing solutions. One way to do this is to clone the [main GitHub repo](https://github.com/callummcdougall/ARENA_2.0) into your working directory, and run it there. When we decide exactly how to give participants access to GPUs, we might use a different workflow, but this should suffice for now. Make sure that you're getting the most updated version of utils at the start of every day (because changes might have been made), and keep an eye out in the `#errata` channel for mistakes which might require you to change parts of the test functions.

## Tips

* To get the most out of these exercises, make sure you understand why all of the assertions should be true, and feel free to add more assertions.
* If you're having trouble writing a batched computation, try doing the unbatched version first.
* If you find these exercises challenging, it would be beneficial to go through them a second time so they feel more natural.

## Feedback

If you have any feedback on this course (e.g. bugs, confusing explanations, parts that you feel could be structured better), please let us know using [this Google Form](https://forms.gle/2ZhdHa87wWsrATjh9).
""", unsafe_allow_html=True)
    # end

# ## Support

# If you ever need help, you can send a message on the ARENA Slack channel `#technical-questions`. You can also reach out to a TA (e.g. Callum) if you'd like a quick videocall to talk through a concept or a problem that you've been having, although there might not always be someone available.

# You can also read the solutions by downloading them from the [GitHub](https://github.com/callummcdougall/arena-v1). However, ***this should be a last resort***. Really try and complete the exercises as a pair before resorting to the solutions. Even if this involves asking a TA for help, this is preferable to reading the solutions. If you do have to read the solutions, then make sure you understand why they work rather than just copying and pasting. 

# At the end of each day, it can be beneficial to look at the solutions. However, these don't always represent the optimal way of completing the exercises; they are just how the author chose to solve them. If you think you have a better solution, we'd be really grateful if you could send it in, so that it can be used to improve the set of exercises for future ARENA iterations.

# Happy coding!

# if is_local or check_password():

section_home()
