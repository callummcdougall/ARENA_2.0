
import os, sys
from pathlib import Path
chapter = r"chapter3_training_at_scale"
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
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/distributed.png" width="350">


Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.



# [3.2] - Distributed training

## Introduction

Welcome to the exciting world of distributed GPU training using PyTorch!

Training deep learning models can be a computationally intensive task, often requiring substantial time and computational resources. However, distributed GPU training provides a solution to this challenge. By leveraging multiple GPUs, either on a single machine or across a network of machines, the training process can be significantly accelerated.

PyTorch, an open-source machine learning library, is one of the leading tools used for distributed training. Its torch.distributed package offers robust support for distributed computing, enabling users to efficiently harness the power of multiple GPUs.

In distributed GPU training, a model's computations are spread across multiple GPUs. Data parallelism is a common technique used in this process, which involves dividing the input data into chunks and processing each chunk on a different GPU in parallel. The computed gradients from each GPU are then synchronized to update the model parameters.

Distributed GPU training is a valuable skill in the modern era of machine learning and deep learning. Mastering it with PyTorch will unlock new levels of computational power and efficiency for your models, enabling you to tackle larger and more complex problems.

## Content & Learning Objectives

### 1️⃣ Basics of distributed programming

> ##### Learning objectives
> 
> * Learn the structure of the PyTorch distributed class
> * Understand what a process, thread, and rank is
> * Explore what might cause race conditions

### 2️⃣ Data parallelism

> ##### Learning objectives
> 
> * Learn about common collective operations
> * Implement broadcast, reduce, and all-reduce
> * Consider the effects of different connection topologies

### 3️⃣ Pipeline parallelism, DDP

> ##### Learning objectives
> 
> * Learn about pipeline parallelism for running across multiple machines
> * Use PyTorch DDP to implement pipeline parallelism

### 4️⃣ Bonus

Includes some suggested bonus exercises and further reading.

""", unsafe_allow_html=True)


section_0()