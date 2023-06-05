
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

st.markdown(r"""
# [1.∞] Reference Page

This page contains links to a bunch of things (blog posts, diagrams, tables) which are useful to have at hand when doing this chapter.

*If you have any other suggestions for this page, please add them on Slack!*

## Logistics

* [Notion page](https://www.notion.so/ARENA-2-0-Virtual-Resources-7934b3cbcfbf4f249acac8842f887a99?pvs=4) for people studying virtually
* [ARENA Slack group invite link](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ)
* [Open Source Mechanistic Interpretability Slack group invite link](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-1qosyh8g3-9bF3gamhLNJiqCL_QqLFrA)

## General

* [Google Drive folder](https://drive.google.com/drive/folders/1N5BbZVh5_pZ3sH1lv4krp-2_wJrB-Ahg) containing Colab versions of all these exercises
* Neel Nanda's [Mech Interp Dynalist notes](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J)
* Neel Nanda's [Concrete Steps to Get Started in Transformer Mechanistic Interpretability](https://www.neelnanda.io/mechanistic-interpretability/getting-started)
* Neel Nanda's [An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers](https://www.neelnanda.io/mechanistic-interpretability/favourite-papers)

## How Transformers Work

* Neel Nanda's Implementing a Transformer Walkthrough: [Part 1/2](https://www.youtube.com/watch?v=bOYE6E8JrtU), [Part 2/2](https://www.youtube.com/watch?v=dsjUDacBw8o)
* Callum McDougall's [An Analogy for Understanding Transformers](https://www.lesswrong.com/posts/euam65XjigaCJQkcN/an-analogy-for-understanding-transformers)
* Callum McDougall's [full transformer excalidraw diagram](https://link.excalidraw.com/l/9KwMnW35Xt8/4kxUsfrPeuS)
* Jay Alammar's [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## TransformerLens

* TransformerLens documentation page:
    * [Homepage](https://neelnanda-io.github.io/TransformerLens/index.html)
    * [Table of model properties](https://neelnanda-io.github.io/TransformerLens/model_properties_table.html)

## Diagrams

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/transformer-full-red.png" width="1200">

Link to excalidraw [here](https://link.excalidraw.com/l/9KwMnW35Xt8/6PEWgOPSxXH).
""", unsafe_allow_html=True)