import os
import streamlit as st
import platform
is_local = (platform.processor() != "")
from pathlib import Path
import sys

import st_dependencies
st_dependencies.styling()

st.sidebar.markdown(r"""
## Links

<a href="https://arena-ch0-fundamentals.streamlit.app/" style="color:black;text-decoration:none">Chapter 0: Fundamentals</a><br>
<a href="https://arena-ch1-transformers.streamlit.app/" style="color:black;text-decoration:none">Chapter 1: Transformers & Mech Interp</a><br>
<a href="https://arena-ch2-rl.streamlit.app/" style="color:black;text-decoration:none">Chapter 2: Reinforcement Learning</a><br>
<a href="https://arena-ch3-training-at-scale.streamlit.app/" style="color:black;text-decoration:none">Chapter 3: Training at Scale</a><br>

<a href="https://mango-ambulance-93a.notion.site/ARENA-2-0-Virtual-Resources-7934b3cbcfbf4f249acac8842f887a99?pvs=4" style="color:black;text-decoration:none">Notion page: Virtual Resources</a><br>
<a href="https://github.com/callummcdougall/ARENA_2.0" style="color:black;text-decoration:none">GitHub page</a><br>
<a href="https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ" style="color:black;text-decoration:none">Slack invite link</a><br>


""", unsafe_allow_html=True)

# start
st.markdown(r"""
# ARENA roadmap

Welcome to our roadmap! 👋 This app shows some projects we're working on or have planned for the future.
            
See our main website to understand more about our broader vision: [arena.education](https://www.arena.education/).
""")

st.info(
r"""
If you have any other suggestions for material or features that you think ARENA could benefit from, please reach out via email `cal.s.mcdougall@gmail.com` or on the [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ).

***Note** - some of the dates below have been moved backwards recently, because I've become quite busy with other fieldbuilding projects and working on a research paper. I've unfortunately had to remove most of the fixed date estimates. If there is anything here you're particularly interested in then please reach out, so we know what to prioritise.*
""")

st.markdown(
r"""

## Timeline

This section details all planned changes, i.e. things we're confident will be implemented and which we do have a good idea of the timeline for. They're roughly ordered by priority / release date (i.e. the first ones in this list will be coming first).

<details>
<summary>🗺️ <b>Smoothing the path from PPO to RLHF</b><br>Planned release: <b>July-August 2023</b></summary>

---

We'd like to make the jump between PPO and RLHF less discontinuous, and we're working on material which will bridge that gap. The diagram on the left shows the current state of the RL curriculum, and the diagram on the right shows the planned final state.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/diagram-of-curriculum-2.png" width="560">

Some more information about each new section:

#### **PPO (Atari / MuJoCo)**

This will extend the basic PPO material.

The most important conceptual ideas Atari introduces relate to the design of the actor / critic architectures (because the observation space is now an image of pixels rather than four values). We'll use CNNs for our agent, and we'll have shared architecture for the actor and critic (an important idea which reappears when we look at transformers later on).

MuJoCo is more challenging than Atari, because it introduces a **continuous action space**. This will require a redesign of several components of the PPO algorithm.

#### **RLHF (Atari / MuJoCo)**

Practice RLHF in a simple environment, before we get to language models. You'll be able to try RLHF on Atari games, or attempt to replicate results like OpenAI's famous "backflipping noodle" experiment.

#### **PPO (transformers)**

Now, we move on to transformer architecture. Before we introduce the human feedback element, we'll use the PPO algorithm to train the transformer to do a simple task (generating more periods, i.e. shorter sentences). This will involve some concepts that came up in the PPO Atari material (e.g. the policy and value networks as having shared architecture), as well as some new ideas (e.g. keeping the transformer on-distribution using a KL divergence penalty term).

#### **RLHF (transformers)**

This section combines the conceptual ideas from the last 2 sections, by having you implement RLHF on transformers. You'll be doing all the coding from the ground up, i.e. with no assistance from libraries like `trlx`.

#### **RLHF (trlx), re-release**

Lastly, we plan to restructure the RLHF & `trlx` chapter, to make it easier to follow & flow better with the rest of the chapter. Learning how to use `trlx` is still valuable, because once you have a conceptual understanding of the RLHF process, there's no need to go through all the low-level implementational details.

</details>

<details>
<summary>🔍 <b>Mechanistic Interpretability - monthly challenges</b><br>Planned release: <b>July 2023</b></summary>

---

We'll start to release a monthly sequence on mechanistic interpretability, in the same vein as Steven Caspar's sequence (they will also be published on LessWrong). Each challenge will involve interpreting a transformer trained on a particular LeetCode problem. The balanced brackets transformer exercises are a good example (although I expect the average task in this sequence to be slightly easier).

**The first one is currently live.**

</details>

<details>
<summary>📚 <b>Mechanistic Interpretability - more exercises</b><br>Planned release: <b>July 2023 - ongoing</b></summary>

We have several more exercise sets planned for the mechanistic interpretability chapter, including:

* ACDC (this will get its own set of exercises)
* Attribution patching (this will be integrated into the IOI exercises)
* Attention head superposition (this will be integrated into the superposition exercises)

</details>

<details>
<summary>⚡ <b>Removal of PyTorch Lightning</b><br>Planned release: <b>July 2023</b></summary>

---

The way we'd been using PyTorch Lightning in the curriculum material (especially the first week) just added confusion, and we no longer think it's worth learning at such an early point in the curriculum. We'll keep the structure of training loops relatively similar to the PyTorch Lightning style (so the exercises won't be totally redesigned), but we'll remove explicit reference to the Lightning library itself.
            
We'll be keeping all the Lightning content on the page, in optional dropdowns, should people still wish to read it. We may also work with Lightning during the redesigned Training at Scale chapter (when it's a more appropriate time to introduce it, and when it can offer more than a standardised way to modularize training loops).

</details>

<details>
<summary>💽 <b>Restructuring the Training at Scale chapter</b><br>Planned release: <b>August 2023</b></summary>

---

After ARENA 2.0 finished, we received feedback that the Training at Scale chapter was somewhat hard to follow & benefit from. We'd like to redesign it so that it feels less like being thrown in at the deep end, and more like you're actually getting useful insights from the material.

</details>


## Speculative

This section lists a few changes we're thinking about making, but which we aren't certain about yet.

* A chapter on diffusion models: GANs, VAEs, diffusion models, stable diffusion.
* Evals (possibly as a subsection of a chapter, or as its own micro section).
* Video walkthroughs of solutions to exercises (this depends on demand).

""", unsafe_allow_html=True)
