
import os, sys
from pathlib import Path
chapter = r"chapter2_rl"
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
    <li class='margtop'><a class='contents-el' href='#introduction-ppo-vs-dqn'>Introduction - PPO vs DQN</a></li>
    <li class='margtop'><a class='contents-el' href='#conceptual-overview-of-ppo'>Conceptual overview of PPO</a></li>
    <li class='margtop'><a class='contents-el' href='#implementational-overview-of-ppo'>Implementational overview of PPO</a></li>
    <li class='margtop'><a class='contents-el' href='#notes-on-today's-workflow'>Notes on today's workflow</a></li>
    <li class='margtop'><a class='contents-el' href='#readings'>Readings</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#optional-reading'>Optional Reading</a></li>
        <li><a class='contents-el' href='#references-not-required-reading'>References (not required reading)</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
    <li class='margtop'><a class='contents-el' href='#ppo-arguments'>PPO Arguments</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""
   
<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/football.jpeg" width="350">

Colab: [**exercises**](https://colab.research.google.com/drive/1USZJy9HCpq8rsAqfD6yYfpAfZvO5YDha) | [**solutions**](https://colab.research.google.com/drive/1f_RBuosHddwQrydZ7-iAnBs6lhkMMMSG)

Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.


# [2.3] - PPO


## Introduction

Proximal Policy Optimization (PPO) is a cutting-edge reinforcement learning algorithm that has gained significant attention in recent years. As an improvement over traditional policy optimization methods, PPO addresses key challenges such as sample efficiency, stability, and robustness in training deep neural networks for reinforcement learning tasks. With its ability to strike a balance between exploration and exploitation, PPO has demonstrated remarkable performance across a wide range of complex environments, including robotics, game playing, and autonomous control systems.

In this section, you'll build your own agent to perform PPO on the CartPole environment. By the end, you should be able to train your agent to near perfect performance in about 30 seconds. You'll also be able to try out other things like **reward shaping**, to make it easier for your agent to learn to balance, or to do fun tricks!

## Content & Learning Objectives


#### 1️⃣ Setting up our agent

> ##### Learning objectives
> 
> * Understand the difference between the actor & critic networks, and what their roles are
> * Learn about & implement generalised advantage estimation
> * Build a replay buffer to store & sample experiences
> * Design an agent class to step through the environment & record experiences

#### 2️⃣ Learning phase

> ##### Learning objectives
>
> * Implement the total objective function (sum of three separate terms)
> * Understand the importance of each of these terms for the overall algorithm
> * Write a function to return an optimizer and learning rate scheduler for your model

#### 3️⃣ PPO: Learning

> ##### Learning objectives
> 
> * Build a full training loop for the PPO algorithm
> * Train our agent, and visualise its performance with Weights & Biases media logger
> * Use reward shaping to improve your agent's training (and make it do tricks!)


## Introduction - PPO vs DQN

Today, we'll be working on PPO (Proximal Policy Optimization). It has some similarities to DQN, but is based on a fundamentally different approach:

|                                        | DQN                                                                                                                                                                                                                                     | PPO                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| What do we learn?                      | We learn the Q-function $Q(s, a)$.                                                                                                                                                                                                      | We learn the policy function $\pi(a \mid s)$.                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Where do our actions come from?                      | Argmaxing $Q(s, a)$ over actions $a$ gives us a deterministic policy. We combine this with an epsilon-greedy algorithm when sampling actions, to enable exploration.                                                                                                                                                                                                       | We directly learn our stochastic policy $\pi$, and we can sample actions from it.                                                                                                                                                                                                                                                                                                                                                                                                                     |
| What networks do we have?              | Our network `q_network` takes $s$ as inputs, and outputs the Q-values for each possible action $a$.  We also had a `target_network`, although this was just a lagged version of `q_network` rather than one that actually gets trained. | We have two networks: `actor` which learns the policy function, and `critic` which learns the value function $V(s)$. These two work in tandem: the `actor` requires the `critic`'s output in order to estimate the policy gradient and perform gradient ascent, and the `critic` tries to learn the value function of the `actor`'s current policy.                                                                                                               |
| Where do our gradients come from?      | We do gradient descent on the squared **TD residual**, i.e. the residual of the Bellman equation (which is only satisfied if we've found the true Q-function).                                                               | For our `actor`, we do gradient ascent on an estimate of the time-discounted future reward stream (i.e. we're directly moving up the **policy gradient**; changing our policy in a way which will lead to higher expected reward). Our `critic` trains by minimising the TD residual.                                                                                                                                                                                                                                                |
| Techniques to improve stability?       | We use a "lagged copy" of our network to sample actions from; in this way we don't update too fast after only having seen a small number of possible states. In the DQN code, this was `q_network` and `target_network`.                | We use a "lagged copy" of our network to sample actions from. In the mathematical notation, this is $\theta$ and $\theta_{\text{old}}$. In the code, we won't actually need to make a different network for this.  We clip the objective function to make sure large policy changes aren't incentivised past a certain point (this is the "proximal" part of PPO). |
| Suitable for continuous action spaces? | No. Our Q-function $Q$ is implemented as a network which takes in states and returns Q-values for each discrete action. It's not even good for large action spaces!                                                                     | Yes. Our policy function $\pi$ can take continuous argument $a$.                                                                                                                                                                                                                                                                                                                                                                                                  |

A quick note on the distinction between **states** and **observations**.

In reality these are two different things (denoted $s_t$ and $o_t$), because our agent might not be able to see all relevant information. However, the games we'll be working with for the rest of this section make no distinction between them. Thus, we will only refer to $s_t$ going forwards.


## Conceptual overview of PPO

Below is an algorithm showing the conceptual overview of PPO. It's split into 2 main phases: **learning** and **rollout**.

In **rollout**, we sample experiences using the current values of our actor and critic networks, and store them in a buffer. This is all done in inference mode. In **learning**, we use our current actor and critic networks (*not* in inference mode) plus these logged experiences to calculate an objective function and use it to update our network.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/ppo-alg-conceptual.png" width="800">


## Implementational overview of PPO

There are 3 main classes you'll be using today:

* `ReplayBuffer`
    * Stores & samples from the buffer of experiences
    * Has a `get_minibatches` method, which samples data from the buffer to actually be used in training
* `Agent`
    * Contains the actor and critic networks, the `play_step` function, and a replay buffer
        * In other words, it contains both the thing doing the inference and the thing which interacts with environment & stores results
        * This is a design choice, other designs might keep these separate
    * Also has a `get_minibatches` method which calls the corresponding buffer method (and uses the agent's current state) 
* `PPOTrainer`
    * This is the main class for training our model, it helps us keep methods like `rollout_phase` and `learning_phase` separate

The image below shows the high-level details of this, and how they relate to the conceptual overview above.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/ppo-alg-objects-6.png" width="1300">

Don't worry if this seems like a lot at first! We'll go through it step-by-step. You might find this diagram useful to return to, throughout the exercises (you're recommended to open it in a separate tab).


## Notes on today's workflow

Your implementation might get huge benchmark scores by the end of the day, but don't worry if it struggles to learn the simplest of tasks. RL can be frustrating because the feedback you get is extremely noisy: the agent can fail even with correct code, and succeed with buggy code. Forming a systematic process for coping with the confusion and uncertainty is the point of today, more so than producing a working PPO implementation.

Some parts of your process could include:

- Forming hypotheses about why it isn't working, and thinking about what tests you could write, or where you could set a breakpoint to confirm the hypothesis.
- Implementing some of the even more basic Gym environments and testing your agent on those.
- Getting a sense for the meaning of various logged metrics, and what this implies about the training process
- Noticing confusion and sections that don't make sense, and investigating this instead of hand-waving over it.


## Readings

* [Spinning Up in Deep RL - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
    * You don't need to follow all the derivations, but try to have a qualitative understanding of what all the symbols represent.
* [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#solving-pong-in-5-minutes-with-ppo--envpool)
    * he good news is that you won't need all 37 of these today, so no need to read to the end.
    * We will be tackling the 13 "core" details, not in the same order as presented here. Some of the sections below are labelled with the number they correspond to in this page (e.g. **Minibatch Update ([detail #6](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Mini%2Dbatch%20Updates))**).
* [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)
    * This paper is a useful reference point for many of the key equations. In particular, you will find up to page 5 useful.

You might find it helpful to make a physical checklist of the 13 items and marking them as you go with how confident you are in your implementation. If things aren't working, this will help you notice if you missed one, or focus on the sections most likely to be bugged.

### Optional Reading

* [Spinning Up in Deep RL - Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html#background)
    * PPO is a fancier version of vanilla policy gradient, so if you're struggling to understand PPO it may help to look at the simpler setting first.
* [Andy Jones - Debugging RL, Without the Agonizing Pain](https://andyljones.com/posts/rl-debugging.html)
    * You've already read this previously but it will come in handy again.
    * You'll want to reuse your probe environments from yesterday, or you can import them from the solution if you didn't implement them all.

### References (not required reading)

- [The Policy of Truth](http://www.argmin.net/2018/02/20/reinforce/) - a contrarian take on why Policy Gradients are actually a "terrible algorithm" that is "legitimately bad" and "never a good idea".
- [Tricks from Deep RL Bootcamp at UC Berkeley](https://github.com/williamFalcon/DeepRLHacks/blob/master/README.md) - more debugging tips that may be of use.
- [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/pdf/2006.05990.pdf) - Google Brain researchers trained over 250K agents to figure out what really affects performance. The answers may surprise you.
- [Lilian Weng Blog](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#ppo)
- [A Closer Look At Deep Policy Gradients](https://arxiv.org/pdf/1811.02553.pdf)
- [Where Did My Optimum Go?: An Empirical Analysis of Gradient Descent Optimization in Policy Gradient Methods](https://arxiv.org/pdf/1810.02525.pdf)
- [Independent Policy Gradient Methods for Competitive Reinforcement Learning](https://papers.nips.cc/paper/2020/file/3b2acfe2e38102074656ed938abf4ac3-Supplemental.pdf) - requirements for multi-agent Policy Gradient to converge to Nash equilibrium.


## Setup


```python
%pip install wandb==0.13.10 # makes sure video works!

import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import random
import time
import sys
import re
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from numpy.random import Generator
import plotly.express as px
import torch as t
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
import gym
from gym.envs.classic_control.cartpole import CartPoleEnv
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from gym.spaces import Discrete
import einops
import copy
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Callable, Optional
from jaxtyping import Float, Int, Bool
import wandb
from IPython.display import clear_output

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part3_ppo"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from part1_intro_to_rl.utils import make_env
from part2_dqn.utils import set_global_seeds
from part2_dqn.solutions import Probe1, Probe2, Probe3, Probe4, Probe5
import part3_ppo.utils as utils
import part3_ppo.tests as tests
from plotly_utils import plot_cartpole_obs_and_dones

# Register our probes from last time
for idx, probe in enumerate([Probe1, Probe2, Probe3, Probe4, Probe5]):
    gym.envs.registration.register(id=f"Probe{idx+1}-v0", entry_point=probe)

Arr = np.ndarray

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

```

## PPO Arguments

Just like for DQN, we've provided you with a dataclass containing arguments for your `train_ppo` function. We've also given you a function from `utils` to display all these arguments (including which ones you've changed).

Don't worry if these don't all make sense right now, they will by the end.


```python
@dataclass
class PPOArgs:
    exp_name: str = "PPO_Implementation"
    seed: int = 1
    cuda: bool = t.cuda.is_available()
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project_name: str = "PPOCart"
    wandb_entity: str = None
    capture_video: bool = True
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 0.00025
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    batches_per_epoch: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 512
    minibatch_size: int = 128

    def __post_init__(self):
        assert self.batch_size % self.minibatch_size == 0, "batch_size must be divisible by minibatch_size"
        self.total_epochs = self.total_timesteps // (self.num_steps * self.num_envs)
        self.total_training_steps = self.total_epochs * self.batches_per_epoch * (self.batch_size // self.minibatch_size)



args = PPOArgs(minibatch_size=256)
utils.arg_help(args)
```



""", unsafe_allow_html=True)


def section_1():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#actor-critic-agent-implementation-detail-2'>Actor-Critic Agent Implementation (detail #2)</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-get-actor-and-critic'><b>Exercise</b> - implement <code>get_actor_and_critic</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#generalized-advantage-estimation-detail-5'>Generalized Advantage Estimation (detail #5)</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-compute-advantages'><b>Exercise</b> - implement <code>compute_advantages</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#replay-buffer'>Replay Buffer</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-minibatch-indices'><b>Exercise</b> - implement <code>minibatch_indices</code></a></li>
        <li><a class='contents-el' href='#exercise-explain-the-values-in-replaybuffersamples'><b>Exercise</b> - explain the values in <code>ReplayBufferSamples</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#ppoagent'>PPOAgent</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-ppoagent'><b>Exercise</b> - implement <code>PPOAgent</code></a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 1️⃣ Setting up our agent


> ##### Learning objectives
> 
> * Understand the difference between the actor & critic networks, and what their roles are
> * Learn about & implement generalised advantage estimation
> * Build a replay buffer to store & sample experiences
> * Design an agent class to step through the environment & record experiences


In this section, we'll do the following:

* Write functions to create our actor and critic networks (which will eventually be stored in our `PPOAgent` instance)
* Write a function to do **generalized advantage estimation** (this will be necessary when computing our objective function during the learning phase)
* Fill in our `ReplayBuffer` class (for storing and sampling experiences)
* Fill in our `PPOAgent` class (a wrapper around our networks and our replay buffer, which will turn them into an agent).


## Actor-Critic Agent Implementation ([detail #2](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Orthogonal%20Initialization%20of%20Weights%20and%20Constant%20Initialization%20of%20biases))

Implement the `Agent` class according to the diagram, inspecting `envs` to determine the observation shape and number of actions. We are doing separate Actor and Critic networks because [detail #13](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Shared%20and%20separate%20MLP%20networks%20for%20policy%20and%20value%20functions) notes that is performs better than a single shared network in simple environments. 

Note that today `envs` will actually have multiple instances of the environment inside, unlike yesterday's DQN which had only one instance inside. From the **37 implementation details** post:

> In this architecture, PPO first initializes a vectorized environment `envs` that runs $N$ (usually independent) environments either sequentially or in parallel by leveraging multi-processes. `envs` presents a synchronous interface that always outputs a batch of $N$ observations from $N$ environments, and it takes a batch of $N$ actions to step the $N$ environments. When calling `next_obs = envs.reset()`, next_obs gets a batch of $N$ initial observations (pronounced "next observation"). PPO also initializes an environment `done` flag variable next_done (pronounced "next done") to an $N$-length array of zeros, where its i-th element `next_done[i]` has values of 0 or 1 which corresponds to the $i$-th sub-environment being *not done* and *done*, respectively.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/screenshot-2.png" width="800">


### Exercise - implement `get_actor_and_critic`

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠🟠⚪⚪

You should spend up to 10-20 minutes on this exercise.
```

Use `layer_init` to initialize each `Linear`, overriding the standard deviation argument `std` according to the diagram. 

<figure style="max-width:510px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNqNkU9LAzEQxb_KMicLW-3W0sOihWC9CR7sbVPKbDJ1A012yZ-DlH53s8buqhV0IMnk8QZ-LzmCaCVBCa8WuybbrLnJYrlQJ-HBKq9EEvt6UobQFhWH1F21tdu5BjvKs-ViwmGbTaerbIOmKap-T_dkno9jy8WFf37hv_3uLyZ3tb1ZOS_vi_Pgc_DcJDwy8gc8E761Izv7Jzz7Qv9x_4ueJXwO_TmIv2YwQe9QeNUaN6aZXc-GQCwmquLaDqEgB01Wo5Lxm469zME3pIlDGVtJewwHz4GbU7SGTqKnR6lidCj3eHCUAwbfvrwZAaW3gc6mtcL4TvrTdXoHrTShmw" /></figure>


```python
def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_actor_and_critic(envs: gym.vector.SyncVectorEnv) -> Tuple[nn.Module, nn.Module]:
    '''
    Returns (actor, critic), the networks used for PPO.
    '''
    obs_shape = envs.single_observation_space.shape
    num_obs = np.array(obs_shape).prod()
    num_actions = envs.single_action_space.n
    pass


tests.test_get_actor_and_critic(get_actor_and_critic)
```

<details>
<summary>Solution</summary>


```python
def get_actor_and_critic(envs: gym.vector.SyncVectorEnv) -> Tuple[nn.Module, nn.Module]:
    '''
    Returns (actor, critic), the networks used for PPO.
    '''
    obs_shape = envs.single_observation_space.shape
    num_obs = np.array(obs_shape).prod()
    num_actions = envs.single_action_space.n
    # SOLUTION

    critic = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1.0)
    ).to(device)

    actor = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, num_actions), std=0.01)
    ).to(device)

    return actor, critic
```
</details>

Question - what do you think is the benefit of using a small standard deviation for the last actor layer?

<details>
<summary>Answer</summary>

The purpose is to center the initial `agent.actor` logits around zero, in other words an approximately uniform distribution over all actions independent of the state. If you didn't do this, then your agent might get locked into a nearly-deterministic policy early on and find it difficult to train away from it.

[Studies suggest](https://openreview.net/pdf?id=nIAxjsniDzg) this is one of the more important initialisation details, and performance is often harmed without it.
</details>


## Generalized Advantage Estimation ([detail #5](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Generalized%20Advantage%20Estimation))

The advantage function $A_\pi(s,a)$ indicates how much better choosing action $a$ would be in state $s$ as compared to the value obtained by letting $\pi$ choose the action (or if $\pi$ is stochastic, compared to the on expectation value by letting $\pi$ decide).
$$
A_\pi(s,a) = Q_\pi(s,a) - V_\pi(s)
$$

There are various ways to compute advantages - follow [detail #5](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Generalized%20Advantage%20Estimation) closely for today.

Given a batch of experiences, we want to compute each `advantage[t][env]`. This is equation $(11)$ of the [PPO paper](https://arxiv.org/pdf/1707.06347.pdf).

Implement `compute_advantages`. I recommend using a reversed for loop over `t` to get it working, and not worrying about trying to completely vectorize it.

Remember that the sum in $(11)$ should be truncated at the first instance when the episode is terminated (i.e. `done=True`). This is another reason why using a for loop is easier than vectorization!

**Gotcha** - if you've imported torch with `import torch as t`, be careful about using `t` as a variable during your iteration! Recommended alternatives are `T`, `t_`, `s`, or `timestep`.

### Exercise - implement `compute_advantages`

```c
Difficulty: 🟠🟠🟠🟠⚪
Importance: 🟠🟠🟠⚪⚪

You should spend up to 20-40 minutes on this exercise.
```

```python
@t.inference_mode()
def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
    '''Compute advantages using Generalized Advantage Estimation.
    next_value: shape (env,)
    next_done: shape (env,)
    rewards: shape (buffer_size, env)
    values: shape (buffer_size, env)
    dones: shape (buffer_size, env)
    Return: shape (buffer_size, env)
    '''
    pass


tests.test_compute_advantages(compute_advantages)
```

<details>
<summary>Help - I'm confused about how to calculate advantages.</summary>

You can calculate all the deltas explicitly, using:

```python
deltas = rewards + gamma * next_values * (1.0 - next_dones) - values
```

where `next_values` and `next_dones` are created by concatenating `(values, next_value)` and `(dones, next_done)` respectively and dropping the first element (i.e. the one at timestep $t=0$).

When calculating the advantages from the deltas, it might help to work backwards, i.e. start with $\hat{A}_{T-1}$ and calculate them recursively. You can go from $\hat{A}_{t}$ to $\hat{A}_{t-1}$ by multiplying by a scale factor (which might be zero depending on the value of `dones[t]`) and adding $\delta_{t-1}$.
</details>

<details>
<summary>Solution</summary>


```python
def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
    '''Compute advantages using Generalized Advantage Estimation.
    next_value: shape (env,)
    next_done: shape (env,)
    rewards: shape (buffer_size, env)
    values: shape (buffer_size, env)
    dones: shape (buffer_size, env)
    Return: shape (buffer_size, env)
    '''
    # SOLUTION
    T = values.shape[0]
    next_values = t.concat([values[1:], next_value.unsqueeze(0)])
    next_dones = t.concat([dones[1:], next_done.unsqueeze(0)])
    deltas = rewards + gamma * next_values * (1.0 - next_dones) - values
    advantages = t.zeros_like(deltas)
    advantages[-1] = deltas[-1]
    for s in reversed(range(1, T)):
        advantages[s-1] = deltas[s-1] + gamma * gae_lambda * (1.0 - dones[s]) * advantages[s]
    return advantages
```
</details>


## Replay Buffer


Our replay buffer has some similarities to the replay buffer from yesterday, as well as some important differences. One difference is the way we sample from the buffer. Rather than iteratively adding things to the buffer and taking a subsample for training (like we did in DQN), we alternate between rollout and learning phases, where in rollout we fill the buffer (i.e. starting from empty) and in learning we update from *all* the experiences in the buffer.

### Exercise - implement `minibatch_indices`

```c
Difficulty: 🟠🟠🟠🟠⚪
Importance: 🟠🟠🟠⚪⚪

You should spend up to 10-15 minutes on this exercise.
```

We'll start by implementing the `minibatch_indices` function, as described in [detail #6](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Mini%2Dbatch%20Updates). This takes a batch size (total number of elements in the buffer, i.e. $N * M$ in detail #6) and minibatch size, and returns a randomly permuted set of indices (which we'll use to index into the buffer).

To make this clearer, we've given you the test code inline (so you can see exactly what your function is required to do).


```python
def minibatch_indexes(rng: Generator, batch_size: int, minibatch_size: int) -> List[np.ndarray]:
    '''
    Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    assert batch_size % minibatch_size == 0
    pass


rng = np.random.default_rng(0)
batch_size = 6
minibatch_size = 2
indexes = minibatch_indexes(rng, batch_size, minibatch_size)

assert np.array(indexes).shape == (batch_size // minibatch_size, minibatch_size)
assert sorted(np.unique(indexes)) == [0, 1, 2, 3, 4, 5]
print("All tests in `test_minibatch_indexes` passed!")
```

<details>
<summary>Solution</summary>


```python
def minibatch_indexes(rng: Generator, batch_size: int, minibatch_size: int) -> List[np.ndarray]:
    '''
    Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    assert batch_size % minibatch_size == 0
    # SOLUTION
    indices = rng.permutation(batch_size)
    indices = einops.rearrange(indices, "(mb_num mb_size) -> mb_num mb_size", mb_size=minibatch_size)
    return list(indices)
```
</details>


Now, we've given you the code for a replay buffer for these exercises. It's pretty similar to the one you used for DQN, except that it stores slightly different variables.

It will store episodes of the form $s_t, a_t, \log{\pi(a_t\mid s_t)}, d_t, r_{t+1}, V_\pi(s_t)$ for $t = 1, ..., T-1$, where $T =$ `arg.num_steps`.



### Exercise - explain the values in `ReplayBufferSamples`

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠🟠🟠⚪

You should spend up to 10-25 minutes on this exercise.

Understanding this is very conceptually important.
```

**Read the [PPO Algorithms paper](https://arxiv.org/pdf/1707.06347.pdf) and the [PPO Implementational Details post](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#solving-pong-in-5-minutes-with-ppo--envpool), then try and figure out what each of the seven items in `ReplayBufferSamples` are and why they are necessary.** If you prefer, you can return to these once you've implmemented all the loss functions (when it might make a bit more sense).

Note - we've omitted `values` and `dones` because they aren't actually used in the learning phase (i.e. we never use them directly in our loss functions). The reason we add them to our `ReplayBufferSamples` is just for logging.


<details>
<summary>obs</summary>

`obs` are the observations from our environment, returned from the `envs.step` function.

These are fed into our `agent.actor` and `agent.critic` to choose our next actions, and get estimates for the value function.

</details>

<details>
<summary>actions</summary>

`actions` are the actions chosen by our policy. These are sampled from the distribution corresponding to the logit output we get from our `agent.actor` network. 

These are passed into the `envs.step` function to generate new observations and rewards. 

</details>

<details>
<summary>logprobs</summary>

`logprobs` are the logit outputs of our `actor.agent` network corresponding to the actions we chose.

These are necessary for calculating the clipped surrogate objective (see equation $(7)$ on page page 3 in the [PPO Algorithms paper](https://arxiv.org/pdf/1707.06347.pdf)).

`logprobs` is the value $\ln(\pi_{\theta_\text{old}}(a_t | s_t))$ in this equation (it was generated during the rollout phase, and is static). $\pi_{\theta}(a_t | s_t)$ is to the output of our `actor.agent` network **which changes as we perform gradient updates on it.**

</details>

<details>
<summary>advantages</summary>

`advantages` are the terms $\hat{A}_t$ used in the calculation of the clipped surrogate objective (again, see equation $(7)$ in the PPO algorithms paper). They are computed using the formula $(11)$ in the paper.

</details>

<details>
<summary>returns</summary>

We mentioned above that `returns = advantages + values`. Returns are used for calculating the value function loss - see [detail #9](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Value%20Function%20Loss%20Clipping) in the PPO implementational details post. This is equivalent to the TD residual loss used in DQN.

</details>



A few notes on the code below:

* The logic to compute `advantages` and `returns` is all contained in the `get_minibatches` method. You should read through this method and make sure you understand what's being computed and why.
    * We store slightly different things in our `ReplayBufferSamples` than we do in our `experiences` list in the replay buffer. The `ReplayBufferSamples` object stores **things we actually use for training**, which are computed **from** the things in the `experiences` list. (For instance, we need `returns`, but not `rewards`, in the learning phase).
* We will take the output of the `get_minibatches` method as our dataset (i.e. one epoch will be us iterating through the minibatches returned by this method). The diagram below illustrates how we take our sampled experiences and turn them into minibatches for training.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/ppo-buffer-sampling-3.png" width="1200">


```python
@dataclass
class ReplayBufferSamples:
    '''
    Samples from the replay buffer, converted to PyTorch for use in neural network training.
    '''
    obs: Float[Tensor, "minibatch_size *obs_shape"]
    dones: Float[Tensor, "minibatch_size"]
    actions: Int[Tensor, "minibatch_size"]
    logprobs: Float[Tensor, "minibatch_size"]
    values: Float[Tensor, "minibatch_size"]
    advantages: Float[Tensor, "minibatch_size"]
    returns: Float[Tensor, "minibatch_size"]


class ReplayBuffer:
    '''
    Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.

    Needs to be initialized with the first obs, dones and values.
    '''
    def __init__(self, args: PPOArgs, envs: gym.vector.SyncVectorEnv):
        '''Defining all the attributes the buffer's methods will need to access.'''
        self.rng = np.random.default_rng(args.seed)
        self.num_envs = envs.num_envs
        self.obs_shape = envs.single_observation_space.shape
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.batch_size = args.batch_size
        self.minibatch_size = args.minibatch_size
        self.num_steps = args.num_steps
        self.batches_per_epoch = args.batches_per_epoch
        self.experiences = []


    def add(self, obs: t.Tensor, actions: t.Tensor, rewards: t.Tensor, dones: t.Tensor, logprobs: t.Tensor, values: t.Tensor) -> None:
        '''
        obs: shape (n_envs, *observation_shape) 
            Observation before the action
        actions: shape (n_envs,) 
            Action chosen by the agent
        rewards: shape (n_envs,) 
            Reward after the action
        dones: shape (n_envs,) 
            If True, the episode ended and was reset automatically
        logprobs: shape (n_envs,)
            Log probability of the action that was taken (according to old policy)
        values: shape (n_envs,)
            Values, estimated by the critic (according to old policy)
        '''
        assert obs.shape == (self.num_envs, *self.obs_shape)
        assert actions.shape == (self.num_envs,)
        assert rewards.shape == (self.num_envs,)
        assert dones.shape == (self.num_envs,)
        assert logprobs.shape == (self.num_envs,)
        assert values.shape == (self.num_envs,)

        self.experiences.append((obs, dones, actions, logprobs, values, rewards))


    def get_minibatches(self, next_value: t.Tensor, next_done: t.Tensor) -> List[ReplayBufferSamples]:
        minibatches = []

        # Turn all experiences to tensors on our device (we only want to do this once, not every time we add a new experience)
        obs, dones, actions, logprobs, values, rewards = [t.stack(arr).to(device) for arr in zip(*self.experiences)]

        # Compute advantages and returns (then get a list of everything we'll need for our replay buffer samples)
        advantages = compute_advantages(next_value, next_done, rewards, values, dones.float(), self.gamma, self.gae_lambda)
        returns = advantages + values
        replaybuffer_args = [obs, dones, actions, logprobs, values, advantages, returns]
        
        # We cycle through the entire buffer `self.batches_per_epoch` times
        for _ in range(self.batches_per_epoch):

            # Get random indices we'll use to generate our minibatches
            indices = minibatch_indexes(self.rng, self.batch_size, self.minibatch_size)

            # Get our new list of minibatches, and add them to the list
            for index in indices:
                minibatch = ReplayBufferSamples(*[
                    arg.flatten(0, 1)[index].to(device) for arg in replaybuffer_args
                ])
                minibatches.append(minibatch)

        # Reset the buffer
        self.experiences = []

        return minibatches

```

Now, like before, here's some code to generate and plot observations. The dotted lines indicate a terminated episode.

Note that we're actually using four environments inside our `envs` object, rather than just one like last time. The 3 solid (non-dotted) lines in the first plot below indicate the transition between different environments in `envs` (which we've stitched together into one long episode in the first plot below).


```python
args = PPOArgs()
envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", i, i, False, "test") for i in range(4)])
next_value = t.zeros(envs.num_envs).to(device)
next_done = t.zeros(envs.num_envs).to(device)
rb = ReplayBuffer(args, envs)
actions = t.zeros(envs.num_envs).int().to(device)
obs = envs.reset()

for i in range(args.num_steps):
    (next_obs, rewards, dones, infos) = envs.step(actions.cpu().numpy())
    real_next_obs = next_obs.copy()
    for (i, done) in enumerate(dones):
        if done: real_next_obs[i] = infos[i]["terminal_observation"]
    logprobs = values = t.zeros(envs.num_envs)
    rb.add(t.from_numpy(obs).to(device), actions, t.from_numpy(rewards).to(device), t.from_numpy(dones).to(device), logprobs, values)
    obs = next_obs

obs, dones, actions, logprobs, values, rewards = [t.stack(arr).to(device) for arr in zip(*rb.experiences)]

plot_cartpole_obs_and_dones(obs, dones, show_env_jumps=True)
```

The next code shows **a single minibatch**, sampled from this replay buffer.


```python
minibatches = rb.get_minibatches(next_value, next_done)

obs = minibatches[0].obs
dones = minibatches[0].dones

plot_cartpole_obs_and_dones(obs, dones)
```

## PPOAgent

As the final task in this section, you should fill in the agent's `play_step` method. This is conceptually similar to what you did during DQN, but with a few key differences.

In DQN, we did the following:

* used the Q-Network and an epsilon greedy policy to select an action based on current observation,
* stepped the environment with this action,
* stored the transition in the replay buffer (using the `add` method of the buffer)

In PPO, you'll do the following:

* use the actor network to return a distribution over actions based on current observation & sample from this distribution to select an action,
* step the environment with this action,
* calculate `logprobs` and `values` (which we'll need during our learning step),
* store the transition in the replay buffer (using the `add` method of the buffer)


### Exercise - implement `PPOAgent`

```c
Difficulty: 🟠🟠🟠🟠⚪
Importance: 🟠🟠🟠🟠⚪

You should spend up to 20-35 minutes on this exercise.
```

A few gotchas:

* Make sure the things you're adding to the buffer are the right shape (otherwise the `add` method of the buffer will throw an error). This includes paying attention to the batch dimension when you put things through the actor and critic networks.
* Don't forget to use inference mode when running your actor and critic networks, since you're only generating data $\theta_\text{old}$ (i.e. you don't want to update the weights of the network based on these values).
* Don't forget to increment the step count `self.steps` by the number of environments (you're stepping once for each env!) in each call to `play_step`.
* At the end of `play_step`, you should update `self.next_obs` and `self.next_done` (because this is where our agent will start next time `play_step` is called).
    * For more on why we need these values, see [this section](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=role%20to%20help%20transition%20between%20phases) of detail #1 in the "37 implementational details" post.

<details>
<summary>Tip - how to sample from distributions (to get <code>actions</code>)</summary>

You might remember using `torch.distributions.categorical.Categorical` when we were sampling from transformers in the previous chapter. We can use this again!

You can define a `Categorical` object by passing in `logits` (the output of the actor network), and then you can:

* Sample from it using the `sample` method,
* Calculate the logprobs of a given action using the `log_prob` method (with the actions you took as input argument to this method).
</details>

For this exercise and others to follow, there's a trade-off in the test functions between being strict and being lax. Too lax and the tests will let failures pass; too strict and they might fail for odd reasons even if your code is mostly correct. If you find youself continually failing tests then you should ask a TA for help.

**Note** - `PPOAgent` subclasses `nn.Module` so that we can call `agent.parameters()` to get the parameters of the actor and critic networks, and feed these params into our optimizer.


```python
class PPOAgent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, args: PPOArgs, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        self.args = args
        self.envs = envs
        self.num_envs = envs.num_envs
        self.obs_shape = envs.single_observation_space.shape
        self.num_obs = np.array(self.obs_shape).prod()
        self.num_actions = envs.single_action_space.n

        # Keep track of global number of steps taken by agent
        self.steps = 0
        # Define actor and critic (using our previous methods)
        self.actor, self.critic = get_actor_and_critic(envs)

        # Define our first (obs, done, value), so we can start adding experiences to our replay buffer
        self.next_obs = t.tensor(self.envs.reset()).to(device)
        self.next_done = t.zeros(self.envs.num_envs).to(device, dtype=t.float)

        # Create our replay buffer
        self.rb = ReplayBuffer(args, envs)


    def play_step(self) -> List[dict]:
        '''
        Carries out a single interaction step between the agent and the environment, and adds results to the replay buffer.
        '''
        pass

    def get_minibatches(self) -> None:
        '''
        Gets minibatches from the replay buffer.
        '''
        with t.inference_mode():
            next_value = self.critic(self.next_obs).flatten()
        return self.rb.get_minibatches(next_value, self.next_done)


tests.test_ppo_agent(PPOAgent)
```

<details>
<summary>Solution</summary>


```python
class PPOAgent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, args: PPOArgs, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        self.args = args
        self.envs = envs
        self.num_envs = envs.num_envs
        self.obs_shape = envs.single_observation_space.shape
        self.num_obs = np.array(self.obs_shape).prod()
        self.num_actions = envs.single_action_space.n

        # Keep track of global number of steps taken by agent
        self.steps = 0
        # Define actor and critic (using our previous methods)
        self.actor, self.critic = get_actor_and_critic(envs)

        # Define our first (obs, done, value), so we can start adding experiences to our replay buffer
        self.next_obs = t.tensor(self.envs.reset()).to(device)
        self.next_done = t.zeros(self.envs.num_envs).to(device, dtype=t.float)

        # Create our replay buffer
        self.rb = ReplayBuffer(args, envs)


    def play_step(self) -> List[dict]:
        '''
        Carries out a single interaction step between the agent and the environment, and adds results to the replay buffer.
        '''
        # SOLUTION
        obs = self.next_obs
        dones = self.next_done
        with t.inference_mode():
            values = self.critic(obs).flatten()
            logits = self.actor(obs)
        
        probs = Categorical(logits=logits)
        actions = probs.sample()
        logprobs = probs.log_prob(actions)
        next_obs, rewards, next_dones, infos = self.envs.step(actions.cpu().numpy())
        rewards = t.from_numpy(rewards).to(device)

        # (s_t, a_t, r_t+1, d_t, logpi(a_t|s_t), v(s_t))
        self.rb.add(obs, actions, rewards, dones, logprobs, values)

        self.next_obs = t.from_numpy(next_obs).to(device)
        self.next_done = t.from_numpy(next_dones).to(device, dtype=t.float)
        self.steps += self.num_envs

        return infos
    

    def get_minibatches(self) -> None:
        '''
        Gets minibatches from the replay buffer.
        '''
        with t.inference_mode():
            next_value = self.critic(self.next_obs).flatten()
        return self.rb.get_minibatches(next_value, self.next_done)
```
</details>




""", unsafe_allow_html=True)


def section_2():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#objective-function'>Objective function</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#clipped-surrogate-objective'>Clipped Surrogate Objective</a></li>
        <li class='margtop'><a class='contents-el' href='#value-function-loss-detail-25'>Value Function Loss (detail #9)</a></li>
        <li><a class='contents-el' href='#exercise-implement-calc-value-function-loss'><b>Exercise</b> - implement <code>calc_value_function_loss</code></a></li>
        <li class='margtop'><a class='contents-el' href='#entropy-bonus-detail-10'>Entropy Bonus (detail #10)</a></li>
        <li><a class='contents-el' href='#exercise-implement-calc-entropy-bonus'><b>Exercise</b> - implement <code>calc_entropy_bonus</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#adam-optimizer-and-scheduler-details'>Adam Optimizer and Scheduler (details #3 and #4)</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-pposcheduler'><b>Exercise</b> - implement <code>PPOScheduler</code></a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 2️⃣ Learning Phase


> ##### Learning objectives
>
> * Implement the total objective function (sum of three separate terms)
> * Understand the importance of each of these terms for the overall algorithm
> * Write a function to return an optimizer and learning rate scheduler for your model


In the last section, we wrote a lot of setup code (including handling most of how our rollout phase will work). Next, we'll turn to the learning phase.

In the next exercises, you'll write code to compute your total objective function. This is given by equation $(9)$ in the paper, and is the sum of three terms - we'll implement each one individually.


Note - the convention we've used in these exercises for signs is that **your function outputs should be the expressions in equation $(9)$**, in other words you will compute $L_t^{CLIP}(\theta)$, $c_1 L_t^{VF}(\theta)$ and $c_2 S[\pi_\theta](s_t)$. We will then perform **gradient ascent** by passing `maximize=True` into our optimizers. An equally valid solution would be to just return the negative of the objective function.


## Objective function


### Clipped Surrogate Objective

For each minibatch, calculate $L^{CLIP}$ from equation $(7)$ in the paper. This will allow us to improve the parameters of our actor.

Note - in the paper, don't confuse $r_{t}$ which is reward at time $t$ with $r_{t}(\theta)$, which is the probability ratio between the current policy (output of the actor) and the old policy (stored in `mb_logprobs`).

Pay attention to the normalization instructions in [detail #7](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Normalization%20of%20Advantages) when implementing this loss function. They add a value of `eps = 1e-8` to the denominator to avoid division by zero, you should also do this.

You can use the `probs.log_prob` method to get the log probabilities that correspond to the actions in `mb_action`.

Note - if you're wondering why we're using a `Categorical` type rather than just using `log_prob` directly, it's because we'll be using them to sample actions later on in our `train_ppo` function. Also, categoricals have a useful method for returning the entropy of a distribution (which will be useful for the entropy term in the loss function).


```python
def calc_clipped_surrogate_objective(
    probs: Categorical, 
    mb_action: Int[Tensor, "minibatch_size"], 
    mb_advantages: Float[Tensor, "minibatch_size"], 
    mb_logprobs: Float[Tensor, "minibatch_size"], 
    clip_coef: float, 
    eps: float = 1e-8
) -> Float[Tensor, ""]:
    '''Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs:
        a distribution containing the actor's unnormalized logits of shape (minibatch_size, num_actions)
    mb_action:
        what actions actions were taken in the sampled minibatch
    mb_advantages:
        advantages calculated from the sampled minibatch
    mb_logprobs:
        logprobs of the actions taken in the sampled minibatch (according to the old policy)
    clip_coef:
        amount of clipping, denoted by epsilon in Eq 7.
    eps:
        used to add to std dev of mb_advantages when normalizing (to avoid dividing by zero)
    '''
    assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape    
    pass


tests.test_calc_clipped_surrogate_objective(calc_clipped_surrogate_objective)
```

<details>
<summary>Solution</summary>


```python
def calc_clipped_surrogate_objective(
    probs: Categorical, 
    mb_action: Int[Tensor, "minibatch_size"], 
    mb_advantages: Float[Tensor, "minibatch_size"], 
    mb_logprobs: Float[Tensor, "minibatch_size"], 
    clip_coef: float, 
    eps: float = 1e-8
) -> Float[Tensor, ""]:
    '''Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs:
        a distribution containing the actor's unnormalized logits of shape (minibatch_size, num_actions)
    mb_action:
        what actions actions were taken in the sampled minibatch
    mb_advantages:
        advantages calculated from the sampled minibatch
    mb_logprobs:
        logprobs of the actions taken in the sampled minibatch (according to the old policy)
    clip_coef:
        amount of clipping, denoted by epsilon in Eq 7.
    eps:
        used to add to std dev of mb_advantages when normalizing (to avoid dividing by zero)
    '''
    assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape    
    # SOLUTION
    logits_diff = probs.log_prob(mb_action) - mb_logprobs

    r_theta = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)

    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1-clip_coef, 1+clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()
```
</details>


## Value Function Loss ([detail #9](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Value%20Function%20Loss%20Clipping))

The value function loss lets us improve the parameters of our critic. Today we're going to implement the simple form: this is just 1/2 the mean squared difference between the following two terms:

* The **critic's prediction**
    * This is $V_\theta(s_t)$ in the paper, and `values` in the diagram earlier (colored blue). 
* The **observed returns**
    * This is $V_t^\text{targ}$ in the paper, and `returns` in the diagram earlier (colored yellow).
    * We defined it as `advantages + values`, where in this case `values` is the item stored in memory rather than the one computed by our network during the learning phase. We can interpret `returns` as a more accurate estimate of values, since the `advantages` term takes into account the rewards $r_{t+1}, r_{t+2}, ...$ which our agent actually accumulated.

The PPO paper did a more complicated thing with clipping, but we're going to deviate from the paper and NOT clip, since [detail #9](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Value%20Function%20Loss%20Clipping) gives evidence that it isn't beneficial.


### Exercise - implement `calc_value_function_loss`

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠🟠⚪⚪

You should spend up to ~10 minutes on this exercise.
```

Implement `calc_value_function_loss` which returns the term denoted $c_1 L_t^{VF}$ in equation $(9)$.


```python
def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size"],
    mb_returns: Float[Tensor, "minibatch_size"],
    vf_coef: float
) -> Float[Tensor, ""]:
    '''Compute the value function portion of the loss function.

    values:
        the value function predictions for the sampled minibatch (using the updated critic network)
    mb_returns:
        the target for our updated critic network (computed as `advantages + values` from the old network)
    vf_coef:
        the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    assert values.shape == mb_returns.shape
    pass


tests.test_calc_value_function_loss(calc_value_function_loss)
```

<details>
<summary>Solution</summary>


```python
def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size"],
    mb_returns: Float[Tensor, "minibatch_size"],
    vf_coef: float
) -> Float[Tensor, ""]:
    '''Compute the value function portion of the loss function.

    values:
        the value function predictions for the sampled minibatch (using the updated critic network)
    mb_returns:
        the target for our updated critic network (computed as `advantages + values` from the old network)
    vf_coef:
        the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    assert values.shape == mb_returns.shape
    # SOLUTION
    return 0.5 * vf_coef * (values - mb_returns).pow(2).mean()
```
</details>


## Entropy Bonus ([detail #10](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Overall%20Loss%20and%20Entropy%20Bonus))

The entropy bonus term is intended to incentivize exploration by increasing the entropy of the actions distribution. For a discrete probability distribution $p$, the entropy $H$ is defined as
$$
H(p) = \sum_x p(x) \ln \frac{1}{p(x)}
$$
If $p(x) = 0$, then we define $0 \ln \frac{1}{0} := 0$ (by taking the limit as $p(x) \to 0$).
You should understand what entropy of a discrete distribution means, but you don't have to implement it yourself: `probs.entropy` computes it using the above formula but in a numerically stable way, and in
a way that handles the case where $p(x) = 0$.

Question: in CartPole, what are the minimum and maximum values that entropy can take? What behaviors correspond to each of these cases?

<details>
<summary>Answer</summary>

The minimum entropy is zero, under the policy "always move left" or "always move right".

The maximum entropy is $\ln(2) \approx 0.693$ under the uniform random policy over the 2 actions.
</details>

Separately from its role in the loss function, the entropy of our action distribution is a useful diagnostic to have: if the entropy of agent's actions is near the maximum, it's playing nearly randomly which means it isn't learning anything (assuming the optimal policy isn't random). If it is near the minimum especially early in training, then the agent might not be exploring enough.


### Exercise - implement `calc_entropy_bonus`

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠🟠⚪⚪

You should spend up to ~10 minutes on this exercise.
```


```python
def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''Return the entropy bonus term, suitable for gradient ascent.

    probs:
        the probability distribution for the current policy
    ent_coef: 
        the coefficient for the entropy loss, which weights its contribution to the overall objective function. Denoted by c_2 in the paper.
    '''
    pass


tests.test_calc_entropy_bonus(calc_entropy_bonus)
```

<details>
<summary>Solution</summary>


```python
def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''Return the entropy bonus term, suitable for gradient ascent.

    probs:
        the probability distribution for the current policy
    ent_coef: 
        the coefficient for the entropy loss, which weights its contribution to the overall objective function. Denoted by c_2 in the paper.
    '''
    # SOLUTION
    return ent_coef * probs.entropy().mean()
```
</details>


## Adam Optimizer and Scheduler (details [#3](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=The%20Adam%20Optimizer%E2%80%99s%20Epsilon%20Parameter) and [#4](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Adam%20Learning%20Rate%20Annealing))

Even though Adam is already an adaptive learning rate optimizer, empirically it's still beneficial to decay the learning rate.

Implement a linear decay from `initial_lr` to `end_lr` over `total_training_steps` steps. Also, make sure you read details details [#3](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=The%20Adam%20Optimizer%E2%80%99s%20Epsilon%20Parameter) and [#4](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Adam%20Learning%20Rate%20Annealing) so you don't miss any of the Adam implementational details.

Note, the training terminates after `num_updates`, so you don't need to worry about what the learning rate will be after this point.


### Exercise - implement `PPOScheduler`

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠⚪⚪⚪

You should spend up to 10-15 minutes on this exercise.
```


```python
class PPOScheduler:
    def __init__(self, optimizer: Optimizer, initial_lr: float, end_lr: float, total_training_steps: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_training_steps = total_training_steps
        self.n_step_calls = 0

    def step(self):
        '''Implement linear learning rate decay so that after total_training_steps calls to step, the learning rate is end_lr.
        '''
        pass

def make_optimizer(agent: PPOAgent, total_training_steps: int, initial_lr: float, end_lr: float) -> Tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_training_steps)
    return (optimizer, scheduler)


tests.test_ppo_scheduler(PPOScheduler)
```

<details>
<summary>Solution</summary>


```python
class PPOScheduler:
    def __init__(self, optimizer: Optimizer, initial_lr: float, end_lr: float, total_training_steps: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_training_steps = total_training_steps
        self.n_step_calls = 0

    def step(self):
        '''Implement linear learning rate decay so that after total_training_steps calls to step, the learning rate is end_lr.
        '''
        # SOLUTION
        self.n_step_calls += 1
        frac = self.n_step_calls / self.total_training_steps
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)


def make_optimizer(agent: PPOAgent, total_training_steps: int, initial_lr: float, end_lr: float) -> Tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_training_steps)
    return (optimizer, scheduler)
```
</details>




""", unsafe_allow_html=True)


def section_3():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#writing-your-training-loop'>Writing your training loop</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#logging'>Logging</a></li>
        <li><a class='contents-el' href='#exercise-complete-the-ppotrainer-classer'><b>Exercise</b> - complete the <code>PPOTrainer</code> class</a></li>
        <li><a class='contents-el' href='#catastrophic-forgetting'>Catastrophic forgetting</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#reward-shaping'>Reward Shaping</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-reward-shaping'><b>Exercise</b> - implement reward shaping</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#bonus'>Bonus</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#trust-region-methods'>Trust Region Methods</a></li>
        <li><a class='contents-el' href='#long-term-replay-buffer'>Long-term Replay Buffer</a></li>
        <li><a class='contents-el' href='#vectorized-advantage-calculation'>Vectorized Advantage Calculation</a></li>
        <li><a class='contents-el' href='#other-discrete-environments'>Other Discrete Environments</a></li>
        <li><a class='contents-el' href='#minigrid-envs-procgen'>Minigrid envs / Procgen</a></li>
        <li><a class='contents-el' href='#continuous-action-spaces'>Continuous Action Spaces</a></li>
        <li><a class='contents-el' href='#atari'>Atari</a></li>
        <li><a class='contents-el' href='#atari'>Multi-Agent PPO</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 3️⃣ Training Loop


> ##### Learning objectives
> 
> * Build a full training loop for the PPO algorithm
> * Train our agent, and visualise its performance with Weights & Biases media logger
> * Use reward shaping to improve your agent's training (and make it do tricks!)


## Writing your training loop


Finally, we can package this all together into our full training loop. 

For a suggested implementation, please refer back to the diagram at the homepage of these exercises (pasted again below for convenience).

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/ppo-alg-objects-6.png" width="1300">

Under this implementation, you have two main methods to implement - `rollout_phase` and `learning_phase`. These will do the following:

* `rollout_phase`
    * Step the agent through the environment for `args.num_steps` steps, collecting data into the replay buffer.
    * If using `wandb`, log any relevant variables.
* `learning_phase`
    * Sample from the replay buffer using `agent.get_minibatches()` (which returns a list of minibatches).
    * Iterate over these minibatches, and for each minibatch:
        * Calculate the total objective function.
        * Backpropagate the loss.
        * Update the agent's parameters using the optimizer.
        * **Clip the gradients** in accordance with [detail #11](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Global%20Gradient%20Clipping%20)
        * Step the scheduler.
    * If using `wandb`, log any relevant variables.

A few notes to help you:

* The `agent.get_minibatches()` function empties the list of experiences stored in the replay buffer, so you don't need to worry about doing this manually.
* You can log variables using `wandb.log({"variable_name": variable_value}, steps=steps)`.
* The `agent.play_step()` method returns a list of dictionaries containing data about the current run. If the agent terminated at that step, then the dictionary will contain `{"episode": {"l": episode_length, "r": episode_reward}}`.
* You might want to create a separate method like `_compute_ppo_objective(minibatch)` which returns the objective function (to keep your code clean for the `learning_phase` method).
* For clipping gradients, you can use `nn.utils.clip_grad_norm(parameters, max_norm)` (see [here](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) for more details).

Finally, you'll put this all together into a training loop. Very roughly, this should look like:

```python
# initialise PPOArgs and PPOTrainer objects
for epoch in args.total_epochs:
    trainer.rollout_phase()
    trainer.learning_phase()
    # log any relevant variables
```

As an indication, the solution's implementation (ignoring logging) has the following properties:
* 2 lines for `rollout_phase`
* 8 lines for `learning_phase`
* 8 lines for `_compute_ppo_objective`
* 5 lines for the full training loop

### Logging

You should only focus on logging once you've got a mininal version of the code working. Once you do, you can try logging variables in the way described by [detail #12](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Debug%20variables). This will involve adding code to the `rollout_phase` and `learning_phase` methods.


### Exercise - complete the `PPOTrainer` class

```c
Difficulty: 🟠🟠🟠🟠🟠
Importance: 🟠🟠🟠🟠🟠

You should spend up to 30-60 minutes on this exercise (including logging).
```

If you get stuck at any point during this implementation, please ask a TA for help!

```python
class PPOTrainer:

    def __init__(self, args: PPOArgs):
        super().__init__()
        self.args = args
        set_global_seeds(args.seed)
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, self.run_name) for i in range(args.num_envs)])
        self.agent = PPOAgent(self.args, self.envs).to(device)
        self.optimizer, self.scheduler = make_optimizer(self.agent, self.args.total_training_steps, self.args.learning_rate, 0.0)
        if args.use_wandb:
            wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, name=args.exp_name)
            if args.capture_video: wandb.gym.monitor()


    def rollout_phase(self):
        '''Should populate the replay buffer with new experiences.'''
        pass


    def learning_phase(self) -> None:
        '''Should get minibatches and iterate through them (performing an optimizer step at each one).'''
        pass
```

<details>
<summary>Solution (full)</summary>

```python
class PPOTrainer:

    def __init__(self, args: PPOArgs):
        super().__init__()
        self.args = args
        set_global_seeds(args.seed)
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, self.run_name) for i in range(args.num_envs)])
        self.agent = PPOAgent(self.args, self.envs).to(device)
        self.optimizer, self.scheduler = make_optimizer(self.agent, self.args.total_training_steps, self.args.learning_rate, 0.0)
        if args.use_wandb:
            wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, name=args.exp_name, config=args)
            if args.capture_video: wandb.gym.monitor()


    def rollout_phase(self):
        '''Should populate the replay buffer with new experiences.'''
        # SOLUTION
        last_episode_len = None
        for step in range(self.args.num_steps):
            infos = self.agent.play_step()
            for info in infos:
                if "episode" in info.keys():
                    last_episode_len = info["episode"]["l"]
                    last_episode_return = info["episode"]["r"]
                    if args.use_wandb: wandb.log({
                        "episode_length": last_episode_len,
                        "episode_return": last_episode_return,
                    }, step=self.agent.steps)
        # Return this for use in the progress bar
        return last_episode_len


    def learning_phase(self) -> None:
        '''Should get minibatches and iterate through them (performing an optimizer step at each one).'''
        # SOLUTION
        minibatches = self.agent.get_minibatches()
        for minibatch in minibatches:
            objective_fn = self._compute_ppo_objective(minibatch)
            objective_fn.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()


    def _compute_ppo_objective(self, minibatch: ReplayBufferSamples) -> Float[Tensor, ""]:
        '''Handles learning phase for a single minibatch. Returns objective function to be maximized.'''
        logits = self.agent.actor(minibatch.obs)
        probs = Categorical(logits=logits)
        values = self.agent.critic(minibatch.obs).squeeze()

        clipped_surrogate_objective = calc_clipped_surrogate_objective(probs, minibatch.actions, minibatch.advantages, minibatch.logprobs, self.args.clip_coef)
        value_loss = calc_value_function_loss(values, minibatch.returns, self.args.vf_coef)
        entropy_bonus = calc_entropy_bonus(probs, self.args.ent_coef)

        total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus

        with t.inference_mode():
            newlogprob = probs.log_prob(minibatch.actions)
            logratio = newlogprob - minibatch.logprobs
            ratio = logratio.exp()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        if args.use_wandb: wandb.log(dict(
            total_steps = self.agent.steps,
            values = values.mean().item(),
            learning_rate = self.scheduler.optimizer.param_groups[0]["lr"],
            value_loss = value_loss.item(),
            clipped_surrogate_objective = clipped_surrogate_objective.item(),
            entropy = entropy_bonus.item(),
            approx_kl = approx_kl,
            clipfrac = np.mean(clipfracs)
        ), step=self.agent.steps)

        return total_objective_function
    


def train(args: PPOArgs) -> PPOAgent:
    '''Implements training loop, used like: agent = train(args)'''

    trainer = PPOTrainer(args)

    progress_bar = tqdm(range(args.total_epochs))

    for epoch in progress_bar:
        last_episode_len = trainer.rollout_phase()        
        if last_episode_len is not None:
            progress_bar.set_description(f"Epoch {epoch:02}, Episode length: {last_episode_len}")

        trainer.learning_phase()
    
    return trainer.agent
```
</details>

<details>
<summary>Solution (simple, no logging)</summary>

```python
class PPOTrainer:

    def __init__(self, args: PPOArgs):
        super().__init__()
        self.args = args
        set_global_seeds(args.seed)
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, self.run_name) for i in range(args.num_envs)])
        self.agent = PPOAgent(self.args, self.envs).to(device)
        self.optimizer, self.scheduler = make_optimizer(self.agent, self.args.total_training_steps, self.args.learning_rate, 0.0)
        if args.use_wandb:
            wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, name=self.run_name, config=args)
            if args.capture_video: wandb.gym.monitor()


    def rollout_phase(self):
        '''Should populate the replay buffer with new experiences.'''
        # SOLUTION
        for step in range(self.args.num_steps):
            infos = self.agent.play_step()


    def learning_phase(self) -> None:
        '''Should get minibatches and iterate through them (performing an optimizer step at each one).'''
        # SOLUTION
        minibatches = self.agent.get_minibatches()
        for minibatch in minibatches:
            objective_fn = self._compute_ppo_objective(minibatch)
            objective_fn.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()


    def _compute_ppo_objective(self, minibatch: ReplayBufferSamples) -> Float[Tensor, ""]:
        '''Handles learning phase for a single minibatch. Returns objective function to be maximized.'''
        logits = self.agent.actor(minibatch.obs)
        probs = Categorical(logits=logits)
        values = self.agent.critic(minibatch.obs).squeeze()

        clipped_surrogate_objective = calc_clipped_surrogate_objective(probs, minibatch.actions, minibatch.advantages, minibatch.logprobs, self.args.clip_coef)
        value_loss = calc_value_function_loss(values, minibatch.returns, self.args.vf_coef)
        entropy_bonus = calc_entropy_bonus(probs, self.args.ent_coef)

        total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus

        return total_objective_function
    


def train(args: PPOArgs) -> PPOAgent:
    '''Implements training loop, used like: agent = train(args)'''

    trainer = PPOTrainer(args)

    progress_bar = tqdm(range(args.total_epochs))

    for epoch in progress_bar:
        last_episode_len = trainer.rollout_phase()
        if last_episode_len is not None:
            progress_bar.set_description(f"Epoch {epoch:02}, Episode length: {last_episode_len}")

        trainer.learning_phase()
    
    return trainer.agent
```
</details>

Here's some code to run your model on the probe environments (and assert that they're all working fine).

```python
def test_probe(probe_idx: int):

    # Define a set of arguments for our probe experiment
    args = PPOArgs(
        env_id=f"Probe{probe_idx}-v0",
        exp_name=f"test-probe-{probe_idx}", 
        total_timesteps=[10000, 10000, 10000, 30000, 50000][probe_idx-1],
        learning_rate=0.001,
        capture_video=False,
        use_wandb=False,
    )

    # YOUR CODE HERE - create a PPOTrainer instance, and train your agent
    agent = None

    # Check that our final results were the ones we expected from this probe
    obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
    expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]], [[args.gamma], [1.0]], [[1.0]], [[1.0], [1.0]]]
    expected_probs_for_probes = [None, None, None, [[0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
    tolerances = [5e-4, 5e-4, 5e-4, 1e-3, 1e-3]
    obs = t.tensor(obs_for_probes[probe_idx-1]).to(device)
    with t.inference_mode():
        value = agent.critic(obs)
        probs = agent.actor(obs).softmax(-1)
    expected_value = t.tensor(expected_value_for_probes[probe_idx-1]).to(device)
    t.testing.assert_close(value, expected_value, atol=tolerances[probe_idx-1], rtol=0)
    expected_probs = expected_probs_for_probes[probe_idx-1]
    if expected_probs is not None:
        t.testing.assert_close(probs, t.tensor(expected_probs).to(device), atol=tolerances[probe_idx-1], rtol=0)
    clear_output()
    print("Probe tests passed!")


for probe_idx in range(1, 6):
    test_probe(probe_idx)
```

Once you've passed the tests for all 5 probe environments, you should test your model on Cartpole.

<details>
<summary>Question - if you've done this correctly (and logged everything), clipped surrogate objective will be close to zero. Does this mean that it's not important in the overall algorithm (compared to the components of the objective function which are larger)?</summary>

No, this doesn't mean that it's not important.

Clipped surrogate objective is a moving target. At each rollout phase, we generate new experiences, and the expected value of the clipped surrogate objective will be zero (because the expected value of advantages is zero). But this doesn't mean that differentiating clipped surrogate objective wrt the policy doesn't have a large gradient!

As we make update steps in the learning phase, the policy values $\pi(a_t \mid s_t)$ will increase for actions which have positive advantages, and decrease for actions which have negative advantages, so the clipped surrogate objective will no longer be zero in expectation. But (thanks to the fact that we're clipping changes larger than $\epsilon$) it will still be very small.

</details>

### Catastrophic forgetting

Note - you might see performance very high initially and then drop off rapidly (before recovering again).

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/cf2.png" width="600">

This is a well-known RL phenomena called **catastrophic forgetting**. It happens when the buffer only contains good experiences, and the agent forgets how to recover from bad experiences. One way to fix this is to change your buffer to keep 10 of experiences from previous epochs, and 90% of experiences from the current epoch. Can you implement this?

(Note - reward shaping can also help fix this problem - see next section.)

## Reward Shaping

Recall the [docs](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) and [source code](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) for the `CartPole` environment.

The current rewards for `CartPole` encourage the agent to keep the episode running for as long as possible, which it then needs to associate with balancing the pole.

Here, we inherit from `CartPoleEnv` so that we can modify the dynamics of the environment.

Try to modify the reward to make the task as easy to learn as possible. Compare this against your performance on the original environment, and see if the agent learns faster with your shaped reward. If you can bound the reward on each timestep between 0 and 1, this will make comparing the results to `CartPole-v1` easier.

<details>
<summary>Help - I'm not sure what I'm meant to return in this function.</summary>

The tuple `(obs, rew, done, info)` is returned from the CartPole environment. Here, `rew` is always 1 unless the episode has terminated.

You should change this, so that `rew` incentivises good behaviour, even if the pole hasn't fallen yet. You can use the information returned in `obs` to construct a new reward function.
</details>

<details>
<summary>Help - I'm confused about how to choose a reward function. (Try and think about this for a while before looking at this dropdown.)</summary>

Right now, the agent always gets a reward of 1 for each timestep it is active. You should try and change this so that it gets a reward between 0 and 1, which is closer to 1 when the agent is performing well / behaving stably, and equals 0 when the agent is doing very poorly.

The variables we have available to us are cart position, cart velocity, pole angle, and pole angular velocity, which I'll denote as $x$, $v$, $\theta$ and $\omega$.

Here are a few suggestions which you can try out:
* $r = 1 - (\theta / \theta_{\text{max}})^2$. This will have the effect of keeping the angle close to zero.
* $r = 1 - (x / x_{\text{max}})^2$. This will have the effect of pushing it back towards the centre of the screen (i.e. it won't tip and fall to the side of the screen).

You could also try using e.g. $|\theta / \theta_{\text{max}}|$ rather than $(\theta / \theta_{\text{max}})^2$. This would still mean reward is in the range (0, 1), but it would result in a larger penalty for very small deviations from the vertical position.

You can also try a linear combination of two or more of these rewards!
</details>


<details>
<summary>Help - my agent's episodic return is smaller than it was in the original CartPole environment.</summary>

This is to be expected, because your reward function is no longer always 1 when the agent is upright. Both your time-discounted reward estimates and your actual realised rewards will be less than they were in the cartpole environment. 

For a fairer test, measure the length of your episodes - hopefully your agent learns how to stay upright for the entire 500 timestep interval as fast as or faster than it did previously.
</details>


### Exercise - implement reward shaping

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠🟠🟠⚪

You should spend up to 15-30 minutes on this exercise.
```

Calculate and return a new reward.


```python
from gym.envs.classic_control.cartpole import CartPoleEnv

class EasyCart(CartPoleEnv):
    def step(self, action):
        (obs, reward, done, info) = super().step(action)
        
        # YOUR CODE HERE - calculate new reward

        return (obs, new_reward, done, info)

        
gym.envs.registration.register(id="EasyCart-v0", entry_point=EasyCart, max_episode_steps=500)
args = PPOArgs(env_id="EasyCart-v0", use_wandb=True)
# YOUR CODE HERE - train agent
```

<details>
<summary>Solution</summary>


```python
class EasyCart(CartPoleEnv):
    def step(self, action):
        (obs, reward, done, info) = super().step(action)
        x, v, theta, omega = obs

        # First reward: angle should be close to zero
        reward_1 = 1 - abs(theta / 0.2095)
        # Second reward: position should be close to the center
        reward_2 = 1 - abs(x / 2.4)

        return (obs, reward_2, done, info)


gym.envs.registration.register(id="EasyCart-v0", entry_point=EasyCart, max_episode_steps=500)
args = PPOArgs(env_id="EasyCart-v0", use_wandb=True)
agent = train(args)
```
</details>

Now, change the environment such that the reward incentivises the agent to spin very fast. You may change the termination conditions of the environment (i.e. return a different value for `done`) if you think this will help.

<details>
<summary>Solution (one possible implementation)</summary>

```python
class SpinCart(CartPoleEnv):

    def step(self, action):
        obs, rew, done, info = super().step(action)
        # YOUR CODE HERE
        x, v, theta, omega = obs
        # Allow for 360-degree rotation (but keep the cart on-screen)
        done = (abs(x) > self.x_threshold)
        # Reward function incentivises fast spinning while staying still & near centre
        rotation_speed_reward = min(1, 0.1*abs(omega))
        stability_penalty = max(1, abs(x/2.5) + abs(v/10))
        reward = rotation_speed_reward - 0.5 * stability_penalty
        return (obs, reward, done, info)


gym.envs.registration.register(id="SpinCart-v0", entry_point=SpinCart, max_episode_steps=500)
args = PPOArgs(env_id="SpinCart-v0", use_wandb=True)
agent = train(args)
```

</details>

Another thing you can try is "dancing". It's up to you to define what qualifies as "dancing" - work out a sensible definition, and the reward function to incentive it. 

## Bonus


Here are a few bonus exercises. They're ordered (approximately) from easiest to hardest.

### Trust Region Methods

Some versions of the PPO algorithm use a slightly different objective function. Rather than our clipped surrogate objective, they use constrained optimization (maximising the surrogate objective subject to a restriction on the [KL divergence](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence) between the old and new policies). 

$$
\begin{array}{ll}
\underset{\theta}{\operatorname{maximize}} & \hat{\mathbb{E}}_t\left[\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old}}}\left(a_t \mid s_t\right)} \hat{A}_t\right] \\
\text { subject to } & \hat{\mathbb{E}}_t\left[\mathrm{KL}\left[\pi_{\theta_{\text {old}}}\left(\cdot \mid s_t\right), \pi_\theta\left(\cdot \mid s_t\right)\right]\right] \leq \delta
\end{array}
$$

The intuition behind this is similar to the clipped surrogate objective. For our clipped objective, we made sure the model wasn't rewarded for deviating from its old policy beyond a certain point (which encourages small updates). Adding an explicit KL constraint accomplishes something similar, because it forces the model to closely adhere to the old policy. For more on KL-divergence and why it's a principled measure, see [this post](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence). We call these algorithms trust-region methods because they incentivise the model to stay in a **trusted region of policy space**, i.e. close to the old policy (where we can be more confident in our results).

The theory behind TRPO actually suggests the following variant - turning the strict constraint into a penalty term, which you should find easier to implement:

$$
\underset{\theta}{\operatorname{maximize}} \, \hat{\mathbb{E}}_t\left[\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old}}}\left(a_t \mid s_t\right)} \hat{A}_t-\beta \mathrm{KL}\left[\pi_{\theta_{\text {old}}}\left(\cdot \mid s_t\right), \pi_\theta\left(\cdot \mid s_t\right)\right]\right]
$$

Rather than forcing the new policy to stay close to the previous policy, this adds a penalty term which incentivises this behaviour (in fact, there is a 1-1 correspondence between constrained optimization problems and the corresponding unconstrained version).

Can you implement this? Does this approach work better than the clipped surrogate objective? What values of $\beta$ work best?

Tip - you can calculate KL divergence using the PyTorch [KL Divergence function](https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.kl). You could also try the approximate version, as described in [detail #12](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Debug%20variables) of the "37 Implementational Details" post.

### Long-term replay buffer

Above, we discussed the problem of **catastrophic forgetting** (where the agent forgets how to recover from bad behaviour, because the buffer only contains good behaviour). One way to fix this is to have a long-term replay buffer, for instance:

* (simple version) You reserve e.g. 10% of your buffer for experiences generated at the start of training.
* (galaxy-brained version) You design a custom scheduled method for removing experiences from the buffer, so that you always have a mix of old and new experiences.

Can you implement one of these, and does it fix the catastrophic forgetting problem (without needing to use reward shaping)?

### Vectorized Advantage Calculation

Try optimizing away the for-loop in your advantage calculation. It's tricky (and quite messy), so an easier version of this is: find a vectorized calculation and try to explain what it does.

<details>
<summary>Hint (for your own implementation)</summary>

*(Assume `num_envs=1` for simplicity)*

Construct a 2D boolean array from `dones`, where the `(i, j)`-th element of the array tells you whether the expression for the `i`-th advantage function should include rewards / values at timestep `j`. You can do this via careful use of `torch.cumsum`, `torch.triu`, and some rearranging.
</details>

There are solutions available in `solutions.py` (commented out).

### Other Discrete Environments

Two environments (supported by gym) which you might like to try are:

* [`Acrobot-v1`](https://www.gymlibrary.dev/environments/classic_control/acrobot/) - this is one of the [Classic Control environments](https://www.gymlibrary.dev/environments/classic_control/), and it's a bit harder to learn than cartpole.
* [`MountainCar-v0`](https://www.gymlibrary.dev/environments/classic_control/mountain_car/) - this is one of the [Classic Control environments](https://www.gymlibrary.dev/environments/classic_control/), and it's much harder to learn than cartpole. This is primarily because of **sparse rewards** (it's really hard to get to the top of the hill), so you'll definitely need reward shaping to get through it!
* [`LunarLander-v2`](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) - this is part of the [Box2d](https://www.gymlibrary.dev/environments/box2d/) environments. It's a bit harder still, because of the added environmental complexity (physics like gravity and friction, and constraints like fuel conservatino). The reward is denser (with the agent receiving rewards for moving towards the landing pad and penalties for moving away or crashing), but the increased complexity makes it overall a harder problem to solve. You might have to perform hyperparameter sweeps to find the best implementation (you can go back and look at the syntax for hyperparameter sweeps [here](https://arena-ch0-fundamentals.streamlit.app/[0.4]_Optimization)). Also, [this page](https://pylessons.com/LunarLander-v2-PPO) might be a useful reference (although the details of their implementation differs from the one we used today). You can look at the hyperparameters they used.

### Minigrid envs / Procgen

There are many more exciting environments to play in, but generally they're going to require more compute and more optimization than we have time for today. If you want to try them out, some we recommend are:

- [Minimalistic Gridworld Environments](https://github.com/Farama-Foundation/gym-minigrid) - a fast gridworld environment for experiments with sparse rewards and natural language instruction.
- [microRTS](https://github.com/santiontanon/microrts) - a small real-time strategy game suitable for experimentation.
- [Megastep](https://andyljones.com/megastep/) - RL environment that runs fully on the GPU (fast!)
- [Procgen](https://github.com/openai/procgen) - A family of 16 procedurally generated gym environments to measure the ability for an agent to generalize. Optimized to run quickly on the CPU.
    - For this one, you might want to read [Jacob Hilton's online DL tutorial](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/6-Reinforcement-Learning.md) (the RL chapter suggests implementing PPO on Procgen), and [Connor Kissane's solutions](https://github.com/ckkissane/deep_learning_curriculum/blob/master/solutions/6_Reinforcement_Learning.ipynb).

### Continuous Action Spaces

The `MountainCar-v0` environment has discrete actions, but there's also a version `MountainCarContinuous-v0` with continuous action spaces. Unlike DQN, PPO can handle continuous actions with minor modifications. Try to adapt your agent; you'll need to handle `gym.spaces.Box` instead of `gym.spaces.Discrete` and make note of the ["9 details for continuous action domains"](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=9%20details%20for%20continuous%20action%20domains) section of the reading.

### [Atari](https://www.gymlibrary.dev/environments/atari/)

The [37 Implementational Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=9%20Atari%2Dspecific%20implementation%20details) post describes how to get PPO working for games like Atari. You've already done a lot of the work here! Try to implement the remaining details and get PPO working on Atari. You'll need to read the Atari-specific implementation details section in the post (and you'll also have to spend some time working with a different library to `gym`). This could be a good capstone project, or just an extended project for the rest of the RL section (after we do RLHF) if you want to get more challenging hands-on engineering experience!

[This post](https://towardsdatascience.com/a-graphic-guide-to-implementing-ppo-for-atari-games-5740ccbe3fbc) might also be useful - it discusses Atari & PPO at a high level, as well as diving into a few technical details about the Atari environment. 

We recommend you focus on some of the easier games, like breakout or sequest. Save [Montezuma's Revenge](https://paperswithcode.com/task/montezumas-revenge) for later!

### Multi-Agent PPO

Multi-Agent PPO (MAPPO) is an extension of the standard PPO algorithm which trains multiple agents at once. It was first described in the paper [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955). Can you implement MAPPO?

Even if you choose not to implement it, you may wish to read some of the material relevant for MAPPO, including [Learning to Deceive in Multi-Agent Hidden Role Games](https://arxiv.org/abs/2209.01551), a paper written by Matthew Aitchison, who will be giving a talk on Tuesday evening.

""", unsafe_allow_html=True)


func_page_list = [
    (section_0, "🏠 Home"),
    (section_1, "1️⃣ Setting up our agent"),
    (section_2, "2️⃣ Learning Phase"),
    (section_3, "3️⃣ Training Loop"), 
]

func_list = [func for func, page in func_page_list]
page_list = [page for func, page in func_page_list]

page_dict = dict(zip(page_list, range(len(page_list))))

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    idx = page_dict[radio]
    func = func_list[idx]
    func()

page()
