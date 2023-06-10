
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
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/football.jpeg" width="350">


Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.


# [2.3] - PPO


## Introduction


This section is designed to get you familiar with basic neural networks: how they are structured, the basic operations like linear layers and convolutions which go into making them, and why they work as well as they do. You'll be using libraries like `einops`, and functions like `torch.as_strided` to get a very low-level picture of how these operations work, which will help build up your overall understanding.


## Content & Learning Objectives


#### 1️⃣ PPO: Introduction

> ##### Learning objectives
>


#### 2️⃣ PPO: Rollout



> ##### Learning objectives
>


#### 3️⃣ PPO: Learning


> ##### Learning objectives
> 


#### 4️⃣ PPO: Full Algorithm


## Setup


```python
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
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import wandb

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
# import part3_ppo.solutions as solutions
from plotly_utils import plot_cartpole_obs_and_dones

for idx, probe in enumerate([Probe1, Probe2, Probe3, Probe4, Probe5]):
    gym.envs.registration.register(id=f"Probe{idx+1}-v0", entry_point=probe)

Arr = np.ndarray

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

```



""", unsafe_allow_html=True)


def section_1():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#notes-on-today's-workflow'>Notes on today's workflow</a></li>
    <li class='margtop'><a class='contents-el' href='#readings'>Readings</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#optional-reading'>Optional Reading</a></li>
        <li><a class='contents-el' href='#references-not-required-reading'>References (not required reading)</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#ppo-overview'>PPO Overview</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#diagram-overview'>Diagram overview</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#ppo-arguments'>PPO Arguments</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 1️⃣ PPO: Introduction


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
    * You might also prefer reading the section **1️⃣ PPO: Background** instead.
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


## PPO Overview

The diagram below shows everything you'll be implementing today. Don't worry if this seems overwhelming - we'll be breaking it down into bite-size pieces as we go through the exercises. You might find it helpful to refer back to this diagram as we proceed, although I'll give a quick overview now. Don't worry if you don't follow all of it straight away.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/ppo-alg-2.png" width="850">


### Diagram overview

We start by initializing the environment, agent, and memory just like we've done previously. The rest of the diagram is mainly split into two sections: the **rollout phase** and the **learning phase**.

In the **rollout phase**, the agent generates a number of trajectories (episodes) by interacting with the environment. Each trajectory is a sequence of states, actions, and rewards. The agent also records the log probability of the actions it took, and the critic network's estimate of the value at that particular state, all of which we'll need for the policy gradient update. 

These variables are logged to an object called **memory**. You have a choice of how to implement the memory object - for instance, you could store it as a dataclass (as the solutions have done) or just a dictionary to keep things simple. You will also need to calculate the **advantages** (see the section on generalized advantage estimation later, or in the reading material).

We repeat the rollout phase until `num_steps` experiences have been logged to memory in total, then move on to the **learning phase**. This mainly consists of randomly sampling the experiences in memory (which we call **minibatches**) and updating the agent's policy and value networks. To connect this to the [PPO paper](https://arxiv.org/pdf/1707.06347.pdf), our old sampled experiences are denoted by $\theta_\text{old}$, and our new parameters (which we update) are $\theta$. In other words, all the yellow objects in memory are generated using $\theta_\text{old}$, and the blue `probs` and `values` are generated by $\theta$. The actor network is updated using the policy gradient update and entropy bonus, and the value network is updated using the TD error. Once these updates are finished, we clear memory, go back to the start, and repeat the process until `total_timesteps`.

We can also see an **agent** box in the diagram, containing the modules **actor** and **critic** as well as the methods **rollout** and **learn**. Again, this is just one suggested implementation of PPO, but you are free to implement it however you like (e.g. having these phases be functions, or loops in a larger function, rather than class methods for the agent).


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
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 512
    minibatch_size: int = 128

    def __post_init__(self):
        self.total_training_steps = self.total_timesteps // self.num_steps



if MAIN:
    args = PPOArgs(minibatch_size=256)
    utils.arg_help(args)

```



""", unsafe_allow_html=True)


def section_2():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#actor-critic-agent-implementation'>Actor-Critic Agent Implementation (detail #2)</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-get-actor-and-critic'><b>Exercise</b> - implement <code>get_actor_and_critic</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#generalized-advantage-estimation'>Generalized Advantage Estimation (detail #5)</a></li>
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

# 2️⃣ PPO: Rollout


## Actor-Critic Agent Implementation ([detail #2](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Orthogonal%20Initialization%20of%20Weights%20and%20Constant%20Initialization%20of%20biases))

Implement the `Agent` class according to the diagram, inspecting `envs` to determine the observation shape and number of actions. We are doing separate Actor and Critic networks because [detail #13](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Shared%20and%20separate%20MLP%20networks%20for%20policy%20and%20value%20functions) notes that is performs better than a single shared network in simple environments. 

Note that today `envs` will actually have multiple instances of the environment inside, unlike yesterday's DQN which had only one instance inside. From the **37 implementation details** post:

> In this architecture, PPO first initializes a vectorized environment `envs` that runs $N$ (usually independent) environments either sequentially or in parallel by leveraging multi-processes. `envs` presents a synchronous interface that always outputs a batch of $N$ observations from $N$ environments, and it takes a batch of $N$ actions to step the $N$ environments. When calling `next_obs = envs.reset()`, next_obs gets a batch of $N$ initial observations (pronounced "next observation"). PPO also initializes an environment `done` flag variable next_done (pronounced "next done") to an $N$-length array of zeros, where its i-th element `next_done[i]` has values of 0 or 1 which corresponds to the $i$-th sub-environment being *not done* and *done*, respectively.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/screenshot-2.png" width="800">


### Exercise - implement `get_actor_and_critic`

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠🟠🟠⚪

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


if MAIN:
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
        # nn.Flatten(),
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
    rewards: shape (T, env)
    values: shape (T, env)
    dones: shape (T, env)
    Return: shape (T, env)
    '''
    pass


if MAIN:
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
    rewards: shape (T, env)
    values: shape (T, env)
    dones: shape (T, env)
    Return: shape (T, env)
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


```python
def minibatch_indexes(rng: Generator, batch_size: int, minibatch_size: int) -> List[np.ndarray]:
    '''Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    assert batch_size % minibatch_size == 0
    pass


if MAIN:
    tests.test_minibatch_indexes(minibatch_indexes)

```

<details>
<summary>Solution</summary>


```python
def minibatch_indexes(rng: Generator, batch_size: int, minibatch_size: int) -> List[np.ndarray]:
    '''Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

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

It will also need to store the terms $s_T, d_T$ and $V_\pi(s_T)$ (i.e. the terms at the end of each episode). **Exercise - can you see why we need all three of these three terms?**

<details>
<summary>Answer</summary>

We need $s_T$ so that we can "pick up where we left off" in the next rollout phase. If we reset each time then we'd never have episodes longer than `arg.num_steps` steps. The default value for `arg.num_steps` is 128, which is smaller than the 500 step maximum for CartPole.

We need $d_T$ and $V_\pi(s_t)$ so that we can calculate the advantages in the GAE section.
</details>

The `add` method works the same as for the DQN implementation - the only difference is that we also add `logprobs` and `values` to the buffer (these are also calculated during rollout phase - see the next exercise). Slightly different variables are returned by the `sample` method (you can see which ones in the `ReplayBufferSamples` class).

You should read through the code for this class, and make sure you understand what it's all doing. In particular, take note of the following:

* In the `sample` method, we flatten each tensor over the first two dimensions. This is because the first is the "buffer dimension" and the second is the "num environments" dimension. From [detail #6](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Mini%2Dbatch%20Updates), we know that we should be flattening over this dimension when we do mini-batch updates.
* In the `sample` method, we compute our advantages and returns. The advantages are computed via the function we defined above; the returns (i.e. the return target for our critic network) are computed as `returns = advantages + values`, in accordance with [detail #5](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Generalized%20Advantage%20Estimation).



### Exercise - explain the values in `ReplayBufferSamples`

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠🟠🟠⚪

You should spend up to 10-25 minutes on this exercise.

Understanding this is very conceptually important.
```

**Read the [PPO Algorithms paper](https://arxiv.org/pdf/1707.06347.pdf) and the [PPO Implementational Details post](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#solving-pong-in-5-minutes-with-ppo--envpool), then try and figure out what each of the six items in `ReplayBufferSamples` are and why they are necessary.** If you prefer, you can return to these once you've implmemented all the loss functions (when it might make a bit more sense).

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

These are necessary for calculating the clipped surrogate objective (see equation $(7)$ on page page 3 in the [PPO Algorithms paper](https://arxiv.org/pdf/1707.06347.pdf)), which is the thing we've called `policy_loss` in this page.

`logprobs` correspond to the term $\pi_{\theta_\text{old}}(a_t | s_t)$ in this equation. $\pi_{\theta}(a_t | s_t)$ corresponds to the output of our `actor.agent` network **which changes as we perform gradient updates on it.**

</details>

<details>
<summary>advantages</summary>

`advantages` are the terms $\hat{A}_t$ used in the calculation of policy loss (again, see equation $(7)$ in the PPO algorithms paper). They are computed using the formula $(11)$ in the paper.

</details>

<details>
<summary>returns</summary>

We mentioned above that `returns = advantages + values`. They are used for calculating the value function loss - see [detail #9](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Value%20Function%20Loss%20Clipping) in the PPO implementational details post.

</details>

<details>
<summary>values</summary>

`values` are the outputs of our `agent.critic` network.

They are required for calculating `advantages`, in our clipped surrogate objective function (see equation $(7)$ on page page 3 in the [PPO Algorithms paper](https://arxiv.org/pdf/1707.06347.pdf)).

</details>


```python
@dataclass
class ReplayBufferSamples:
    '''
    Samples from the replay buffer, converted to PyTorch for use in neural network training.
    '''
    obs: Float[Tensor, "sampleSize *obsShape"]
    dones: Float[Tensor, "sampleSize"]
    actions: Int[Tensor, "sampleSize"]
    logprobs: Float[Tensor, "sampleSize"]
    values: Float[Tensor, "sampleSize"]
    advantages: Float[Tensor, "sampleSize"]
    returns: Float[Tensor, "sampleSize"]


class ReplayBuffer:
    '''
    Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.

    Needs to be initialized with the first obs, dones and values.
    '''
    rng: Generator

    def __init__(
        self,
        args: PPOArgs,
        num_environments: int,
    ):
        self.num_environments = num_environments
        self.rng = np.random.default_rng(args.seed)
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.batch_size = args.batch_size
        self.minibatch_size = args.minibatch_size
        self.num_steps = args.num_steps
        self.update_epochs = args.update_epochs
        self.experiences = []
        self.minibatches = []

    def add(self, obs: Arr, actions: Arr, rewards: Arr, dones: Arr, logprobs: Arr, values: Arr) -> None:
        '''
        obs: shape (num_environments, *observation_shape) 
            Observation before the action
        actions: shape (num_environments,) 
            Action chosen by the agent
        rewards: shape (num_environments,) 
            Reward after the action
        dones: shape (num_environments,) 
            If True, the episode ended and was reset automatically
        next_obs: shape (num_environments, *observation_shape) 
            Observation after the action
            If done is True, this should be the terminal observation, NOT the first observation of the next episode.
        logprobs: shape (num_environments,)
            Log probability of the action that was taken (according to old policy)
        values: shape (num_environments,)
            Values, estimated by the critic (according to old policy)
        '''
        assert obs.shape[0] == self.num_environments
        assert actions.shape == (self.num_environments,)
        assert rewards.shape == (self.num_environments,)
        assert dones.shape == (self.num_environments,)
        assert logprobs.shape == (self.num_environments,)
        assert values.shape == (self.num_environments,)

        self.experiences.append((obs, dones, actions, logprobs, values, rewards))


    def get_minibatches(self, next_value: t.Tensor, next_done: t.Tensor):

        obs, dones, actions, logprobs, values, rewards = [t.stack(arr).to(device) for arr in zip(*self.experiences)]

        self.minibatches = []
        for _ in range(self.update_epochs):
            indices = minibatch_indexes(self.rng, self.batch_size, self.minibatch_size)
            advantages = compute_advantages(next_value, next_done, rewards, values, dones.float(), self.gamma, self.gae_lambda)
            returns = advantages + values
            replaybuffer_args = [obs, dones, actions, logprobs, values, advantages, returns]
            self.minibatches.extend([
                ReplayBufferSamples(*[arg.flatten(0, 1)[index].to(device) for arg in replaybuffer_args])
                for index in indices
            ])

```

Now, like before, here's some code to generate and plot observations. Note that we're actually using four environments inside our `envs` object, rather than just one like last time. The 3 thick lines in the first plot below indicate the transition between different environments in `envs` (which we've stitched together into one long episode).


```python

if MAIN:
    args = PPOArgs()
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", i, i, False, "test") for i in range(4)])
    next_value = t.zeros(envs.num_envs).to(device)
    next_done = t.zeros(envs.num_envs).to(device)
    rb = ReplayBuffer(args, envs.num_envs)
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
    
    plot_cartpole_obs_and_dones(obs.flip(0), dones.flip(0), show_env_jumps=True)

```

```python

if MAIN:
    rb.get_minibatches(next_value, next_done)
    
    obs = rb.minibatches[0].obs
    dones = rb.minibatches[0].dones
    
    plot_cartpole_obs_and_dones(obs.flip(0), dones.flip(0))

```

## PPOAgent

As the final task in this section, you should fill in the agent's `play_step` method. This is conceptually similar to what you did during DQN, but with a few key differences.

In DQN, we did the following:

* used the Q-Network and an epsilon greedy policy to select an action based on current observation,
* stepped the environment with this action,
* stored the transition in the replay buffer.

In PPO, we do the following:

* use the actor network to return a distribution over actions based on current observation,
* sample from this distribution to select an action,
* calculate the logprobs of the selected action (according to the returned distribution),
* use the critic to estimate the value of the current observation,
* store the transition in the replay buffer (same variables as in DQN, but with the addition of the logprobs and value estimate).


### Exercise - implement `PPOAgent`

```c
Difficulty: 🟠🟠🟠🟠⚪
Importance: 🟠🟠🟠🟠⚪

You should spend up to 20-35 minutes on this exercise.
```

A few gotchas:

* Make sure the things you're adding to the buffer are the right shape. This includes paying attention to the batch dimension when you put things through the actor and critic networks.
* Don't forget to use inference mode when running your actor and critic networks.
* Don't forget to increment the step count.
* At the end of `play_step`, you should set the `next_obs` and `next_done` to be the values returned by the environment (i.e. so the agent always knows where it is, for the next time it's called).

<details>
<summary>Tip - how to sample from distributions</summary>

You might remember using `torch.distributions.categorical.Categorical` when we were sampling from transformers in the previous chapter. We can use this again!

You can define a `Categorical` object by passing in `logits` (the output of the actor network), and then you can:

* Sample from it using the `sample` method,
* Calculate the logprobs of a given action using the `log_prob` method (with the actions you took as input argument to this method).
</details>

An additional note - for this exercise and others to follow, there's a trade-off in the test functions between being strict and being lax. Too lax and the tests will let failures pass; too strict and they might fail for odd reasons even if your code is mostly correct. If you find youself continually failing tests then you should ask a TA for help.


```python
class PPOAgent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, args: PPOArgs, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        self.args = args
        self.envs = envs
        self.obs_shape = envs.single_observation_space.shape
        self.num_obs = np.array(self.obs_shape).prod()
        self.num_actions = envs.single_action_space.n
        self.next_obs = None
        self.next_done = None
        self.next_value = None

        self.steps = 0
        self.actor, self.critic = get_actor_and_critic(envs)

        self.rb = ReplayBuffer(args, envs.num_envs)
        self.reset()

    def play_step(self) -> List[dict]:
        '''
        Carries out a single interaction step between the agent and the environment, and adds results to the replay buffer.
        '''
        pass

    def reset(self) -> None:
        '''
        Resets the agent's memories (except current state, because this will roll over into next batch of memories).
        '''
        self.rb.experiences = []
        if self.next_obs is None:
            self.next_obs = t.tensor(self.envs.reset()).to(device)
            self.next_done = t.zeros(self.envs.num_envs).to(device, dtype=t.float)
            with t.inference_mode():
                self.next_value = self.critic(self.next_obs).flatten()


# tests.test_ppo_agent(PPOAgent)

```



""", unsafe_allow_html=True)


def section_3():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#objective-function'>Objective function</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#clipped-surrogate-objective'>Clipped Surrogate Objective</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#value-function-loss'>Value Function Loss (detail #9)</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-calc-value-function-loss'><b>Exercise</b> - implement <code>calc_value_function_loss</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#entropy-bonus'>Entropy Bonus (detail #10)</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-calc-entropy-bonus'><b>Exercise</b> - implement <code>calc_entropy_bonus</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#adam-optimizer-and-scheduler-details-#125-and-#10'>Adam Optimizer and Scheduler (details #3 and #4)</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 3️⃣ PPO: Learning


Now, we'll turn to the learning phase. Firstly, we'll work on computing our objective function. This is given by equation $(9)$ in the paper, and is the sum of three terms - we'll implement each term individually.


Note - the convention we've used in these exercises for signs is that **your function outputs should be the expressions in equation $(9)$**, in other words you will compute $L_t^{CLIP}(\theta)$, $c_1 L_t^{VF}(\theta)$ and $c_2 S[\pi_\theta](s_t)$. You can then either perform gradient descent on the **negative** of the expression in $(9)$, or perform **gradient ascent** on the expression by passing `maximize=True` into your Adam optimizer when you initialise it.


## Objective function


### Clipped Surrogate Objective

For each minibatch, calculate $L^{CLIP}$ from equation $(7)$ in the paper. We will refer to this function as `policy_loss`. This will allow us to improve the parameters of our actor.

Note - in the paper, don't confuse $r_{t}$ which is reward at time $t$ with $r_{t}(\theta)$, which is the probability ratio between the current policy (output of the actor) and the old policy (stored in `mb_logprobs`).

Pay attention to the normalization instructions in [detail #7](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Normalization%20of%20Advantages) when implementing this loss function. They add a value of `eps = 1e-8` to the denominator to avoid division by zero, you should also do this.

You can use the `probs.log_prob` method to get the log probabilities that correspond to the actions in `mb_action`.

Note - if you're wondering why we're using a `Categorical` type rather than just using `log_prob` directly, it's because we'll be using them to sample actions later on in our `train_ppo` function. Also, categoricals have a useful method for returning the entropy of a distribution (which will be useful for the entropy term in the loss function).


```python
def calc_clipped_surrogate_objective(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float, eps: float = 1e-8
) -> t.Tensor:
    '''Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs: a distribution containing the actor's unnormalized logits of shape (minibatch, num_actions)

    clip_coef: amount of clipping, denoted by epsilon in Eq 7.
    '''
    pass


if MAIN:
    tests.test_calc_clipped_surrogate_objective(calc_clipped_surrogate_objective)

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
        self.obs_shape = envs.single_observation_space.shape
        self.num_obs = np.array(self.obs_shape).prod()
        self.num_actions = envs.single_action_space.n
        self.next_obs = None
        self.next_done = None
        self.next_value = None

        self.steps = 0
        self.actor, self.critic = get_actor_and_critic(envs)

        self.rb = ReplayBuffer(args, envs.num_envs)
        self.reset()

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
        self.steps += 1

        return infos
    
    def reset(self) -> None:
        '''
        Resets the agent's memories (except current state, because this will roll over into next batch of memories).
        '''
        self.rb.experiences = []
        if self.next_obs is None:
            self.next_obs = t.tensor(self.envs.reset()).to(device)
            self.next_done = t.zeros(self.envs.num_envs).to(device, dtype=t.float)
            with t.inference_mode():
                self.next_value = self.critic(self.next_obs).flatten()


# tests.test_ppo_agent(PPOAgent)

def calc_clipped_surrogate_objective(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float, eps: float = 1e-8
) -> t.Tensor:
    '''Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs: a distribution containing the actor's unnormalized logits of shape (minibatch, num_actions)

    clip_coef: amount of clipping, denoted by epsilon in Eq 7.
    '''
    # SOLUTION
    logits_diff = probs.log_prob(mb_action) - mb_logprobs
    assert logits_diff.shape == mb_advantages.shape, "Shape mismatch between logits_diff and mb_advantages"

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
def calc_value_function_loss(values: t.Tensor, mb_returns: t.Tensor, vf_coef: float) -> t.Tensor:
    '''Compute the value function portion of the loss function.

    vf_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    assert values.shape == mb_returns.shape, "Shape mismatch between values and returns"
    pass


if MAIN:
    tests.test_calc_value_function_loss(calc_value_function_loss)

```

<details>
<summary>Solution</summary>


```python
def calc_value_function_loss(values: t.Tensor, mb_returns: t.Tensor, vf_coef: float) -> t.Tensor:
    '''Compute the value function portion of the loss function.

    vf_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    assert values.shape == mb_returns.shape, "Shape mismatch between values and returns"
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

The minimum entropy is $\ln(2) \approx 0.693$ under the uniform random policy over the 2 actions.
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

    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall objective function. Denoted by c_2 in the paper.
    '''
    pass


if MAIN:
    tests.test_calc_entropy_bonus(calc_entropy_bonus)

```

<details>
<summary>Solution</summary>


```python
def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''Return the entropy bonus term, suitable for gradient ascent.

    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall objective function. Denoted by c_2 in the paper.
    '''
    # SOLUTION
    return ent_coef * probs.entropy().mean()
```
</details>


## Adam Optimizer and Scheduler (details #3 and #4)

Even though Adam is already an adaptive learning rate optimizer, empirically it's still beneficial to decay the learning rate.

Implement a linear decay from `initial_lr` to `end_lr` over num_updates steps. Also, make sure you read details #3 and #4 so you don't miss any of the Adam implementational details.

Note, the training terminates after `num_updates`, so you don't need to worry about what the learning rate will be after this point.

Remember to pass the parameter `maximize=True` into Adam, if you defined the loss functions in the way we suggested above.


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
        self.n_step_calls += 1
        frac = self.n_step_calls / self.total_training_steps
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)


def make_optimizer(agent: PPOAgent, total_training_steps: int, initial_lr: float, end_lr: float) -> tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
        pass


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
        self.n_step_calls += 1
        frac = self.n_step_calls / self.total_training_steps
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)


def make_optimizer(agent: PPOAgent, total_training_steps: int, initial_lr: float, end_lr: float) -> tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    # SOLUTION
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_training_steps)
    return (optimizer, scheduler)
```
</details>




""", unsafe_allow_html=True)


def section_4():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#learning-function'>Learning function</a></li>
    <li class='margtop'><a class='contents-el' href='#reward-shaping'>Reward Shaping</a></li>
    <li class='margtop'><a class='contents-el' href='#bonus'>Bonus</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#continuous-action-spaces'>Continuous Action Spaces</a></li>
        <li><a class='contents-el' href='#vectorized-advantage-calculation'>Vectorized Advantage Calculation</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 4️⃣ PPO: Full Algorithm


## Learning function

Finally, we can package this all together into our full training loop. 

Most of this has already been provided for you. We have the following methods:

* `__init__` - there's much less code here than was requried for your DQN implementation, because most of the complexity is in the initialisation functions for our PPO Agent.
* `_log`, `configure_optimizers`, `on_train_epoch_end`, `train_dataloader` - these are all basically the same as last time.
* `training_step` - this function gets called once per gradient update step. An explanation of the code here:
    * It checks whether there are any remaining minibatches in the agent's memory.
    * If not, it does the following:
        * Steps the schedulers (i.e. updates the optimizer's learning rate) since this happens at the end of every batch
        * Calls the `rollout_phase` method to repopulate the agent's memory
    * After this, there will definitely be minibatches stored in memory, so we pop out the first minibatch and use our `learning_phase` method to return our objective function (which we perform a gradient update step on).

You only have two functions to implement - `rollout_phase` and `learning_phase`. These should do the following:

#### `rollout_phase`

Reset the agent's memory, and repopulate it with `num_steps` new experiences. You should also populate `agent.rb.minibatches` with the sampled minibatches of data from these experiences (this can be done with the inplace method `agent.rb.sample_minibatches` - see the code which produced our graphs earlier).

*The solution is 4 lines of code (not including logging or comments).*

#### `learning_phase`

Calculate the three different terms that go into the objective function, and return the total objective function.

*The solution is 8 lines of code (not including logging or comments).*

#### Logging

You should only focus on logging once you've got a mininal version of the code working. Once you do, you can try logging variables in the way described by [detail #12](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Debug%20variables). This will involve adding code to the `rollout_phase` and `learning_phase` methods.



Again, you can implement this as a method for your agent (the suggested option) or in any other way you like.

Your function will need to do the following:
- For `n = 1, 2, ..., args.update_epochs`, you should:
    - Use previously-written functions to make minibatches from the data stored in `memory`
    - For each minibatch:
        - Use your `agent.actor` to calculate new probabilities $\pi_\theta(a_t \mid s_t)$
        - Use these (and the data in memory) to calculate your objective function
        - Perform a gradient ascent step on your total objective function
            - Here, you should also clip gradients, in the way suggested by [detail #11](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=Global%20Gradient%20Clipping)
- Step your scheduler
- Take the last minibatch, and log variables for debugging in accordance with  (see the next section for more on this explanation. You can skip this point for now until the rest of your code is working)


```python
class DQNLightning(pl.LightningModule):
    agent: PPOAgent

    def __init__(self, args: PPOArgs):
        super().__init__()
        self.args = args
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, self.run_name) for i in range(args.num_envs)]
        )
        self.agent = PPOAgent(self.args, self.envs)


    def rollout_phase(self):
        all_infos = []
        self.agent.reset()
        for step in range(self.args.num_steps):
            infos = self.agent.play_step()
            all_infos.extend(infos)
        for info in all_infos:
            if "episode" in info.keys():
                self.log("episodic_return", info["episode"]["r"])
                self.log("episodic_length", info["episode"]["l"])
                break


    def learning_phase(self, mb: ReplayBufferSamples):
        logits = self.agent.actor(mb.obs)
        probs = Categorical(logits=logits)
        # print(self.agent.critic[0].weight.pow(2).sum().item())
        values = self.agent.critic(mb.obs).squeeze()
        self.log("values", values.mean().item())
        clipped_surrogate_objective = calc_clipped_surrogate_objective(probs, mb.actions, mb.advantages, mb.logprobs, self.args.clip_coef)
        # print(values, mb.returns)
        value_loss = calc_value_function_loss(values, mb.returns, self.args.vf_coef)
        entropy_bonus = calc_entropy_bonus(probs, self.args.ent_coef)
        total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus

        with t.inference_mode():
            newlogprob = probs.log_prob(mb.actions)
            logratio = newlogprob - mb.logprobs
            ratio = logratio.exp()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        self.log_dict(dict(
            learning_rate = self.optimizers().param_groups[0]["lr"],
            value_loss = value_loss.item(),
            clipped_surrogate_objective = clipped_surrogate_objective.item(),
            entropy = entropy_bonus.item(),
            approx_kl = approx_kl,
            clipfrac = np.mean(clipfracs)
        ))

        return total_objective_function


    def training_step(self, batch: Any, batch_idx) -> Float[Tensor, ""]:
        '''Handles rollout and learning phases. Returns objective function to be maximized.'''

        if not self.agent.rb.minibatches:

            # last_roll = getattr(self, "last_roll", None)
            # if last_roll is not None:
            #     print(f"Rollout took {time.time() - last_roll} seconds")
            # self.last_roll = time.time()
            # Step our scheduler once
            self.scheduler.step()
            # print("about to roll")
            # print("rolling")
            # Fill memory (by stepping agent through environment)
            self.rollout_phase()
            # Get minibatches from memory
            self.agent.rb.get_minibatches(self.agent.next_value, self.agent.next_done)
            # print("rolled")
        # print("learning")
            
        # Get the next minibatch
        minibatch = self.agent.rb.minibatches.pop()

        # Compute the objective function from this minibatch
        total_objective_function = self.learning_phase(minibatch)

        return total_objective_function
    
    

    def configure_optimizers(self):
        optimizer, scheduler = make_optimizer(self.agent, self.args.total_training_steps, self.args.learning_rate, 0.0)
        self.scheduler = scheduler 
        return optimizer
    

    def on_train_epoch_end(self):
        obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
        expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]], [[args.gamma], [1.0]], [[1.0]], [[1.0], [1.0]]]
        expected_probs_for_probes = [None, None, None, [[0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
        tolerances = [5e-4, 5e-4, 5e-4, 1e-3, 1e-3]
        match = re.match(r"Probe(\d)-v0", args.env_id)
        if match:
            probe_idx = int(match.group(1)) - 1
            obs = t.tensor(obs_for_probes[probe_idx]).to(device)
            # print("Obs: ", obs)
            # print("shape = ", obs.shape)
            with t.inference_mode():
                value = self.agent.critic(obs)
                probs = self.agent.actor(obs).softmax(-1)
            print("Value: ", value)
            expected_value = t.tensor(expected_value_for_probes[probe_idx]).to(device)
            # print("Expected value: ", expected_value)
            t.testing.assert_close(value, expected_value, atol=tolerances[probe_idx], rtol=0)
            expected_probs = t.tensor(expected_probs_for_probes[probe_idx]).to(device)
            if expected_probs is not None:
                t.testing.assert_close(probs, expected_probs, atol=tolerances[probe_idx], rtol=0)
            print("Probe tests passed!")
        self.envs.close()


    def train_dataloader(self):
        '''We don't use a trainloader in the traditional sense, so we'll just have this.'''
        return range(self.args.total_training_steps)

```

<details>
<summary>Solution</summary>


</details>

Here's some code to run your model on the probe environments (and assert that they're all working fine):


```python

if MAIN:
    probe_idx = 5
    
    args = PPOArgs(
        env_id=f"Probe{probe_idx}-v0",
        exp_name=f"test-probe-{probe_idx}", 
        total_timesteps=20000 if probe_idx <= 3 else 50000,
        learning_rate=0.001,
        capture_video=False,
        use_wandb=False,
    )
    model = DQNLightning(args).to(device)
    logger = CSVLogger(save_dir=args.log_dir, name=model.run_name)
    
    trainer = pl.Trainer(
        max_steps=args.total_training_steps,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=args.max_grad_norm,
    )
    trainer.fit(model=model)
    
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    px.line(metrics, y="values", labels={"x": "Step"}, title="Probe 1 (if you're seeing this, then you passed the tests!)", width=600, height=400)

```

```python

if MAIN:
    wandb.finish()
    
    args = PPOArgs()
    logger = WandbLogger(save_dir=args.log_dir, project=args.wandb_project_name, name=model.run_name)
    if args.use_wandb: wandb.gym.monitor() # Makes sure we log video!
    model = DQNLightning(args).to(device)
    
    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=args.total_training_steps,
        logger=logger,
        log_every_n_steps=10,
    )
    trainer.fit(model=model)

```

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
</details>

You could also try using e.g. $|\theta / \theta_{\text{max}}|$ rather than $(\theta / \theta_{\text{max}})^2$. This would still mean reward is in the range (0, 1), but it would result in a larger penalty for very small deviations from the vertical position.

You can also try a linear combination of two or more of these rewards!

<details>
<summary>Help - my agent's episodic return is smaller than it was in the original CartPole environment.</summary>

This is to be expected, because your reward function is no longer always 1 when the agent is upright. Both your time-discounted reward estimates and your actual realised rewards will be less than they were in the cartpole environment. 

For a fairer test, measure the length of your episodes - hopefully your agent learns how to stay upright for the entire 500 timestep interval as fast as or faster than it did previously.
</details>


```python
class EasyCart(CartPoleEnv):
    def step(self, action):
        (obs, rew, done, info) = super().step(action)

        x, v, theta, omega = obs

        # First reward: angle should be close to zero
        reward_1 = 1 - abs(theta / 0.2095)
        # Second reward: position should be close to the center
        reward_2 = 1 - abs(x / 2.4)

        # reward = 0.3 * reward_1 + 0.7 * reward_2
        reward = reward_2

        return (obs, reward, done, info)

```

```python

if MAIN:
    gym.envs.registration.register(id="EasyCart-v0", entry_point=EasyCart, max_episode_steps=500)
    
    args = PPOArgs(env_id="EasyCart-v0", gamma=0.995, total_timesteps = 1_000_000)
    
    wandb.finish()
    
    logger = WandbLogger(save_dir=args.log_dir, project=args.wandb_project_name, name=model.run_name)
    if args.use_wandb: wandb.gym.monitor() # Makes sure we log video!
    model = DQNLightning(args).to(device)
    
    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=args.total_timesteps,
        logger=logger,
        log_every_n_steps=10,
    )
    trainer.fit(model=model)

```

<details>
<summary>One possible solution</summary>

```python
class EasyCart(CartPoleEnv):
    def step(self, action):
        (obs, rew, done, info) = super().step(action)

        x, v, theta, omega = obs

        # First reward: angle should be close to zero
        reward_1 = 1 - abs(theta / 0.2095)
        # Second reward: position should be close to the center
        reward_2 = 1 - abs(x / 2.4)

        reward = 0.3 * reward_1 + 0.7 * reward_2

        return (obs, reward, done, info)
```

if MAIN:
    gym.envs.registration.register(id="EasyCart-v0", entry_point=EasyCart, max_episode_steps=500)
    args = PPOArgs()
    args.env_id = "EasyCart-v0"
    # args.track = False
    args.gamma = 0.995
    train_ppo(args)
</details>


Now, change the environment such that the reward incentivises the agent to "dance".

It's up to you to define what qualifies as "dancing". Work out a sensible definition, and the reward function to incentive it. You may change the termination conditions of the environment if you think it helps teaching the cart to dance.


## Bonus


### Continuous Action Spaces

The `MountainCar-v0` environment has discrete actions, but there's also a version `MountainCarContinuous-v0` with continuous action spaces. Unlike DQN, PPO can handle continuous actions with minor modifications. Try to adapt your agent; you'll need to handle `gym.spaces.Box` instead of `gym.spaces.Discrete` and make note of the "9 details for continuous action domains" section of the reading.



### Vectorized Advantage Calculation

Try optimizing away the for-loop in your advantage calculation. It's tricky, so an easier version of this is: find a vectorized calculation and try to explain what it does.

There are solutions available in `solutions.py` (commented out).


```python
from gym.envs.classic_control.cartpole import CartPoleEnv
import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled
import math

class EasyCart(CartPoleEnv):
    def step(self, action):
        (obs, rew, done, info) = super().step(action)

        x, v, theta, omega = obs

        # First reward: angle should be close to zero
        reward_1 = 1 - abs(theta / 0.2095)
        # Second reward: position should be close to the center
        reward_2 = 1 - abs(x / 2.4)

        reward = 0.3 * reward_1 + 0.7 * reward_2

        return (obs, reward, done, info)



if MAIN:
    if MAIN:
        gym.envs.registration.register(id="EasyCart-v0", entry_point=EasyCart, max_episode_steps=500)
        args = PPOArgs()
        args.env_id = "EasyCart-v0"
        # args.track = False
        args.gamma = 0.995
        train_ppo(args)
    
    # %%
    
class SpinCart(CartPoleEnv):

    def step(self, action):
        obs, rew, done, info = super().step(action)
        # YOUR CODE HERE
        x, v, theta, omega = obs
        # Allow for 360-degree rotation
        done = (abs(x) > self.x_threshold)
        # Reward function incentivises fast spinning while staying still & near centre
        rotation_speed_reward = min(1, 0.1*abs(omega))
        stability_penalty = max(1, abs(x/2.5) + abs(v/10))
        reward = rotation_speed_reward - 0.5 * stability_penalty
        return (obs, reward, done, info)




if MAIN:
    if MAIN:
        gym.envs.registration.register(id="SpinCart-v0", entry_point=SpinCart, max_episode_steps=500)
        args = PPOArgs()
        args.env_id = "SpinCart-v0"
        train_ppo(args)

```



""", unsafe_allow_html=True)


func_page_list = [
    (section_0, "🏠 Home"),     (section_1, "1️⃣ PPO: Introduction"),     (section_2, "2️⃣ PPO: Rollout"),     (section_3, "3️⃣ PPO: Learning"),     (section_4, "4️⃣ PPO: Full Algorithm"), 
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
