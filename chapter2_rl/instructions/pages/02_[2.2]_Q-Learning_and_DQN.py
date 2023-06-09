
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

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/dqn.png" width="350">

Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.


# [2.2] - Q-Learning and DQN


## Introduction


This section is designed to get you familiar with basic neural networks: how they are structured, the basic operations like linear layers and convolutions which go into making them, and why they work as well as they do. You'll be using libraries like `einops`, and functions like `torch.as_strided` to get a very low-level picture of how these operations work, which will help build up your overall understanding.


## Content & Learning Objectives


#### 1️⃣ Q-Learning

Now, we deal with situations where the environment is a black-box, and the agent must learn the rules of the world via interaction with it. This is different from everything else we've done so far, e.g. in the previous section we could calculate optimal policies by using the tensors $R$ and $T$, which we will now assume the agent doesn't have direct knowledge of.

We call algorithms which have access to the transition probability distribution and reward function **model-based algorithms**. **Q-learning** is a **model-free algorithm**. From the original paper introducing Q-learning:

*[Q-learning] provides agents with the capability of learning to act optimally in Markovian domains by experiencing the consequences of actions, without requiring them to build maps of the domains.*

> ##### Learning objectives
> 
> - Understand the basic Q-learning algorithm
> - Implement SARSA and Q-Learning, and compare them on different envionments
> - Understand the difference between model-based and model-free algorithms
> - Learn more about exploration vs exploitation, and create an epsilon-greedy policy based on your Q-values

#### 2️⃣ DQN

In this section, you'll implement Deep Q-Learning, often referred to as DQN for "Deep Q-Network". This was used in a landmark paper Playing Atari with [Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

You'll apply the technique of DQN to master the famous CartPole environment (below), and then (if you have time) move on to harder challenges like Acrobot and MountainCar.

> ##### Learning objectives
> 
> - Understand the DQN algorithm
> - Learn more about RL debugging, and build probe environments to debug your agents
> - Create a replay buffer to store environment transitions
> - Implement DQN using PyTorch Lightning, on the CartPole environment


## Setup


```python
import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np
import gym
import gym.spaces
import gym.envs.registration
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm, trange
import sys
import time
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Tuple
import torch as t
from torch import nn, Tensor
from gym.spaces import Discrete, Box
from numpy.random import Generator
import pandas as pd
import wandb
import pandas as pd
from pathlib import Path
from jaxtyping import Float, Int, Bool
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger

Arr = np.ndarray

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_dqn"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from part1_intro_to_rl.utils import make_env
from part1_intro_to_rl.solutions import Environment, Toy, Norvig, find_optimal_policy
import part2_dqn.utils as utils
import part2_dqn.tests as tests
from plotly_utils import line, cliffwalk_imshow, plot_cartpole_obs_and_dones

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

```



""", unsafe_allow_html=True)


def section_1():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#readings'>Readings</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#optional-readings'>Optional Readings</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#recap-of-gym-env'>Recap of <code>gym.Env</code></a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#the-step-method'>The <code>step</code> method</a></li>
        <li><a class='contents-el' href='#the-render-method'>The <code>render()</code> method</a></li>
        <li><a class='contents-el' href='#observation-and-action-types'>Observation and Action Types</a></li>
        <li><a class='contents-el' href='#registering-an-environment'>Registering an Environment</a></li>
        <li><a class='contents-el' href='#timelimit-wrapper'>TimeLimit Wrapper</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#cheater-agent'>Cheater Agent</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-cheater'><b>Exercise</b> - implement <code>Cheater</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#sarsa:-on-policy-td-control'>SARSA: On-Policy TD Control</a></li>
    <li class='margtop'><a class='contents-el' href='#q-learning:-off-policy-td-control'>Q-Learning: Off-policy TD Control</a></li>
    <li class='margtop'><a class='contents-el' href='#explore-vs-exploit'>Explore vs. Exploit</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-q-learning-and-sarsa'><b>Exercise</b> - Implement Q-learning and SARSA</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#other-environments'>Other Environments</a></li>
    <li class='margtop'><a class='contents-el' href='#tabular-methods'>Tabular Methods</a></li>
    <li class='margtop'><a class='contents-el' href='#bonus'>Bonus</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-build-your-own-cliffwalking-environment'><b>Exercise</b> - build your own CliffWalking environment</a></li>
        <li><a class='contents-el' href='#monte-carlo-q-learning'>Monte-Carlo Q-learning</a></li>
        <li><a class='contents-el' href='#lr-scheduler'>LR scheduler</a></li>
        <li><a class='contents-el' href='#other-environments'>Other environments</a></li>
        <li><a class='contents-el' href='#double-q-learning'>Double-Q learning</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 1️⃣ Q-Learning


> ##### Learning objectives
> 
> - Understand the basic Q-learning algorithm
> - Implement SARSA and Q-Learning, and compare them on different envionments
> - Understand the difference between model-based and model-free algorithms
> - Learn more about exploration vs exploitation, and create an epsilon-greedy policy based on your Q-values


Now, we deal with situations where the environment is a black-box, and the agent must learn the rules of the world via interaction with it. This is different from everything else we've done so far, e.g. in the previous section we could calculate optimal policies by using the tensors $R$ and $T$, which we will now assume the agent doesn't have direct knowledge of.

We call algorithms which have access to the transition probability distribution and reward function **model-based algorithms**. **Q-learning** is a **model-free algorithm**. From the original paper introducing Q-learning:

*[Q-learning] provides agents with the capability of learning to act optimally in Markovian domains by experiencing the consequences of actions, without requiring them to build maps of the domains.*

The "Q" part of Q-learning refers to the function $Q$ which we encountered yesterday - the expected rewards for an action $a$ taken in a particular state $s$, based on some policy $\pi$.



## Readings

Don't worry about absorbing every detail, we will repeat a lot of the details here. Don't worry too much about the maths, we will also cover that here.

- [Sutton and Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
    - Chapter 6, Section 6.1, 6.3 (Especially Example 6.4)
    - Note that Section 6.1 talks about temporal difference (TD) updates for the value function $V$. We will instead be using TD updates for the Q-value $Q$.
    - Don't worry about the references to Monte Carlo in Chapter 5.

### Optional Readings

- [Q-Learning](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf) The original paper where Q-learning is first described.


Today and tomorrow, we'll be using OpenAI Gym, which provides a uniform interface to many different RL environments including Atari games. Gym was released in 2016 and details of the API have changed significantly over the years. We are using version 0.23.1, so ensure that any documentation you use refers to the same version.

<details>
<summary>What's the difference between observation and state?</summary>

We use the word *observation* here as some environments are *partially observable*, the agent receives not an exact description of the state they are in, but merely an observation giving a partial description (for our gridworld, it could be a description of which cells directly adjacent to the agent are free to move into, rather than precisely which state they are in). This means that the agent would be unable to distinguish the cell north of the wall from the cell south of the wall. Returning the state as the observation is a special case, and we will often refer to one or the other as required.
</details>

Again, we'll be using NumPy for this section, and we'll start off with our gridworld environment from last week:

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/gridworld.png" width="300">

but this time we'll use it within the `gym` framework. 

## Recap of `gym.Env`

Let's have a speed recap of the key features the `gym.Env` class provides, and see how we can use it to wrap our gridworld environment from last week.

### The `step` method

The environment's `step` method takes the action selected by the agent and returns four values: `obs`, `reward`, `done`, and the `info` dictionary.

`obs` and `reward` is the next observation and reward that the agent receives based on the action chosen.

`done` indicates if the environment has entered a terminal state and ended. Here, both the goal-states (+1 and -1) are terminal. Early termination is equivalent to an infinite trajectory where the agent remains trapped for all future states, and always receives reward zero.

`info` can contain anything extra that doesn't fit into the uniform interface - it's up to the environment what to put into it. A good use of this is for debugging information that the agent isn't "supposed" to see, like the dynamics of the environment. Agents that cheat and peek at `info` are helpful because we know that they should obtain the maximum possible rewards; if they aren't, then there's a bug. We will throw the entire underlying environment into `info`, from which an agent could cheat by peeking at the values for `T` and `R`.

### The `render()` method

Render is only used for debugging or entertainment, and what it does is up to the environment. It might render a little popup window showing the Atari game, or it might give you a RGB image tensor, or just some ASCII text describing what's happening.

### Observation and Action Types

A `gym.Env` is a generic type: both the type of the observations and the type of the actions depends on the specifics of the environment.

We're only dealing with the simplest case: a discrete set of actions which are the same in every state. In general, the actions could be continuous, or depend on the state.

---

Below, we define a class that allows us to use our old environment definition from last week, and wrap it in a `gym.Env` instance so we can learn from experience instead.

Read the code below carefully and make sure you understand how the Gym environment API works.


```python
ObsType = int
ActType = int
    
class DiscreteEnviroGym(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete

    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
        self.observation_space = gym.spaces.Discrete(env.num_states)
        self.action_space = gym.spaces.Discrete(env.num_actions)
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        '''
        Samples from the underlying dynamics of the environment
        '''
        (states, rewards, probs) = self.env.dynamics(self.pos, action)
        idx = self.np_random.choice(len(states), p=probs)
        (new_state, reward) = (states[idx], rewards[idx])
        self.pos = new_state
        done = self.pos in self.env.terminal
        return (new_state, reward, done, {"env": self.env})

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)
        self.pos = self.env.start
        return (self.pos, {"env": self.env}) if return_info else self.pos

    def render(self, mode="human"):
        assert mode == "human", f"Mode {mode} not supported!"

```

### Registering an Environment

User code normally won't use the constructor of an `Env` directly for two reasons:

- Usually, we want to wrap our `Env` in one or more wrapper classes.
- If we want to test our agent on a variety of environments, it's annoying to have to import all the `Env` classes directly.

The `register` function stores information about our `Env` in a registry so that a later call to `gym.make` can look it up using the `id` string that is passed in.

By convention, the `id` strings have a suffix with a version number. There can be multiple versions of the "same" environment with different parameters, and benchmarks should always report the version number for a fair comparison. For instance, `id="NorvigGrid-v0"` below.

### TimeLimit Wrapper

As defined, our environment might take a very long time to terminate: A policy that actively avoids
the terminal states and hides in the bottom-left corner would almost surely terminate through
a long sequence of slippery moves, but this could take a long time.
By setting `max_episode_steps` here, we cause our `env` to be wrapped in a `TimeLimit` wrapper class which terminates the episode after that number of steps.

Note that the time limit is also an essential part of the problem definition: if it were larger or shorter, there would be more or less time to explore, which means that different algorithms (or at least different hyperparameters) would then have improved performance. We would obviously want to choose a rather
conservative value such that any reasonably strong policy would be able to reach a terminal state in time.

For our toy gridworld environment, we choose the (rather pessimistic) bound of 100 moves.


```python
gym.envs.registration.register(
    id="NorvigGrid-v0",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=100,
    nondeterministic=True,
    kwargs={"env": Norvig(penalty=-0.04)},
)

gym.envs.registration.register(
    id="ToyGym-v0", 
    entry_point=DiscreteEnviroGym, 
    max_episode_steps=2, 
    nondeterministic=False, 
    kwargs={"env": Toy()}
)
```

Provided is the `RandomAgent` subclass which should pick an action at random, using the random number generator provided by gym. This is useful as a baseline to ensure the environment has no bugs. If your later agents are doing worse than random, you have a bug!



```python
@dataclass
class Experience:
    '''A class for storing one piece of experience during an episode run'''
    obs: ObsType
    act: ActType
    reward: float
    new_obs: ObsType
    new_act: Optional[ActType] = None

@dataclass
class AgentConfig:
    '''Hyperparameters for agents'''
    epsilon: float = 0.1
    lr: float = 0.05
    optimism: float = 0


defaultConfig = AgentConfig()
    
class Agent:
    '''Base class for agents interacting with an environment (you do not need to add any implementation here)'''
    rng: np.random.Generator

    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma: float = 0.99, seed: int = 0):
        self.env = env
        self.reset(seed)
        self.config = config
        self.gamma = gamma
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        self.name = type(self).__name__

    def get_action(self, obs: ObsType) -> ActType:
        raise NotImplementedError()

    def observe(self, exp: Experience) -> None:
        '''
        Agent observes experience, and updates model as appropriate.
        Implementation depends on type of agent.
        '''
        pass

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def run_episode(self, seed) -> List[int]:
        '''
        Simulates one episode of interaction, agent learns as appropriate
        Inputs:
            seed : Seed for the random number generator
        Outputs:
            The rewards obtained during the episode
        '''
        rewards = []
        obs = self.env.reset(seed=seed)
        self.reset(seed=seed)
        done = False
        while not done:
            act = self.get_action(obs)
            (new_obs, reward, done, info) = self.env.step(act)
            exp = Experience(obs, act, reward, new_obs)
            self.observe(exp)
            rewards.append(reward)
            obs = new_obs
        return rewards

    def train(self, n_runs=500):
        '''
        Run a batch of episodes, and return the total reward obtained per episode
        Inputs:
            n_runs : The number of episodes to simulate
        Outputs:
            The discounted sum of rewards obtained for each episode
        '''
        all_rewards = []
        for seed in trange(n_runs):
            rewards = self.run_episode(seed)
            all_rewards.append(utils.sum_rewards(rewards, self.gamma))
        return all_rewards

class Random(Agent):
    def get_action(self, obs: ObsType) -> ActType:
        return self.rng.integers(0, self.num_actions)

```

## Cheater Agent

Just like yesterday, you'll implement a cheating agent that peeks at the info and finds the optimal policy directly using your previous code. If your agent gets more than this in the long run, you have a bug!


### Exercise - implement `Cheater`

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠🟠⚪⚪

You should spend up to 10-15 minutes on this exercise.
```


You should solve for the optimal policy once when `Cheater` is initalised, and then use that to define `get_action`.

Check that your cheating agent outperforms the random agent. The cheating agent represents the best possible behavior, as it omnisciently always knows to play optimally.

On the environment `ToyGym-v0`, (assuming $\gamma = 0.99$) the cheating agent should always get reward $2 \gamma = 1.98$,
and the random agent should get a fluctuating reward, with average $\frac{2 \gamma + 1}{2} = 1.49$. 

Hint: Use `env.unwrapped.env` to extract the `Environment` wrapped inside `gym.Env`, to get access to the underlying dynamics.

<details>
<summary>Help - I get 'AttributeError: 'DiscreteEnviroGym' object has no attribute 'num_states''.</summary>

This is probably because you're passing the `DiscreteEnviroGym` object to your `find_optimal_policy` function. In the following line of code:

```python
env_toy = gym.make("ToyGym-v0")
```

the object `env_toy` wraps around the `Toy` environment you used last week. As mentioned, you'll need to use `env.unwrapped.env` to access this environment, and its dynamics.
</details>

<details>
<summary>Help - I'm not sure what I should be doing here.</summary>

You should be using your `find_optimal_policy` function fom yesterday. Recall that the first argument of this function was the `Environment` object; you can find this by using `env.unwrapped.env`.

When getting an action in response to an observation, you can just extract the corresponding action from this optimal policy (i.e. by indexing into your policy array). Remember that your policy is a 1D array containing the best action for each observation.
</details>


```python
class Cheater(Agent):
    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma=0.99, seed=0):
        super().__init__(env, config, gamma, seed)
        pass

    def get_action(self, obs):
        pass


env_toy = gym.make("ToyGym-v0")
agents_toy: List[Agent] = [Cheater(env_toy), Random(env_toy)]
returns_list = []
names_list = []
for agent in agents_toy:
    returns = agent.train(n_runs=100)
    returns_list.append(utils.cummean(returns))
    names_list.append(agent.name)

line(returns_list, names=names_list, title=f"Avg. reward on {env_toy.spec.name}")
```

<details>
<summary>Solution</summary>

```python
class Cheater(Agent):
    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma=0.99, seed=0):
        super().__init__(env, config, gamma, seed)
        # SOLUTION
        self.pi_opt = find_optimal_policy(self.env.unwrapped.env, self.gamma)

    def get_action(self, obs):
        # SOLUTION
        return self.pi_opt[obs]
```
</details>

## SARSA: On-Policy TD Control

Now we wish to train an agent on the same gridworld environment as before, but this time it doesn't have access to the underlying dynamics (`T` and `R`). The rough idea here is to try and estimate the Q-value function directly from samples received from the environment. Recall that the optimal Q-value function satisfies 
$$
Q^*(s,a) = \mathbb{E}_{\pi^*} \left[ \sum_{i=t+1}^\infty \gamma^{i-t}r_i  \mid s_t = s, a_t = a\right]
= \mathbb{E}_{\pi^*} \left[r + \gamma \max_{a'} Q^*(s', a') \right]
= \sum_{s'} T(s' \mid s,a) \left( R(s,a,s') + \gamma \max_{a'} Q^*(s',a') \right)
$$
where
* $s'$ represents the next state after $s$,
* $a'$ the next action after $a$
* $r$ is the reward obtained from taking action $a$ in state $s$
* the expectation $\mathbb{E}_{\pi^*}$ is with respect to both the optimal policy $\pi^*$, as well as the stochasticity in the environment itself.

So, for any particular episode $s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2, r_3,\ldots$ we have that
*on average* the value of $Q^*(s_t, a_t)$ should be equal to the *actual reward*
$r_t$ recieved when choosing action $a_t$ in state $s_t$, plus $\gamma$ times the
Q-value of the next state $s_{t+1}$ and next action $a_{t+1}$.
$$
Q^*(s_t,a_t) =
\mathbb{E}_{\pi^*} \left[r + \gamma \max_{a'} Q^*(s', a') \right]
\approx r_{t+1} + \gamma  Q^*(s_{t+1}, a_{t+1})
$$
where $a_{t+1} = \pi^*(s_{t+1}) = \argmax_{a} Q^*(s_{t+1}, a)$.


Letting $Q$ denote our best current estimate of $Q^*$, the error $\delta_t := r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t)$  in this "guess" is called the **TD error**, and tells us in which direction we should bump our estimate of $Q^*$.
Of course, this estimate might be wildly inaccurate (even for the same state-action pair!), due to the stochasticity of the environment, or poor estimates of $Q$. So, we update our estimate slightly in the direction of $\delta_t$, much like stochastic gradient descent does. The update rule for Q-learning (with learning rate $\eta > 0$) is
$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \eta \left( r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t) \right)
$$
This update depends on the information $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$, and so is called **SARSA** learning. Note that SARSA learns *on-policy*, in that it only learns from data that was actually generated by the current policy $\pi$, derived from the current estimate of $Q$, $\pi(s) = \argmax_a Q(s,a)$.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/sarsa.png" width="700">

## Q-Learning: Off-policy TD Control

At the end of the day, what SARSA is essentially doing is estimating $Q^\pi$ by using the rewards gathered by following policy $\pi$. But we don't actually care about $Q^\pi$, what we care about is $Q^*$. Q-Learning provides a slight modification to SARSA, by modifying the TD-error $\delta_t$ to use the action that $\pi$ *should* have taken in state $s_t$ (namely $\argmax_a Q(s_t,a)$) rather than the action $a_t = \pi(s_t)$ that was actually taken.
$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \eta \left( r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t,a_t) \right)
$$
Note that each Q-learning update depends on the information $(s_t, a_t, r_{t+1}, s_{t+1})$.
This means that Q-learning tries to estimate $Q^*$ directly, regardless of what policy $\pi$ generated the episode, and so Q-Learning learns *off-policy*.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/qlearn.png" width="700">


## Explore vs. Exploit

Lastly, methods to learn the Q-value often have trouble exploring. If a state-action pair $(s,a)$ with low Q-value $Q^*(s,a)$ just so happens to return a high reward by chance, the greedy policy with respect to $Q$ will often choose action $a$ in state $s$ instead of exploring potentially other good actions first. To remedy this, we use instead an $\epsilon$-greedy policy with respect to the current Q-value estimates: With probability $\epsilon$, a random action is chosen, and with probability $1-\epsilon$ the greedy action $\argmax_a Q(s,a)$ is chosen. The exploration probability $\epsilon$ is a hyperparameter that for now we will set to a constant $\epsilon = 0.1$, but more sophisticated techniques include the use of a schedule to start exploring often early, and then decay exploration as times goes on.

We also have the choice of how the estimate $Q(s,a)$ is initialized. By choosing "optimistic" values (initial values that are much higher than what we expect $Q^*(s,a)$ to actually be), this will encourage the greedy policy to hop between different actions in each state when they discover they weren't as valuable as first thought.

We will implement an `EpsilonGreedy` agent that keeps track of the current Q-value estimates, and selects an action based on the epsilon greedy policy.

Both `SARSA` and `QLearning` will inherit from `EpsilonGreedy`, and differ in how they update the Q-value estimates.

- Keep track of an estimate of the Q-value of each state-action pair.
- Epsilon greedy exploration: with probability `epsilon`, take a random action; otherwise take the action with the highest average observed reward (according to your current Q-value estimates).
    - Remember that your `AgentConfig` object contains epsilon, as well as the optimism value and learning rate.
- Optimistic initial values: initialize each arm's reward estimate with the `optimism` value.
- Compare the performance of your Q-learning and SARSA agent again the random and cheating agents.
- Try and tweak the hyperparameters from the default values of `epsilon = 0.1`, `optimism = 1`, `lr = 0.1` to see what effect this has. How fast can you get your
agents to perform?


### Exercise - Implement Q-learning and SARSA

```c
Difficulty: 🟠🟠🟠🟠⚪
Importance: 🟠🟠🟠🟠⚪

You should spend up to 25-45 minutes on these exercises.

They are difficult, so you should use the hints if you need.
```

#### Tips

- Use `self.rng.random()` to generate random numbers in the range $[0,1)$, and `self.rng.integers(0, n)` for random integers in the range $0, 1, \ldots, n-1$.
- The random agent results in very long episodes, which slows evaluation. You can remove them from the experiment once you've convinced yourself that your agents are doing something intelligent and outperforming a random agent.
- Leave $\gamma =0.99$ for now.

<details>
<summary>Help - I'm not sure what methods I should be rewriting in QLearning and SARSA.</summary>

Your `EpsilonGreedy` agent already has a method for getting actions, which should use the `self.Q` object. This will be the same for `QLearning` and `SARSA`.

The code you need to add is the `observe` method. Recall from the code earlier that `observe` takes an `Experience` object, which stores data `obs`, `act`, `reward`, `new_obs`, and `new_act`. In mathematical notation, these correspond to $s_t$, $a_t$, $r_{t+1}$, $s_{t+1}$ and $a_{t+1}$.

For `SARSA`, there's an added complication. We want SARSA to directly update the Q-value estimates after action $a_{t+1}$ is taken, as we need all of $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$ to perform a SARSA update. This means you will need to override the `run_episode` function to adjust when the Q-values are updated.
</details>

<details>
<summary>Help - I'm still confused about the <code>run_episode</code> function.</summary>

The main loop of the original `run_episode` function looked like this:

```python
while not done:
    act = self.get_action(obs)
    (new_obs, reward, done, info) = self.env.step(act)
    exp = Experience(obs, act, reward, new_obs)
    self.observe(exp)
    rewards.append(reward)
    obs = new_obs
```

The problem here is that we don't have `new_act` in our `Experience` dataclass, because we only keep track of one action at a time. We can fix this by defining `new_act = self.get_action(new_obs)` **after** `new_obs` is defined. In this way, we can pass all of `(obs, act, reward, new_obs, new_act)` into our `Experience` dataclass.

```python
while not done:
    (new_obs, reward, done, info) = self.env.step(act)
    new_act = self.get_action(new_obs)
    exp = Experience(obs, act, reward, new_obs, new_act)
    self.observe(exp)
    rewards.append(reward)
    obs = new_obs
    act = new_act
```
</details>

<details>
<summary>What output should I expect to see?</summary>


</details>


```python
class EpsilonGreedy(Agent):
    '''
    A class for SARSA and Q-Learning to inherit from.
    '''
    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma: float = 0.99, seed: int = 0):
        pass

    def get_action(self, obs: ObsType) -> ActType:
        '''
        Selects an action using epsilon-greedy with respect to Q-value estimates
        '''
        pass

class QLearning(EpsilonGreedy):
    def observe(self, exp: Experience) -> None:
        pass

class SARSA(EpsilonGreedy):
    def observe(self, exp: Experience):
        pass

    def run_episode(self, seed) -> List[int]:
        pass


n_runs = 1000
gamma = 0.99
seed = 1
env_norvig = gym.make("NorvigGrid-v0")
config_norvig = AgentConfig()
args_norvig = (env_norvig, config_norvig, gamma, seed)
agents_norvig: List[Agent] = [Cheater(*args_norvig), QLearning(*args_norvig), SARSA(*args_norvig), Random(*args_norvig)]
returns_norvig = {}
fig = go.Figure(layout=dict(
    title_text=f"Avg. reward on {env_norvig.spec.name}", 
    template="simple_white",
    xaxis_range=[-30, n_runs+30]
))
for agent in agents_norvig:
    returns = agent.train(n_runs)
    fig.add_trace(go.Scatter(y=utils.cummean(returns), name=agent.name))
fig.show()
```

<details>
<summary>Solution</summary>


```python
class EpsilonGreedy(Agent):
    '''
    A class for SARSA and Q-Learning to inherit from.
    '''
    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma: float = 0.99, seed: int = 0):
        # SOLUTION
        super().__init__(env, config, gamma, seed)
        self.Q = np.zeros((self.num_states, self.num_actions)) + self.config.optimism

    def get_action(self, obs: ObsType) -> ActType:
        '''
        Selects an action using epsilon-greedy with respect to Q-value estimates
        '''
        # SOLUTION
        if self.rng.random() < self.config.epsilon:
            return self.rng.integers(0, self.num_actions)
        else:
            return self.Q[obs].argmax()

class QLearning(EpsilonGreedy):
    def observe(self, exp: Experience) -> None:
        # SOLUTION
        s, a, r_new, s_new = exp.obs, exp.act, exp.reward, exp.new_obs
        self.Q[s,a] += self.config.lr * (r_new + self.gamma * np.max(self.Q[s_new]) - self.Q[s, a])

class SARSA(EpsilonGreedy):
    def observe(self, exp: Experience):
        # SOLUTION
        s, a, r_new, s_new, a_new = exp.obs, exp.act, exp.reward, exp.new_obs, exp.new_act
        self.Q[s,a] += self.config.lr * (r_new + self.gamma * self.Q[s_new, a_new] - self.Q[s, a])

    def run_episode(self, seed) -> List[int]:
        # SOLUTION
        rewards = []
        obs = self.env.reset(seed=seed)
        act = self.get_action(obs)
        self.reset(seed=seed)
        done = False
        while not done:
            (new_obs, reward, done, info) = self.env.step(act)
            new_act = self.get_action(new_obs)
            exp = Experience(obs, act, reward, new_obs, new_act)
            self.observe(exp)
            rewards.append(reward)
            obs = new_obs
            act = new_act
        return rewards
```
</details>


Compare the performance of SARSA and Q-Learning on the gridworld environment v.s. the cheating agent and the random agent. Try to tune the hyperparameters to get the best performance you can.

- Which seems to work better? SARSA or Q-Learning?
- Does the optimism parameter seems to help?
- What's the best choice of exploration parameter $\epsilon$?
- The feedback from the environment is very noisy. At the moment, the code provided plots the cumulative average reward per episode. You might want to try plotting a sliding average instead, or an exponential weighted moving average (see `part2_dqn/utils.py`).


## Other Environments

`gym` provides a large set of environments with which to test agents against. We can see all available environments by running `gym.envs.registry.all()`

Have a look at [the gym library](https://www.gymlibrary.dev/environments/toy_text/) for descriptions of these environments. As written, our SARSA and Q-Learning agents will only work with environments that have both discrete observation and discrete action spaces.

We'll modify the above code to use environment `gym.make("CliffWalking-v0")` instead (see [this link](https://www.gymlibrary.dev/environments/toy_text/cliff_walking/)). We have the following graph from Sutton & Barto, Example 6.6, that displays the sum of reward obtained for each episode, as well as the policies obtained (SARSA takes the safer path, Q-Learning takes the optimal path). You may want to check out [this post](https://towardsdatascience.com/walking-off-the-cliff-with-off-policy-reinforcement-learning-7fdbcdfe31ff).

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/cliff_pi.png" width="400">

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/cliff.png" width="400">

Do you get a similar result when you run the code below?

Some notes:

* Use $\gamma = 1$ as described in Sutton & Barto, Example 6.6.
* Try tweaking the learning rate and epsilon (start with $\epsilon = 0.1$) to try and cause SARSA to take the cautious path, while Q-Learning takes the risky path.
* We've included some helper functions to display the value of each state, as well as the policy an agent would take, given the Q-value function.
* One of the bonus exercises we've suggested is to write your own version of `CliffWalking-v0` by writing a class similar to the `Norvig` class you have been working with. If you do this correctly, then you'll also be able to make a cheating agent.
* We've defined a `cliffwalk_imshow` helper function for you, which visualises your agent's path (and reward at each square).

<details>
<summary>Question - why is it okay to use gamma=1 here?</summary>

The penalty term `-1` makes sure that the agent continually penalised until it hits the terminal state. Unlike our `Norvig` environment, there is no wall to get stuck in perpetually, rather hitting the cliff will send you back to the start, so the agent must eventually reach the terminal state.
</details>


```python
gamma = 1
seed = 0

config_cliff = AgentConfig(epsilon=0.1, lr = 0.1, optimism=0)
env = gym.make("CliffWalking-v0")
n_runs = 2500
args_cliff = (env, config_cliff, gamma, seed)

returns_list = []
name_list = []
agents: List[Union[QLearning, SARSA]] = [QLearning(*args_cliff), SARSA(*args_cliff)]

for agent in agents:
    returns = agent.train(n_runs)[1:]
    returns_list.append(utils.cummean(returns))
    name_list.append(agent.name)
    V = agent.Q.max(axis=-1).reshape(4, 12)
    pi = agent.Q.argmax(axis=-1).reshape(4, 12)
    cliffwalk_imshow(V, pi, title=f"CliffWalking: {agent.name} Agent")

line(
    returns_list, 
    names=name_list, 
    template="simple_white",
    title="Q-Learning vs SARSA on CliffWalking-v0",
    labels={"x": "Episode", "y": "Avg. reward", "variable": "Agent"},
)

```

## Tabular Methods

The methods used here are called tabular methods, because they create a lookup table from `(state, action)` to the Q value. This is pure memorization, and if our reward function was sampled from the space of all functions, this is usually the best you can do because there's no structure that you can exploit to do better.

We can hope to do better on most "natural" reward functions that do have structure. For example in a game of poker, there is structure in both of the actions (betting £100 will have a similar reward to betting £99 or £101), and between states (having a pair of threes in your hand is similar to having a pair of twos or fours). We need to take advantage of this, otherwise there are just too many states and actions to have any hope of training an agent.

One idea is to use domain knowledge to hand-code a function that "buckets" states or actions into a smaller number of equivalence classes and use those as the states and actions in a smaller version of the problem (see Sutton and Barto, Section 9.5). This was one component in the RL agent [Libratus: The Superhuman AI for No-Limit Poker](https://www.cs.cmu.edu/~noamb/papers/17-IJCAI-Libratus.pdf). The details are beyond the scope of today's material, but I found them fascinating.

If you don't have domain knowledge to leverage, or you care specifically about making your algorithm "general", you can follow the approach that we'll be using in Part 2️⃣: make a neural network that takes in a state (technically, an observation) and outputs a value for each action. We then train the neural network using environmental interaction as training data.


## Bonus - build your own CliffWalking environment


```c
Difficulty: 🟠🟠🟠🟠🟠
Importance: 🟠🟠⚪⚪⚪

You should spend up to 30-60 minutes on this exercise.
```

You should return to this exercise at the end if you want to. For now, you should progress to part 2️⃣.

You can modify the code used to define the `Norvig` class to define your own version of `CliffWalking-v0`. 

You can do this without guidance, or you can get some more guidance from the dropdowns below. **Hint 1** offers vague guidance, **Hint 2** offers more specific direction.

Some notes for this task:

* The random agent will take a *very* long time to accidentally stumble into the goal state, and will slow down learning. You should probably neglect it.
* As soon as you hit the cliff, you should immediately move back to the start square, i.e. in pseudocode:
    ```python
    if new_state in cliff:
        new_state = start_state
        reward -= 100
    ```
    This means you'll never calculate Q from the cliff, so your Q-values will always be zero here.

<details>


```python
class CliffWalking(Environment):

    def __init__(self, penalty=-1):
        pass

    def dynamics(self, state : int, action : int) -> Tuple[Arr, Arr, Arr]:
        pass

    @staticmethod
    def render(Q: Arr, name: str):
        V = Q.max(axis=-1).reshape(4, 12)
        pi = Q.argmax(axis=-1).reshape(4, 12)
        cliffwalk_imshow(V, pi, title=f"CliffWalking: {name} Agent")



gym.envs.registration.register(
    id="CliffWalking-myversion",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=200,
    nondeterministic=True,
    kwargs={"env": CliffWalking(penalty=-1)},
)
gamma = 0.99
seed = 0
config_cliff = AgentConfig(epsilon=0.1, lr = 0.1, optimism=0)
env = gym.make("CliffWalking-myversion")
n_runs = 500
args_cliff = (env, config_cliff, gamma,seed)

agents = [Cheater(*args_cliff), QLearning(*args_cliff), SARSA(*args_cliff), Random(*args_cliff)]
returns_list = []
name_list = []

for agent in agents:
    returns = agent.train(n_runs)[1:]
    returns_list.append(utils.cummean(returns))
    name_list.append(agent.name)

line(
    returns_list, 
    names=name_list, 
    template="simple_white",
    title="Q-Learning vs SARSA on CliffWalking-v0",
    labels={"x": "Episode", "y": "Avg. reward", "variable": "Agent"},
)
```


<summary>Hints (for both methods)</summary>

The main way in which the `CliffWalking` environment differs from the `Norvig` gridworld is that the former has cliffs while the latter has walls. Cliffs and walls have different behaviour; you can see how the cliffs affect the agent by visiting the documentation page for `CliffWalking-v0`.

#### `__init__`

This mainly just involves changing the dimensions of the space, position of the start and terminal states, and parameters like `penalty`. Also, rather than walls, you'll need to define the position of the **cliffs** (which behave differently).

#### `dynamics`

You'll need to modify `dynamics` in the following two ways:

* Remove the slippage probability (although it would be interesting to experiment with this and see what effect it has!)
* Remove the "when you hit a wall, you get trapped forever" behaviour, and replace it with "when you hit a cliff, you get a reward of -100 and go back to the start state".

</details>



<details>
<summary>Solution</summary>


```python
class CliffWalking(Environment):

    def __init__(self, penalty=-1):
        # SOLUTION
        self.height = 4
        self.width = 12
        self.penalty = penalty
        num_states = self.height * self.width
        num_actions = 4
        self.states = np.array([[x, y] for y in range(self.height) for x in range(self.width)])
        self.actions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])  # up, right, down, left
        self.dim = (self.height, self.width)

        # special states: tuples of state and reward
        # all other states get penalty
        start = 36
        terminal = np.array([47], dtype=int)
        self.cliff = np.arange(37, 47, dtype=int)
        self.goal_rewards = np.array([1.0, -1.0])

        super().__init__(num_states, num_actions, start=start, terminal=terminal)


    def dynamics(self, state : int, action : int) -> Tuple[Arr, Arr, Arr]:
        # SOLUTION
        def state_index(state):
            assert 0 <= state[0] < self.width and 0 <= state[1] < self.height, print(state)
            pos = state[0] + state[1] * self.width
            assert 0 <= pos < self.num_states, print(state, pos)
            return pos

        pos = self.states[state]
        move = self.actions[action]

        if state in self.terminal:
            return (np.array([state]), np.array([0]), np.array([1]))

        # No slipping; each action is deterministic
        out_probs = np.zeros(self.num_actions)
        out_probs[action] = 1

        out_states = np.zeros(self.num_actions, dtype=int) + self.num_actions
        out_rewards = np.zeros(self.num_actions) + self.penalty
        new_states = [pos + x for x in self.actions]

        for i, s_new in enumerate(new_states):

            if not (0 <= s_new[0] < self.width and 0 <= s_new[1] < self.height):
                out_states[i] = state
                continue

            new_state = state_index(s_new)

            # Check if would hit the cliff, if so then get -100 penalty and go back to start
            if new_state in self.cliff:
                out_states[i] = self.start
                out_rewards[i] -= 100

            else:
                out_states[i] = new_state

            for idx in range(len(self.terminal)):
                if new_state == self.terminal[idx]:
                    out_rewards[i] = self.goal_rewards[idx]

        return (out_states, out_rewards, out_probs)

    @staticmethod
    def render(Q: Arr, name: str):
        V = Q.max(axis=-1).reshape(4, 12)
        pi = Q.argmax(axis=-1).reshape(4, 12)
        cliffwalk_imshow(V, pi, title=f"CliffWalking: {name} Agent")
```
</details>




### Monte-Carlo Q-learning

Implement Monte-Carlo Q-learning (Chapter 5 Sutton and Barto) and $\text{TD}(\lambda)$ with eligibility traces (Chapter 7 Sutton and Barto).

### LR scheduler

Try using a schedule for the exploration rate $\epsilon$ (Large values early to encourage exploration, low values later once the agent has sufficient statistics to play optimally).

Would Q-Learning or SARSA be better off with a scheduled exploration rate?

The Sutton book mentions that if $\epsilon$ is gradually reduced, both methods asymptotically converge to the optimal policy. Is this what you find?

### Other environments

Try other environments like [Frozen Lake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) and [BlackJack](https://www.gymlibrary.dev/environments/toy_text/blackjack/). Note that BlackJack uses `Tuple(Discrete(32), Discrete(11), Discrete(2))` as it's observation space, so you will have to write some glue code to convert this back and forth between an observation space of `Discrete(32 * 11 * 2)` to work with our agents as written.

### Double-Q learning

Read Sutton and Barto Section 6.7 Maximisation Bias and Double Learning. Implement Double-Q learning, and compare it's performance against SARSA and Q-Learning.




""", unsafe_allow_html=True)


def section_2():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#readings'>Readings</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#interesting-resources-not-required-reading'>Interesting Resources (not required reading)</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#fast-feedback-loops'>Fast Feedback Loops</a></li>
    <li class='margtop'><a class='contents-el' href='#cartpole'>CartPole</a></li>
    <li class='margtop'><a class='contents-el' href='#outline-of-the-exercises'>Outline of the exercises</a></li>
    <li class='margtop'><a class='contents-el' href='#the-q-network'>The Q-Network</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-qnetwork'><b>Exercise</b> - implement <code>QNetwork</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#replay-buffer'>Replay Buffer</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#correlated-states'>Correlated States</a></li>
        <li><a class='contents-el' href='#uniform-sampling'>Uniform Sampling</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#environment-resets'>Environment Resets</a></li>
    <li class='margtop'><a class='contents-el' href='#exploration'>Exploration</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#reward-shaping'>Reward Shaping</a></li>
        <li><a class='contents-el' href='#reward-hacking'>Reward Hacking</a></li>
        <li><a class='contents-el' href='#advanced-exploration'>Advanced Exploration</a></li>
        <li><a class='contents-el' href='#exercise-implement-linear-scheduler'><b>Exercise</b> - implement linear scheduler</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#epsilon-greedy-policy'>Epsilon Greedy Policy</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-the-epsilon-greedy-policy'><b>Exercise</b> - implement the epsilon greedy policy</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#probe-environments'>Probe Environments</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#a-note-on-action-spaces'>A note on action spaces</a></li>
        <li><a class='contents-el' href='#exercise-implement-additional-probe-environments'><b>Exercise</b> - implement additional probe environments</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#weights-and-biases'>Weights and Biases</a></li>
    <li class='margtop'><a class='contents-el' href='#main-dqn-algorithm'>Main DQN Algorithm</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-fill-in-the-agent-class'><b>Exercise</b> - fill in the agent class</a></li>
        <li><a class='contents-el' href='#exercise-write-dqn-training-loop'><b>Exercise</b> - write DQN training loop</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#beyond-cartpole'>Beyond CartPole</a></li>
    <li class='margtop'><a class='contents-el' href='#bonus'>Bonus</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#target-network'>Target Network</a></li>
        <li><a class='contents-el' href='#shrink-the-brain'>Shrink the Brain</a></li>
        <li><a class='contents-el' href='#dueling-dqn'>Dueling DQN</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 2️⃣ Deep Q-Learning


> ##### Learning objectives
> 
> - Understand the DQN algorithm
> - Learn more about RL debugging, and build probe environments to debug your agents
> - Create a replay buffer to store environment transitions
> - Implement DQN using PyTorch Lightning, on the CartPole environment


In this section, you'll implement Deep Q-Learning, often referred to as DQN for "Deep Q-Network". This was used in a landmark paper [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

At the time, the paper was very exicitng: The agent would play the game by only looking at the same screen pixel data that a human player would be looking at, rather than a description of where the enemies in the game world are. The idea that convolutional neural networks could look at Atari game pixels and "see" gameplay-relevant features like a Space Invader was new and noteworthy. In 2022, we take for granted that convnets work, so we're going to focus on the RL aspect solely, and not the vision component.



## Readings

* [Deep Q Networks Explained](https://www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained)
    * A high-level distillation as to how DQN works.
* [Andy Jones - Debugging RL, Without the Agonizing Pain](https://andyljones.com/posts/rl-debugging.html)
    * Useful tips for debugging your code when it's not working.
    * The "probe environments" (a collection of simple environments of increasing complexity) section will be our first line of defense against bugs.

### Interesting Resources (not required reading)

- [An Outsider's Tour of Reinforcement Learning](http://www.argmin.net/2018/06/25/outsider-rl/) - comparison of RL techniques with the engineering discipline of control theory.
- [Towards Characterizing Divergence in Deep Q-Learning](https://arxiv.org/pdf/1903.08894.pdf) - analysis of what causes learning to diverge
- [Divergence in Deep Q-Learning: Tips and Tricks](https://amanhussain.com/post/divergence-deep-q-learning/) - includes some plots of average returns for comparison
- [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures) - 2017 bootcamp with video and slides. Good if you like videos.
- [DQN debugging using OpenAI gym Cartpole](https://adgefficiency.com/dqn-debugging/) - random dude's adventures in trying to get it to work.
- [CleanRL DQN](https://github.com/vwxyzjn/cleanrl) - single file implementations of RL algorithms. Your starter code today is based on this; try not to spoiler yourself by looking at the solutions too early!
- [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html) - 2018 article describing difficulties preventing industrial adoption of RL.
- [Deep Reinforcement Learning Works - Now What?](https://tesslerc.github.io/posts/drl_works_now_what/) - 2020 response to the previous article highlighting recent progress.
- [Seed RL](https://github.com/google-research/seed_rl) - example of distributed RL using Docker and GCP.


## Fast Feedback Loops

We want to have faster feedback loops, and learning from Atari pixels doesn't achieve that. It might take 15 minutes per training run to get an agent to do well on Breakout, and that's if your implementation is relatively optimized. Even waiting 5 minutes to learn Pong from pixels is going to limit your ability to iterate, compared to using environments that are as simple as possible.


## CartPole

The classic environment "CartPole-v1" is simple to understand, yet hard enough for a RL agent to be interesting, by the end of the day your agent will be able to do this and more! (Click to watch!)


[![CartPole](https://img.youtube.com/vi/46wjA6dqxOM/0.jpg)](https://www.youtube.com/watch?v=46wjA6dqxOM "CartPole")

If you like, run `python play_cartpole.py` (locally, not on the remote machine)
to try having a go at the task yourself! Use Left/Right to move the cart, R to reset,
and Q to quit. By default, the cart will alternate Left/Right actions (there's no no-op action)
if you're not pressing a button.



The description of the task is [here](https://www.gymlibrary.dev/environments/classic_control/cart_pole/). Note that unlike the previous environments, the observation here is now continuous. You can see the source for CartPole [here](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py); don't worry about the implementation but do read the documentation to understand the format of the actions and observations.

The simple physics involved would be very easy for a model-based algorithm to fit, (this is a common assignment in control theory using [proportional-integral-derivative](https://en.wikipedia.org/wiki/PID_controller) (PID) controllers) but today we're doing it model-free: your agent has no idea that these observations represent positions or velocities, and it has no idea what the laws of physics are. The network has to learn in which direction to bump the cart in response to the current state of the world.

Each environment can have different versions registered to it. By consulting [the Gym source](https://github.com/openai/gym/blob/master/gym/envs/__init__.py) you can see that CartPole-v0 and CartPole-v1 are the same environment, except that v1 has longer episodes. Again, a minor change like this can affect what algorithms score well; an agent might consistently survive for 200 steps in an unstable fashion that means it would fall over if ran for 500 steps.


## Outline of the exercises

- Implement the Q-network that maps a state to an estimated value for each action.
- Implement the policy which chooses actions based on the Q-network, plus epsilon greedy randomness
to encourage exploration.
- Implement a replay buffer to store experiences $e_t = (s_t, a_t, r_{t+1}, s_{t+1})$.
- Piece everything together into a training loop and train your agent.


## The Q-Network

The Q-Network takes in an observation and outputs a number for each available action predicting how good it is, mimicking he behaviour of our Q-value table from yesterday.
For best results, the architecture of the Q-network can be customized to each particular problem. For example, [the architecture of OpenAI Five](https://cdn.openai.com/research-covers/openai-five/network-architecture.pdf) used to play DOTA 2 is pretty complex and involves LSTMs.

For learning from pixels, a simple convolutional network and some fully connected layers does quite well. Where we have already processed features here, it's even easier: an MLP of this size should be plenty large for any environment today. Your code should support running the network on either GPU or CPU, but for CartPole it was actually faster to use CPU on my hardware.

Implement the Q-network using a standard MLP, constructed of alternating Linear and ReLU layers.
The size of the input will match the dimensionality of the observation space, and the size of the output will match the number of actions to choose from (associating a reward to each.)
The dimensions of the hidden_sizes are provided.

Here is a diagram of what our particular Q-Network will look like for CartPole:

<figure style="max-width:250px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNplkEtrwzAMgP-K0WkDB5ash5LDTtmhUDa616UeRY3VxSx-4AdjlP73Oc3WpkwHIaTvA0l7aK0kqOHDo-vYSyMMyxHSdmw0RI6tigeKX9Z_jsMhFmYtYGFciuzKbsMmdOiIXwt4Z0Vxx5bKEPoyM2M1YVhZ3Zy4J1q-lushT73q7GWYs_nsQqj-CbdnYT7jzCS9wTYqa8JJXL1hn6nHFI87T5Dj1uNlZCRw0OQ1Kpmfsh_aAmJHmgTUuZS0w9RHAcIcMpqcxEj3UkXrod5hH4gDpmifv00LdfSJ_qBGYX6p_qUOP7mqd2g" /></figure>

<details>
<summary>Why do we not include a ReLU at the end?</summary>

If you end with a ReLU, then your network can only predict 0 or positive Q-values. This will cause problems as soon as you encounter an environment with negative rewards, or you try to do some scaling of the rewards.
</details>

<details>
<summary>CartPole-v1 gives +1 reward on every timestep. Why would the network not just learn the constant +1 function regardless of observation?</summary>

The network is learning Q-values (the sum of all future expected discounted rewards from this state/action pair), not rewards. Correspondingly, once the agent has learned a good policy, the Q-value associated with state action pair (pole is slightly left of vertical, move cart left) should be large, as we would expect a long episode (and correspondingly lots of reward) by taking actions to help to balance the pole. Pairs like (cart near right boundary, move cart right) cause the episode to terminate, and as such the network will learn low Q-values.
</details>


### Exercise - implement `QNetwork`

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠🟠⚪⚪

You should spend up to 10-15 minutes on this exercise.
```


```python
class QNetwork(nn.Module):
    '''For consistency with your tests, please wrap your modules in a `nn.Sequential` called `layers`.'''
    layers: nn.Sequential

    def __init__(
        self, 
        dim_observation: int, 
        num_actions: int, 
        hidden_sizes: List[int] = [120, 84]
    ):
        super().__init__()
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        pass


net = QNetwork(dim_observation=4, num_actions=2)
n_params = sum((p.nelement() for p in net.parameters()))
assert isinstance(getattr(net, "layers", None), nn.Sequential)
print(net)
print(f"Total number of parameters: {n_params}")
print("You should manually verify network is Linear-ReLU-Linear-ReLU-Linear")
assert n_params == 10934
```

<details>
<summary>Solution</summary>


```python
class QNetwork(nn.Module):
    '''For consistency with your tests, please wrap your modules in a `nn.Sequential` called `layers`.'''
    layers: nn.Sequential

    def __init__(
        self, 
        dim_observation: int, 
        num_actions: int, 
        hidden_sizes: List[int] = [120, 84]
    ):
        super().__init__()
        # SOLUTION
        in_features_list = [dim_observation] + hidden_sizes
        out_features_list = hidden_sizes + [num_actions]
        layers = []
        for i, (in_features, out_features) in enumerate(zip(in_features_list, out_features_list)):
            layers.append(nn.Linear(in_features, out_features))
            if i < len(in_features_list) - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        # SOLUTION
        return self.layers(x)
```
</details>


## Replay Buffer

The goal of DQN is to reduce the reinforcement learning problem to a supervised learning problem.
In supervised learning, training examples should be drawn **i.i.d**. from some distribution, and we hope to generalize to future examples from that distribution.

In RL, the distribution of experiences $e_t = (s_t, a_t, r_{t+1}, s_{t+1})$ to train from depend on the policy $\pi$ followed, which depends on the current state of the Q-value network, so DQN is always chasing a moving target. This is why the training loss curve isn't going to have a nice steady decrease like in supervised learning. We will extend experiences to $e_t = (o_t, a_t, r_{t+1}, o_{t+1}, d_{t+1})$. Here, $d_{t+1}$ is a boolean indicating that $o_{t+1}$ is a terminal observation, and that no further interaction happened beyond $s_{t+1}$ in the episode from which it was generated.

### Correlated States

Due to DQN using a neural network to learn the Q-values, the value of many state-action pairs are aggregated together (unlike tabular Q-learning which learns independently the value of each state-action pair). For example, consider a game of chess. The board will have some encoding as a vector, but visually similar board states might have wildly different consequences for the best move. Another problem is that states within an episode are highly correlated and not i.i.d. at all. A few bad moves from the start of the game might doom the rest of the game regardless how well the agent tries to recover, whereas a few bad moves near the end of the game might not matter if the agent has a very strong lead, or is so far behind the game is already lost. Training mostly on an episode where the agent opened the game poorly might disincentive good moves to recover, as these too will have poor Q-value estimates.

### Uniform Sampling

To recover from this problem and make the environment look "more i.i.d", a simple strategy that works decently well is to pick a buffer size, store experiences and uniformly sample out of that buffer. Intuitively, if we want the policy to play well in all sorts of states, the sampled batch should be a representative sample of all the diverse scenarios that can happen in the environment.

For complex environments, this implies a very large batch size (or doing something better than uniform sampling). [OpenAI Five](https://cdn.openai.com/dota-2.pdf) used batch sizes of over 2 million experiences for Dota 2.

The capacity of the replay buffer is yet another hyperparameter; if it's too small then it's just going to be full of recent and correlated examples. But if it's too large, we pay an increasing cost in memory usage and the information may be too old to be relevant.

Implement `ReplayBuffer`. It only needs to handle a discrete action space, and you can assume observations are some shape of dtype `np.float32`, and actions are of dtype `np.int64`. The replay buffer will store experiences $e_t = (o_t, a_t, r_{t+1}, o_{t+1}, d_{t+1})$ in a circular queue. If the buffer is already full, the oldest experience is overwritten.

You should also include objects `self.observations`, `self.actions`, etc in your `ReplayBuffer` class. This is just so that you can plot them against your shuffled replay buffer, and verify that the outputs look reasonable (see the next section).


```python
@dataclass
class ReplayBufferSamples:
    '''
    Samples from the replay buffer, converted to PyTorch for use in neural network training.
    '''
    observations: Float[Tensor, "sampleSize *obsShape"]
    actions: Int[Tensor, "sampleSize"]
    rewards: Float[Tensor, "sampleSize"]
    dones: Bool[Tensor, "sampleSize"]
    next_observations: Float[Tensor, "sampleSize *obsShape"]


class ReplayBuffer:
    '''
    Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.
    '''
    rng: Generator
    observations: t.Tensor
    actions: t.Tensor
    rewards: t.Tensor
    dones: t.Tensor
    next_observations: t.Tensor

    def __init__(self, buffer_size: int, num_environments: int, seed: int):
        assert num_environments == 1, "This buffer only supports SyncVectorEnv with 1 environment inside."
        pass

    def add(
        self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, next_obs: np.ndarray
    ) -> None:
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
        '''
        pass

    def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
        '''
        Uniformly sample sample_size entries from the buffer and convert them to PyTorch tensors on device.
        Sampling is with replacement, and sample_size may be larger than the buffer size.
        '''
        pass


tests.test_replay_buffer_single(ReplayBuffer)
tests.test_replay_buffer_deterministic(ReplayBuffer)
tests.test_replay_buffer_wraparound(ReplayBuffer)
```

<details>
<summary>Solution</summary>


```python
class ReplayBuffer:
    '''
    Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.
    '''
    rng: Generator
    observations: t.Tensor
    actions: t.Tensor
    rewards: t.Tensor
    dones: t.Tensor
    next_observations: t.Tensor

    def __init__(self, buffer_size: int, num_environments: int, seed: int):
        assert num_environments == 1, "This buffer only supports SyncVectorEnv with 1 environment inside."
        # SOLUTION
        self.buffer_size = buffer_size
        self.rng = np.random.default_rng(seed)
        self.buffer = [None for _ in range(5)]

    def add(
        self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, next_obs: np.ndarray
    ) -> None:
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
        '''
        # SOLUTION
        for i, (arr, arr_list) in enumerate(zip([obs, actions, rewards, dones, next_obs], self.buffer)):
            if arr_list is None:
                self.buffer[i] = arr
            else:
                self.buffer[i] = np.concatenate((arr, arr_list))
            if self.buffer[i].shape[0] > self.buffer_size:
                self.buffer[i] = self.buffer[i][:self.buffer_size]

        self.observations, self.actions, self.rewards, self.dones, self.next_observations = [t.as_tensor(arr) for arr in self.buffer]

    def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
        '''
        Uniformly sample sample_size entries from the buffer and convert them to PyTorch tensors on device.
        Sampling is with replacement, and sample_size may be larger than the buffer size.
        '''
        # SOLUTION
        indices = self.rng.integers(0, self.buffer[0].shape[0], sample_size)
        samples = [t.as_tensor(arr_list[indices], device=device) for arr_list in self.buffer]
        return ReplayBufferSamples(*samples)
```
</details>


## Environment Resets

There's a subtlety to the Gym API around what happens when the agent fails and the episode is terminated. Our environment is set up to automatically reset at the end of an episode, but when this happens the `next_obs` returned from `step` is actually the initial observation of the new episode.

What we want to store in the replay buffer is the final observation of the old episode. The code to do this is shown below.

- Run the code and inspect the replay buffer contents. Referring back to the [CartPole source](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py), do the numbers make sense?
- Look at the sample, and see if it looks properly shuffled.


```python
rb = ReplayBuffer(buffer_size=256, num_environments=1, seed=0)
envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 0, 0, False, "test")])
obs = envs.reset()
for i in range(256):
    actions = np.array([0])
    (next_obs, rewards, dones, infos) = envs.step(actions)
    real_next_obs = next_obs.copy()
    for (i, done) in enumerate(dones):
        if done:
            real_next_obs[i] = infos[i]["terminal_observation"]
    rb.add(obs, actions, rewards, dones, next_obs)
    obs = next_obs

plot_cartpole_obs_and_dones(rb.observations.flip(0), rb.dones.flip(0))

sample = rb.sample(256, t.device("cpu"))
plot_cartpole_obs_and_dones(sample.observations.flip(0), sample.dones.flip(0))
```

<details>
<summary>You should be getting graphs which look like this:</summary>

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/shuffled_and_un.png" width="800">

Explanations - each of the dotted lines (the values $t^*$ where $d_{t^*}=1$) corresponds to a state $s_{t^*}$ where the pole's angle goes over the [bounds](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) of `+=0.2095` (note that it doesn't stay upright nearly long enough to hit the positional bounds). If you zoom in on one of these points, then you'll see that we never actually record observations when the pole is out of bounds. At $s_{t^*-1}$ we are still within bounds, and once we go over bounds the next observation is taken from the reset environment.

</details>


## Exploration

DQN makes no attempt to explore intelligently. The exploration strategy is the same as
for Q-Learning: agents take a random action with probability epsilon, but now we gradually
decrease epsilon. The Q-network is also randomly initialized (rather than initialized with zeros),
so its predictions of what is the best action to take are also pretty random to start.

Some games like [Montezuma's Revenge](https://paperswithcode.com/task/montezumas-revenge) have sparse rewards that require more advanced exploration methods to obtain. The player is required to collect specific keys to unlock specific doors, but unlike humans, DQN has no prior knowledge about what a key or a door is, and it turns out that bumbling around randomly has too low of a probability of correctly matching a key to its door. Even if the agent does manage to do this, the long separation between finding the key and going to the door makes it hard to learn that picking the key up was important.

As a result, DQN scored an embarrassing 0% of average human performance on this game.

### Reward Shaping

One solution to sparse rewards is to use human knowledge to define auxillary reward functions that are more dense and made the problem easier (in exchange for leaking in side knowledge and making
the algorithm more specific to the problem at hand). What could possibly go wrong?

The canonical example is for a game called [CoastRunners](https://openai.com/blog/faulty-reward-functions/), where the goal was given to maximize the
score (hoping that the agent would learn to race around the map). Instead, it found it could
gain more score by driving in a loop picking up power-ups just as they respawn, crashing and
setting the boat alight in the process.

### Reward Hacking

For Montezuma's Revenge, the reward was shaped by giving a small reward for
picking up the key.
One time this was tried, the reward was given slightly too early and the agent learned it could go close to the key without quite picking it up, obtain the auxillary reward, and then back up and repeat.

[![Montezuma Reward Hacking](https://img.youtube.com/vi/_sFp1ffKIc8/0.jpg)](https://www.youtube.com/watch?v=_sFp1ffKIc8 "Montezuma Reward Hacking")

A collected list of examples of Reward Hacking can be found [here](https://docs.google.com/spreadsheets/d/e/2PACX-1vRPiprOaC3HsCf5Tuum8bRfzYUiKLRqJmbOoC-32JorNdfyTiRRsR7Ea5eWtvsWzuxo8bjOxCG84dAg/pubhtml).


### Advanced Exploration

It would be better if the agent didn't require these auxillary rewards to be hardcoded by humans,
but instead reply on other signals from the environment that a state might be worth exploring. One idea is that a state which is "surprising" or "novel" (according to the agent's current belief
of how the environment works) in some sense might be valuable. Designing an agent to be
innately curious presents a potential solution to exploration, as the agent will focus exploration
in areas it is unfamiliar with. In 2018, OpenAI released [Random Network Distillation](https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/) which made progress in formalizing this notion, by measuring the agent's ability to predict the output of a neural network
on visited states. States that are hard to predict are poorly explored, and thus highly rewarded.
In 2019, an excellent paper [First return, then explore](https://arxiv.org/pdf/2004.12919v6.pdf) found an even better approach. Such reward shaping can also be gamed, leading to the
noisy TV problem, where agents that seek novelty become entranced by a source of randomness in the
environment (like a analog TV out of tune displaying white noise), and ignore everything else
in the environment.


### Exercise - implement linear scheduler

```c
Difficulty: 🟠⚪⚪⚪⚪
Importance: 🟠🟠⚪⚪⚪

You should spend up to 5-10 minutes on this exercise.
```

For now, implement the basic linearly decreasing exploration schedule.


```python
def linear_schedule(
    current_step: int, start_e: float, end_e: float, exploration_fraction: float, total_timesteps: int
) -> float:
    '''Return the appropriate epsilon for the current step.

    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).

    It should stay at end_e for the rest of the episode.
    '''
    pass


epsilons = [
    linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500)
    for step in range(500)
]
line(epsilons, labels={"x": "steps", "y": "epsilon"}, title="Probability of random action")

tests.test_linear_schedule(linear_schedule)
```

<details>
<summary>Plot of the Intended Schedule</summary>

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/newplot.png" width="560">
</details>

<details>
<summary>Solution</summary>


```python
def linear_schedule(
    current_step: int, start_e: float, end_e: float, exploration_fraction: float, total_timesteps: int
) -> float:
    '''Return the appropriate epsilon for the current step.

    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).

    It should stay at end_e for the rest of the episode.
    '''
    # SOLUTION
    return start_e + (end_e - start_e) * min(current_step / (exploration_fraction * total_timesteps), 1)
```
</details>


## Epsilon Greedy Policy

In DQN, the policy is implicitly defined by the Q-network: we take the action with the maximum predicted reward. This gives a bias towards optimism. By estimating the maximum of a set of values $v_1, \ldots, v_n$ using the maximum of some noisy estimates $\hat{v}_1, \ldots, \hat{v}_n$ with $\hat{v}_i \approx v$, we get unlucky and get very large positive noise on some samples, which the maximum then chooses. Hence, the agent will choose actions that the Q-network is overly optimistic about.

See Sutton and Barto, Section 6.7 if you'd like a more detailed explanation, or the original [Double Q-Learning](https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf) paper which notes this maximisation bias, and introduces a method to correct for it using two separate Q-value estimators, each used to update the other.



### Exercise - implement the epsilon greedy policy

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠🟠⚪⚪

You should spend up to 10-20 minutes on this exercise.
```

- Don't forget to convert the result back to a `np.darray`.
- Use `rng.random()` to generate random numbers in the range $[0,1)$, and `rng.integers(0, n, size)` for an array of shape `size` random integers in the range $0, 1, \ldots, n-1$.
- Use `envs.single_action_space.n` to retrieve the number of possible actions.


```python
def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv, q_network: QNetwork, rng: Generator, obs: t.Tensor, epsilon: float
) -> np.ndarray:
    '''With probability epsilon, take a random action. Otherwise, take a greedy action according to the q_network.
    Inputs:
        envs : gym.vector.SyncVectorEnv, the family of environments to run against
        q_network : QNetwork, the network used to approximate the Q-value function
        obs : The current observation
        epsilon : exploration percentage
    Outputs:
        actions: (n_environments, ) the sampled action for each environment.
    '''
    pass


tests.test_epsilon_greedy_policy(epsilon_greedy_policy)
```

<details>
<summary>Solution</summary>


```python
def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv, q_network: QNetwork, rng: Generator, obs: t.Tensor, epsilon: float
) -> np.ndarray:
    '''With probability epsilon, take a random action. Otherwise, take a greedy action according to the q_network.
    Inputs:
        envs : gym.vector.SyncVectorEnv, the family of environments to run against
        q_network : QNetwork, the network used to approximate the Q-value function
        obs : The current observation
        epsilon : exploration percentage
    Outputs:
        actions: (n_environments, ) the sampled action for each environment.
    '''
    # SOLUTION
    num_actions = envs.single_action_space.n
    if rng.random() < epsilon:
        return rng.integers(0, num_actions, size = (envs.num_envs,))
    else:
        q_scores = q_network(obs)
        return q_scores.argmax(-1).detach().cpu().numpy()
```
</details>


## Probe Environments

Extremely simple probe environments are a great way to debug your algorithm. The first one is given to you.

Let's try and break down how this environment works. We see that the function `step` always returns the same thing. The observation and reward are always the same, and `done` is always true (i.e. the episode always terminates after one action). We expect the agent to rapidly learn that the value of the constant observation `[0.0]` is `+1`.

### A note on action spaces

The action space we're using here is `gym.spaces.Box`. This means we're dealing with real-valued quantities, i.e. continuous not discrete. The first two arguments of `Box` are `low` and `high`, and these define a box in $\mathbb{R}^n$. For instance, if these arrays are `(0, 0)` and `(1, 1)` respectively, this defines the box $0 \leq x, y \leq 1$ in 2D space.


```python
ObsType = np.ndarray
ActType = int
    
class Probe1(gym.Env):
    '''One action, observation of [0.0], one timestep long, +1 reward.

    We expect the agent to rapidly learn that the value of the constant [0.0] observation is +1.0. Note we're using a continuous observation space for consistency with CartPole.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([0]), np.array([0]))
        self.action_space = Discrete(1)
        self.seed()
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        return (np.array([0]), 1.0, True, {})

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)
        if return_info:
            return (np.array([0.0]), {})
        return np.array([0.0])


gym.envs.registration.register(id="Probe1-v0", entry_point=Probe1)
env = gym.make("Probe1-v0")
assert env.observation_space.shape == (1,)
assert env.action_space.shape == ()
```

### Exercise - implement additional probe environments

```c
Difficulty: 🟠🟠🟠🟠⚪
Importance: 🟠⚪⚪⚪⚪

You should spend up to 10-30 minutes on this exercise.

You should look at solutions if you're stuck. It's very important to have working probe environments to debug your algorithm.
```

Feel free to skip ahead for now, and implement these as needed to debug your model. 

Each implementation should be very similar to `Probe1` above. If you aren't sure whether you've implemented them correctly, you can check them against the solutions. Make sure you check them againts the solutions eventually - it's hard enough debugging your RL algorithms without also worrying about whether the probe environments you're using for debugging are correct!


```python
class Probe2(gym.Env):
    '''One action, observation of [-1.0] or [+1.0], one timestep long, reward equals observation.

    We expect the agent to rapidly learn the value of each observation is equal to the observation.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        pass

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        pass


gym.envs.registration.register(id="Probe2-v0", entry_point=Probe2)
    
    
class Probe3(gym.Env):
    '''One action, [0.0] then [1.0] observation, two timesteps, +1 reward at the end.

    We expect the agent to rapidly learn the discounted value of the initial observation.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        pass

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        pass


gym.envs.registration.register(id="Probe3-v0", entry_point=Probe3)
    
    
class Probe4(gym.Env):
    '''Two actions, [0.0] observation, one timestep, reward is -1.0 or +1.0 dependent on the action.

    We expect the agent to learn to choose the +1.0 action.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        pass

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        pass


gym.envs.registration.register(id="Probe4-v0", entry_point=Probe4)
    
    
class Probe5(gym.Env):
    '''Two actions, random 0/1 observation, one timestep, reward is 1 if action equals observation otherwise -1.

    We expect the agent to learn to match its action to the observation.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        pass

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        pass


gym.envs.registration.register(id="Probe5-v0", entry_point=Probe5)
```

<details>
<summary>Solution</summary>


```python
class Probe2(gym.Env):
    '''One action, observation of [-1.0] or [+1.0], one timestep long, reward equals observation.

    We expect the agent to rapidly learn the value of each observation is equal to the observation.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        # SOLUTION
        super().__init__()
        self.observation_space = Box(np.array([-1.0]), np.array([+1.0]))
        self.action_space = Discrete(1)
        self.reset()
        self.reward = None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # SOLUTION
        assert self.reward is not None
        return np.array([self.observation]), self.reward, True, {}

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        # SOLUTION
        super().reset(seed=seed)
        self.reward = 1.0 if self.np_random.random() < 0.5 else -1.0
        self.observation = self.reward
        if return_info:
            return np.array([self.reward]), {}
        return np.array([self.reward])

class Probe3(gym.Env):
    '''One action, [0.0] then [1.0] observation, two timesteps, +1 reward at the end.

    We expect the agent to rapidly learn the discounted value of the initial observation.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        # SOLUTION
        super().__init__()
        self.observation_space = Box(np.array([-0.0]), np.array([+1.0]))
        self.action_space = Discrete(1)
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # SOLUTION
        self.n += 1
        if self.n == 1:
            return np.array([1.0]), 0.0, False, {}
        elif self.n == 2:
            return np.array([0.0]), 1.0, True, {}
        raise ValueError(self.n)

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        # SOLUTION
        super().reset(seed=seed)
        self.n = 0
        if return_info:
            return np.array([0.0]), {}
        return np.array([0.0])

class Probe4(gym.Env):
    '''Two actions, [0.0] observation, one timestep, reward is -1.0 or +1.0 dependent on the action.

    We expect the agent to learn to choose the +1.0 action.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        # SOLUTION
        self.observation_space = Box(np.array([-0.0]), np.array([+0.0]))
        self.action_space = Discrete(2)
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # SOLUTION
        reward = -1.0 if action == 0 else 1.0
        return np.array([0.0]), reward, True, {}

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        # SOLUTION
        super().reset(seed=seed)
        if return_info:
            return np.array([0.0]), {}
        return np.array([0.0])

class Probe5(gym.Env):
    '''Two actions, random 0/1 observation, one timestep, reward is 1 if action equals observation otherwise -1.

    We expect the agent to learn to match its action to the observation.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        # SOLUTION
        self.observation_space = Box(np.array([-1.0]), np.array([+1.0]))
        self.action_space = Discrete(2)
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # SOLUTION
        reward = 1.0 if action == self.obs else -1.0
        return np.array([self.obs]), reward, True, {}

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        # SOLUTION
        super().reset(seed=seed)
        self.obs = 1.0 if self.np_random.random() < 0.5 else 0.0
        if return_info:
            return np.array([self.obs], dtype=float), {}
        return np.array([self.obs], dtype=float)
```
</details>


## Weights and Biases

In previous parts, we've just trained the agent, and then plotted the reward per episode after training. For small toy examples that train in a few seconds this is fine, but for longer runs we'd like to watch the run live and make sure the agent is doing something interesting (especially if we were planning to run the model overnight.)

Luckily, **Weights and Biases** has got us covered! When you run your experiments, you'll be able to view not only *live plots* of the loss and average reward per episode while the agent is training - you can also log and view animations, which visualise your agent's progress in real time! The code below will handle all logging.


## Main DQN Algorithm

We now combine all the elements we have designed thus far into the final DQN algorithm. Here, we assume the environment returns three parameters $(s_{new}, r, d)$, a new state $s_{new}$, a reward $r$ and a boolean $d$ indicating whether interaction has terminated yet.

Our Q-value function $Q(s,a)$ is now a network $Q(s,a ; \theta)$ parameterised by weights $\theta$. The key idea, as in Q-learning, is to ensure the Q-value function satisfies the optimal Bellman equation
$$
Q(s,a ; \theta)
= \mathbb{E}_{s',r \sim p(\cdot \mid s,a)} \left[r + \gamma \max_{a'} Q(s', a' ;\theta) \right]
$$

$$
\delta_t = \mathbb{E} \left[ r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$
Letting $y_t = r_t + \gamma \max_a Q(s_{t+1}, a)$, we can use the expected squared TD-Error $\delta_t^2 = (y_t - Q(s_t, a_t))^2$ as the loss function to optimize against. Since we want the modle to learn from a variety of experiences (recall that supervised learning is assuming i.i.d) we approximate the expectation by sampling a batch $B = \{s^i, a^i, r^i, s^i_\text{new}\}$ of experiences from the replay buffer, and try to adjust $\theta$ to make the loss
$$
L(\theta) = \frac{1}{|B|} \sum_{i=1}^B \left( r^i +
\gamma \max_a Q(s^i_\text{new}, a ; \theta_\text{target}) - Q(s^i, a^i ; \theta) \right)^2
$$
smaller via gradient descent. Here, $\theta_\text{target}$ is a previous copy of the parameters $\theta$. Every so often, we then update the target $\theta_\text{target} \leftarrow \theta$ as the agent improves it's Q-values from experience.

The image below uses the notation $\llbracket S \rrbracket$ - this means 1 if $S$ is True, and 0 if $S$ is False.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/dqn_algo.png" width="550">

On line 13, we need to check if the environment is still running before adding the $\gamma \max_{a'}Q(s^i_\text{new}, a' ; \theta_\text{target})$ term, as terminal states don't have future rewards, so we zero the second term if $d^i = \text{True}$, indicating that the episode has terminated.


Below is a dataclass for training your DQN. You can use the `arg_help` method to see a description of each argument (it will also highlight any arguments which have ben changed from their default values).

The exact breakdown of training is as follows:

* The agent takes `total_timesteps` steps in the environment during the training loop.
* The first `buffer_size` of these steps are used to fill the replay buffer (we don't update gradients until the buffer is full).
* After this point, we perform an optimizer step every `train_frequency` steps of our agent.

This is shown in the diagram below (not to scale!).

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/dqn_breakdown.png" width="500">


```python
@dataclass
class DQNArgs:
    exp_name: str = "DQN_implementation"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = t.cuda.is_available()
    log_dir: str = "logs"
    use_wandb: bool = True
    wandb_project_name: str = "CartPoleDQN"
    wandb_entity: Optional[str] = None
    capture_video: bool = True
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500_000
    learning_rate: float = 0.00025
    buffer_size: int = 10_000
    gamma: float = 0.99
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.1
    exploration_fraction: float = 0.2
    train_frequency: int = 10
    log_frequency: int = 50

    def __post_init__(self):
        assert self.total_timesteps - self.buffer_size >= self.train_frequency
        self.total_training_steps = (self.total_timesteps - self.buffer_size) // self.train_frequency



args = DQNArgs(batch_size=256)
utils.arg_help(args)
```

### Exercise - fill in the agent class

```c
Difficulty: 🟠🟠🟠🟠⚪
Importance: 🟠🟠🟠🟠⚪

You should spend up to 25-49 minutes on this exercise.
```

You should now fill in the methods for the `DQNAgent` class below. This is a class which is designed to handle taking steps in the environment (with an epsilon greedy policy), and updating the buffer.

The `play_step` function should do the following:

* Get a new set of actions via the `self.get_actions` method (taking `self.next_obs` as our current observation)
* Step the environment, via `self.envs.step` (which returns a new set of experiences)
* Add the new experiences to the buffer
* Set `self.next_obs` to the new observations (this is so the agent knows where it is for the next step)
* Increment the global step counter
* Return the diagnostic information from the new experiences (i.e. the `infos` dicts)

The `get_actions` function should do the following:

* Set `self.epsilon` according to the linear schedule, and current timestep
* Sample actions according to the epsilon-greedy policy

Note - for this exercise and others to follow, there's a trade-off in the test functions between being strict and being lax. Too lax and the tests will let failures pass; too strict and they might fail for odd reasons even if your code is mostly correct. If you find youself continually failing tests then you should ask a TA for help.

```python
class DQNAgent:
    '''Base Agent class handeling the interaction with the environment.'''

    def __init__(
        self, 
        envs: gym.vector.SyncVectorEnv, 
        args: DQNArgs, 
        rb: ReplayBuffer,
        q_network: QNetwork,
        target_network: QNetwork,
        rng: np.random.Generator
    ):
        self.envs = envs
        self.args = args
        self.rb = rb
        self.next_obs = self.envs.reset() # Need a starting observation!
        self.steps = 0
        self.epsilon = args.start_e
        self.q_network = q_network
        self.target_network = target_network
        self.rng = rng

    def play_step(self) -> List[dict]:
        '''
        Carries out a single interaction step between the agent and the environment, and adds results to the replay buffer.
        
        Returns `infos` (list of dictionaries containing info we will log).
        '''
        pass

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        '''
        Samples actions according to the epsilon-greedy policy using the linear schedule for epsilon.
        '''
        pass


tests.test_agent(DQNAgent)
```

<details>
<summary>Solution</summary>


```python
class DQNAgent:
    '''Base Agent class handeling the interaction with the environment.'''

    def __init__(
        self, 
        envs: gym.vector.SyncVectorEnv, 
        args: DQNArgs, 
        rb: ReplayBuffer,
        q_network: QNetwork,
        target_network: QNetwork,
        rng: np.random.Generator
    ):
        self.envs = envs
        self.args = args
        self.rb = rb
        self.next_obs = self.envs.reset() # Need a starting observation!
        self.steps = 0
        self.epsilon = args.start_e
        self.q_network = q_network
        self.target_network = target_network
        self.rng = rng

    def play_step(self) -> List[dict]:
        '''
        Carries out a single interaction step between the agent and the environment, and adds results to the replay buffer.
        
        Returns `infos` (list of dictionaries containing info we will log).
        '''
        # SOLUTION
        obs = self.next_obs
        actions = self.get_actions(obs)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.rb.add(obs, actions, rewards, dones, next_obs)

        self.next_obs = next_obs
        self.steps += 1
        return infos

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        '''
        Samples actions according to the epsilon-greedy policy using the linear schedule for epsilon.
        '''
        # SOLUTION
        self.epsilon = linear_schedule(self.steps, args.start_e, args.end_e, args.exploration_fraction, args.total_timesteps)
        actions = epsilon_greedy_policy(self.envs, self.q_network, self.rng, t.tensor(obs).to(device), self.epsilon)
        assert actions.shape == (len(self.envs.envs),)
        return actions
```
</details>


### Exercise - write DQN training loop

```c
Difficulty: 🟠🟠🟠🟠🟠
Importance: 🟠🟠🟠🟠🟠

You should spend up to 30-50 minutes on this exercise.
```

If you did the material on [PyTorch Lightning](https://arena-ch0-fundamentals.streamlit.app/[0.3]_ResNets#pytorch-lightning) during the first week, this should all be familiar to you. If not, a little refresher:

<details>
<summary>Click here for a basic refresher on PyTorch Lightning & Weights and Biases</summary>

PyTorch Lightining (which we'll import as `pl`) is a useful tool for cleaning up and modularizing your PyTorch training code.

We can define a class which inherits from `pl.LightningModule`, which will contain both our model and instructions for what the training steps look like. This class should have at minimum the following 2 methods:

* `training_step` - compute and return the training loss on a single batch (plus optionally log metrics)
* `configure_optimizers` - return the optimizers you want to use for training

Other optional methods include:

* `forward` - which acts like `forward` for a regular `nn.Module` object
* `on_train_epoch_end` - runs once when the training epoch ends
* `train_dataloader` - returns a dataloader (or other iterable) which is iterated over to give us the `batch` argument in `training_step`

Once you have your class, you need to take the following steps to run your training loop:

* Create an instance of that class, e.g. `model = LitTransformer(...)`
* Define a trainer, e.g. `trainer = pl.Trainer(max_epochs=...)`

Weights and Biases is a useful service which visualises training runs and performs hyperparameter sweeps. If you want to log to Weights and Biases, you need to amend the following:

* Define `logger = WandbLogger(save_dir=..., project=...)`, and pass this to your `Trainer` instance (along with other optional arguments e.g. `log_every_n_steps`).
* Remember to call `wandb.finish()` at the end of your training instance.

</details>

There are 3 methods you'll need to fill in below:

#### `__init__`

We've given you some of the code here. You should add code to do the following:

* Defining all the objects which are type-annotated just below the class definition (i.e. `q_network`, `target_network`, etc).
    * Make sure you match the weights in `q_network` and `target_network` at the start of training (you can use `net2.load_state_dict(net1.state_dict())` to do this).
* Run the first `args.buffer_size` steps of the agent (this fills your buffer).
    * You don't need to do anything with the `infos` dicts returned here.

#### `training_step`

This method contains most of the logic for DQN. You should do the following:

* Step the agent, for `args.train_frequency` steps.
    * This is because the `training_step` method corresponds to one step of the optimizer for our model, and we need to take `train_frequency` steps per model update.
* Get a new sample fro your buffer, of size `args.batch_size` **(line 12 of algorithm 1)**.
* Calculate your loss **(lines, 13, 14)**.
* Optionally log variables, using the `_log` method we've given you.
* If the current global step divides `args.target_network_frequency`, then update the target network weights to match the current network weights **(lines 17 & 18)**.
* Return the loss. PyTorch Lightning will implicitly handle the optimizer step **(line 15)** for you.

Note - because we don't really have a dataloader in the conventional sense (we're just sampling repeatedly from our buffer), the `train_dataloader` method just returns a `range` object which is iterated over but not used in our `training_step` method.

#### `configure_optimizers`

Return the Adam optimizer, with learning rate from `args`. Make sure you're passing the right model parameters to the optimizer!

#### A few other notes on this code

* We've given you a `on_train_epoch_end` method which runs when the epoch finishes. The main purpose of this is to test your probe environments - it checks whether you used one of the probes, and if you did then it checks that the value of each state has converged to the correct values for that probe. Below, we've also given you some boilerplate code to test one of the probe environments. If you didn't finish implementing your probes (or you've not yet compared them to the solutions), now would be a great time to go and do that!


Don't be discouraged if your code takes a while to work - it's normal for debugging RL to take longer than you would expect. Add asserts or your own tests, implement an appropriate probe environment, try anything in the Andy Jones post that sounds promising, and try to notice confusion. Reinforcement Learning is often so tricky as even if the algorithm has bugs, the agent might still learn something useful regardless (albeit maybe not as well), or even if everything is correct, the agent might just fail to learn anything useful (like how DQN failed to do anything on Montezuma's Revenge.)

Since the environment is already know to be one DQN can solve, and we've already provided hyperparameters that work for this environment, hopefully that's isolated a lot of the problems one would usually have with solving real world problems with RL.

<details>
<summary>Expected Behavior of the Loss</summary>

In supervised learning, we want our loss to always be decreasing and it's a bad sign if it's actually increasing, as that's the only metric we care about. In RL, it's the total reward per epsiode that should be (noisily) increasing over time.

Our agent's loss function just reflects how close together the Q-network's estimates are to the experiences currently sampled from the replay buffer, which might not adequately represent what the world actually looks like.

This means that once the agent starts to learn something and do better at the problem, it's expected for the loss to increase. The loss here is just the TD-error, the difference between how valuable the agent thinks the (state-action) is, v.s. the best  current bootstrapped estimate of the actual Q-value.

For example, the Q-network initially learned some state was bad, because an agent that reached them was just flapping around randomly and died shortly after. But now it's getting evidence that the same state is good, now that the agent that reached the state has a better idea what to do next. A higher loss is thus actually a good sign that something is happening (the agent hasn't stagnated), but it's not clear if it's learning anything useful without also checking how the total reward per episode has changed.
</details>


```python
class DQNLightning(pl.LightningModule):
    q_network: QNetwork
    target_network: QNetwork
    rb: ReplayBuffer
    agent: DQNAgent

    def __init__(self, args: DQNArgs):
        super().__init__()
        self.args = args
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, self.run_name)])
        self.start_time = time.time()
        self.rng = np.random.default_rng(args.seed)

        # YOUR CODE HERE!
        pass

        
    def _log(self, predicted_q_vals: t.Tensor, epsilon: float, loss: Float[Tensor, ""], infos: List[dict]) -> None:
        log_dict = {"td_loss": loss, "q_values": predicted_q_vals.mean().item(), "SPS": int(self.agent.steps / (time.time() - self.start_time))}
        for info in infos:
            if "episode" in info.keys():
                log_dict.update({"episodic_return": info["episode"]["r"], "episodic_length": info["episode"]["l"], "epsilon": epsilon})
        self.log_dict(log_dict)


    def training_step(self, batch: Any) -> Float[Tensor, ""]:
        # YOUR CODE HERE!
        pass

        
    def configure_optimizers(self):
        # YOUR CODE HERE!
        pass
        

    def on_train_epoch_end(self):
        obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
        expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]], [[args.gamma], [1.0]], [[-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]
        tolerances = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]
        match = re.match(r"Probe(\d)-v0", args.env_id)
        if match:
            probe_idx = int(match.group(1)) - 1
            obs = t.tensor(obs_for_probes[probe_idx]).to(device)
            value = self.q_network(obs)
            print("Value: ", value)
            expected_value = t.tensor(expected_value_for_probes[probe_idx]).to(device)
            t.testing.assert_close(value, expected_value, atol=tolerances[probe_idx], rtol=0)
            print("Probe tests passed!")
        self.envs.close()


    def train_dataloader(self):
        '''We don't use a trainloader in the traditional sense, so we'll just have this.'''
        return range(self.args.total_training_steps)


```

<details>
<summary>Solution</summary>


```python
class DQNLightning(pl.LightningModule):
    q_network: QNetwork
    target_network: QNetwork
    rb: ReplayBuffer
    agent: DQNAgent

    def __init__(self, args: DQNArgs):
        super().__init__()
        self.args = args
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, self.run_name)])
        self.start_time = time.time()
        self.rng = np.random.default_rng(args.seed)

        # SOLUTION
        num_actions = self.envs.single_action_space.n
        obs_shape = self.envs.single_observation_space.shape
        num_observations = np.array(obs_shape, dtype=int).prod()

        self.q_network = QNetwork(num_observations, num_actions).to(device)
        self.target_network = QNetwork(num_observations, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.rb = ReplayBuffer(args.buffer_size, len(self.envs.envs), args.seed)
        self.agent = DQNAgent(self.envs, self.args, self.rb, self.q_network, self.target_network, self.rng)
        
        for step in tqdm(range(args.buffer_size), desc="Filling initial replay buffer"):
            infos = self.agent.play_step()


    def _log(self, predicted_q_vals: t.Tensor, epsilon: float, loss: Float[Tensor, ""], infos: List[dict]) -> None:
        log_dict = {"td_loss": loss, "q_values": predicted_q_vals.mean().item(), "SPS": int(self.agent.steps / (time.time() - self.start_time))}
        for info in infos:
            if "episode" in info.keys():
                log_dict.update({"episodic_return": info["episode"]["r"], "episodic_length": info["episode"]["l"], "epsilon": epsilon})
        self.log_dict(log_dict)


    def training_step(self, batch: Any) -> Float[Tensor, ""]:
        # SOLUTION

        for step in range(args.train_frequency):
            infos = self.agent.play_step()

        data = self.rb.sample(args.batch_size, device)
        s, a, r, d, s_new = data.observations, data.actions, data.rewards, data.dones, data.next_observations

        with t.inference_mode():
            self.target_network.requires_grad_(False)
            target_max = self.target_network(s_new).max(-1).values
        predicted_q_vals = self.q_network(s)[range(args.batch_size), a.flatten()]

        td_error = r.flatten() + args.gamma * target_max * (1 - d.float().flatten()) - predicted_q_vals
        loss = td_error.pow(2).mean()
        
        self._log(predicted_q_vals, self.agent.epsilon, loss, infos)

        if self.agent.steps % args.target_network_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss


    def configure_optimizers(self):
        # SOLUTION
        return t.optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
    

    def on_train_epoch_end(self):
        obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
        expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]], [[args.gamma], [1.0]], [[-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]
        tolerances = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]
        match = re.match(r"Probe(\d)-v0", args.env_id)
        if match:
            probe_idx = int(match.group(1)) - 1
            obs = t.tensor(obs_for_probes[probe_idx]).to(device)
            value = self.q_network(obs)
            print("Value: ", value)
            expected_value = t.tensor(expected_value_for_probes[probe_idx]).to(device)
            t.testing.assert_close(value, expected_value, atol=tolerances[probe_idx], rtol=0)
            print("Probe tests passed!")
        self.envs.close()


    def train_dataloader(self):
        '''We don't use a trainloader in the traditional sense, so we'll just have this.'''
        return range(self.args.total_training_steps)
```
</details>


Here's some boilerplate code to run one of your probes (change the `probe_idx` variable to try out different probes):


```python
probe_idx = 1

args = DQNArgs(
    env_id=f"Probe{probe_idx}-v0",
    exp_name=f"test-probe-{probe_idx}", 
    total_timesteps=3000,
    learning_rate=0.001,
    buffer_size=500,
    capture_video=False,
    use_wandb=False
)
model = DQNLightning(args).to(device)
logger = CSVLogger(save_dir=args.log_dir, name=model.run_name)

trainer = pl.Trainer(
    max_steps=args.total_training_steps,
    logger=logger,
    log_every_n_steps=1,
)
trainer.fit(model=model)

metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
px.line(metrics, y="q_values", labels={"x": "Step"}, title="Probe 1 (if you're seeing this, then you passed the tests!)", width=600, height=400)
```

And here's some code to run the full version of your model, using weights and biases. Note the `wandb.gym.monitor()` line, which makes sure that we log media to weights and biases. This needs to be called *after* `wandb.init()` (which is called implicitly when we define our logger).


```python
wandb.finish()

args = DQNArgs()
logger = WandbLogger(save_dir=args.log_dir, project=args.wandb_project_name, name=model.run_name)
if args.use_wandb: wandb.gym.monitor() # Makes sure we log video!
model = DQNLightning(args).to(device)

trainer = pl.Trainer(
    max_epochs=1,
    max_steps=args.total_timesteps,
    logger=logger,
    log_every_n_steps=args.log_frequency,
)
trainer.fit(model=model)
```

## Beyond CartPole

If things go well and your agent masters CartPole, the next harder challenges are [Acrobot-v1](https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py), and [MountainCar-v0](https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py). These also have discrete action spaces, which are the only type we're dealing with today. Feel free to Google for appropriate hyperparameters for these other problems - in a real RL problem you would have to do hyperparameter search using the techniques we learned on a previous day because bad hyperparameters in RL often completely fail to learn, even if the algorithm is perfectly correct.

There are many more exciting environments to play in, but generally they're going to require more compute and more optimization than we have time for today. If you finish the main material, some ones I like are:

- [Minimalistic Gridworld Environments](https://github.com/Farama-Foundation/gym-minigrid) - a fast gridworld environment for experiments with sparse rewards and natural language instruction.
- [microRTS](https://github.com/santiontanon/microrts) - a small real-time strategy game suitable for experimentation.
- [Megastep](https://andyljones.com/megastep/) - RL environment that runs fully on the GPU (fast!)
- [Procgen](https://github.com/openai/procgen) - A family of 16 procedurally generated gym environments to measure the ability for an agent to generalize. Optimized to run quickly on the CPU.



## Bonus

### Target Network

Why have the target network? Modify the DQN code above, but this time use the same network for both the target and the Q-value network, rather than updating the target every so often.

Compare the performance of this against using the target network.

### Shrink the Brain

Can DQN still learn to solve CartPole with a Q-network with fewer parameters? Could we get away with three-quarters or even half as many parameters? Try comparing the resulting training curves with a shrunken version of the Q-network. What about the same number of parameters, but with more/less layers, and less/more parameters per layer?

### Dueling DQN

Implement dueling DQN according to [the paper](https://arxiv.org/pdf/1511.06581.pdf) and compare its performance.




""", unsafe_allow_html=True)


func_page_list = [
    (section_0, "🏠 Home"),     (section_1, "1️⃣ Q-Learning"),     (section_2, "2️⃣ Deep Q-Learning"), 
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
