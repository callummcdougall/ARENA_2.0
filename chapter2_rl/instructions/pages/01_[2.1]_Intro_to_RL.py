
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
<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/bandit.png" width="350">

Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.


# [2.1] - Intro to RL


## Introduction


This section is designed to bring you up to speed with the basics of reinforcement learning. Before we cover the big topics like PPO and RLHF, we need to build a strong foundation by understanding what RL is and what kinds of problems it was designed to solve.

## Content & Learning Objectives


#### 1️⃣ Multi-Armed Bandit

In Part 1 we'll study the multi-armed bandit problem, which is simple yet captures introduces many of the difficulties in RL. Many practical problems can be posed in terms of generalizations of the multi-armed bandit. For example, the Hyperband algorithm for hyperparameter optimization is based on the multi-armed bandit with an infinite number of arms.

> ##### Learning Objectives
> 
> * Understand the anatomy of a `gym.Env`, so that you feel comfortable using them and writing your own
> * Practice RL in the tabular (lookup table) setting before adding the complications of neural networks
> * Understand the difficulty of optimal exploration
> * Understand that performance is extremely variable in RL, and how to measure performance

#### 2️⃣ Tabular RL & Policy Improvement

Next, we'll start to get into the weeds of some of the mathematical formulations of RL. Some important concepts to cover are **Markov processes**, the **value function** and the **Bellman equation**. 

We'll then put this into practice on some basic gridworld environments.

> ##### Learning Objectives
> 
> * Understand the tabular RL problem, for known environments. 
> * Learn how to numerically evaluate a given policy (via an iterative update formula).
> * Understand the policy improvement theorem, and understand what circumstances can allow us to directly solve for the optimal policy.


## Setup


```python
!pip install gym==0.23.1
!pip install pygame

import os
import sys
from typing import Optional, Union, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import random
from tqdm import tqdm
import einops
from pathlib import Path
import matplotlib.pyplot as plt
import gym
import gym.envs.registration
import gym.spaces

Arr = np.ndarray
max_episode_steps = 1000
N_RUNS = 200

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_intro_to_rl"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import part1_intro_to_rl.utils as utils
import part1_intro_to_rl.tests as tests
from plotly_utils import imshow

MAIN = __name__ == "__main__"

```



""", unsafe_allow_html=True)


def section_1():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#readings'>Readings</a></li>
    <li class='margtop'><a class='contents-el' href='#intro-to-openai-gym'>Intro to OpenAI Gym</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#the-info-dictionary'>The <code>info</code> dictionary</a></li>
        <li><a class='contents-el' href='#the-render-method'>The <code>render()</code> method</a></li>
        <li><a class='contents-el' href='#observation-and-action-types'>Observation and Action Types</a></li>
        <li><a class='contents-el' href='#registering-an-environment'>Registering an Environment</a></li>
        <li><a class='contents-el' href='#timelimit-wrapper'>TimeLimit Wrapper</a></li>
        <li><a class='contents-el' href='#a-note-on-pseudo-rngs'>A Note on (pseudo) RNGs</a></li>
        <li><a class='contents-el' href='#exercise-implement-randomagent'><b>Exercise</b> - implement <code>RandomAgent</code></a></li>
        <li><a class='contents-el' href='#exercise-implement-reward-averaging'><b>Exercise</b> - implement reward averaging</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#cheater-agent'>Cheater Agent</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-cheater-agent'><b>Exercise</b> - implement cheater agent</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#the-authentic-rl-experience'>The Authentic RL Experience</a></li>
    <li class='margtop'><a class='contents-el' href='#ucbactionselection'>UCBActionSelection</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-ucbaselection'><b>Exercise</b> - implement <code>UCBASelection</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#bonus'>Bonus</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 1️⃣ Multi-Armed Bandit


> ##### Learning Objectives
> 
> * Understand the anatomy of a `gym.Env`, so that you feel comfortable using them and writing your own
> * Practice RL in the tabular (lookup table) setting before adding the complications of neural networks
> * Understand the difficulty of optimal exploration
> * Understand that performance is extremely variable in RL, and how to measure performance


## Readings

* [Sutton and Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf), Chapter 2
    * Section 2.1: A k-armed Bandit Problem through to Section 2.7 Upper-Confidence-Bound Action Section
    * We won't cover Gradient Bandits. Don't worry about these.
    * Don't worry too much about all the math for the moment if you can't follow it.


## Intro to OpenAI Gym

Today and tomorrow, we'll be using OpenAI Gym, which provides a uniform interface to many different RL environments including Atari games. Gym was released in 2016 and details of the API have changed significantly over the years. We are using version 0.23.1, so ensure that any documentation you use refers to the same version.

Below, we've provided a simple environment for the multi-armed bandit described in the Sutton and Barto reading. Here, an action is an integer indicating the choice of arm to pull, and an observation is the constant integer 0, since there's nothing to observe here. Even though the agent does in some sense observe the reward, the reward is always a separate variable from the observation.

We're using NumPy for this section. PyTorch provides GPU support and autograd, neither of which is of any use to environment code or the code we're writing today.

Read the `MultiArmedBandit` code carefully and make sure you understand how the Gym environment API works.


### The `info` dictionary

The environment's `step` method returns four values: `obs`, `reward`, `done`, and the `info` dictionary.

`info` can contain anything extra that doesn't fit into the uniform interface - it's up to the environment what to put into it. A good use of this is for debugging information that the agent isn't "supposed" to see. In this case, we'll return the index of the actual best arm. This would allow us to measure how often the agent chooses the best arm, but it would also allow us to build a "cheating" agent that peeks at this information to make its decision.

Cheating agents are helpful because we know that they should obtain the maximum possible rewards; if they aren't, then there's a bug.

### The `render()` method

Render is only used for debugging or entertainment, and what it does is up to the environment. It might render a little popup window showing the Atari game, or it might give you a RGB image tensor, or just some ASCII text describing what's happening. In this case, we'll just make a little plot showing the distribution of rewards for each arm of the bandit.

### Observation and Action Types

A `gym.Env` is a generic type: both the type of the observations and the type of the actions depends on the specifics of the environment.

We're only dealing with the simplest case: a discrete set of actions which are the same in every state. In general, the actions could be continuous, or depend on the state.


```python
ObsType = int
ActType = int
    
class MultiArmedBandit(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    num_arms: int
    stationary: bool
    arm_reward_means: np.ndarray

    def __init__(self, num_arms=10, stationary=True):
        super().__init__()
        self.num_arms = num_arms
        self.stationary = stationary
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(num_arms)
        self.reset()

    def step(self, arm: ActType) -> Tuple[ObsType, float, bool, dict]:
        '''
        Note: some documentation references a new style which has (termination, truncation) bools in place of the done bool.
        '''
        assert self.action_space.contains(arm)
        if not self.stationary:
            q_drift = self.np_random.normal(loc=0.0, scale=0.01, size=self.num_arms)
            self.arm_reward_means += q_drift
            self.best_arm = int(np.argmax(self.arm_reward_means))
        reward = self.np_random.normal(loc=self.arm_reward_means[arm], scale=1.0)
        obs = 0
        done = False
        info = dict(best_arm=self.best_arm)
        return (obs, reward, done, info)

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)
        if self.stationary:
            self.arm_reward_means = self.np_random.normal(loc=0.0, scale=1.0, size=self.num_arms)
        else:
            self.arm_reward_means = np.zeros(shape=[self.num_arms])
        self.best_arm = int(np.argmax(self.arm_reward_means))
        if return_info:
            return (0, dict())
        else:
            return 0

    def render(self, mode="human"):
        assert mode == "human", f"Mode {mode} not supported!"
        bandit_samples = []
        for arm in range(self.action_space.n):
            bandit_samples += [np.random.normal(loc=self.arm_reward_means[arm], scale=1.0, size=1000)]
        plt.violinplot(bandit_samples, showmeans=True)
        plt.xlabel("Bandit Arm")
        plt.ylabel("Reward Distribution")
        plt.show()

```

### Registering an Environment

User code normally won't use the constructor of an `Env` directly for two reasons:

- Usually, we want to wrap our `Env` in one or more wrapper classes.
- If we want to test our agent on a variety of environments, it's annoying to have to import all the `Env` classes directly.

The `register` function stores information about our `Env` in a registry so that a later call to `gym.make` can look it up using the `id` string that is passed in.

By convention, the `id` strings have a suffix with a version number. There can be multiple versions of the "same" environment with different parameters, and benchmarks should always report the version number for a fair comparison. For instance, `id="ArmedBanditTestbed-v0"` below.


### TimeLimit Wrapper

As defined, our environment never terminates; the `done` flag is always False so the agent would keep playing forever. By setting `max_episode_steps` here, we cause our env to be wrapped in a `TimeLimit` wrapper class which terminates the episode after that number of steps.

The time limit is also an essential part of the problem definition: if it were larger or shorter, there would be more or less time to explore, which means that different algorithms (or at least different hyperparameters) would then have improved performance.

*Note - the `gym` library is well known for being pretty janky and having annoying errors and warnings! You should generally ignore these warnings unless they're causing you problems e.g. you're failing tests.*


```python
gym.envs.registration.register(
    id="ArmedBanditTestbed-v0",
    entry_point=MultiArmedBandit,
    max_episode_steps=max_episode_steps,
    nondeterministic=True,
    reward_threshold=1.0,
    kwargs={"num_arms": 10, "stationary": True},
)

env = gym.make("ArmedBanditTestbed-v0")
print(f"Our env inside its wrappers looks like: {env}")
```

### A Note on (pseudo) RNGs

The PRNG that `gym.Env` provides as `self.np_random` is from the [PCG family](https://www.pcg-random.org/index.html). In RL code, you often need massive quantities of pseudorandomly generated numbers, so it's important to have a generator that is both very fast and has good quality output.

When you call `np.random.randint` or similar, you're using the old-school Mersenne Twister algorithm which is both slower and has inferior quality output to PCG. Since Numpy 1.17, you can use `np.random.default_rng()` to get a PCG generator and then use its `integers` method to get random integers.


### Exercise - implement `RandomAgent`

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠⚪⚪⚪

You should spend up to ~10 minutes on this exercise.
```

- Implement the `RandomAgent` subclass which should pick an arm at random.
    - This is useful as a baseline to ensure the environment has no bugs. If your later agents are doing worse than random, you have a bug!
- Run the code below, in order to:
    - Verify that `RandomAgent` pulls the optimal arm with frequency roughly `1/num_arms`.
    - Verify that the average reward is very roughly zero. This is the case since the mean reward for each arm is centered on zero.


```python
class Agent:
    '''
    Base class for agents in a multi-armed bandit environment
    
    (you do not need to add any implementation here)
    '''
    rng: np.random.Generator

    def __init__(self, num_arms: int, seed: int):
        self.num_arms = num_arms
        self.reset(seed)

    def get_action(self) -> ActType:
        raise NotImplementedError()

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        pass

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)


def run_episode(env: gym.Env, agent: Agent, seed: int):

    (rewards, was_best) = ([], [])

    env.reset(seed=seed)
    agent.reset(seed=seed)

    done = False
    while not done:
        arm = agent.get_action()
        (obs, reward, done, info) = env.step(arm)
        agent.observe(arm, reward, info)
        rewards.append(reward)
        was_best.append(1 if arm == info["best_arm"] else 0)
    
    rewards = np.array(rewards, dtype=float)
    was_best = np.array(was_best, dtype=int)
    return (rewards, was_best)


def run_agent(env: gym.Env, agent: Agent, n_runs=200, base_seed=1):
    all_rewards = []
    all_was_bests = []
    base_rng = np.random.default_rng(base_seed)
    for n in tqdm(range(n_runs)):
        seed = base_rng.integers(low=0, high=10_000, size=1).item()
        (rewards, corrects) = run_episode(env, agent, seed)
        all_rewards.append(rewards)
        all_was_bests.append(corrects)
    return (np.array(all_rewards), np.array(all_was_bests))


class RandomAgent(Agent):

    def get_action(self) -> ActType:
        pass

    def __repr__(self):
        # Useful when plotting multiple agents with `plot_rewards`
        return "RandomAgent"


num_arms = 10
stationary = True
env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms, stationary=stationary)
agent = RandomAgent(num_arms, 0)
all_rewards, all_corrects = run_agent(env, agent)

print(f"Expected correct freq: {1/10}, actual: {all_corrects.mean():.6f}")
assert np.isclose(all_corrects.mean(), 1/10, atol=0.05), "Random agent is not random enough!"

print(f"Expected average reward: 0.0, actual: {all_rewards.mean():.6f}")
assert np.isclose(all_rewards.mean(), 0, atol=0.05), "Random agent should be getting mean arm reward, which is zero."

print("All tests passed!")
```

<details>
<summary>Solution</summary>


```python
class RandomAgent(Agent):

    def get_action(self) -> ActType:
        # SOLUTION
        return self.rng.integers(low=0, high=self.num_arms)

    def __repr__(self):
        # Useful when plotting multiple agents with `plot_rewards`
        return "RandomAgent"
```
</details>


### Exercise - implement reward averaging

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠⚪⚪⚪

You should spend up to 20-30 minutes on this exercise.
```

You should now complete the methods for the `RewardAveraging` agent, which applies the reward averaging algorithm as detailed in Sutton and Barto section 2.4, "Incremental Implementation":

- Track the moving average of observed reward for each arm.
- Epsilon greedy exploration: with probability `epsilon`, take a random action; otherwise take the action with the highest average observed reward.
- Optimistic initial values: initialize each arm's reward estimate with the `optimism` value.

Remember to call the init function for your `Agent` superclass with `num_arms` and `seed` in your `__init__()` method.

<details>
<summary>Hint - average reward formula</summary>

$$Q_k = Q_{k-1} + \frac{1}{k}[R_k - Q_{k-1}]$$

Where $k$ is the number of times the action has been taken, $R_k$ is the reward from the kth time the action was taken, and $Q_{k-1}$ is the average reward from the previous times this action was taken (this notation departs slightly from the S&B notation, but may be more helpful for our implementation).

**Important - $k$ is not the total number of timesteps, it's the total number of times you've taken this particular action.**
</details>

We've given you a function for plotting multiple agents' reward trajectories on the same graph, with an optional moving average parameter to make the graph smoother.


```python
class RewardAveraging(Agent):
    def __init__(self, num_arms: int, seed: int, epsilon: float, optimism: float):
        pass

    def get_action(self):
        pass

    def observe(self, action, reward, info):
        pass

    def reset(self, seed: int):
        pass

    def __repr__(self):
        # For the legend, when plotting
        return f"RewardAveraging(eps={self.epsilon}, optimism={self.optimism})"


num_arms = 10
stationary = True
names = []
all_rewards = []
env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms, stationary=stationary)

for optimism in [0, 5]:
    agent = RewardAveraging(num_arms, 0, epsilon=0.01, optimism=optimism)
    (rewards, num_correct) = run_agent(env, agent, n_runs=N_RUNS, base_seed=1)
    all_rewards.append(rewards)
    names.append(str(agent))
    print(agent)
    print(f" -> Frequency of correct arm: {num_correct.mean():.4f}")
    print(f" -> Average reward: {rewards.mean():.4f}")

utils.plot_rewards(all_rewards, names, moving_avg_window=15)
```

<details>
<summary>Solution</summary>


```python
class RewardAveraging(Agent):
    def __init__(self, num_arms: int, seed: int, epsilon: float, optimism: float):
        # SOLUTION
        self.epsilon = epsilon
        self.optimism = optimism
        super().__init__(num_arms, seed)

    def get_action(self):
        # SOLUTION
        if self.rng.random() < self.epsilon:
            return self.rng.integers(low=0, high=self.num_arms).item()
        else:
            return np.argmax(self.Q)
        
    def observe(self, action, reward, info):
        # SOLUTION
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def reset(self, seed: int):
        # SOLUTION
        super().reset(seed)
        self.N = np.zeros(self.num_arms)
        self.Q = np.full(self.num_arms, self.optimism, dtype=float)

    def __repr__(self):
        # For the legend, when plotting
        return f"RewardAveraging(eps={self.epsilon}, optimism={self.optimism})"
```
</details>

<details>
<summary>Question - can you interpret these results?</summary>

At the very start, the more optimistic agent performs worse, because it explores more and exploits less. Its estimates are wildly over-optimistic, so even if it finds a good arm, its Q-value for that arm will decrease. On the other hand, if the realistic agent finds a good arm early on, it'll probably return to exploit it.

However, the optimistic agent eventually outperforms the realistic agent, because its increased exploration means it's more likely to converge on the best arm.
</details>

<details>
<summary>Question - how do you think these results would change if epsilon was decreased for both agents?</summary>

You should expect the optimistic agent to outperform the realistic agent even more. The smaller epsilon is, the more necessary optimism is (because without it the agent won't explore enough).
</details>


## Cheater Agent

The cheater agent will always choose the best arm. It's a good idea to implement, because you can compare your other agents to it to make sure they're not doing better than the cheater (if they are, you probably have a bug!).


### Exercise - implement cheater agent

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠🟠⚪⚪

You should spend up to 10-25 minutes on this exercise.

It's important for you to understand why cheater agents are important for debugging.
```

You should fill in the four methods below.

<details>
<summary>Help - I'm not sure how to implement the cheater.</summary>

Recall that your `env.step` method returns a tuple of `(obs, reward, done, info)`, and the `info` dict contains `best_arm`.

In `observe`, you should read the `best_arm` object from the `info` dictionary, and store it. Then, in `get_action`, you should always return this best arm.

Note, you'll have to make a special case for when best arm hasn't been observed yet (in this case, your action doesn't matter).
</details>


```python
class CheatyMcCheater(Agent):
    def __init__(self, num_arms: int, seed: int):
        pass

    def get_action(self):
        pass

    def observe(self, action: int, reward: float, info: dict):
        pass

    def __repr__(self):
        return "Cheater"



cheater = CheatyMcCheater(num_arms, 0)
reward_averaging = RewardAveraging(num_arms, 0, epsilon=0.1, optimism=0)
random = RandomAgent(num_arms, 0)

names = []
all_rewards = []

for agent in [cheater, reward_averaging, random]:
    (rewards, num_correct) = run_agent(env, agent, n_runs=N_RUNS, base_seed=1)
    names.append(str(agent))
    all_rewards.append(rewards)

utils.plot_rewards(all_rewards, names, moving_avg_window=15)

assert (all_rewards[0] < all_rewards[1]).mean() < 0.001, "Cheater should be better than reward averaging"
print("Tests passed!")
```

<details>
<summary>Solution</summary>


```python
class CheatyMcCheater(Agent):
    def __init__(self, num_arms: int, seed: int):
        # SOLUTION
        super().__init__(num_arms, seed)
        self.best_arm = 0

    def get_action(self):
        # SOLUTION
        return self.best_arm

    def observe(self, action: int, reward: float, info: dict):
        # SOLUTION
        self.best_arm = info["best_arm"]

    def __repr__(self):
        return "Cheater"
```
</details>


## The Authentic RL Experience

It would be nice if we could say something like "optimistic reward averaging is a good/bad feature that improves/decreases performance in bandit problems." Unfortunately, we can't justifiably claim either at this point.

Usually, RL code fails silently, which makes it difficult to be confident that you don't have any bugs. I had a bug in my first attempt that made both versions appear to perform equally, and there were only 13 lines of code, and I had written similar code before.

The number of hyperparameters also grows rapidly, and hyperparameters have interactions with each other. Even in this simple problem, we already have two different ways to encourage exploration (`epsilon` and `optimism`), and it's not clear whether it's better to use one or the other, or both in some combination. It's actually worse than that, because `epsilon` should probably be annealed down at some rate.

Even in this single comparison, we trained 200 agents for each version. Is that a big number or a small number to estimate the effect size? Probably we should like, compute some statistics? And test with a different number of arms - maybe we need more exploration for more arms? The time needed for a rigorous evaluation is going to increase quickly.

We're using 0.23.1 of `gym`, which is not the latest version. 0.24.0 and 0.24.1 according to the [release notes](https://github.com/openai/gym/releases) have "large bugs" and the maintainers "highly discourage using these releases". How confident are we in the quality of the library code we're relying on?

As we continue onward to more complicated algorithms, keep an eye out for small discrepancies or minor confusions. Look for opportunities to check and cross-check everything, and be humble with claims.


## UCBActionSelection

Once you feel good about your `RewardAveraging` implementation, you should implement `UCBActionSelection`.

This should store the same moving average rewards for each action as `RewardAveraging` did, but instead of taking actions using the epsilon-greedy strategy it should use Equation 2.8 in Section 2.6 to select actions using the upper confidence bound. It's also useful to add a small epsilon term to the $N_t(a)$ term, to handle instances where some of the actions haven't been taken yet.

You should expect to see a small improvement over `RewardAveraging` using this strategy.


### Exercise - implement `UCBASelection`

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠⚪⚪⚪

You should spend up to 20-40 minutes on this exercise.
```


```python
class UCBActionSelection(Agent):
    def __init__(self, num_arms: int, seed: int, c: float, eps: float = 1e-6):
        pass

    def get_action(self):
        pass

    def observe(self, action, reward, info):
        pass

    def reset(self, seed: int):
        pass

    def __repr__(self):
        return f"UCB(c={self.c})"



cheater = CheatyMcCheater(num_arms, 0)
reward_averaging = RewardAveraging(num_arms, 0, epsilon=0.1, optimism=0)
reward_averaging_optimism = RewardAveraging(num_arms, 0, epsilon=0.1, optimism=5)
ucb = UCBActionSelection(num_arms, 0, c=2.0)
random = RandomAgent(num_arms, 0)

names = []
all_rewards = []

for agent in [cheater, reward_averaging, reward_averaging_optimism, ucb, random]:
    (rewards, num_correct) = run_agent(env, agent, n_runs=N_RUNS, base_seed=1)
    names.append(str(agent))
    all_rewards.append(rewards)

utils.plot_rewards(all_rewards, names, moving_avg_window=15)
```

<details>
<summary>Solution</summary>


```python
class UCBActionSelection(Agent):
    def __init__(self, num_arms: int, seed: int, c: float, eps: float = 1e-6):
        # SOLUTION
        self.c = c
        self.eps = eps
        super().__init__(num_arms, seed)

    def get_action(self):
        # SOLUTION
        # This method avoids division by zero errors, and makes sure N=0 entries are chosen by argmax
        ucb = self.Q + self.c * np.sqrt(np.log(self.t) / (self.N + self.eps))
        return np.argmax(ucb)

    def observe(self, action, reward, info):
        # SOLUTION
        self.t += 1
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def reset(self, seed: int):
        # SOLUTION
        super().reset(seed)
        self.t = 1
        self.N = np.zeros(self.num_arms)
        self.Q = np.zeros(self.num_arms)

    def __repr__(self):
        return f"UCB(c={self.c})"
```
</details>


## Bonus

Here are a few bonus exercises you can try if you're interested. You could instead progress to the second section, and return to them if you have time.

* Implement the gradient bandit algorithm.
* Implement an environment and an agent for the contextual bandit problem.
* Complete the exercises at the end of Chapter 2 of Sutton and Barto.




""", unsafe_allow_html=True)


def section_2():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#reinforcement-learning-known-environments'>Reinforcement Learning, Known Environments</a></li>
    <li class='margtop'><a class='contents-el' href='#readings'>Readings</a></li>
    <li class='margtop'><a class='contents-el' href='#what-is-reinforcement-learning'>What is Reinforcement Learning?</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#pedantic-note-1'>Pedantic note #1</a></li>
        <li><a class='contents-el' href='#pedantic-note-2'>Pedantic note #2</a></li>
        <li><a class='contents-el' href='#theory-exercises'>Theory Exercises</a></li>
        <li><a class='contents-el' href='#exercise-compute-the-value-v-s0-v-pi-s-0-v-s0-for-l-pi-pi-l-l-and-r-pi-pi-r-r'><b>Exercise</b> - compute the value V_π(s<sub>0</sub>) for π = π<sub>L</sub> and π = π<sub>R</sub></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#tabular-rl-known-environments'>Tabular RL, Known Environments</a></li>
    <li class='margtop'><a class='contents-el' href='#policy-evaluation'>Policy Evaluation</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-policy-eval-numerical'><b>Exercise</b> - implement <code>policy_eval_numerical</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#exact-policy-evaluation'>Exact Policy Evaluation</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-policy-eval-exact'><b>Exercise</b> - implement <code>policy_eval_exact</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#policy-improvement'>Policy Improvement</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-policy-improvement'><b>Exercise</b>: implement <code>policy_improvement</code></a></li>
        <li><a class='contents-el' href='#exercise-implement-find-optimal-policy'><b>Exercise</b> - implement <code>find_optimal_policy</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#bonus'>Bonus</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 2️⃣ Tabular RL & Policy Improvement


> ##### Learning Objectives
> 
> * Understand the tabular RL problem, for known environments. 
> * Learn how to numerically evaluate a given policy (via an iterative update formula).
> * Understand the policy improvement theorem, and understand what circumstances can allow us to directly solve for the optimal policy.


## Reinforcement Learning, Known Environments

We are presented with a environment, and the transition function that indicates how the environment will transition from state to state based on the action chosen. We will see how we can turn the problem of finding the best agent into an optimization problem, which we can then solve iteratively.

Here, we assume environments small enough where visiting all pairs of states and actions is tractable. These types of models don't learn any relationships between states that can be treated similarly, but keep track of an estimate
of how valuable each state is in a large lookup table. 


## Readings

Don't worry about absorbing every detail, we will repeat a lot of the details here.
Don't worry too much about the maths, we will also cover that here.

- [Sutton and Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- Chapter 3, Sections 3.1, 3.2, 3.3, 3.5, 3.6
- Chapter 4, Sections 4.1, 4.2, 4.3, 4.4


## What is Reinforcement Learning?

In reinforcement learning, the agent interacts with an environment in a loop: The agent in state $s$ issues an **action** $a$ to the environment, and the environment replies with **state, reward** pairs $(s',r)$. We will assume the environment is **Markovian**, in that the next state $s'$ and reward $r$ depend solely on the current state $s$ and the action $a$ chosen by the agent (as opposed to environments which may depend on actions taken far in the past.)


### Pedantic note #1

Some authors disagree on precisely where the next timestep occours. We (following Sutton and Barto) define the timestep transition to be when the environment
acts, so given state $s_t$ the agent returns action $a_t$ (in the same time step), but given $(s_t, a_t)$ the environment generates $(s_{t+1}, r_{t+1})$.

Other authors notate the history as
$$
s_0, a_0, r_0, s_1, a_1, r_1, s_2, a_2, r_2, \ldots
$$
often when the reward $r_t$ is a deterministic function of the current state $s_t$ and action chosen $a_t$, and the environment's only job is to select the next state $s_{t+1}$ given $(s_t, a_t)$.

There's nothing much we can do about this, so be aware of the difference when reading other sources.


The agent chooses actions using a policy $\pi$, which we can think of as either a deterministic function $a = \pi(s)$ from states to actions, or more generally a stochastic function from which actions are sampled, $a \sim \pi(\cdot | s)$.

Implicitly, we have assumed that the agent need only be Markovian as well. Is this a reasonable assumption?

<details>
<summary>Answer</summary>

The environment dynamics depend only on the current state, so the agent needs only the current state to decide how to act optimally (for example, in a game of tic-tac-toe, knowledge of the current state of the board is sufficient to determine the optimal move, how the board got to that state is irrelevant.)
</details>


The environment samples (state, reward) pairs from a probability distribution conditioned on the current state $s$ and the action $a$ the policy chose in that state, $(s', r) \sim p(\cdot | s, a)$. In the case where the environment is also deterministic, we write $(s', r) = p(s, a)$.

The goal of the agent is to choose a policy that maximizes the **expected discounted return**, the sum of rewards it would expect to obtain by following it's currently chosen **policy** $\pi$. We call the expected discounted return from a state $s$ following policy $\pi$ the **state value function** $V_{\pi}(s)$, or simply **value function**, as 
$$
V_{\pi}(s) = \mathbb{E}_{\pi} \left[ \sum_{i=t}^\infty \gamma^{i-t} r_{i+1} \Bigg| s_t = s \right]
$$
where the expectation is with respect to sampling actions from $\pi$, and (implicitly) sampling states and rewards from $p$.


### Pedantic note #2

Technically $V_\pi$ also depends on the choice of environment $p$ and discount factor $\gamma$, but usually during training the only thing we are optimizing for is the choice of policy $\pi$ (the environment $p$ is fixed and the discount $\gamma$ is either provided or fixed as a hyperparameter before training), so we ignore these dependencies instead of writing $V_{\pi, p, \gamma}$.

<details>
<summary>Why do we discount?</summary>

We would like a way to signal to the agent that reward now is better than reward later.
If we didn't discount, the sum $\sum_{i=t}^\infty r_i$ may diverge.
This leads to strange behavior, as we can't meaningfully compare the returns for sequences of rewards that diverge. Trying to sum the sequence $1,1,1,\ldots$ or $2,2,2,2\ldots$ or even $2, -1, 2 , -1, \ldots$ all diverge to positive infinity, and so are all "equally good" even though it's clear that $2,2,2,2\ldots$ would be the most desirable.

An agent with a policy that leads to infinite expected return might become lazy, as waiting around doing nothing for a thousand years, and then actually optimally leads to the same return (infinite) as playing optimal from the first time step.

Worse still, we can have reward sequences for which the sum may never approach any finite value, nor diverge to $\pm \infty$ (like summing $-1, 1, -1 ,1 , \ldots$).
We could otherwise patch this by requiring that the agent has a finite number of interactions with the environment (this is often true, and is called an **episodic** environment that we will see later) or restrict ourselves to environments for which the expected return for the optimal policy is finite, but these can often be undesirable constraints to place.
</details>

<details>
<summary>Why do we discount geometrically?</summary>

In general, we could consider a more general discount function $\Gamma : \mathbb{N} \to [0,1)$ and define the discounted return as $\sum_{i=t}^\infty \Gamma(i) r_i$. The geometric discount $\Gamma(i) = \gamma^i$ is commonly used, but other discounts include the hyperbolic discount $\Gamma(i) = \frac{1}{1 + iD}$, where $D>0$ is a hyperparameter. (Humans are [often said](https://chris-said.io/2018/02/04/hyperbolic-discounting/) to act as if they use a hyperbolic discount.) Other than  being mathematically convenient to work with (as the sum of geometric discounts has an elegant closed form expression $\sum_{i=0}^\infty \gamma^i  = \frac{1}{1-\gamma}$), geometric is preferred as it is *time consistant*, that is, the discount from one time step to the next remains constant. 
Rewards at timestep $t+1$ are always worth a factor of $\gamma$ less than rewards on timestep $t$
$$
\frac{\Gamma(t+1)}{\Gamma(t)} = \frac{\gamma^{t+1}}{\gamma^t} = \gamma
$$
whereas for the hyperbolic reward, the amount discounted from one step to the next
is a function of the timestep itself, and decays to 1 (no discount)
$$
\frac{\Gamma(t+1)}{\Gamma(t)} =  \frac{1+tD}{1+(t+1)D} \to 1
$$
so very little discounting is done once rewards are far away enough in the future. 

Hence, using geometric discount lends a certain symmetry to the problem. If after $N$ actions the agent ends up in exactly the same state as it was in before, then there will be less total value remaining (because of the decay rate), but the optimal policy won't change (because the rewards the agent gets if it takes a series of actions will just be a scaled-down version of the rewards it would have gotten for those actions $N$ steps ago).
</details>


> Note we can write the value function in the following recursive matter:
> $$
> V_\pi(s) = \sum_a \pi(a | s) \sum_{s', r} p(s',r \mid s, a) \left( r + \gamma V_\pi(s') \right)
> $$
> 
> (Optional) Try to prove this for yourself!
> 
> <details>
> <summary>Hint</summary>
> 
> Start by writing the expectation as a sum over all possible actions $a$ that can be taken from state $s$. Then, try to separate $r_{t+1}$ and the later reward terms inside the sum.
> </details>
> 
> <details>
> <summary>Answer</summary>
> 
> First, we write it out as a sum over possible next actions, using the **policy function** $\pi$:
> $$
> \begin{aligned}
> V_\pi(s) &=\mathbb{E}_\pi\left[\sum_{i=t}^{\infty} \gamma^{i-t} r_{i+1} \mid s_t=s\right] \\
> &=\sum_a \pi(a \mid s) \mathbb{E}_\pi\left[\sum_{i=t}^{\infty} \gamma^{i-t} r_{i+1} \mid s_t=s, a_t=a\right]
> \end{aligned}
> $$
> In other words, this is an average of the possible expected reward streams after a particular action, weighted by the probability that this action is chosen.
> 
> We can then separate out the term for the rewward at step $t+1$:
> $$
> =\sum_a \pi(a \mid s) \mathbb{E}_\pi\left[r_{t+1}+\sum_{i=t+1}^{\infty} \gamma^{i-t} r_{i+1} \mid s_t=s, a_t=a\right]
> $$
> And then we expand the sum over all possible state-reward pairs $(s', r)$ which we might evolve to, following state-action pair $(s, a)$:
> $$
> \begin{aligned}
> &=\sum_{a, s^{\prime}, r} \pi(a \mid s) p\left(s^{\prime}, r \mid s, a\right)\left(r+\gamma \cdot \mathbb> {E}_\pi\left[\sum_{i=t+1}^{\infty} \gamma^{i-(t+1)} r_{i+1} \mid s_{t+1}=s^{\prime}\right]\right) \\
> \end{aligned}
> $$
> Finally, we can notice that the sum term looks a lot like our original expression for $V_{\pi}(s)$, just starting from $s'$ rather than $s$. So we can write this as:
> $$
> \begin{aligned}
> &=\sum_{a, s, r} \pi(a \mid s) p\left(s^{\prime}, r \mid s, a\right)\left(r+\gamma V_\pi\left(s^{\prime}\right)\right) \\
> \end{aligned}
> $$
> And finally, we rearrange the sum terms:
> $$
> \begin{aligned}
> &=\sum_a \pi(a \mid s) \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left(r+\gamma V_\pi\left(s^{\prime}\right)\right)
> \end{aligned}
> $$
> as required.
> 
> How should we interpret this? This formula tells us that the total value from present time can be written as sum of next-timestep rewards and value terms discounted by a factor of $\gamma$. Recall earlier in our discussion of **geometric** vs **hyperbolic** discounting, we argued that geometric discounting has a symmetry through time, because of the constant discount factor. This is exactly what this formula shows, just on the scale of a single step.
> 
> </details>

This recursive formulation of the value function is called the **Bellman equation**, and can be thought of as "The value of the current state is the reward the agent gets now, plus the value for the next state."

We can also define the **action-value function**, or **Q-value** of state $s$ and action $a$ following policy $\pi$:
$$
Q_\pi(s,a) = \mathbb{E}\left[ \sum_{i=t}^\infty \gamma^{i-t}r_{t+1} \Bigg| s_t=s, a_t=a   \right]
$$
which can be written recursively much like the value function can:
$$
Q_\pi(s,a) = \sum_{s',r} p(s',r \mid s,a) \left( r + \gamma \sum_{a'} \pi(a' \mid s') Q_\pi(s', a') \right)
$$


### Theory Exercises

(These are mostly to check your understanding of the readings on RL. Feel free to skip ahead if you're already happy with the definition.)

Consider the following environment: There are two actions $A = \{a_L, a_R\}$, three states $S = \{s_0, s_L, s_R\}$ and three rewards $R = \{0,1,2\}$. The environment is deterministic, and can be represented by the following transition diagram:

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/markov-diagram.png" width="400">

The edges represent the state transitions given an action, as well as the reward received. For example, in state $s_0$, taking action $a_L$ means the new state is $s_L$, and the reward received is $+1$. (The transitions for $s_L$ and $s_R$ are independent of action taken.)

We say that two policies $\pi_1$ and $\pi_2$ are **equivalent** if $\forall s \in S. V_{\pi_1}(s) = V_{\pi_2}(s)$.

This gives us effectively two choices of deterministic policies, $\pi_L(s_0) = s_L$ and $\pi_R(s_0) = s_R$. (It is irrelevant what those policies do in the other states.)

A policy $\pi_1$ is **better** than $\pi_2$ (denoted $\pi_1 \geq \pi_2$) if
$\forall s \in S. V_{\pi_1}(s) \geq V_{\pi_2}(s)$.


### Exercise - compute the value $V_{\pi}(s_0)$ for $\pi = \pi_L$ and $\pi = \pi_R$.

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠🟠⚪⚪

You should spend up to 15-25 minutes on this exercise.

This is a good exercise to build your intuition for these formulas and expressions.
```

Which policy is better? Does the answer depend on the choice of discount factor $\gamma$? If so, how?

<details>
<summary>Answer</summary>

Following the first policy, this gives
$$
V_{\pi_L}(s_0) = 1 + \gamma V_{\pi_L}(s_L) = 1 + \gamma(0 + \gamma V_{\pi_L}(s_0)) = 1 +\gamma^2 V_{\pi_L}(s_0)
$$
Rearranging, this gives
$$
V_{\pi_L}(s_0) = \frac{1}{1-\gamma^2}
$$
Following the second policy, this gives
$$
V_{\pi_R}(s_0) = 0 + \gamma V_{\pi_R}(s_R) = \gamma(2 + \gamma V_{\pi_R}(s_0)) = 2 \gamma + \gamma^2 V_{\pi_R}(s_0)
$$
Rearranging, this gives
$$
V_{\pi_R}(s_0) = \frac{2\gamma}{1-\gamma^2}
$$
Therefore,
$$
\pi^* = \begin{cases}
\pi_L & \gamma < 1/2 \\
\pi_L \text{ or } \pi_R & \gamma = 1/2 \\
\pi_R & \gamma > 1/2
\end{cases}
$$
which makes sense, an agent that discounts heavily ($\gamma < 1/2$) is shortsighted,
and will choose the reward 1 now, over the reward 2 later.
</details>


An **optimal** policy (denoted $\pi^*$) is a policy that is better than all other policies. There may be more than one optimal policy, so we refer to any of them as $\pi^*$, with the understanding that since all optimal policies have the same value $V_{\pi^*}$ for all states, it doesn't actually matter which is chosen.

It is possible to prove that, for any environment, an optimal policy exists, but we won't go into detail on this proof today.


## Tabular RL, Known Environments

For the moment, we focus on environments for which the agent has access to $p$, the function describing the underlying dynamics of the environment, which will allow us to solve the Bellman equation explicitly. While unrealistic, it means we can explicitly solve the Bellman equation. Later on we will remove this assumption and treat the environment as a black box from which the agent can sample from.

We will simplify things a bit further, and assume that the environment samples states from a probability distribution $T(\cdot | s, a)$ conditioned on the current state $s$ and the action $a$ the policy $\pi$ chose in that state.

Normally, the reward is also considered to be a stochastic function of both state and action $r \sim  R(\cdot \mid s,a)$, but we will assume the reward is a deterministic function $R(s,a,s')$ of the current state, action and next state, and offload the randomness in the rewards to the next state $s'$ sampled from $T$.

This (together with assuming $\pi$ is deterministic) gives a simpler recursive form of the value function
$$
V_\pi(s) = \sum_{s'} T(s' \mid s, a) \Big( R(s,a,s') + \gamma V_\pi (s') \Big)
\text{ where } a = \pi(s)
$$

Below, we've provided a simple environment for a gridworld taken from [Russell and Norvig](http://aima.cs.berkeley.edu/). The agent can choose one of four actions: `up` (0), `right` (1), `down` (2) and `left` (3), encoded as numbers. The observation is just the state that the agent is in (encoded as a number from 0 to 11).

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/gridworld.png" width="300">

The result of choosing an action from an empty cell will move the agent in that direction, unless they would bump into a wall or walk outside the grid, in which case the next state is unchanged. Both the terminal states "trap" the agent, and any movement from one of the terminal states leaves the agent in the same state, and no reward is received. The agent receives a small penalty for each move $r = -0.04$ to encourage them to move to a terminal state quickly, unless the agent is moving into a terminal state, in which case they recieve reward either $+1$ or $-1$ as appropriate.

Lastly, the environment is slippery, and the agent only has a 70\% chance of moving in the direction chosen, with a 10% chance each of attempting to move in the other three cardinal directions.

<details>
<summary>There are two possible routes to +1, starting from the cell immediately to the <b>right of START</b>. Which one do you think the agent will prefer? How do you think changing the penalty value r=-0.04 to something else might affect this?
</summary>

The two routes are moving clockwise and anticlockwise around the center obstacle.

The clockwise route requires more steps, which (assuming $r<0$) will be penalised more.

But the anticlockwise route has a different disadvantage. The agent has a 70% probability of slipping, and the anticlockwise route brings them into closer proximity with the $-1$ terminal state.

We can guess that there is a certain threshold value $r^* < 0$ such that the optimal policy is clockwise when $r<r^*$ (because movement penalty is large, making the anticlockwise route less attractive), and anticlockwise when $r>r^*$. 

You can test this out (and find the approximate value of $r^*$) when you run your RL agent!
</details>

Provided is a class that allows us to define environments with known dynamics. The only parts you should be concerned
with are

* `.num_states`, which gives the number of states,
* `.num_actions`, which gives the number of actions
* `.T`, a 3-tensor of shape  `(num_states, num_actions, num_states)` representing the probability $T(s_{next} \mid s, a)$ = `T[s,a,s_next]`
* `.R`, the reward function, encoded as a vector of shape `(num_states, num_actions, num_states)` that returns the reward $R(s,a,s_{next})$ = `R[s,a,s_next]` associated with entering state $s_{next}$ from state $s$ by taking action $a$.

This environment also provides two additional parameters which we will not use now, but need for part 3 where the environment is treated as a black box, and agents learn from interaction.
* `.start`, the state with which interaction with the environment begins. By default, assumed to be state zero.
* `.terminal`, a list of all the terminal states (with which interaction with the environment ends). By default, terminal states are empty.


```python
class Environment:
    def __init__(self, num_states: int, num_actions: int, start=0, terminal=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.start = start
        self.terminal = np.array([], dtype=int) if terminal is None else terminal
        (self.T, self.R) = self.build()

    def build(self):
        '''
        Constructs the T and R tensors from the dynamics of the environment.
        Outputs:
            T : (num_states, num_actions, num_states) State transition probabilities
            R : (num_states, num_actions, num_states) Reward function
        '''
        num_states = self.num_states
        num_actions = self.num_actions
        T = np.zeros((num_states, num_actions, num_states))
        R = np.zeros((num_states, num_actions, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                (states, rewards, probs) = self.dynamics(s, a)
                (all_s, all_r, all_p) = self.out_pad(states, rewards, probs)
                T[s, a, all_s] = all_p
                R[s, a, all_s] = all_r
        return (T, R)

    def dynamics(self, state: int, action: int) -> Tuple[Arr, Arr, Arr]:
        '''
        Computes the distribution over possible outcomes for a given state
        and action.
        Inputs:
            state : int (index of state)
            action : int (index of action)
        Outputs:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        '''
        raise NotImplementedError

    def render(pi: Arr):
        '''
        Takes a policy pi, and draws an image of the behavior of that policy, if applicable.
        Inputs:
            pi : (num_actions,) a policy
        Outputs:
            None
        '''
        raise NotImplementedError

    def out_pad(self, states: Arr, rewards: Arr, probs: Arr):
        '''
        Inputs:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        Outputs:
            states  : (num_states,) all the next states
            rewards : (num_states,) rewards for each next state transition
            probs   : (num_states,) likelihood of each state-reward pair (including zero-prob outcomes.)
        '''
        out_s = np.arange(self.num_states)
        out_r = np.zeros(self.num_states)
        out_p = np.zeros(self.num_states)
        for i in range(len(states)):
            idx = states[i]
            out_r[idx] += rewards[i]
            out_p[idx] += probs[i]
        return (out_s, out_r, out_p)

```

For example, here is the toy environment from the diagram earlier implemented in this format.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/markov-diagram.png" width="400">


```python
class Toy(Environment):
    def dynamics(self, state: int, action: int):
        (S0, SL, SR) = (0, 1, 2)
        LEFT = 0
        num_states = 3
        num_actions = 2
        assert 0 <= state < self.num_states and 0 <= action < self.num_actions
        if state == S0:
            if action == LEFT:
                (next_state, reward) = (SL, 1)
            else:
                (next_state, reward) = (SR, 0)
        elif state == SL:
            (next_state, reward) = (S0, 0)
        elif state == SR:
            (next_state, reward) = (S0, 2)
        return (np.array([next_state]), np.array([reward]), np.array([1]))

    def __init__(self):
        super().__init__(num_states=3, num_actions=2)

```

Given a definition for the `dynamics` function, the `Environment` class automatically generates `T` and `R` for us.


```python
toy = Toy()

actions = ["a_L", "a_R"]
states = ["s_L", "S_0", "S_R"]

imshow(
    toy.T, # dimensions (s, a, s_next)
    title="Transition probabilities T(s_next | s, a) for toy environment", 
    facet_col=-1, facet_labels=[f"s_next = {s}" for s in states], y=states, x=actions,
    labels = {"x": "Action", "y": "State", "color": "Transition<br>Probability"},
)

imshow(
    toy.R, # dimensions (s, a, s_next)
    title="Rewards R(s, a, s_next) for toy environment", 
    facet_col=-1, facet_labels=[f"s_next = {s}" for s in states], y=states, x=actions,
    labels = {"x": "Action", "y": "State", "color": "Reward"},
)
```

A few notes:

* This environment is deterministic; the transition probabilities are always 1.
* Given that there are some "illegal moves", some of the reward function values don't really make sense (e.g. in this environment we can't have $s_{next} = s$ because every move has to be to a new state). But this is fine, because (see the formula above) we'll always be multiplying these reward values by the corresponding transition probabilities of zero, so they won't change any of the computation we're doing.


We also provide an implementation of the gridworld environment above. We include a definition of `render`, which given a policy, prints out a grid showing the direction the policy will try to move in from each cell.


```python
class Norvig(Environment):
    def dynamics(self, state: int, action: int) -> Tuple[Arr, Arr, Arr]:
        def state_index(state):
            assert 0 <= state[0] < self.width and 0 <= state[1] < self.height, print(state)
            pos = state[0] + state[1] * self.width
            assert 0 <= pos < self.num_states, print(state, pos)
            return pos

        pos = self.states[state]
        move = self.actions[action]
        if state in self.terminal or state in self.walls:
            return (np.array([state]), np.array([0]), np.array([1]))
        out_probs = np.zeros(self.num_actions) + 0.1
        out_probs[action] = 0.7
        out_states = np.zeros(self.num_actions, dtype=int) + self.num_actions
        out_rewards = np.zeros(self.num_actions) + self.penalty
        new_states = [pos + x for x in self.actions]
        for (i, s_new) in enumerate(new_states):
            if not (0 <= s_new[0] < self.width and 0 <= s_new[1] < self.height):
                out_states[i] = state
                continue
            new_state = state_index(s_new)
            if new_state in self.walls:
                out_states[i] = state
            else:
                out_states[i] = new_state
            for idx in range(len(self.terminal)):
                if new_state == self.terminal[idx]:
                    out_rewards[i] = self.goal_rewards[idx]
        return (out_states, out_rewards, out_probs)

    def render(self, pi: Arr):
        assert len(pi) == self.num_states
        emoji = ["⬆️", "➡️", "⬇️", "⬅️"]
        grid = [emoji[act] for act in pi]
        grid[3] = "🟩"
        grid[7] = "🟥"
        grid[5] = "⬛"
        print("".join(grid[0:4]) + "\n" + "".join(grid[4:8]) + "\n" + "".join(grid[8:]))

    def __init__(self, penalty=-0.04):
        self.height = 3
        self.width = 4
        self.penalty = penalty
        num_states = self.height * self.width
        num_actions = 4
        self.states = np.array([[x, y] for y in range(self.height) for x in range(self.width)])
        self.actions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        self.dim = (self.height, self.width)
        terminal = np.array([3, 7], dtype=int)
        self.walls = np.array([5], dtype=int)
        self.goal_rewards = np.array([1.0, -1])
        super().__init__(num_states, num_actions, start=8, terminal=terminal)

```

## Policy Evaluation

At the moment, we would like to determine the value function $V_\pi$ of some policy $\pi$. **We will assume policies are deterministic**, and encode policies as a lookup table from states to actions (so $\pi$ will be a vector of shape `(num_states,)`, where each element is an integer `a` in the range `0 <= a < num_actions`, representing one of the possible actions to choose for that state.)

Firstly, we will use the Bellman equation as an update rule: Given a current estimate $\hat{V}_\pi$ of the value function $V_{\pi}$, we can obtain a better estimate by using the Bellman equation, sweeping over all states.
$$
\forall s. \hat{V}_\pi(s) \leftarrow \sum_{s'} T(s' \,|\, s, a) \left( R(s,a,s') + \gamma \hat{V}_\pi(s') \right) \;\text{ where } a = \pi(s)
$$
Recall that the true value function $V_\pi$ satisfies this as an equality, i.e. it is a fixed point of the iteration. 

(Also, remember how we defined our objects: `T[s, a, s']` $= T(s' \,|\, a, s)$, and `R[s, a, s']` $= R(s, a, s')$.)

We continue looping this update rule until the result stabilizes: $\max_s |\hat{V}^{new}(s) - \hat{V}^{old}(s)| < \epsilon$ for some small $\epsilon > 0$. Use $\hat{V}_\pi(s) = 0$ as your initial guess.


### Exercise - implement `policy_eval_numerical`

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠🟠⚪⚪

You should spend up to 15-30 minutes on this exercise.
```

The `policy_eval_numerical` function takes in a deterministic policy $\pi$, and transition and reward matrices $T$ and $R$ (stored as `env.T` and `env.R`), and computes the value function $V_\pi$ using the Bellman equation as an update rule. It keeps iterating until it reaches `max_iterations`, or the resut stabilizes - which ever comes first.


```python
def policy_eval_numerical(env: Environment, pi: Arr, gamma=0.99, eps=1e-8, max_iterations=10_000) -> Arr:
    '''
    Numerically evaluates the value of a given policy by iterating the Bellman equation
    Inputs:
        env: Environment
        pi : shape (num_states,) - The policy to evaluate
        gamma: float - Discount factor
        eps  : float - Tolerance
        max_iterations: int - Maximum number of iterations to run
    Outputs:
        value : float (num_states,) - The value function for policy pi
    '''
    pass


tests.test_policy_eval(policy_eval_numerical, exact=False)
```

<details>
<summary>Solution</summary>


```python
def policy_eval_numerical(env: Environment, pi: Arr, gamma=0.99, eps=1e-8, max_iterations=10_000) -> Arr:
    '''
    Numerically evaluates the value of a given policy by iterating the Bellman equation
    Inputs:
        env: Environment
        pi : shape (num_states,) - The policy to evaluate
        gamma: float - Discount factor
        eps  : float - Tolerance
        max_iterations: int - Maximum number of iterations to run
    Outputs:
        value : float (num_states,) - The value function for policy pi
    '''
    # SOLUTION
    num_states = env.num_states
    T = env.T
    R = env.R
    value = np.zeros((num_states,))  

    for i in range(max_iterations):    
        new_v = np.zeros_like(value)
        for s in range(num_states):
            new_v[s] = np.dot(T[s, pi[s]], R[s, pi[s]] + gamma * value)
        delta = np.abs(new_v - value).sum()
        if delta < eps:
            break
        value = np.copy(new_v)
    else:
        print(f"Failed to converge after {max_iterations} steps.")
        
    return value
```
</details>


## Exact Policy Evaluation

Essentially what we are doing in the previous step is numerically solving the Bellman equation. Since the Bellman equation essentially gives us a set of simultaneous equations, we can solve it explicitly rather than iterating the Bellman update rule.

Given a policy $\pi$, consider $\mathbf{v} \in \mathbb{R}^{|S|}$ to be a vector representing the value function $V_\pi$ for each state.
$$
\mathbf{v} = [V_\pi(s_1), \ldots, V_\pi(s_{N})]
$$
and recall the Bellman equation:
$$
\mathbf{v}_i = \sum_{s'} T(s' \mid s_i,\pi(s)) \left( R(s,\pi(s),s') + \gamma V_\pi(s') \right)
$$
We can define two matrices $P^\pi$ and $R^\pi$, both of shape `(num_states, num_states)`
as
$$
P^\pi_{i,j} = T(j \,|\, \pi(i), i) \quad R^\pi_{i,j} = R(i, \pi(i), j)
$$
$P^\pi$ can be thought of as a probability transition matrix from current state to next state, and $R^\pi$ is the reward function given current state and next state, assuming actions are chosen by $\pi$.
$$
\mathbf{v}_i = \sum_{j} P^\pi_{i,j} \left( R^\pi_{i,j} + \gamma \mathbf{v}_j \right)
$$
$$
\mathbf{v}_i = \sum_{j} P^\pi_{i,j}  R^\pi_{i,j} +  \gamma \sum_{j} P^\pi_{i,j} \mathbf{v}_j
$$
We can define $\mathbf{r}^\pi_i = \sum_{j} P^\pi_{i,j}  (R^\pi)_{i,j} = (P^\pi  (R^\pi)^T)_{ii}$
$$
\mathbf{v}_i = \mathbf{r}_i^\pi + \gamma (P^\pi \mathbf{v})_i
$$
A little matrix algebra, and we obtain:
$$
\mathbf{v} = (I - \gamma P^\pi)^{-1} \mathbf{r}^\pi
$$
which gives us a closed form solution for the value function $\mathbf{v}$.

Is the inverse $(I - \gamma P^\pi)^{-1}$ guaranteed to exist? Discuss with your partner, and ask Callum if you're not sure.


### Exercise - implement `policy_eval_exact`

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠🟠⚪⚪

You should spend up to 10-30 minutes on this exercise.
```

<details>
<summary>Help - I'm not sure where to start.</summary>

You can define the matrix $P^\pi$ indexing `env.T` with the states and corresponding actions `pi` (i.e. you have a 2D array whose elements are $T(s' \mid s, \pi(s))$ for all $s, s'$). Same for defining $R^\pi$ from the reward matrix.

</details>


```python
def policy_eval_exact(env: Environment, pi: Arr, gamma=0.99) -> Arr:
    '''
    Finds the exact solution to the Bellman equation.
    '''
    pass


tests.test_policy_eval(policy_eval_exact, exact=True)
```

<details>
<summary>Solution</summary>


```python
def policy_eval_exact(env: Environment, pi: Arr, gamma=0.99) -> Arr:
    '''
    Finds the exact solution to the Bellman equation.
    '''
    # SOLUTION
    states = np.arange(env.num_states)
    actions = pi
    transition_matrix = env.T[states, actions, :]
    reward_matrix = env.R[states, actions, :]

    r = einops.einsum(transition_matrix, reward_matrix, "s s_next, s s_next -> s")

    mat = np.eye(env.num_states) - gamma * transition_matrix

    return np.linalg.solve(mat, r)
```
</details>


## Policy Improvement

So, we now have a way to compute the value of a policy. What we are really interested in is finding better policies. One way we can do this is to compare how well $\pi$ performs on a state versus the value obtained by choosing another action instead, that is, the Q-value. If there is an action $a'$ for which $Q_{\pi}(s,a') > V_\pi(s) \equiv Q_\pi(s, \pi(s))$, then we would prefer that $\pi$ take action $a'$ in state $s$ rather than whatever action $\pi(s)$ currently is. In general, we can select the action that maximizes the Bellman equation:
$$
\argmax_a \sum_{s'} T(s' \mid s, a) (R(s,a,s') + \gamma V_{\pi}(s')) \geq \sum_{s',r}
T(s' \mid s, \pi(s)) \left( R(s,\pi(s),s') + \gamma V_{\pi}(s') \right) = V_{\pi}(s')
$$

This gives us an update rule for the policy. Given the value function $V_{\pi}(s)$
for policy $\pi$, we define an improved policy $\pi^\text{better}$ as follows:
$$
\pi^\text{better}(s) = \argmax_a \sum_{s'} T(s' \mid s, a) (R(s,a,s') + \gamma V_{\pi}(s'))
$$
Note that we are simultaneously applying this improvement over all states $s$.


### Exercise - implement `policy_improvement`

```c
Difficulty: 🟠🟠⚪⚪⚪
Importance: 🟠🟠🟠🟠⚪

You should spend up to 10-25 minutes on this exercise.
```

The `policy_improvement` function takes in a value function $V$, and transition and reward matrices $T$ and $R$ (stored as `env.T` and `env.R`), and returns a new policy $\pi^\text{better}$.


```python
def policy_improvement(env: Environment, V: Arr, gamma=0.99) -> Arr:
    '''
    Inputs:
        env: Environment
        V  : (num_states,) value of each state following some policy pi
    Outputs:
        pi_better : vector (num_states,) of actions representing a new policy obtained via policy iteration
    '''
    pass


tests.test_policy_improvement(policy_improvement)
```

<details>
<summary>Solution</summary>


```python
def policy_improvement(env: Environment, V: Arr, gamma=0.99) -> Arr:
    '''
    Inputs:
        env: Environment
        V  : (num_states,) value of each state following some policy pi
    Outputs:
        pi_better : vector (num_states,) of actions representing a new policy obtained via policy iteration
    '''
    # SOLUTION
    sum_term = sum([
        einops.einsum(env.T, env.R, "s a s_next, s a s_next -> s a"),
        gamma * einops.einsum(env.T, V, "s a s_next, s_next -> s a")
    ])
    pi_new = np.argmax(sum_term, axis=1)
    return pi_new
```
</details>


Putting these together, we now have an algorithm to find the optimal policy for an environment.
$$
\pi_0 \overset{E}{\to} V_{\pi_0}
\overset{I}{\to} \pi_1 \overset{E}{\to} V_{\pi_1}
\overset{I}{\to} \pi_2 \overset{E}{\to} V_{\pi_2} \overset{I}{\to}  \ldots
$$
We alternate policy evaluation ($\overset{E}{\to}$) and policy improvement ($\overset{I}{\to}$), each step a monotonic improvement, until the policy no longer changes ($\pi_n = \pi_{n+1}$), at which point we have an optimal policy, as our current policy will satisfy the optimal Bellman equations:
$$
V_{\pi^*} = \argmax_a \sum_{s',r} T(s' \mid s,a) (R(s,a,s') + \gamma V_{\pi^*}(s'))
$$


### Exercise - implement `find_optimal_policy`

```c
Difficulty: 🟠🟠🟠⚪⚪
Importance: 🟠🟠🟠🟠⚪

You should spend up to 15-30 minutes on this exercise.
```

Implement the `find_optimal_policy` function below. This should iteratively do the following: 

* find the correct value function $V_{\pi}$ for your given policy $\pi$ using your `policy_eval_exact` function,
* update your policy using `policy_improvement`,

where we start from any arbitrary value of $\pi_0$ (we've picked all zeros).

A few things to note:

* Don't forget that policies should be of `dtype=int`, rather than floats! $\pi$ in this case represents **the deterministic action you take in each state**.
* Since the optimal policy is not unique, the automated tests will merely check that your optimal policy has the same value function as the optimal policy found by the solution.


```python
def find_optimal_policy(env: Environment, gamma=0.99, max_iterations=10_000):
    '''
    Inputs:
        env: environment
    Outputs:
        pi : (num_states,) int, of actions represeting an optimal policy
    '''
    pi = np.zeros(shape=env.num_states, dtype=int)
    pass


tests.test_find_optimal_policy(find_optimal_policy)

penalty = -0.04
norvig = Norvig(penalty)
pi_opt = find_optimal_policy(norvig, gamma=0.99)
norvig.render(pi_opt)
```

<details>
<summary>Solution</summary>


```python
def find_optimal_policy(env: Environment, gamma=0.99, max_iterations=10_000):
    '''
    Inputs:
        env: environment
    Outputs:
        pi : (num_states,) int, of actions represeting an optimal policy
    '''
    pi = np.zeros(shape=env.num_states, dtype=int)
    # SOLUTION

    for i in range(max_iterations):
        V = policy_eval_exact(env, pi, gamma)
        pi_new = policy_improvement(env, V, gamma)
        if np.array_equal(pi_new, pi):
            return pi_new
        else:
            pi = pi_new
    else:
        print(f"Failed to converge after {max_iterations} steps.")
        return pi
```
</details>


Once you've passed the tests, you should play around with the `penalty` value for the gridworld environment and see how this affects the optimal policy found. Which squares change their optimal strategy when the penalty takes different negative values? Can you see why this happens?


## Bonus

- Implement and test your policy evaluation method on other environments.
- Complete some exercises in Chapters 3 and 4 of Sutton and Barto.
- Modify the tabular RL solvers to allow stochastic policies or to allow $\gamma=1$ on episodic environments (may need to change how environments are defined.)




""", unsafe_allow_html=True)


func_page_list = [
    (section_0, "🏠 Home"),     (section_1, "1️⃣ Multi-Armed Bandit"),     (section_2, "2️⃣ Tabular RL & Policy Improvement"), 
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
