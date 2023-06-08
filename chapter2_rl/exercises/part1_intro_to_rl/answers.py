#%%
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

!pip install gym==0.23.1
!pip install pygame

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

# %%
ObsType = int
ActType = int

class MultiArmedBandit(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    num_arms: int
    stationary: bool
    arm_reward_means: np.ndarray
    arm_star: int

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
# %%
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
# %%
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
        return self.rng.integers(0, self.num_arms)

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


# %%

class RewardAveraging(Agent):
    def __init__(self, num_arms: int, seed: int, epsilon: float, optimism: float):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.optimism = optimism
        self.reset(seed)


    def get_action(self):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.num_arms)
        else:
            return np.argmax(self.average_rewards) 

    def observe(self, action, reward, info):
        self.arm_num_samples[action] += 1
        new_average = self.average_rewards[action] + 1 / self.arm_num_samples[action] * (reward - self.average_rewards[action])
        
        self.average_rewards[action] = new_average

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.arm_num_samples = np.zeros(shape=(num_arms,))
        self.average_rewards = np.ones(shape=(num_arms,)) * self.optimism


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
# %%
class CheatyMcCheater(Agent):
    def __init__(self, num_arms: int, seed: int):
        self.num_arms = num_arms
        self.best_arm = 0

    def get_action(self):
        return self.best_arm

    def observe(self, action: int, reward: float, info: dict):
        self.best_arm = info["best_arm"]

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
# %%
class UCBActionSelection(Agent):
    def __init__(self, num_arms: int, seed: int, c: float, eps: float = 1e-6):
        self.c = c
        self.num_arms = num_arms
        self.eps = eps
        self.reset(seed)
        

    def get_action(self):
        Q = self.R/(self.N + self.eps)
        UCB_vals = Q + self.c * np.sqrt(np.log(self.t)/ (self.N + self.eps))
        return np.argmax(UCB_vals)


    def observe(self, action, reward, info):
        self.t += 1
        self.N[action] += 1
        self.R[action] += reward
        pass

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.t = 1
        self.N = np.zeros(shape=(self.num_arms,))
        self.R = np.zeros(shape=(self.num_arms,))



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
# %%
