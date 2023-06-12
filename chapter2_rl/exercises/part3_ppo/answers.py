# %%
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import wandb

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part3_ppo"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

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


# %%
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
        assert (
            self.batch_size % self.minibatch_size == 0
        ), "batch_size must be divisible by minibatch_size"
        self.total_epochs = self.total_timesteps // (self.num_steps * self.num_envs)
        self.total_training_steps = (
            self.total_epochs
            * self.batches_per_epoch
            * (self.batch_size // self.minibatch_size)
        )


args = PPOArgs(minibatch_size=256)
utils.arg_help(args)


# %%
def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_actor_and_critic(envs: gym.vector.SyncVectorEnv) -> Tuple[nn.Module, nn.Module]:
    """
    Returns (actor, critic), the networks used for PPO.
    """
    obs_shape = envs.single_observation_space.shape
    num_obs = np.array(obs_shape).prod()
    num_actions = envs.single_action_space.n

    mid_dim = 64
    final_std = 0.01
    actor = t.nn.Sequential(
        layer_init(t.nn.Linear(in_features=num_obs, out_features=mid_dim)),
        t.nn.Tanh(),
        layer_init(t.nn.Linear(in_features=mid_dim, out_features=mid_dim)),
        t.nn.Tanh(),
        layer_init(
            t.nn.Linear(in_features=mid_dim, out_features=num_actions), std=final_std
        ),
    )

    critic = t.nn.Sequential(
        layer_init(t.nn.Linear(in_features=num_obs, out_features=mid_dim)),
        t.nn.Tanh(),
        layer_init(t.nn.Linear(in_features=mid_dim, out_features=mid_dim)),
        t.nn.Tanh(),
        layer_init(t.nn.Linear(in_features=mid_dim, out_features=1), std=1),
    )
    return actor, critic


tests.test_get_actor_and_critic(get_actor_and_critic)


# %%


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
    """Compute advantages using Generalized Advantage Estimation.
    next_value: shape (env,)
    next_done: shape (env,)
    rewards: shape (buffer_size, env)
    values: shape (buffer_size, env)
    dones: shape (buffer_size, env)
    Return: shape (buffer_size, env)
    """
    # A_pi(state, action) = Q_pi(state, action) - V_pi(state, action)
    # A estimate_t = delta_t + delta_{t+1} + ... delta_{T - 1}

    # TD errors: r_{t+1} + gamma * V_pi(S_{t+1}) - V_pi(S_t)
    buffer_size, env = rewards.shape

    deltas = t.zeros_like(rewards)
    next_values = t.concat([values, next_value.unsqueeze(0)], dim=0)[1:]
    next_dones = t.concat([dones, next_done.unsqueeze(0)], dim=0)[1:]
    still_alives = 1.0 - next_dones

    assert values.shape[0] == buffer_size
    assert dones.shape[0] == buffer_size
    assert still_alives.shape[0] == buffer_size

    deltas = rewards + gamma * next_values * still_alives - values

    assert deltas.shape[0] == buffer_size
    assert deltas.shape[1] == env

    A_estimates = t.zeros_like(rewards)
    A_estimates[-1] = deltas[-1]

    for timestep in range(buffer_size - 2, -1, -1):
        scale = gamma * gae_lambda * (1 - dones[timestep + 1])
        A_estimates[timestep] = deltas[timestep] + scale * A_estimates[timestep + 1]

    return A_estimates


tests.test_compute_advantages(compute_advantages)
# %%


def minibatch_indexes(
    rng: Generator, batch_size: int, minibatch_size: int
) -> List[np.ndarray]:
    """
    Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    """
    assert batch_size % minibatch_size == 0
    all_indexes = np.arange(batch_size)
    rng.shuffle(all_indexes)
    result = []
    for i in range(0, batch_size, minibatch_size):
        result.append(all_indexes[i : i + minibatch_size])
    return result


rng = np.random.default_rng(0)
batch_size = 6
minibatch_size = 2
indexes = minibatch_indexes(rng, batch_size, minibatch_size)

assert np.array(indexes).shape == (batch_size // minibatch_size, minibatch_size)
assert sorted(np.unique(indexes)) == [0, 1, 2, 3, 4, 5]
print("All tests in `test_minibatch_indexes` passed!")


# %%
@dataclass
class ReplayBufferSamples:
    """
    Samples from the replay buffer, converted to PyTorch for use in neural network training.
    """

    obs: Float[Tensor, "minibatch_size *obs_shape"]
    dones: Float[Tensor, "minibatch_size"]
    actions: Int[Tensor, "minibatch_size"]
    logprobs: Float[Tensor, "minibatch_size"]
    values: Float[Tensor, "minibatch_size"]
    advantages: Float[Tensor, "minibatch_size"]
    returns: Float[Tensor, "minibatch_size"]


class ReplayBuffer:
    """
    Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.

    Needs to be initialized with the first obs, dones and values.
    """

    def __init__(self, args: PPOArgs, envs: gym.vector.SyncVectorEnv):
        """Defining all the attributes the buffer's methods will need to access."""
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

    def add(
        self,
        obs: Arr,
        actions: Arr,
        rewards: Arr,
        dones: Arr,
        logprobs: Arr,
        values: Arr,
    ) -> None:
        """
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
        """
        assert obs.shape == (self.num_envs, *self.obs_shape)
        assert actions.shape == (self.num_envs,)
        assert rewards.shape == (self.num_envs,)
        assert dones.shape == (self.num_envs,)
        assert logprobs.shape == (self.num_envs,)
        assert values.shape == (self.num_envs,)

        self.experiences.append((obs, dones, actions, logprobs, values, rewards))

    def get_minibatches(
        self, next_value: t.Tensor, next_done: t.Tensor
    ) -> List[ReplayBufferSamples]:
        minibatches = []

        # Turn all experiences to tensors on our device (we only want to do this
        # once, not every time we add a new experience)
        obs, dones, actions, logprobs, values, rewards = [
            t.stack(arr).to(device) for arr in zip(*self.experiences)
        ]

        # Compute advantages and returns (then get a list of everything we'll need
        # for our replay buffer samples)
        advantages = compute_advantages(
            next_value,
            next_done,
            rewards,
            values,
            dones.float(),
            self.gamma,
            self.gae_lambda,
        )
        returns = advantages + values
        replaybuffer_args = [obs, dones, actions, logprobs, values, advantages, returns]

        # We cycle through the entire buffer `self.batches_per_epoch` times
        for _ in range(self.batches_per_epoch):
            # Get random indices we'll use to generate our minibatches
            indices = minibatch_indexes(self.rng, self.batch_size, self.minibatch_size)

            # Get our new list of minibatches, and add them to the list
            for index in indices:
                minibatch = ReplayBufferSamples(
                    *[arg.flatten(0, 1)[index].to(device) for arg in replaybuffer_args]
                )
                minibatches.append(minibatch)

        # Reset the buffer
        self.experiences = []

        return minibatches


# %%

args = PPOArgs()
envs = gym.vector.SyncVectorEnv(
    [make_env("CartPole-v1", i, i, False, "test") for i in range(4)]
)
next_value = t.zeros(envs.num_envs).to(device)
next_done = t.zeros(envs.num_envs).to(device)
rb = ReplayBuffer(args, envs)
# actions = t.zeros(envs.num_envs).int().to(device)
actions = t.randint(low=0, high=2, size=(envs.num_envs,)).int().to(device)

obs = envs.reset()

for i in range(args.num_steps):
    (next_obs, rewards, dones, infos) = envs.step(actions.cpu().numpy())
    real_next_obs = next_obs.copy()
    for i, done in enumerate(dones):
        if done:
            real_next_obs[i] = infos[i]["terminal_observation"]
    logprobs = values = t.zeros(envs.num_envs)
    rb.add(
        t.from_numpy(obs).to(device),
        actions,
        t.from_numpy(rewards).to(device),
        t.from_numpy(dones).to(device),
        logprobs,
        values,
    )
    obs = next_obs

obs, dones, actions, logprobs, values, rewards = [
    t.stack(arr).to(device) for arr in zip(*rb.experiences)
]

plot_cartpole_obs_and_dones(obs, dones, show_env_jumps=True)
# %%
minibatches = rb.get_minibatches(next_value, next_done)

obs = minibatches[0].obs
dones = minibatches[0].dones

plot_cartpole_obs_and_dones(obs, dones)


# %%
class PPOAgent(nn.Module):
    # dim_obs -> 1 (value)
    critic: nn.Sequential
    # dim_obs -> dim_actions
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
        """
        Carries out a single interaction step between the agent and the environment, and adds results to the replay buffer.
        """
        with t.inference_mode():
            logits = self.actor(self.next_obs)
            logprobs = t.log_softmax(logits, dim=-1)
            values = self.critic(self.next_obs)
            # next_done???
        sampler = t.distributions.categorical.Categorical(logits=logits)
        actions = sampler.sample(sample_shape=(self.num_envs,)).squeeze().cpu()
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.rb.add(
            obs=self.next_obs,
            actions=actions,
            rewards=rewards,
            dones=self.next_done,
            logprobs=logprobs,
            values=values,
        )
        self.next_obs = next_obs
        self.next_dones = dones
 
        self.steps += 1
        return infos

    def get_minibatches(self):
        """
        Gets minibatches from the replay buffer.
        """
        with t.inference_mode():
            next_value = self.critic(self.next_obs).flatten()
        return self.rb.get_minibatches(next_value, self.next_done)


tests.test_ppo_agent(PPOAgent)
# %%
