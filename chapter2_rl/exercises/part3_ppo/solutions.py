import os
import random
import time
import sys
import re
from dataclasses import dataclass
import numpy as np
from numpy.random import Generator
import torch as t
from torch import Tensor
import gym
from gym.envs.classic_control.cartpole import CartPoleEnv
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from gym.spaces import Discrete
import einops
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Callable, Optional
from jaxtyping import Float, Int, Bool

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part3_ppo"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from part1_intro_to_rl.utils import make_env
from part2_dqn.utils import set_global_seeds
import part3_ppo.utils as utils
import part3_ppo.tests as tests
from plotly_utils import plot_cartpole_obs_and_dones

Arr = np.ndarray

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"




@dataclass
class PPOArgs:
    exp_name: str = "PPO_Implementation"
    seed: int = 1
    cuda: bool = True
    track: bool = True
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


if MAIN:
    args = PPOArgs(minibatch_size=256)
    utils.arg_help(args)



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
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



if MAIN:
    tests.test_compute_advantages(compute_advantages)


# %%

def minibatch_indexes(rng: Generator, batch_size: int, minibatch_size: int) -> List[np.ndarray]:
    '''Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    assert batch_size % minibatch_size == 0
    # SOLUTION
    indices = rng.permutation(batch_size)
    indices = einops.rearrange(indices, "(mb_num mb_size) -> mb_num mb_size", mb_size=minibatch_size)
    return list(indices)


@dataclass
class ReplayBufferSamples:
    '''
    Samples from the replay buffer, converted to PyTorch for use in neural network training.
    '''
    obs: Float[Tensor, "sampleSize *obsShape"]
    dones: Int[Tensor, "sampleSize"]
    actions: Int[Tensor, "sampleSize"]

    logprobs: Float[Tensor, "sampleSize"]
    advantages: Float[Tensor, "sampleSize"]
    returns: Float[Tensor, "sampleSize"]
    values: Float[Tensor, "sampleSize"]


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
        self.buffer = [None for _ in range(6)]
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

        for i, (arr, arr_list) in enumerate(zip([obs, actions, rewards, dones, logprobs, values], self.buffer)):
            assert arr.shape[0] == self.num_environments
            if isinstance(arr, np.ndarray):
                arr = t.from_numpy(arr)
            arr = arr[None, :].cpu()
            if arr_list is None:
                self.buffer[i] = arr
            else:
                self.buffer[i] = t.concat((arr, arr_list))
            if self.buffer[i].shape[0] > self.num_steps:
                self.buffer[i] = self.buffer[i][:self.num_steps]

        self.obs, self.actions, self.rewards, self.dones, self.logprobs, self.values = self.buffer


    def get_minibatches(self, next_value: t.Tensor, next_done: t.Tensor):
        indices = minibatch_indexes(self.rng, self.batch_size, self.minibatch_size)
        advantages = compute_advantages(next_value, next_done, self.rewards, self.values, self.dones.float(), self.gamma, self.gae_lambda)
        returns = advantages + self.values
        replaybuffer_args = [self.obs, self.dones, self.actions, self.logprobs, advantages, returns, self.values]
        self.minibatches = [
            ReplayBufferSamples(*[arg.flatten(0, 1)[index].to(device) for arg in replaybuffer_args])
            for index in indices
        ]
    

    def reset(self) -> None:
        '''
        Reset the buffer to empty.
        '''
        self.buffer = [None for _ in range(6)]
    
# %%


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
        with t.inference_mode():
            values = self.critic(obs).flatten()
            logits = self.actor(obs)
        
        probs = Categorical(logits=logits)
        actions = probs.sample()
        logprobs = probs.log_prob(actions)
        next_obs, rewards, next_dones, infos = self.envs.step(actions.cpu().numpy())
        rewards = t.from_numpy(rewards).to(device)

        # (s_t, d_t, a_t, logpi(a_t|s_t), r_t+1, v(s_t))
        self.rb.add(obs, actions, rewards, next_dones, logprobs, values)

        self.next_obs = t.from_numpy(next_obs).to(device)
        self.next_done = t.from_numpy(next_dones).to(device, dtype=t.float)
        self.steps += 1

        return infos
    
    def reset(self) -> None:
        '''
        Reset the agent's state (called at the end of each batch).
        '''
        self.next_obs = t.tensor(self.envs.reset()).to(device)
        self.next_done = t.zeros(self.envs.num_envs).to(device, dtype=t.float)
        self.next_value = self.critic(self.next_obs).flatten().detach()
        self.rb.reset()


# %%

def calc_clipped_surrogate_objective(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float
) -> t.Tensor:
    '''Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs: a distribution containing the actor's unnormalized logits of shape (minibatch, num_actions)

    clip_coef: amount of clipping, denoted by epsilon in Eq 7.
    '''
    # SOLUTION
    logits_diff = probs.log_prob(mb_action) - mb_logprobs

    r_theta = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_advantages.std()

    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1-clip_coef, 1+clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()


def calc_value_function_loss(values: t.Tensor, mb_returns: t.Tensor, vf_coef: float) -> t.Tensor:
    '''Compute the value function portion of the loss function.

    vf_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    return 0.5 * vf_coef * (values - mb_returns).pow(2).mean()


def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''Return the entropy bonus term, suitable for gradient ascent.

    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall loss. Denoted by c_2 in the paper.
    '''
    return ent_coef * probs.entropy().mean()


# %%

# def train_ppo(args):

#     run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
#     if args.track:
#         wandb.init(
#             project=args.wandb_project_name,
#             entity=args.wandb_entity,
#             config=vars(args), # vars is equivalent to args.__dict__
#             name=run_name,
#             monitor_gym=True,
#             save_code=True,
#         )
#     set_global_seeds(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
#     envs = gym.vector.SyncVectorEnv(
#         [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
#     )
#     assert envs.single_action_space.shape is not None
#     assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"
#     agent = Agent(envs).to(device)
#     num_updates = args.total_timesteps // args.batch_size
#     (optimizer, scheduler) = make_optimizer(agent, num_updates, args.learning_rate, 0.0)
    
#     "YOUR CODE HERE: initialise your memory object"
#     memory = Memory(envs, args, device)

#     progress_bar = tqdm(range(num_updates))
    
#     for _ in progress_bar:

#         "YOUR CODE HERE: perform rollout and learning steps, and optionally log vars"
#         agent.rollout(memory, args, envs)
#         agent.learn(memory, args, optimizer, scheduler)
        
#         if args.track:
#             memory.log()
        
#         desc = memory.get_progress_bar_description()
#         if desc:
#             progress_bar.set_description(desc)

#         memory.reset()

#     # If running one of the Probe environments, test if learned critic values are sensible after training
#     obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
#     expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]], [[args.gamma], [1.0]], [[-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]
#     tolerances = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]
#     match = re.match(r"Probe(\d)-v0", args.env_id)
#     if match:
#         probe_idx = int(match.group(1)) - 1
#         obs = t.tensor(obs_for_probes[probe_idx]).to(device)
#         value = agent.critic(obs)
#         print("Value: ", value)
#         expected_value = t.tensor(expected_value_for_probes[probe_idx]).to(device)
#         t.testing.assert_close(value, expected_value, atol=tolerances[probe_idx], rtol=0)

#     envs.close()
#     if args.track:
#         wandb.finish()


# # %%

# import sys
# # sys.path.append(r"C:\ffmpeg\bin")
# # sys.path.append(r"C:\ffmpeg")

# if MAIN:
#     args = PPOArgs()
#     # args.track = False
#     arg_help(args)
#     train_ppo(args)

# # %%

# from gym.envs.classic_control.cartpole import CartPoleEnv
# import gym
# from gym import logger, spaces
# from gym.error import DependencyNotInstalled
# import math

# class EasyCart(CartPoleEnv):
#     def step(self, action):
#         (obs, rew, done, info) = super().step(action)

#         x, v, theta, omega = obs

#         # First reward: angle should be close to zero
#         reward_1 = 1 - abs(theta / 0.2095)
#         # Second reward: position should be close to the center
#         reward_2 = 1 - abs(x / 2.4)

#         reward = 0.3 * reward_1 + 0.7 * reward_2

#         return (obs, reward, done, info)


# if MAIN:
#     gym.envs.registration.register(id="EasyCart-v0", entry_point=EasyCart, max_episode_steps=500)
#     args = PPOArgs()
#     args.env_id = "EasyCart-v0"
#     # args.track = False
#     args.gamma = 0.995
#     train_ppo(args)

# # %%

# class SpinCart(CartPoleEnv):

#     def step(self, action):
#         obs, rew, done, info = super().step(action)
#         # YOUR CODE HERE
#         x, v, theta, omega = obs
#         # Allow for 360-degree rotation
#         done = (abs(x) > self.x_threshold)
#         # Reward function incentivises fast spinning while staying still & near centre
#         rotation_speed_reward = min(1, 0.1*abs(omega))
#         stability_penalty = max(1, abs(x/2.5) + abs(v/10))
#         reward = rotation_speed_reward - 0.5 * stability_penalty
#         return (obs, reward, done, info)



# if MAIN:
#     gym.envs.registration.register(id="SpinCart-v0", entry_point=SpinCart, max_episode_steps=500)
#     args = PPOArgs()
#     args.env_id = "SpinCart-v0"
#     train_ppo(args)

# # %%