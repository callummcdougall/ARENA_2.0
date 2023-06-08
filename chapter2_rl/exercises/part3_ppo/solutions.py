# %%
import os
import random
import time
import sys
os.chdir(r"C:\Users\calsm\Documents\AI Alignment\ARENA\arena-v1-ldn-exercises-restructured")
sys.path.append(r"C:\Users\calsm\Documents\AI Alignment\ARENA\arena-v1-ldn-exercises-restructured")
from dataclasses import dataclass, field
import re
import numpy as np
import torch
import torch as t
import gym
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from gym.spaces import Discrete
from einops import rearrange, repeat
import wandb
import plotly.express as px
from typing import Optional, Any, Tuple, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict

from w4d3_chapter4_ppo.utils import make_env, PPOArgs, arg_help, plot_cartpole_obs_and_dones, set_global_seeds

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %%

@dataclass
class Minibatch:
    obs: t.Tensor
    actions: t.Tensor
    logprobs: t.Tensor
    advantages: t.Tensor
    returns: t.Tensor
    values: t.Tensor

class Memory():

    def __init__(self, envs: gym.vector.SyncVectorEnv, args: PPOArgs, device):
        self.envs = envs
        self.args = args
        self.next_obs = None
        self.next_done = None
        self.next_value = None
        self.device = device
        self.global_step = 0
        self.reset()

    def add(self, *data):
        '''Adds an experience to storage. Called during the rollout phase.
        '''
        info, *experiences = data
        self.experiences.append(experiences)
        for item in info:
            if "episode" in item.keys():
                self.episode_lengths.append(item["episode"]["l"])
                self.episode_returns.append(item["episode"]["r"])
                self.add_vars_to_log(
                    episode_length = item["episode"]["l"],
                    episode_return = item["episode"]["r"],
                )
            self.global_step += 1

    def get_minibatches(self) -> List[Minibatch]:
        '''Computes advantages, and returns minibatches to be used in the 
        learning phase.
        '''
        obs, dones, actions, logprobs, rewards, values = [t.stack(arr) for arr in zip(*self.experiences)]
        advantages = compute_advantages(self.next_value, self.next_done, rewards, values, dones, self.device, self.args.gamma, self.args.gae_lambda)
        returns = advantages + values
        return make_minibatches(
            obs, actions, logprobs, advantages, values, returns, self.args.batch_size, self.args.minibatch_size
        )

    def get_progress_bar_description(self) -> Optional[str]:
        '''Creates a progress bar description, if any episodes have terminated. 
        If not, then the bar's description won't change.
        '''
        if self.episode_lengths:
            global_step = self.global_step
            avg_episode_length = np.mean(self.episode_lengths)
            avg_episode_return = np.mean(self.episode_returns)
            return f"{global_step=:<06}, {avg_episode_length=:<3.0f}, {avg_episode_return=:<3.0f}"

    def reset(self) -> None:
        '''Function to be called at the end of each rollout period, to make 
        space for new experiences to be generated.
        '''
        self.experiences = []
        self.vars_to_log = defaultdict(dict)
        self.episode_lengths = []
        self.episode_returns = []
        if self.next_obs is None:
            self.next_obs = torch.tensor(self.envs.reset()).to(self.device)
            self.next_done = torch.zeros(self.envs.num_envs).to(self.device, dtype=t.float)

    def add_vars_to_log(self, **kwargs):
        '''Add variables to storage, for eventual logging (if args.track=True).
        '''
        self.vars_to_log[self.global_step] |= kwargs

    def log(self):
        '''Logs variables to wandb.
        '''
        for step, vars_to_log in self.vars_to_log.items():
            wandb.log(vars_to_log, step=step)

# %%
class PPOScheduler:
    def __init__(self, optimizer, initial_lr: float, end_lr: float, num_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        '''Implement linear learning rate decay so that after num_updates calls to step, the learning rate is end_lr.
        '''
        self.n_step_calls += 1
        frac = self.n_step_calls / self.num_updates
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)

# %%

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        self.obs_shape = envs.single_observation_space.shape
        self.num_obs = np.array(self.obs_shape).item()
        self.num_actions = envs.single_action_space.n
        self.critic = nn.Sequential(
            # nn.Flatten(),
            layer_init(nn.Linear(self.num_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.num_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.num_actions), std=0.01)
        )

    def rollout(self, memory: Memory, args: PPOArgs, envs: gym.vector.SyncVectorEnv) -> None:
        '''Performs the rollout phase, as described in '37 Implementational Details'.
        '''

        device = memory.device

        obs = memory.next_obs
        done = memory.next_done

        for step in range(args.num_steps):

            # Generate the next set of new experiences (one for each env)
            with t.inference_mode():
                value = self.critic(obs).flatten()
                logits = self.actor(obs)
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            reward = t.from_numpy(reward).to(device)

            # (s_t, d_t, a_t, logpi(a_t|s_t), r_t+1, v(s_t))
            memory.add(info, obs, done, action, logprob, reward, value)

            obs = t.from_numpy(next_obs).to(device)
            done = t.from_numpy(next_done).to(device, dtype=t.float)

        # Store last (obs, done, value) tuple, since we need it to compute advantages
        memory.next_obs = obs
        memory.next_done = done
        # Compute advantages, and store them in memory
        with t.inference_mode():
            memory.next_value = self.critic(obs).flatten()

    def learn(self, memory: Memory, args: PPOArgs, optimizer: optim.Adam, scheduler: PPOScheduler) -> None:
        '''Performs the learning phase, as described in '37 Implementational 
        Details'.
        '''

        for _ in range(args.update_epochs):
            minibatches = memory.get_minibatches()
            # Compute loss on each minibatch, and step the optimizer
            for mb in minibatches:
                logits = self.actor(mb.obs)
                probs = Categorical(logits=logits)
                values = self.critic(mb.obs)
                clipped_surrogate_objective = calc_clipped_surrogate_objective(probs, mb.actions, mb.advantages, mb.logprobs, args.clip_coef)
                value_loss = calc_value_function_loss(values, mb.returns, args.vf_coef)
                entropy_bonus = calc_entropy_bonus(probs, args.ent_coef)
                total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus
                optimizer.zero_grad()
                total_objective_function.backward()
                nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                optimizer.step()

        # Step the scheduler
        scheduler.step()

        # Get debug variables, for just the most recent minibatch
        if args.track:
            with t.inference_mode():
                newlogprob = probs.log_prob(mb.actions)
                logratio = newlogprob - mb.logprobs
                ratio = logratio.exp()
                approx_kl = (ratio - 1 - logratio).mean().item()
                clipfracs = [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
            memory.add_vars_to_log(
                learning_rate = optimizer.param_groups[0]["lr"],
                value_loss = value_loss.item(),
                clipped_surrogate_objective = clipped_surrogate_objective.item(),
                entropy = entropy_bonus.item(),
                approx_kl = approx_kl,
                clipfrac = np.mean(clipfracs)
            )

# %%

def make_optimizer(agent: Agent, num_updates: int, initial_lr: float, end_lr: float) -> Tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.
    '''
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, num_updates)
    return (optimizer, scheduler)

# %%

def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    device: t.device,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
    '''Compute advantages using Generalized Advantage Estimation.
    next_value: shape (env,)
    next_done: shape (env,)
    rewards: shape (t, env)
    values: shape (t, env)
    dones: shape (t, env)
    Return: shape (t, env)
    '''
    T = values.shape[0]
    next_values = torch.concat([values[1:], next_value.unsqueeze(0)])
    next_dones = torch.concat([dones[1:], next_done.unsqueeze(0)])
    deltas = rewards + gamma * next_values * (1.0 - next_dones) - values
    advantages = torch.zeros_like(deltas).to(device)
    advantages[-1] = deltas[-1]
    for t in reversed(range(1, T)):
        advantages[t-1] = deltas[t-1] + gamma * gae_lambda * (1.0 - dones[t]) * advantages[t]
    return advantages


# def shift_rows(arr):
#     """
#     Helper function for compute_advantages_vectorized

#     Given a 1D array like:
#         [1, 2, 3]
#     this function will return:
#         [[1, 2, 3],
#          [0, 1, 2],
#          [0, 0, 1]]

#     If the array has >1D, it treats the later dimensions as batch dims
#     """
#     L = arr.shape[0]
#     output = t.zeros(L, 2*L, *arr.shape[1:]).to(dtype=arr.dtype)
#     output[:, :L] = arr[None, :]
#     output = rearrange(output, "t1 t2 ... -> (t1 t2) ...")
#     output = output[:L*(2*L-1)]
#     output = rearrange(output, "(t1 t2) ... -> t1 t2 ...", t1=L)
#     output = output[:, :L]

#     return output

# def compute_advantages_vectorized(
#     next_value: t.Tensor,
#     next_done: t.Tensor,
#     rewards: t.Tensor,
#     values: t.Tensor,
#     dones: t.Tensor,
#     device: t.device,
#     gamma: float,
#     gae_lambda: float,
# ) -> t.Tensor:
#     """
#     Basic idea (assuming num_envs=1 in this description, but the case generalises):

#         create a matrix of discount factors (gamma*lmda)**l, shape (t, l), suitably shifted
#         create a matrix of deltas, shape (t, l), suitably shifted
#         mask the deltas after the "done" points
#         multiply two matrices and sum over l (second dim)
#     """
#     T, num_envs = rewards.shape
#     next_values = torch.concat([values[1:], next_value.unsqueeze(0)])
#     next_dones = torch.concat([dones[1:], next_done.unsqueeze(0)])
#     deltas = rewards + gamma * next_values * (1.0 - next_dones) - values

#     deltas_repeated = repeat(deltas, "t2 env -> t1 t2 env", t1=T)
#     mask = repeat(next_dones, "t2 env -> t1 t2 env", t1=T).to(device)
#     mask_uppertri = repeat(t.triu(t.ones(T, T)), "t1 t2 -> t1 t2 env", env=num_envs).to(device)
#     mask = mask * mask_uppertri
#     mask = 1 - (mask.cumsum(dim=1) > 0).float()
#     mask = t.concat([t.ones(T, 1, num_envs).to(device), mask[:, :-1]], dim=1)
#     mask = mask * mask_uppertri
#     deltas_masked = mask * deltas_repeated

#     discount_factors = (gamma * gae_lambda) ** t.arange(T).to(device)
#     discount_factors_repeated = repeat(discount_factors, "t -> t env", env=num_envs)
#     discount_factors_shifted = shift_rows(discount_factors_repeated).to(device)

#     advantages = (discount_factors_shifted * deltas_masked).sum(dim=1)
#     return advantages



# %%

def minibatch_indexes(batch_size: int, minibatch_size: int) -> List[np.ndarray]:
    '''Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    assert batch_size % minibatch_size == 0
    indices = np.random.permutation(batch_size)
    indices = rearrange(indices, "(mb_num mb_size) -> mb_num mb_size", mb_size=minibatch_size)
    return list(indices)

def make_minibatches(
    obs: t.Tensor,
    actions: t.Tensor,
    logprobs: t.Tensor,
    advantages: t.Tensor,
    values: t.Tensor,
    returns: t.Tensor,
    batch_size: int,
    minibatch_size: int,
) -> List[Minibatch]:
    '''Flatten the environment and steps dimension into one batch dimension, then shuffle and split into minibatches.'''

    return [
        Minibatch(
            obs.flatten(0, 1)[ind], 
            actions.flatten(0, 1)[ind], 
            logprobs.flatten(0, 1)[ind], 
            advantages.flatten(0, 1)[ind], 
            values.flatten(0, 1)[ind],
            returns.flatten(0, 1)[ind], 
        )
        for ind in minibatch_indexes(batch_size, minibatch_size)
    ]


if MAIN:
    num_envs = 4
    run_name = "test-run"
    envs = gym.vector.SyncVectorEnv(
        [make_env("CartPole-v1", i, i, False, run_name) for i in range(num_envs)]
    )
    args = PPOArgs()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    memory = Memory(envs, args, device)
    agent = Agent(envs).to(device)
    agent.rollout(memory, args, envs)

    obs = t.stack([e[0] for e in memory.experiences])
    done = t.stack([e[1] for e in memory.experiences])
    plot_cartpole_obs_and_dones(obs, done)

    # def write_to_html(fig, filename):
    #     with open(f"{filename}.html", "w") as f:
    #         f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    
    # write_to_html(fig, '090 Trig Loss Ratio.html')

# %%

def calc_clipped_surrogate_objective(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float
) -> t.Tensor:
    '''Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs: a distribution containing the actor's unnormalized logits of shape (minibatch, num_actions)

    clip_coef: amount of clipping, denoted by epsilon in Eq 7.
    '''
    logits_diff = probs.log_prob(mb_action) - mb_logprobs

    r_theta = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_advantages.std()

    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1-clip_coef, 1+clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()

# %%

def calc_value_function_loss(values: t.Tensor, mb_returns: t.Tensor, vf_coef: float) -> t.Tensor:
    '''Compute the value function portion of the loss function.

    vf_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    return 0.5 * vf_coef * (values - mb_returns).pow(2).mean()

# %%

def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''Return the entropy bonus term, suitable for gradient ascent.

    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall loss. Denoted by c_2 in the paper.
    '''
    return ent_coef * probs.entropy().mean()


# %%

def train_ppo(args):

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args), # vars is equivalent to args.__dict__
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    set_global_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert envs.single_action_space.shape is not None
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"
    agent = Agent(envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    (optimizer, scheduler) = make_optimizer(agent, num_updates, args.learning_rate, 0.0)
    
    "YOUR CODE HERE: initialise your memory object"
    memory = Memory(envs, args, device)

    progress_bar = tqdm(range(num_updates))
    
    for _ in progress_bar:

        "YOUR CODE HERE: perform rollout and learning steps, and optionally log vars"
        agent.rollout(memory, args, envs)
        agent.learn(memory, args, optimizer, scheduler)
        
        if args.track:
            memory.log()
        
        desc = memory.get_progress_bar_description()
        if desc:
            progress_bar.set_description(desc)

        memory.reset()

    # If running one of the Probe environments, test if learned critic values are sensible after training
    obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
    expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]], [[args.gamma], [1.0]], [[-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]
    tolerances = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]
    match = re.match(r"Probe(\d)-v0", args.env_id)
    if match:
        probe_idx = int(match.group(1)) - 1
        obs = t.tensor(obs_for_probes[probe_idx]).to(device)
        value = agent.critic(obs)
        print("Value: ", value)
        expected_value = t.tensor(expected_value_for_probes[probe_idx]).to(device)
        t.testing.assert_close(value, expected_value, atol=tolerances[probe_idx], rtol=0)

    envs.close()
    if args.track:
        wandb.finish()


# %%

import sys
# sys.path.append(r"C:\ffmpeg\bin")
# sys.path.append(r"C:\ffmpeg")

if MAIN:
    args = PPOArgs()
    # args.track = False
    arg_help(args)
    train_ppo(args)

# %%

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
    gym.envs.registration.register(id="SpinCart-v0", entry_point=SpinCart, max_episode_steps=500)
    args = PPOArgs()
    args.env_id = "SpinCart-v0"
    train_ppo(args)

# %%