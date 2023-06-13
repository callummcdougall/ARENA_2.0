
# %%

# %pip install wandb==0.13.10

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
from plotly_utils import plot_cartpole_obs_and_dones

# Register our probes from last time
for idx, probe in enumerate([Probe1, Probe2, Probe3, Probe4, Probe5]):
    gym.envs.registration.register(id=f"Probe{idx+1}-v0", entry_point=probe)

Arr = np.ndarray

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


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

# %%

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
    
    actor = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, num_actions), std=0.01)
    ).to(device)
    critic = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1)
    ).to(device)

    return (actor, critic)



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
    '''Compute advantages using Generalized Advantage Estimation.
    next_value: shape (env,)
    next_done: shape (env,)
    rewards: shape (buffer_size, env)
    values: shape (buffer_size, env)
    dones: shape (buffer_size, env)
    Return: shape (buffer_size, env)
    '''
    T, n_env = rewards.shape

    # print(dones)
    next_values = t.cat([values[1:, :], next_value[None]])
    next_dones = t.cat([dones[1:, :], next_done[None]])

    delta = rewards + gamma*(1 - next_dones).float()*next_values - values # Perhaps drop the dones at the end
    advantage = t.zeros(T+1, n_env).to(rewards.device)

    for t_ in range(T-1, -1, -1):
        advantage[t_] = delta[t_] + (1 - next_dones[t_]).float()*(gamma*gae_lambda)*advantage[t_+1] 

    return advantage[:-1]

tests.test_compute_advantages(compute_advantages)

# %%

def minibatch_indexes(rng: Generator, batch_size: int, minibatch_size: int) -> List[np.ndarray]:
    '''
    Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    assert batch_size % minibatch_size == 0
    
    permuted_idx = rng.permutation(batch_size)
    return [permuted_idx[i*minibatch_size:(i+1)*minibatch_size] for i in range(batch_size//minibatch_size)]


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


    def add(self, obs: Arr, actions: Arr, rewards: Arr, dones: Arr, logprobs: Arr, values: Arr) -> None:
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

# %%

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
# %%

minibatches = rb.get_minibatches(next_value, next_done)

obs = minibatches[0].obs
dones = minibatches[0].dones

plot_cartpole_obs_and_dones(obs, dones)
# %%

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


    # def play_step(self) -> List[dict]:
    #     '''
    #     Carries out a single interaction step between the agent and the environment, and adds results to the replay buffer.
    #     '''
    #     with t.inference_mode():
    #         obs = (t.from_numpy(self.next_obs) if isinstance(self.next_obs, np.ndarray) else self.next_obs).to(device)
    #         action_logits = self.actor(obs)
    #         values = self.critic(obs).squeeze(1)
    #     actions = t.distributions.categorical.Categorical(logits=action_logits).sample()
    #     next_obs, rewards, dones, infos = self.envs.step(actions.cpu().numpy())
    #     action_probs = t.log_softmax(action_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    #     self.rb.add(obs, actions, t.from_numpy(rewards), t.from_numpy(dones), action_probs, values)
    #     self.steps += self.num_envs
    #     self.next_obs = next_obs
    #     self.next_done = dones

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

        for obs_i, info_i in zip(obs, infos):
            info_i['obs'] = obs_i 

        return infos

    def get_minibatches(self) -> None:
        '''
        Gets minibatches from the replay buffer.
        '''
        with t.inference_mode():
            next_value = self.critic(self.next_obs).flatten()
        return self.rb.get_minibatches(next_value, self.next_done)


tests.test_ppo_agent(PPOAgent)
# %%

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
    action_probs = probs.probs.gather(1, mb_action.unsqueeze(1)).squeeze(1)
    old_action_probs = t.exp(mb_logprobs)
    r_t = action_probs/(old_action_probs + eps)
    adv_normalized = (mb_advantages - t.mean(mb_advantages)) / (t.std(mb_advantages) + eps)
    return t.mean(t.min(r_t * adv_normalized, t.clip(r_t, min=1-clip_coef, max=1+clip_coef)*adv_normalized))


tests.test_calc_clipped_surrogate_objective(calc_clipped_surrogate_objective)
# %%
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
    return vf_coef * 0.5 * (values - mb_returns).pow(2).mean()


tests.test_calc_value_function_loss(calc_value_function_loss)
# %%
def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''Return the entropy bonus term, suitable for gradient ascent.

    probs:
        the probability distribution for the current policy
    ent_coef: 
        the coefficient for the entropy loss, which weights its contribution to the overall objective function. Denoted by c_2 in the paper.
    '''
    return probs.entropy().mean() * ent_coef


tests.test_calc_entropy_bonus(calc_entropy_bonus)
# %%
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
        lr = self.initial_lr + (self.end_lr - self.initial_lr) * (self.n_step_calls / self.total_training_steps)
        for i in self.optimizer.param_groups:
            i["lr"] = lr

def make_optimizer(agent: PPOAgent, total_training_steps: int, initial_lr: float, end_lr: float) -> Tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_training_steps)
    return (optimizer, scheduler)


tests.test_ppo_scheduler(PPOScheduler)
# %%
class MyDataset(Dataset):
    def __init__(self, batches: List[ReplayBufferSamples]):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class PPOLightning(pl.LightningModule):
    agent: PPOAgent

    def __init__(self, args: PPOArgs):
        super().__init__()
        self.args = args
        set_global_seeds(args.seed)
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, self.run_name) for i in range(args.num_envs)])
        self.agent = PPOAgent(self.args, self.envs).to(device)
        self.rollout_phase()


    def on_train_epoch_end(self) -> None:
        self.rollout_phase()


    def rollout_phase(self) -> None:
        '''Should populate the replay buffer with new experiences.'''
        for _ in range(self.args.num_steps):
            infos = self.agent.play_step()
            self.log('new reward ratio', np.mean(infos[0]['new_reward_ratio']))
            self.log('Position', infos[0]['obs'][0])
            self.log('Velocity', infos[0]['obs'][1])
            if 'episode' in infos[0].keys():
                self.log('Episode duration', infos[0]['episode']['l'])
                self.log('Reward in episode', infos[0]['episode']['r'])

    def training_step(self, minibatch: ReplayBufferSamples, minibatch_idx: int) -> Float[Tensor, ""]:
        '''Handles learning phase for a single minibatch. Returns objective function to be maximized.'''
        probs = t.distributions.categorical.Categorical(logits=self.agent.actor(minibatch.obs))
        values = self.agent.critic(minibatch.obs)
        L_clip = calc_clipped_surrogate_objective(probs, minibatch.actions, minibatch.advantages, minibatch.logprobs, self.args.clip_coef)
        L_vf = calc_value_function_loss(values.squeeze(1), minibatch.returns, self.args.vf_coef)
        L_h = calc_entropy_bonus(probs, self.args.ent_coef)
        self.log('clip', L_clip)
        self.log('vf', L_vf)
        self.log('entropy', L_h)
        self.log('Mean Values', t.mean(minibatch.values))
        self.scheduler.step()
        return L_clip - L_vf + L_h

    def configure_optimizers(self):
        '''Returns optimizer and scheduler (sets scheduler as attribute, so we can call self.scheduler.step() during each training step)'''
        optimizer, scheduler = make_optimizer(self.agent, self.args.total_training_steps, self.args.learning_rate, 0.0)
        self.scheduler = scheduler
        return optimizer


    def train_dataloader(self):
        return MyDataset(self.agent.get_minibatches())
    
# %%
# probe_idx = 1

# # Define a set of arguments for our probe experiment
# args = PPOArgs(
#     env_id=f"Probe{probe_idx}-v0",
#     exp_name=f"test-probe-{probe_idx}", 
#     total_timesteps=10000 if probe_idx <= 3 else 30000,
#     learning_rate=0.001,
#     capture_video=False,
#     use_wandb=False,
# )
# model = PPOLightning(args).to(device)
# logger = CSVLogger(save_dir=args.log_dir, name=args.exp_name)

# # Run our experiment
# trainer = pl.Trainer(
#     max_epochs=args.total_epochs,
#     logger=logger,
#     log_every_n_steps=10,
#     gradient_clip_val=args.max_grad_norm,
#     reload_dataloaders_every_n_epochs=1,
#     enable_progress_bar=False,
# )
# trainer.fit(model=model)

# # Check that our final results were the ones we expected from this probe
# obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
# expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]], [[args.gamma], [1.0]], [[1.0]], [[1.0], [1.0]]]
# expected_probs_for_probes = [None, None, None, [[0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
# tolerances = [5e-4, 5e-4, 5e-4, 1e-3, 1e-3]
# obs = t.tensor(obs_for_probes[probe_idx-1]).to(device)
# model.to(device)
# with t.inference_mode():
#     value = model.agent.critic(obs)
#     probs = model.agent.actor(obs).softmax(-1)
# expected_value = t.tensor(expected_value_for_probes[probe_idx-1]).to(device)
# t.testing.assert_close(value, expected_value, atol=tolerances[probe_idx-1], rtol=0)
# expected_probs = expected_probs_for_probes[probe_idx-1]
# if expected_probs is not None:
#     t.testing.assert_close(probs, t.tensor(expected_probs).to(device), atol=tolerances[probe_idx-1], rtol=0)
# print("Probe tests passed!")

# # Use the code below to inspect your most recent logged results
# try:
#     metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
#     metrics.tail()
# except:
#     print("No logged metrics found. You can log things using `self.log(metric_name, metric_value)` or `self.log_dict(d)` where d is a dict of {name: value}.")
# %%
##############################
# Acrobot
##############################

from gym.envs.classic_control.acrobot import AcrobotEnv

@dataclass
class AcrobotArgs:
    exp_name: str = "EasyAcrobot_Implementation"
    seed: int = 1
    cuda: bool = t.cuda.is_available()
    log_dir: str = "logs"
    use_wandb: bool = True
    wandb_project_name: str = "EasyAcrobot"
    wandb_entity: str = None
    capture_video: bool = True
    env_id: str = "EasyAcrobot-v1"
    # total_timesteps: int = 500000
    total_timesteps: int = 200000
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

class EasyAcrobot(AcrobotEnv):
    def step(self, action):
        (obs, reward, done, info) = super().step(action)
        # new_reward_addn = abs(obs[4]) * 0.01 / (abs(obs[1])+1e-2)
        new_reward_addn = abs(obs[4] + obs[5])
        info['new_reward_ratio'] = abs(new_reward_addn / reward)
        reward += new_reward_addn
        return (obs, reward, done, info)
gym.envs.registration.register(id="EasyAcrobot-v1", entry_point=EasyAcrobot, max_episode_steps=500)

probe_idx = 1

# Define a set of arguments for our probe experiment
args = AcrobotArgs(
    # env_id=f"Probe{probe_idx}-v0",
    # exp_name=f"test-probe-{probe_idx}", 
    # total_timesteps=10000 if probe_idx <= 3 else 30000,
    # learning_rate=0.001,
    capture_video=True,
    use_wandb=True,
)
utils.arg_help(args)
wandb.finish()
model = PPOLightning(args).to(device)
logger = WandbLogger(project=args.wandb_project_name, save_dir=args.log_dir)
# logger = CSVLogger(save_dir=args.log_dir)

# Run our experiment
trainer = pl.Trainer(
    max_epochs=args.total_epochs,
    logger=logger,
    log_every_n_steps=10,
    gradient_clip_val=args.max_grad_norm,
    reload_dataloaders_every_n_epochs=1,
    enable_progress_bar=False,
)
trainer.fit(model=model)
# %%


from gym.envs.classic_control.mountain_car import MountainCarEnv

@dataclass
class MountainArgs:
    exp_name: str = "EasyMountain_Implementation"
    seed: int = 1
    cuda: bool = t.cuda.is_available()
    log_dir: str = "logs"
    use_wandb: bool = True
    wandb_project_name: str = "EasyMountain"
    wandb_entity: str = None
    capture_video: bool = True
    env_id: str = "EasyMountain-v1"
    # total_timesteps: int = 500000
    total_timesteps: int = 1000000
    learning_rate: float = 0.00025
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    batches_per_epoch: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01 # Changed
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 512
    minibatch_size: int = 128

    def __post_init__(self):
        assert self.batch_size % self.minibatch_size == 0, "batch_size must be divisible by minibatch_size"
        self.total_epochs = self.total_timesteps // (self.num_steps * self.num_envs)
        self.total_training_steps = self.total_epochs * self.batches_per_epoch * (self.batch_size // self.minibatch_size)

class EasyMountain(MountainCarEnv):
    def step(self, action):
        (obs, reward, done, info) = super().step(action)
        # new_reward_addn = abs(obs[4]) * 0.01 / (abs(obs[1])+1e-2)
        new_reward_addn = 1 * (20 * obs[1]**2 + 10 * ((obs[0] + 0.5)*(obs[0] + 0.6)))
        info['new_reward_ratio'] = abs(new_reward_addn / reward)
        reward += new_reward_addn
        return (obs, reward, done, info)

gym.envs.registration.register(id="EasyMountain-v1", entry_point=EasyMountain, max_episode_steps=500)


# Define a set of arguments for our probe experiment
args = MountainArgs()
wandb.finish()
model = PPOLightning(args).to(device)
logger = WandbLogger(project=args.wandb_project_name, save_dir=args.log_dir)
# logger = CSVLogger(save_dir=args.log_dir)

# Run our experiment
trainer = pl.Trainer(
    max_epochs=args.total_epochs,
    logger=logger,
    log_every_n_steps=10,
    gradient_clip_val=args.max_grad_norm,
    reload_dataloaders_every_n_epochs=1,
    enable_progress_bar=False,
)
trainer.fit(model=model)