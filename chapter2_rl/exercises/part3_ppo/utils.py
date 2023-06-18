# %%
import gym
import numpy as np
from typing import List
import argparse
import os
import random
import torch as t
from typing import Optional
from dataclasses import dataclass
import pandas as pd
from IPython.display import display
Arr = np.ndarray
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from einops import rearrange

# %%
def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    """Return a function that returns an environment after setting up boilerplate."""
    
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env, 
                    f"videos/{run_name}", 
                    step_trigger=lambda x : x % 5000 == 0 # Video every 5000 steps for env #1
                )
        obs = env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return thunk

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)

def window_avg(arr: Arr, window: int):
    """
    Computes sliding window average
    """
    return np.convolve(arr, np.ones(window), mode="valid") / window

def cummean(arr: Arr):
    """
    Computes the cumulative mean
    """
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)

# Taken from https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
# See https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
def ewma(arr : Arr, alpha : float):
    '''
    Returns the exponentially weighted moving average of x.
    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}
    Returns:
    --------
    ewma: numpy array
          the exponentially weighted moving average
    '''
    # Coerce x to an array
    s = np.zeros_like(arr)
    s[0] = arr[0]
    for i in range(1,len(arr)):
        s[i] = alpha * arr[i] + (1-alpha)*s[i-1]
    return s


def sum_rewards(rewards : List[int], gamma : float = 1):
    """
    Computes the total discounted sum of rewards for an episode.
    By default, assume no discount
    Input:
        rewards [r1, r2, r3, ...] The rewards obtained during an episode
        gamma: Discount factor
    Output:
        The sum of discounted rewards 
        r1 + gamma*r2 + gamma^2 r3 + ...
    """
    total_reward = 0
    for r in rewards[:0:-1]: #reverse, excluding first
        total_reward += r
        total_reward *= gamma
    total_reward += rewards[0]
    return total_reward

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

arg_help_strings = dict(
    exp_name = "the name of this experiment",
    seed = "seed of the experiment",
    cuda = "if toggled, cuda will be enabled by default",
    log_dir = "the directory where the logs will be stored",
    use_wandb = "if toggled, this experiment will be tracked with Weights and Biases",
    wandb_project_name = "the wandb's project name",
    wandb_entity = "the entity (team) of wandb's project",
    capture_video = "whether to capture videos of the agent performances (check out `videos` folder)",
    env_id = "the id of the environment",
    total_timesteps = "total timesteps of the experiments",
    learning_rate = "the learning rate of the optimizer",
    num_envs = "number of synchronized vector environments in our `envs` object",
    num_steps = "number of steps taken in the rollout phase",
    gamma = "the discount factor gamma",
    gae_lambda = "the discount factor used in our GAE estimation",
    batches_per_epoch = "how many times you loop through the data generated in rollout",
    clip_coef = "the epsilon term used in the clipped surrogate objective function",
    ent_coef = "coefficient of entropy bonus term",
    vf_coef = "cofficient of value loss function",
    max_grad_norm = "value used in gradient clipping",
    batch_size = "number of random samples we take from the rollout data",
    minibatch_size = "size of each minibatch we perform a gradient step on",
)

def arg_help(args: Optional[PPOArgs], print_df=False):
    """Prints out a nicely displayed list of arguments, their default values, and what they mean."""
    if args is None:
        args = PPOArgs()
        changed_args = []
    else:
        default_args = PPOArgs()
        changed_args = [key for key in default_args.__dict__ if getattr(default_args, key) != getattr(args, key)]
    df = pd.DataFrame([arg_help_strings]).T
    df.columns = ["description"]
    df["default value"] = [repr(getattr(args, name)) for name in df.index]
    df.index.name = "arg"
    df = df[["default value", "description"]]
    if print_df:
        df.insert(1, "changed?", ["yes" if i in changed_args else "" for i in df.index])
        with pd.option_context(
            'max_colwidth', 0, 
            'display.width', 150, 
            'display.colheader_justify', 'left'
        ):
            print(df)
    else:
        s = (
            df.style
            .set_table_styles([
                {'selector': 'td', 'props': 'text-align: left;'},
                {'selector': 'th', 'props': 'text-align: left;'}
            ])
            .apply(lambda row: ['background-color: red' if row.name in changed_args else None] + [None,] * (len(row) - 1), axis=1)
        )
        with pd.option_context("max_colwidth", 0):
            display(s)

# %%

def plot_cartpole_obs_and_dones(obs: t.Tensor, done: t.Tensor):
    """
    obs: shape (n_steps, n_envs, n_obs)
    dones: shape (n_steps, n_envs)

    Plots the observations and the dones.
    """
    obs = rearrange(obs, "step env ... -> (env step) ...").cpu().numpy()
    done = rearrange(done, "step env -> (env step)").cpu().numpy()
    done_indices = np.nonzero(done)[0]
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Cart x-position", "Pole angle"])
    fig.update_layout(template="simple_white", title="CartPole experiences (dotted lines = termination)", showlegend=False)
    d = dict(zip(['posn', 'speed', 'angle', 'angular_velocity'], obs.T))
    d["posn_min"] = np.full_like(d["posn"], -2.4)
    d["posn_max"] = np.full_like(d["posn"], +2.4)
    d["angle_min"] = np.full_like(d["posn"], -0.2095)
    d["angle_max"] = np.full_like(d["posn"], +0.2095)
    for i, (name0, color, y) in enumerate(zip(["posn", "angle"], px.colors.qualitative.D3, [2.4, 0.2095]), 1):
        for name1 in ["", "_min", "_max"]:
            fig.add_trace(go.Scatter(y=d[name0+name1], name=name0+name1, mode="lines", marker_color=color), col=1, row=i)
        for x in done_indices:
            fig.add_vline(x=x, y1=1, y0=0, line_width=2, line_color="black", line_dash="dash", col=1, row=i)
    for sign, text0 in zip([-1, 1], ["Min", "Max"]):
        for row, (y, text1) in enumerate(zip([2.4, 0.2095], ["posn", "angle"]), 1):
            fig.add_annotation(text=" ".join([text0, text1]), xref="paper", yref="paper", x=550, y=sign*y, showarrow=False, row=row, col=1)
    fig.show()

def set_global_seeds(seed):
    '''Sets random seeds in several different ways (to guarantee reproducibility)
    '''
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.backends.cudnn.deterministic = True