import numpy as np
import torch as t
import random
from typing import Optional, Union, List
import gym
import gym.spaces
import gym.envs.registration
from gym.utils import seeding
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Tuple
from dataclasses import asdict
from gettext import find
from tqdm import tqdm
from PIL import Image, ImageDraw
import einops
import os

MAIN = __name__ == "__main__"
Arr = np.ndarray

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)


def moving_avg(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_rewards(
    all_rewards: List[np.ndarray], 
    names: List[str],
    moving_avg_window: Optional[int] = 15,
):
    fig = go.Figure(layout=dict(template="simple_white", title_text="Mean reward over all runs"))
    for rewards, name in zip(all_rewards, names):
        rewards_avg = rewards.mean(axis=0)
        if moving_avg_window is not None:
            rewards_avg = moving_avg(rewards_avg, moving_avg_window)
        fig.add_trace(go.Scatter(y=rewards_avg, mode="lines", name=name))
    fig.show()


def linear_schedule(current_step: int, start_e: float, end_e: float, exploration_fraction: float, total_timesteps: int) -> float:
    """Return the appropriate epsilon for the current step.
    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).
    """
    "SOLUTION"
    duration = exploration_fraction * total_timesteps
    slope = (end_e - start_e) / duration
    return max(slope * current_step + start_e, end_e)



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

# %%
