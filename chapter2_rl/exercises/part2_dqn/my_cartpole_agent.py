# %%
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
import copy

Arr = np.ndarray

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_dqn"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part1_intro_to_rl.utils import make_env
from part1_intro_to_rl.solutions import Environment, Toy, Norvig, find_optimal_policy
import part2_dqn.utils as utils
import part2_dqn.tests as tests
from plotly_utils import line, cliffwalk_imshow, plot_cartpole_obs_and_dones

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %%
class QNetwork(nn.Module):
    """For consistency with your tests, please wrap your modules in a `nn.Sequential` called `layers`."""

    layers: nn.Sequential

    def __init__(
        self,
        dim_observation: int,
        num_actions: int,
        hidden_sizes: List[int] = [120, 84],
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(dim_observation, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layers(x)


net = QNetwork(dim_observation=4, num_actions=2)
n_params = sum((p.nelement() for p in net.parameters()))
assert isinstance(getattr(net, "layers", None), nn.Sequential)
print(net)
print(f"Total number of parameters: {n_params}")
print("You should manually verify network is Linear-ReLU-Linear-ReLU-Linear")
assert n_params == 10934


# %%
@dataclass
class ReplayBufferSamples:
    """
    Samples from the replay buffer, converted to PyTorch for use in neural network training.
    """

    observations: Float[Tensor, "sampleSize *obsShape"]
    actions: Int[Tensor, "sampleSize"]
    rewards: Float[Tensor, "sampleSize"]
    dones: Bool[Tensor, "sampleSize"]
    next_observations: Float[Tensor, "sampleSize *obsShape"]


class ReplayBuffer:
    """
    Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.
    """

    rng: Generator
    observations: t.Tensor
    actions: t.Tensor
    rewards: t.Tensor
    dones: t.Tensor
    next_observations: t.Tensor

    def __init__(
        self, buffer_size: int, num_environments: int, seed: int, observation_shape=4
    ):
        assert (
            num_environments == 1
        ), "This buffer only supports SyncVectorEnv with 1 environment inside."
        self.num_environments = num_environments
        self.observations = t.empty(buffer_size, num_environments, observation_shape)
        self.actions = t.empty(buffer_size, num_environments, dtype=t.long)
        self.rewards = t.empty(buffer_size, num_environments)
        self.dones = t.empty(buffer_size, num_environments)
        self.next_observations = t.empty(
            buffer_size, num_environments, observation_shape
        )
        self.buffer_size = buffer_size
        self.n_filled = 0
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        """
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
        """
        assert obs.shape[0] == self.num_environments
        assert actions.shape == (self.num_environments,)
        assert rewards.shape == (self.num_environments,)
        assert dones.shape == (self.num_environments,)
        assert next_obs.shape[0] == self.num_environments

        self.observations[self.ptr] = t.tensor(obs)
        self.actions[self.ptr] = t.tensor(actions)
        self.rewards[self.ptr] = t.tensor(rewards)
        self.dones[self.ptr] = t.tensor(dones)
        self.next_observations[self.ptr] = t.tensor(next_obs)

        self.n_filled += 1
        self.n_filled = min(self.buffer_size, self.n_filled)

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.ptr = 0

    def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
        """
        Uniformly sample sample_size entries from the buffer and convert them to PyTorch tensors on device.
        Sampling is with replacement, and sample_size may be larger than the buffer size.
        """
        sample_indices = self.rng.integers(0, self.n_filled, size=(sample_size,))
        buffer_samples = ReplayBufferSamples(
            observations=self.observations[sample_indices, 0].to(device),
            actions=self.actions[sample_indices, 0].to(device),
            rewards=self.rewards[sample_indices, 0].to(device),
            dones=self.dones[sample_indices, 0].to(device),
            next_observations=self.next_observations[sample_indices, 0].to(device),
        )
        return buffer_samples


tests.test_replay_buffer_single(ReplayBuffer)
tests.test_replay_buffer_deterministic(ReplayBuffer)
tests.test_replay_buffer_wraparound(ReplayBuffer)
# %%
rb = ReplayBuffer(buffer_size=256, num_environments=1, seed=0)
envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 0, 0, False, "test")])
obs = envs.reset()
for i in range(256):
    actions = np.array([0])
    (next_obs, rewards, dones, infos) = envs.step(actions)
    real_next_obs = next_obs.copy()
    for i, done in enumerate(dones):
        if done:
            real_next_obs[i] = infos[i]["terminal_observation"]
    rb.add(obs, actions, rewards, dones, next_obs)
    obs = next_obs


plot_cartpole_obs_and_dones(rb.observations, rb.dones)

sample = rb.sample(256, t.device("cpu"))
plot_cartpole_obs_and_dones(sample.observations, sample.dones)


# %%
def linear_schedule(
    current_step: int,
    start_e: float,
    end_e: float,
    exploration_fraction: float,
    total_timesteps: int,
) -> float:
    """Return the appropriate epsilon for the current step.

    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).

    It should stay at end_e for the rest of the episode.
    """
    explore_timesteps = exploration_fraction * total_timesteps
    slope = (end_e - start_e) / explore_timesteps
    return max(start_e + slope * current_step, end_e)


epsilons = [
    linear_schedule(
        step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500
    )
    for step in range(500)
]
line(
    epsilons,
    labels={"x": "steps", "y": "epsilon"},
    title="Probability of random action",
)

tests.test_linear_schedule(linear_schedule)


# %%
def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv,
    q_network: QNetwork,
    rng: Generator,
    obs: t.Tensor,
    epsilon: float,
) -> np.ndarray:
    """With probability epsilon, take a random action. Otherwise, take a greedy action according to the q_network.
    Inputs:
        envs : gym.vector.SyncVectorEnv, the family of environments to run against
        q_network : QNetwork, the network used to approximate the Q-value function
        obs : The current observation
        epsilon : exploration percentage
    Outputs:
        actions: (n_environments, ) the sampled action for each environment.
    """
    if rng.uniform(0, 1, size=1).item() < epsilon:  # Explore
        num_actions = envs.single_action_space.n
        return rng.integers(low=0, high=num_actions, size=envs.num_envs)
    else:  # Exploit
        Q_values = q_network(obs)  # check one-hot encoded
        return Q_values.argmax(dim=-1).detach().cpu().numpy()


tests.test_epsilon_greedy_policy(epsilon_greedy_policy)
# %%
ObsType = np.ndarray
ActType = int

# ---START------------------ CUSTOM CARTPOLE ENV ----------------------START---
# https://github.com/openai/gym/issues/1788

import math
from gym.envs.classic_control.cartpole import CartPoleEnv
import pygame
from pygame import gfxdraw


class MyCartPoleEnv(CartPoleEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            # or theta < -self.theta_threshold_radians
            # or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 160  # TOP OF CART
        # carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

gym.envs.registration.register(id="MyCartPole", entry_point=MyCartPoleEnv)

# ----END------------------- CUSTOM CARTPOLE ENV -----------------------END----

# %%
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
    env_id: str = "MyCartPole"
    # env_id: str = "CartPole-v1"
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
        self.total_training_steps = (
            self.total_timesteps - self.buffer_size
        ) // self.train_frequency


args = DQNArgs(batch_size=256)
utils.arg_help(args)


# %%
class DQNAgent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        args: DQNArgs,
        rb: ReplayBuffer,
        q_network: QNetwork,
        target_network: QNetwork,
        rng: np.random.Generator,
    ):
        self.envs = envs
        self.args = args
        self.rb = rb
        self.next_obs = self.envs.reset()  # Need a starting observation!
        self.steps = 0
        self.epsilon = args.start_e
        self.q_network = q_network
        self.target_network = target_network
        self.rng = rng

    def play_step(self) -> List[dict]:
        """
        Carries out a single interaction step between the agent and the environment, and adds results to the replay buffer.

        Returns `infos` (list of dictionaries containing info we will log).
        """
        actions = self.get_actions(self.next_obs)
        obs, rewards, dones, infos = self.envs.step(actions)

        # Set rewards to be proportional to angular velocity
        rewards = obs[..., 3]
        # Also incentivise being in the middle of the screen
        # angular_velocity = obs[..., 3]
        # cart_pos = obs[..., 0]
        # cart_pos_reward_scale = 30 - cart_pos ** 2
        # rewards = angular_velocity * cart_pos_reward_scale

        self.rb.add(self.next_obs, actions, rewards, dones, obs)
        self.next_obs = obs
        self.steps += 1
        return infos

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        """
        Samples actions according to the epsilon-greedy policy using the linear schedule for epsilon.
        """
        self.epsilon = linear_schedule(
            self.steps,
            self.args.start_e,
            self.args.end_e,
            self.args.exploration_fraction,
            self.args.total_timesteps,
        )

        if self.rng.uniform(0, 1, size=1).item() < self.epsilon:  # Explore
            return self.rng.integers(low=0, high=self.envs.single_action_space.n, size=1)
        else:  # Exploit
            obs = t.tensor(obs, device=device)
            Q_values = self.q_network(obs)  # check one-hot encoded
            return Q_values.argmax(dim=-1).detach().cpu().numpy()


tests.test_agent(DQNAgent)


# %%
class DQNLightning(pl.LightningModule):
    q_network: QNetwork
    target_network: QNetwork
    rb: ReplayBuffer
    agent: DQNAgent

    def __init__(self, args: DQNArgs):
        super().__init__()
        self.args = args
        self.run_name = (
            f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        )
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed, 0, args.capture_video, self.run_name)]
        )
        self.start_time = time.time()
        self.rng = np.random.default_rng(args.seed)

        dim_observations = self.envs.single_observation_space.shape[0]
        num_actions = self.envs.single_action_space.n
        self.q_network = QNetwork(dim_observations, num_actions).to(device)
        self.target_network = QNetwork(dim_observations, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.rb = ReplayBuffer(
            args.buffer_size,
            self.envs.num_envs,
            args.seed,
            observation_shape=dim_observations,
        )
        self.agent = DQNAgent(
            self.envs,
            self.args,
            self.rb,
            self.q_network,
            self.target_network,
            self.rng,
        )

        # Run the agent `buffer_size` times
        for _ in range(args.buffer_size):
            self.agent.play_step()

    def _log(
        self,
        predicted_q_vals: t.Tensor,
        epsilon: float,
        loss: Float[Tensor, ""],
        infos: List[dict],
    ) -> None:
        log_dict = {
            "td_loss": loss,
            "q_values": predicted_q_vals.mean().item(),
            "SPS": int(self.agent.steps / (time.time() - self.start_time)),
        }
        for info in infos:
            if "episode" in info.keys():
                log_dict.update(
                    {
                        "episodic_return": info["episode"]["r"],
                        "episodic_length": info["episode"]["l"],
                        "epsilon": epsilon,
                    }
                )
        self.log_dict(log_dict)

    def training_step(self, batch: Any) -> Float[Tensor, ""]:
        # Do some training steps for the agent
        infos = [{}]
        for _ in range(self.args.train_frequency):
            infos = self.agent.play_step()

        # Collect a batch of samples
        buffer_samples = self.rb.sample(self.args.batch_size, device=device)
        obs = buffer_samples.observations
        acts = buffer_samples.actions
        rewards = buffer_samples.rewards
        dones = buffer_samples.dones
        next_obs = buffer_samples.next_observations

        # Calculate loss for the DQN
        gamma = self.args.gamma
        target_Q_values = self.target_network(next_obs).max(dim=-1).values
        y = rewards + (gamma * target_Q_values) * (1 - dones)
        Q_values = self.q_network(obs)
        loss = (y - Q_values[range(acts.shape[0]), acts]).pow(2).mean()

        # Update the target network
        if self.agent.steps % self.args.target_network_frequency == 0:
            for param_q, param_t in zip(self.q_network.parameters(), self.target_network.parameters()):
                param_t.data = param_q.data.clone()

        # Log stuff
        self._log(Q_values, self.agent.epsilon, loss, infos)
        return loss

    def configure_optimizers(self):
        return t.optim.Adam(self.q_network.parameters(), lr=self.args.learning_rate)

    def on_train_epoch_end(self):
        obs_for_probes = [
            [[0.0]],
            [[-1.0], [+1.0]],
            [[0.0], [1.0]],
            [[0.0]],
            [[0.0], [1.0]],
        ]
        expected_value_for_probes = [
            [[1.0]],
            [[-1.0], [+1.0]],
            [[args.gamma], [1.0]],
            [[-1.0, 1.0]],
            [[1.0, -1.0], [-1.0, 1.0]],
        ]
        tolerances = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]
        match = re.match(r"Probe(\d)-v0", args.env_id)
        if match:
            probe_idx = int(match.group(1)) - 1
            obs = t.tensor(obs_for_probes[probe_idx]).to(device)
            value = self.q_network(obs)
            print("Value: ", value)
            expected_value = t.tensor(expected_value_for_probes[probe_idx]).to(device)
            t.testing.assert_close(
                value, expected_value, atol=tolerances[probe_idx], rtol=0
            )
            print("Probe tests passed!")
        self.envs.close()

    def train_dataloader(self):
        """We don't use a trainloader in the traditional sense, so we'll just have this."""
        return range(self.args.total_training_steps)

# %%
# wandb.finish()

args = DQNArgs()
logger = WandbLogger(save_dir=args.log_dir, project=args.wandb_project_name, name="RL_model")
# if args.use_wandb: wandb.gym.monitor() # Makes sure we log video!
model = DQNLightning(args).to(device)

trainer = pl.Trainer(
    max_epochs=1,
    max_steps=args.total_timesteps,
    logger=logger,
    log_every_n_steps=args.log_frequency,
)
trainer.fit(model=model)
