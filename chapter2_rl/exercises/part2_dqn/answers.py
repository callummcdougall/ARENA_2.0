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

Arr = np.ndarray

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(
    f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_dqn"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from part1_intro_to_rl.utils import make_env
from part1_intro_to_rl.solutions import Environment, Toy, Norvig, find_optimal_policy
import part2_dqn.utils as utils
import part2_dqn.tests as tests
from plotly_utils import line, cliffwalk_imshow, plot_cartpole_obs_and_dones

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# %%
ObsType = int
ActType = int


class DiscreteEnviroGym(gym.Env[ActType, ObsType]):
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

    def reset(self,
              seed: Optional[int] = None,
              return_info=False,
              options=None) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)
        self.pos = self.env.start
        return (self.pos, {"env": self.env}) if return_info else self.pos

    def render(self, mode="human"):
        assert mode == "human", f"Mode {mode} not supported!"


# %%
gym.envs.registration.register(
    id="NorvigGrid-v0",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=100,
    nondeterministic=True,
    kwargs={"env": Norvig(penalty=-0.04)},
)

gym.envs.registration.register(id="ToyGym-v0",
                               entry_point=DiscreteEnviroGym,
                               max_episode_steps=2,
                               nondeterministic=False,
                               kwargs={"env": Toy()})


# %%
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

    def __init__(self,
                 env: DiscreteEnviroGym,
                 config: AgentConfig = defaultConfig,
                 gamma: float = 0.99,
                 seed: int = 0):
        self.env = env
        self.config = config
        self.gamma = gamma
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        self.name = type(self).__name__
        self.reset(seed)

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

    def run_episode(self, seed: int) -> List[int]:
        '''
        Simulates one episode of interaction, agent learns as appropriate
        Inputs:
            seed: Seed for the random number generator
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

    def __repr__(self):
        return self.name + '()'


class Random(Agent):

    def get_action(self, obs: ObsType) -> ActType:
        return self.rng.integers(0, self.num_actions)


# %% Define the Cheater
class Cheater(Agent):

    def __init__(self,
                 env: DiscreteEnviroGym,
                 config: AgentConfig = defaultConfig,
                 gamma=0.99,
                 seed=0):
        super().__init__(env, config, gamma, seed)
        self.pi_opt = find_optimal_policy(env.unwrapped.env, gamma)

    def get_action(self, obs):
        return self.pi_opt[obs]


env_toy = gym.make("ToyGym-v0")
agents_toy: List[Agent] = [Cheater(env_toy), Random(env_toy)]
returns_list = []
names_list = []
for agent in agents_toy:
    returns = agent.train(n_runs=1000)
    returns_list.append(utils.cummean(returns))
    names_list.append(agent.name)

line(returns_list,
     names=names_list,
     title=f"Avg. reward on {env_toy.spec.name}")


# %%
class EpsilonGreedy(Agent):
    '''
    A class for SARSA and Q-Learning to inherit from.
    '''

    def __init__(self,
                 env: DiscreteEnviroGym,
                 config: AgentConfig = defaultConfig,
                 gamma: float = 0.99,
                 seed: int = 0):
        super().__init__(env, config, gamma, seed)

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.Q = np.zeros((self.num_states, self.num_actions)) + self.config.optimism


    def get_action(self, obs: ObsType) -> ActType:
        '''
        Selects an action using epsilon-greedy with respect to Q-value estimates
        '''
        if self.rng.random() < self.config.epsilon:
            return self.rng.integers(0, self.num_actions)
        else:
            return np.argmax(self.Q[obs])

    def __repr__(self):
        return f"{self.name}(epsilon={self.config.epsilon:.3f}, lr={self.config.lr:.3f}, optimism={self.config.optimism:.3f})"


class QLearning(EpsilonGreedy):

    def observe(self, exp: Experience) -> None:
        future_reward = self.gamma * np.max(self.Q[exp.new_obs])
        td_error = exp.reward + future_reward - self.Q[exp.obs, exp.act]
        self.Q[exp.obs, exp.act] += self.config.lr * td_error



class SARSA(EpsilonGreedy):

    def observe(self, exp: Experience):
        new_action = self.get_action(exp.new_obs)
        future_reward = self.gamma * self.Q[exp.new_obs, new_action]
        td_error = exp.reward + future_reward - self.Q[exp.obs, exp.act]
        self.Q[exp.obs, exp.act] += self.config.lr * td_error


n_runs = 500
gamma = 0.99
seed = 1

import numpy.random as rd

env_norvig = gym.make("NorvigGrid-v0")

agents_norvig: List[Agent] = [
    Cheater(env_norvig),
    Random(env_norvig),
]


def eval(eps, lr, opt):
    config = AgentConfig(epsilon=eps, lr=lr, optimism=opt)
    args_norvig = (env_norvig, config, gamma, seed)
    agent = QLearning(*args_norvig)
    returns = agent.train(n_runs)
    return np.mean(returns)


# %%
fig = go.Figure(
    layout=dict(title_text=f"Avg. reward on {env_norvig.spec.name}",
                template="simple_white",
                xaxis_range=[-30, n_runs + 30]))
for agent in agents_norvig:
    returns = agent.train(n_runs)
    fig.add_trace(go.Scatter(y=utils.cummean(returns), name=str(agent)))
fig.show()
# %%

gamma = 1
seed = 0

config_cliff = AgentConfig(epsilon=0.1, lr=0.1, optimism=0)
env = gym.make("CliffWalking-v0")
n_runs = 2500
args_cliff = (env, config_cliff, gamma, seed)

returns_list = []
name_list = []
agents: List[Union[QLearning,
                   SARSA]] = [QLearning(*args_cliff),
                              SARSA(*args_cliff)]

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
    labels={
        "x": "Episode",
        "y": "Avg. reward",
        "variable": "Agent"
    },
)


# %%
class SARSA(EpsilonGreedy):

    def observe(self, exp: Experience):
        # SOLUTION
        s, a, r_new, s_new, a_new = exp.obs, exp.act, exp.reward, exp.new_obs, exp.new_act
        self.Q[s, a] += self.config.lr * (
            r_new + self.gamma * self.Q[s_new, a_new] - self.Q[s, a])

    def run_episode(self, seed) -> List[float]:
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


# %%
class QNetwork(nn.Module):
    '''For consistency with your tests, please wrap your modules in a `nn.Sequential` called `layers`.'''
    layers: nn.Sequential

    def __init__(self,
                 dim_observation: int,
                 num_actions: int,
                 hidden_sizes: List[int] = [120, 84]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_observation, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(*hidden_sizes),
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
        self.num_environments = num_environments
        self.buffer_size = buffer_size
        self.rng = np.random.default_rng(seed)

        self.buffers = [None] * 5
        self.write_head = 0

    @property
    def observations(self) -> t.Tensor:
        return self.buffers[0]

    @property
    def actions(self) -> t.Tensor:
        return self.buffers[1]

    @property
    def rewards(self) -> t.Tensor:
        return self.buffers[2]

    @property
    def dones(self) -> t.Tensor:
        return self.buffers[3]

    @property
    def next_observations(self) -> t.Tensor:
        return self.buffers[4]

    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            dones: np.ndarray, next_obs: np.ndarray) -> None:
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
        assert obs.shape[0] == self.num_environments
        assert actions.shape == (self.num_environments, )
        assert rewards.shape == (self.num_environments, )
        assert dones.shape == (self.num_environments, )
        assert next_obs.shape[0] == self.num_environments

        # print('Adding to buffer. Write head:', self.write_head, 'Buffer size:', self.buffer_size)

        new_data = [obs, actions, rewards, dones, next_obs]
        dtypes = [t.float32, t.int64, t.float32, t.bool, t.float32]

        if self.buffers[0] is None:
            # define all buffers
            for i, (data, dtype) in enumerate(zip(new_data, dtypes)):
                self.buffers[i] = t.zeros((self.buffer_size, *data.shape[1:]),
                                          dtype=dtype)

        # add new data to buffers
        for i, data in enumerate(new_data):
            self.buffers[i][self.write_head %
                            self.buffer_size] = t.from_numpy(data)
        self.write_head += 1

    def sample(self, sample_size: int,
               device: t.device) -> ReplayBufferSamples:
        '''
        Uniformly sample sample_size entries from the buffer and convert them to PyTorch tensors on device.
        Sampling is with replacement, and sample_size may be larger than the buffer size.
        '''
        # print('Sampling from buffer. Write head:', self.write_head, 'Buffer size:', self.buffer_size,
        #       'Sample size:', sample_size,
        #       'Buffer length:', len(self))
        indices = self.rng.integers(0, len(self), sample_size)
        data = [buffer[indices].to(device) for buffer in self.buffers]
        return ReplayBufferSamples(*data)

    def __len__(self) -> int:
        return min(self.write_head, self.buffer_size)

    @property
    def full(self) -> bool:
        return len(self) == self.buffer_size


tests.test_replay_buffer_single(ReplayBuffer)
tests.test_replay_buffer_deterministic(ReplayBuffer)
tests.test_replay_buffer_wraparound(ReplayBuffer)
# %%
if False:
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
# %%


def linear_schedule(current_step: int, start_e: float, end_e: float,
                    exploration_fraction: float,
                    total_timesteps: int) -> float:
    '''Return the appropriate epsilon for the current step.

    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).

    It should stay at end_e for the rest of the episode.
    '''
    if current_step > (exploration_fraction * total_timesteps):
        return end_e
    t = current_step / (exploration_fraction * total_timesteps)
    return start_e + t * (end_e - start_e)


epsilons = [
    linear_schedule(step,
                    start_e=1.0,
                    end_e=0.05,
                    exploration_fraction=0.5,
                    total_timesteps=500) for step in range(500)
]
line(epsilons,
     labels={
         "x": "steps",
         "y": "epsilon"
     },
     title="Probability of random action")

tests.test_linear_schedule(linear_schedule)


# %%
def epsilon_greedy_policy(envs: gym.vector.SyncVectorEnv, q_network: QNetwork,
                          rng: Generator, obs: t.Tensor,
                          epsilon: float) -> np.ndarray:
    '''With probability epsilon, take a random action. Otherwise, take a greedy action according to the q_network.
    Inputs:
        envs : gym.vector.SyncVectorEnv, the family of environments to run against
        q_network : QNetwork, the network used to approximate the Q-value function
        obs : The current observation
        epsilon : exploration percentage
    Outputs:
        actions: (n_environments, ) the sampled action for each environment.
    '''

    if rng.random() < epsilon:
        actions = rng.integers(0,
                               envs.single_action_space.n,
                               size=envs.num_envs)
    else:
        actions = q_network(obs).argmax(dim=-1).cpu().numpy()
    return actions


tests.test_epsilon_greedy_policy(epsilon_greedy_policy)
# %%
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
        return np.array([0]), 1.0, True, {}

    def reset(self,
              seed: Optional[int] = None,
              return_info=False,
              options=None) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)
        if return_info:
            return (np.array([0.0]), {})
        return np.array([0.0])


gym.envs.registration.register(id="Probe1-v0", entry_point=Probe1)
env = gym.make("Probe1-v0")
assert env.observation_space.shape == (1, )
assert env.action_space.shape == ()


# %%
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


gym.envs.registration.register(id="Probe2-v0", entry_point=Probe2)
gym.envs.registration.register(id="Probe3-v0", entry_point=Probe3)
gym.envs.registration.register(id="Probe4-v0", entry_point=Probe4)
gym.envs.registration.register(id="Probe5-v0", entry_point=Probe5)

#%% DQN


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
        assert self.target_network_frequency % self.train_frequency == 0
        assert self.log_frequency % self.train_frequency == 0

        assert self.total_timesteps - self.buffer_size >= self.train_frequency
        self.total_training_steps = (self.total_timesteps -
                                     self.buffer_size) // self.train_frequency

    @property
    def device(self) -> t.device:
        return t.device("cuda" if self.cuda else "cpu")


args = DQNArgs(batch_size=256)
utils.arg_help(args)

#%%


class DQNAgent:
    '''Base Agent class handeling the interaction with the environment.'''

    def __init__(self, envs: gym.vector.SyncVectorEnv, args: DQNArgs,
                 rb: ReplayBuffer, q_network: QNetwork,
                 target_network: QNetwork, rng: np.random.Generator):
        self.envs = envs
        self.args = args
        self.steps = 0
        self.epsilon = args.start_e
        self.rng = rng

        self.rb = rb
        self.q_network = q_network
        self.target_network = target_network

        self.next_obs = self.envs.reset()  # Need a starting observation!

    def play_step(self) -> List[dict]:
        '''
        Carries out a single interaction step between the agent and the environment, and adds results to the replay buffer.

        Returns `infos` (list of dictionaries containing info we will log).
        '''
        # Get a new set of actions via the self.get_actions method (taking self.next_obs as our current observation)
        actions = self.get_actions(self.next_obs)

        # Step the environment, via self.envs.step (which returns a new set of experiences)
        # experiences = observations, rewards, dones, infos
        observations, rewards, dones, infos = self.envs.step(actions)

        # Add the new experiences to the buffer
        self.rb.add(
            self.next_obs,
            actions,
            rewards,
            dones,
            next_obs=observations,
        )

        # Set self.next_obs to the new observations (this is so the agent knows where it is for the next step)
        self.next_obs = observations

        # Increment the global step counter
        self.steps += 1

        # Return the diagnostic information from the new experiences (i.e. the infos dicts)
        return infos

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        '''
        Samples actions according to the epsilon-greedy policy using the linear schedule for epsilon.
        '''
        # Set self.epsilon according to the linear schedule, and current timestep
        self.epsilon = linear_schedule(
            current_step=self.steps,
            start_e=self.args.start_e,
            end_e=self.args.end_e,
            exploration_fraction=self.args.exploration_fraction,
            total_timesteps=self.args.total_timesteps)

        # Sample actions according to the epsilon-greedy policy
        actions = epsilon_greedy_policy(
            envs=self.envs,
            q_network=self.q_network,
            rng=self.rng,
            obs=t.as_tensor(obs, dtype=t.float32, device=self.args.device),
            epsilon=self.epsilon,
        )
        return actions


# tests.test_agent(DQNAgent)
print('ahaha')


# %%
class DQNLightning(pl.LightningModule):
    q_network: QNetwork
    target_network: QNetwork
    rb: ReplayBuffer
    agent: DQNAgent

    def __init__(self, args: DQNArgs):
        super().__init__()
        self.args = args
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv([
            make_env(args.env_id, args.seed, 0, args.capture_video,
                     self.run_name)
        ])
        self.start_time = time.time()
        self.rng = np.random.default_rng(args.seed)

        num_actions = self.envs.single_action_space.n
        num_observations = np.prod(self.envs.single_observation_space.shape)

        self.q_network = QNetwork(num_observations, num_actions).to(args.device)
        self.target_network = QNetwork(num_observations, num_actions).to(args.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.rb = ReplayBuffer(args.buffer_size, len(self.envs.envs), args.seed)
        self.agent = DQNAgent(self.envs, args, self.rb, self.q_network,
                              self.target_network, self.rng)

        for _ in trange(args.buffer_size):
            self.agent.play_step()

    def _log(self, predicted_q_vals: t.Tensor, epsilon: float,
             loss: Float[Tensor, ""], infos: List[dict]) -> None:
        log_dict = {
            "td_loss": loss,
            "q_values": predicted_q_vals.mean().item(),
            "SPS": int(self.agent.steps / (time.time() - self.start_time))
        }
        for info in infos:
            if "episode" in info.keys():
                log_dict.update({
                    "episodic_return": info["episode"]["r"],
                    "episodic_length": info["episode"]["l"],
                    "epsilon": epsilon
                })
        self.log_dict(log_dict)


    def training_step(self, batch: Any) -> Float[Tensor, ""]:
        # YOUR CODE HERE!

        for _ in range(self.args.train_frequency):
            infos = self.agent.play_step()

        minibatch: ReplayBufferSamples = self.rb.sample(self.args.batch_size, self.args.device)

        with t.inference_mode():
            future_value = self.q_network(minibatch.next_observations)
            # future_value = self.target_network(minibatch.next_observations)
            max_future_value = future_value.max(dim=1).values
            max_future_value[minibatch.dones] = 0.0
            y = minibatch.rewards + self.args.gamma * max_future_value

        action_value = self.q_network(minibatch.observations)  # (batch_size, num_actions)

        # print("---")
        # print("obs:", minibatch.observations.item())
        # print("action_value:", action_value.item())
        # print('future_value:', future_value.item())
        # print('max_future_value:', max_future_value.item())

        action_taken_value = action_value[range(self.args.batch_size), minibatch.actions]
        loss = t.mean((y - action_taken_value)**2)

        if self.agent.steps % self.args.target_network_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.agent.steps % self.args.log_frequency == 0:
            self._log(action_value, self.agent.epsilon, loss, infos)

        self.log("loss", loss)
        self.log("mean_q_values", action_value.mean())
        return loss

    def configure_optimizers(self):
        # YOUR CODE HERE!
        return t.optim.Adam(self.q_network.parameters(),
                            lr=self.args.learning_rate)

    def on_train_epoch_end(self):
        obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]],
                          [[0.0], [1.0]]]
        expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]],
                                     [[args.gamma], [1.0]], [[-1.0, 1.0]],
                                     [[1.0, -1.0], [-1.0, 1.0]]]
        tolerances = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]
        match = re.match(r"Probe(\d)-v0", args.env_id)
        if match:
            probe_idx = int(match.group(1)) - 1
            obs = t.tensor(obs_for_probes[probe_idx]).to(device)
            value = self.q_network(obs)
            print("Value: ", value)
            expected_value = t.tensor(
                expected_value_for_probes[probe_idx]).to(device)
            t.testing.assert_close(value,
                                   expected_value,
                                   atol=tolerances[probe_idx],
                                   rtol=0)
            print("Probe tests passed!")
        self.envs.close()


    def train_dataloader(self):
        '''We don't use a trainloader in the traditional sense, so we'll just have this.'''
        return range(self.args.total_training_steps)
# %%

probe_idx = 3

args = DQNArgs(
    env_id=f"Probe{probe_idx}-v0",
    exp_name=f"test-probe-{probe_idx}",
    total_timesteps=3000,
    learning_rate=0.001,
    buffer_size=500,
    capture_video=False,
    use_wandb=False,
    batch_size=1
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
px.line(metrics, y="q_values", labels={"x": "Step"}, title=f"Probe {probe_idx} (if you're seeing this, then you passed the tests!)", width=600, height=400)
# %%
# %%

wandb.finish()

args = DQNArgs()
logger = WandbLogger(save_dir=args.log_dir, project=args.wandb_project_name, name=model.run_name)
# if args.use_wandb: wandb.gym.monitor() # Makes sure we log video!
model = DQNLightning(args).to(device)

trainer = pl.Trainer(
    max_epochs=1,
    max_steps=args.total_timesteps,
    logger=logger,
    log_every_n_steps=args.log_frequency,
)
trainer.fit(model=model)
# %%