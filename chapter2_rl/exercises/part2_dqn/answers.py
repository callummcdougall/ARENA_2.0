#%%
import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_dqn"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from part1_intro_to_rl.utils import make_env
from part1_intro_to_rl.solutions import Environment, Toy, Norvig, find_optimal_policy
import part2_dqn.utils as utils
import part2_dqn.tests as tests
from plotly_utils import line, cliffwalk_imshow, plot_cartpole_obs_and_dones

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
ObsType = int
ActType = int

class DiscreteEnviroGym(gym.Env):
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

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
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

gym.envs.registration.register(
    id="ToyGym-v0", 
    entry_point=DiscreteEnviroGym, 
    max_episode_steps=2, 
    nondeterministic=False, 
    kwargs={"env": Toy()}
)
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

    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma: float = 0.99, seed: int = 0):
        self.env = env
        self.reset(seed)
        self.config = config
        self.gamma = gamma
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        self.name = type(self).__name__

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

    def run_episode(self, seed) -> List[int]:
        '''
        Simulates one episode of interaction, agent learns as appropriate
        Inputs:
            seed : Seed for the random number generator
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

class Random(Agent):
    def get_action(self, obs: ObsType) -> ActType:
        return self.rng.integers(0, self.num_actions)
# %%
class Cheater(Agent):
    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma=0.99, seed=0):
        super().__init__(env, config, gamma, seed)
        self.pi = find_optimal_policy(env.unwrapped.env, gamma)

    def get_action(self, obs):
        return self.pi[obs]


env_toy = gym.make("ToyGym-v0")
agents_toy: List[Agent] = [Cheater(env_toy), Random(env_toy)]
returns_list = []
names_list = []
for agent in agents_toy:
    returns = agent.train(n_runs=100)
    returns_list.append(utils.cummean(returns))
    names_list.append(agent.name)

line(returns_list, names=names_list, title=f"Avg. reward on {env_toy.spec.name}")
# %%
class EpsilonGreedy(Agent):
    '''
    A class for SARSA and Q-Learning to inherit from.
    '''
    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma: float = 0.99, seed: int = 0):
        super().__init__(env, config, gamma, seed)
        self.Q = np.full((self.num_states, self.num_actions), config.optimism, dtype=float)

    def get_action(self, obs: ObsType) -> ActType:
        '''
        Selects an action using epsilon-greedy with respect to Q-value estimates
        '''
        if np.random.random() > self.config.epsilon:
            return np.argmax(self.Q[obs])
        else:
            return self.rng.integers(0, self.num_actions)

class QLearning(EpsilonGreedy):
    def observe(self, exp: Experience) -> None:
        self.Q[exp.obs, exp.act] += self.config.lr * (
            exp.reward + self.gamma * np.max(self.Q[exp.new_obs]) - self.Q[exp.obs, exp.act]
        )


class SARSA(EpsilonGreedy):
    def observe(self, exp: Experience):
        self.Q[exp.obs, exp.act] += self.config.lr * (
            exp.reward + self.gamma * self.Q[exp.new_obs, exp.new_act] - self.Q[exp.obs, exp.act]
        )

    def run_episode(self, seed) -> List[int]:
        rewards = []
        state = self.env.reset(seed=seed)
        action = self.get_action(state)
        for _ in range(100_000):
            new_state, reward, done, info = self.env.step(action)
            rewards.append(reward)
            new_action = self.get_action(new_state)
            exp = Experience(state, action, reward, new_state, new_action)
            self.observe(exp)
            state = new_state
            action = new_action
            if done:
                break
        return rewards


n_runs = 1000
gamma = 0.99
seed = 1
env_norvig = gym.make("NorvigGrid-v0")
config_norvig = AgentConfig()
args_norvig = (env_norvig, config_norvig, gamma, seed)
agents_norvig: List[Agent] = [Cheater(*args_norvig), QLearning(*args_norvig), SARSA(*args_norvig), Random(*args_norvig)]
returns_norvig = {}
fig = go.Figure(layout=dict(
    title_text=f"Avg. reward on {env_norvig.spec.name}", 
    template="simple_white",
    xaxis_range=[-30, n_runs+30]
))
for agent in agents_norvig:
    returns = agent.train(n_runs)
    fig.add_trace(go.Scatter(y=utils.cummean(returns), name=agent.name))
fig.show()
# %%
gamma = 1
seed = 0

config_cliff = AgentConfig(epsilon=0.1, lr = 0.1, optimism=0)
env = gym.make("CliffWalking-v0")
n_runs = 2500
args_cliff = (env, config_cliff, gamma, seed)

returns_list = []
name_list = []
agents: List[Union[QLearning, SARSA]] = [QLearning(*args_cliff), SARSA(*args_cliff)]

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
    labels={"x": "Episode", "y": "Avg. reward", "variable": "Agent"},
)
# %%
class QNetwork(nn.Module):
    '''For consistency with your tests, please wrap your modules in a `nn.Sequential` called `layers`.'''
    layers: nn.Sequential

    def __init__(
        self, 
        dim_observation: int, 
        num_actions: int, 
        hidden_sizes: List[int] = [120, 84]
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim_observation, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_actions)

    def forward(self, x: t.Tensor) -> t.Tensor:
        out = nn.functional.relu(self.fc1(x))
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


net = QNetwork(dim_observation=4, num_actions=2)
n_params = sum((p.nelement() for p in net.parameters()))
# assert isinstance(getattr(net, "layers", None), nn.Sequential)
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


# class ReplayBuffer:
#     '''
#     Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.
#     '''
#     rng: Generator
#     observations: t.Tensor
#     actions: t.Tensor
#     rewards: t.Tensor
#     dones: t.Tensor
#     next_observations: t.Tensor

#     def __init__(self, buffer_size: int, num_environments: int, seed: int):
#         assert num_environments == 1, "This buffer only supports SyncVectorEnv with 1 environment inside."
#         self.num_environments = num_environments
#         self.rng = np.random.default_rng(seed)
#         self.observations = t.empty((buffer_size, 4), dtype=t.float32)
#         self.actions = t.empty((buffer_size,), dtype=t.int64)
#         self.rewards = t.empty((buffer_size,), dtype=t.float32)
#         self.dones = t.empty((buffer_size,), dtype=t.bool)
#         self.next_observations = t.empty((buffer_size, 4), dtype=t.float32)

#     def add(
#         self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, next_obs: np.ndarray
#     ) -> None:
#         '''
#         obs: shape (num_environments, *observation_shape) 
#             Observation before the action
#         actions: shape (num_environments,) 
#             Action chosen by the agent
#         rewards: shape (num_environments,) 
#             Reward after the action
#         dones: shape (num_environments,) 
#             If True, the episode ended and was reset automatically
#         next_obs: shape (num_environments, *observation_shape) 
#             Observation after the action
#             If done is True, this should be the terminal observation, NOT the first observation of the next episode.
#         '''
#         assert obs.shape[0] == self.num_environments
#         assert actions.shape == (self.num_environments,)
#         assert rewards.shape == (self.num_environments,)
#         assert dones.shape == (self.num_environments,)
#         assert next_obs.shape[0] == self.num_environments
#         self.observations = t.tensor(obs, dtype=t.float32)
#         self.actions = t.tensor(actions, dtype=t.int64)
#         self.rewards = t.tensor(rewards, dtype=t.float32)
#         self.dones = t.tensor(dones, dtype=t.bool)
#         self.next_observations = t.tensor(next_obs, dtype=t.float32)

#     def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
#         '''
#         Uniformly sample sample_size entries from the buffer and convert them to PyTorch tensors on device.
#         Sampling is with replacement, and sample_size may be larger than the buffer size.
#         '''
#         idx = self.rng.choice(len(self.observations), size=sample_size, replace=True)
#         return ReplayBufferSamples(
#             observations=self.observations[idx].to(device),
#             actions=self.actions[idx].to(device),
#             rewards=self.rewards[idx].to(device),
#             dones=self.dones[idx].to(device),
#             next_observations=self.next_observations[idx].to(device),
#         )
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
        # SOLUTION
        self.buffer_size = buffer_size
        self.rng = np.random.default_rng(seed)
        self.buffer = [None for _ in range(5)]

    def add(
        self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, next_obs: np.ndarray
    ) -> None:
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
        assert actions.shape == (self.num_environments,)
        assert rewards.shape == (self.num_environments,)
        assert dones.shape == (self.num_environments,)
        assert next_obs.shape[0] == self.num_environments

        # SOLUTION
        for i, (arr, arr_list) in enumerate(zip([obs, actions, rewards, dones, next_obs], self.buffer)):
            if arr_list is None:
                self.buffer[i] = arr
            else:
                self.buffer[i] = np.concatenate((arr, arr_list))
            if self.buffer[i].shape[0] > self.buffer_size:
                self.buffer[i] = self.buffer[i][:self.buffer_size]

        self.observations, self.actions, self.rewards, self.dones, self.next_observations = [t.as_tensor(arr) for arr in self.buffer]

    def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
        '''
        Uniformly sample sample_size entries from the buffer and convert them to PyTorch tensors on device.
        Sampling is with replacement, and sample_size may be larger than the buffer size.
        '''
        # SOLUTION
        indices = self.rng.integers(0, self.buffer[0].shape[0], sample_size)
        samples = [t.as_tensor(arr_list[indices], device=device) for arr_list in self.buffer]
        return ReplayBufferSamples(*samples)

tests.test_replay_buffer_single(ReplayBuffer)
tests.test_replay_buffer_deterministic(ReplayBuffer)
tests.test_replay_buffer_wraparound(ReplayBuffer)
# %%
def linear_schedule(
    current_step: int, start_e: float, end_e: float, exploration_fraction: float, total_timesteps: int
) -> float:
    '''Return the appropriate epsilon for the current step.

    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).

    It should stay at end_e for the rest of the episode.
    '''
    fraction = current_step / (exploration_fraction * total_timesteps)
    return fraction*end_e + (1-fraction)*start_e


epsilons = [
    linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500)
    for step in range(500)
]
line(epsilons, labels={"x": "steps", "y": "epsilon"}, title="Probability of random action")

tests.test_linear_schedule(linear_schedule)
# %%
def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv, q_network: QNetwork, rng: Generator, obs: t.Tensor, epsilon: float
) -> np.ndarray:
    '''With probability epsilon, take a random action. Otherwise, take a greedy action according to the q_network.
    Inputs:
        envs : gym.vector.SyncVectorEnv, the family of environments to run against
        q_network : QNetwork, the network used to approximate the Q-value function
        obs : The current observation
        epsilon : exploration percentage
    Outputs:
        actions: (n_environments, ) the sampled action for each environment.
    '''
    n_a = envs.single_action_space.n
    if rng.random() < epsilon:
        return rng.integers(0, n_a, size=(envs.num_envs,))
    else:
        return q_network(obs).argmax(-1).cpu().numpy()


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
        return (np.array([0]), 1.0, True, {})

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)
        if return_info:
            return (np.array([0.0]), {})
        return np.array([0.0])


gym.envs.registration.register(id="Probe1-v0", entry_point=Probe1)
env = gym.make("Probe1-v0")
assert env.observation_space.shape == (1,)
assert env.action_space.shape == ()
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
        assert self.total_timesteps - self.buffer_size >= self.train_frequency
        self.total_training_steps = (self.total_timesteps - self.buffer_size) // self.train_frequency


args = DQNArgs(batch_size=256)
utils.arg_help(args)
# %%
class DQNAgent:
    '''Base Agent class handeling the interaction with the environment.'''

    def __init__(
        self, 
        envs: gym.vector.SyncVectorEnv, 
        args: DQNArgs, 
        rb: ReplayBuffer,
        q_network: QNetwork,
        target_network: QNetwork,
        rng: np.random.Generator
    ):
        self.envs = envs
        self.args = args
        self.rb = rb
        self.next_obs = self.envs.reset() # Need a starting observation!
        self.steps = 0
        self.epsilon = args.start_e
        self.q_network = q_network
        self.target_network = target_network
        self.rng = rng

    def play_step(self) -> List[dict]:
        '''
        Carries out a single interaction step between the agent and the environment, and adds results to the replay buffer.

        Returns `infos` (list of dictionaries containing info we will log).
        '''
        actions = self.get_actions(self.next_obs)
        (new_obs, rewards, dones, infos) = self.envs.step(actions)
        self.rb.add(self.next_obs, actions, rewards, dones, new_obs)
        self.next_obs = new_obs
        self.steps += 1
        return infos


    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        '''
        Samples actions according to the epsilon-greedy policy using the linear schedule for epsilon.
        '''
        self.epsilon = linear_schedule(self.steps, self.args.start_e,
                                       self.args.end_e, self.args.exploration_fraction,
                                       self.args.total_timesteps)
        return epsilon_greedy_policy(self.envs, self.q_network, self.rng,
                                     t.as_tensor(obs, device=device), self.epsilon)


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
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, self.run_name)])
        self.start_time = time.time()
        self.rng = np.random.default_rng(args.seed)

        self.q_network = QNetwork(self.envs.single_observation_space.shape[0],
                                  self.envs.single_action_space.n).to(device)
        self.target_network = QNetwork(self.envs.single_observation_space.shape[0],
                                       self.envs.single_action_space.n).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.rb = ReplayBuffer(args.buffer_size, self.envs.num_envs, args.seed)
        self.agent = DQNAgent(self.envs, args, self.rb, self.q_network,
                              self.target_network, self.rng)
        for _ in range(args.buffer_size):
            self.agent.play_step()


    def _log(self, predicted_q_vals: t.Tensor, epsilon: float, loss: Float[Tensor, ""], infos: List[dict]) -> None:
        log_dict = {"td_loss": loss, "q_values": predicted_q_vals.mean().item(), "SPS": int(self.agent.steps / (time.time() - self.start_time))}
        for info in infos:
            if "episode" in info.keys():
                log_dict.update({"episodic_return": info["episode"]["r"], "episodic_length": info["episode"]["l"], "epsilon": epsilon})
        self.log_dict(log_dict)


    def training_step(self, batch: Any) -> Float[Tensor, ""]:
        for _ in range(self.args.train_frequency):
            self.agent.play_step()
        samples = self.rb.sample(self.args.batch_size, device)
        preds = self.q_network(samples.observations).gather(-1, samples.actions.unsqueeze(-1)).squeeze(-1)

        with t.no_grad():
            next_q_vals = self.target_network(samples.next_observations).max(-1).values
        not_dones = 1 - samples.dones.int()
        targets = samples.rewards + self.args.gamma * not_dones * next_q_vals

        loss = nn.functional.mse_loss(preds, targets)
        self._log(preds, self.agent.epsilon, loss, self.agent.play_step())
        if batch % self.args.target_network_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return loss

    def configure_optimizers(self):
        return t.optim.Adam(self.q_network.parameters(), lr=self.args.learning_rate)


    def on_train_epoch_end(self):
        obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
        expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]], [[args.gamma], [1.0]], [[-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]
        tolerances = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]
        match = re.match(r"Probe(\d)-v0", args.env_id)
        if match:
            probe_idx = int(match.group(1)) - 1
            obs = t.tensor(obs_for_probes[probe_idx]).to(device)
            value = self.q_network(obs)
            print("Value: ", value)
            expected_value = t.tensor(expected_value_for_probes[probe_idx]).to(device)
            t.testing.assert_close(value, expected_value, atol=tolerances[probe_idx], rtol=0)
            print("Probe tests passed!")
        self.envs.close()


    def train_dataloader(self):
        '''We don't use a trainloader in the traditional sense, so we'll just have this.'''
        return range(self.args.total_training_steps)
# %%
probe_idx = 1

args = DQNArgs(
    env_id=f"Probe{probe_idx}-v0",
    exp_name=f"test-probe-{probe_idx}", 
    total_timesteps=3000,
    learning_rate=0.001,
    buffer_size=500,
    capture_video=False,
    use_wandb=False
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
px.line(metrics, y="q_values", labels={"x": "Step"}, title="Probe 1 (if you're seeing this, then you passed the tests!)", width=600, height=400)
# %%
