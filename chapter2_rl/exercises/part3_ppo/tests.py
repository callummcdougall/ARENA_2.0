import gym
import numpy as np
import torch as t
from torch import nn
from torch.distributions.categorical import Categorical
Arr = np.ndarray

import os, sys
os.chdir(r"C:\Users\calsm\Documents\AI Alignment\ARENA\arena-v1-ldn-exercises-restructured")
sys.path.append(r"C:\Users\calsm\Documents\AI Alignment\ARENA\arena-v1-ldn-exercises-restructured")

from w4d3_chapter4_ppo.utils import make_env
from w4d3_chapter4_ppo import solutions

def test_agent(Agent):
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", i, i, False, "test-run") for i in range(5)])
    agent = Agent(envs)
    assert sum(p.numel() for p in agent.critic.parameters()) == 4545
    assert sum(p.numel() for p in agent.actor.parameters()) == 4610
    for name, param in agent.named_parameters():
        if "bias" in name:
            t.testing.assert_close(param.pow(2).sum(), t.tensor(0.0))

def test_compute_advantages_single(compute_advantages, dones_false, single_env):
    print("".join([
        "Testing with ",
        "all dones=False" if dones_false else "episode termination",
        ", ",
        "single environment" if single_env else "multiple environments",
        " ... "
    ]), end="")
    t_ = 5
    env_ = 1 if single_env else 12
    next_value = t.randn(env_)
    next_done = t.zeros(env_) if dones_false else t.randint(0, 2, (env_,)) 
    rewards = t.randn(t_, env_)
    values = t.randn(t_, env_)
    dones = t.zeros(t_, env_) if dones_false else t.randint(0, 2, (t_, env_))
    device = t.device("cpu")
    gamma = 0.95
    gae_lambda = 0.9
    args = (next_value, next_done, rewards, values, dones, device, gamma, gae_lambda)
    actual = compute_advantages(*args)
    expected = solutions.compute_advantages(*args)
    # print(actual, expected)
    t.testing.assert_close(actual, expected)
    print("passed.")

def test_compute_advantages(compute_advantages):

    for dones_false in [True, False]:
        for single_env in [True, False]:
            test_compute_advantages_single(compute_advantages, dones_false, single_env)
    print("All tests in `test_compute_advantages_single` passed!")

def test_calc_clipped_surrogate_objective(calc_clipped_surrogate_objective):

    minibatch = 3
    num_actions = 4
    probs = Categorical(logits=t.randn((minibatch, num_actions)))
    mb_action = t.randint(0, num_actions, (minibatch,))
    mb_advantages = t.randn((minibatch,))
    mb_logprobs = t.randn((minibatch,))
    clip_coef = 0.01
    expected = solutions.calc_clipped_surrogate_objective(probs, mb_action, mb_advantages, mb_logprobs, clip_coef)
    actual = calc_clipped_surrogate_objective(probs, mb_action, mb_advantages, mb_logprobs, clip_coef)
    t.testing.assert_close(actual.pow(2), expected.pow(2))
    if actual * expected < 0:
        print("Warning: you have calculated the negative of the policy loss, suitable for gradient descent.")
    print("All tests in `test_calc_clipped_surrogate_objective` passed.")

def test_calc_value_function_loss(calc_value_function_loss):
    critic = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
    mb_obs = t.randn(5, 3)
    mb_returns = t.randn(5, 4)
    vf_coef = 0.5
    with t.inference_mode():
        expected = solutions.calc_value_function_loss(critic, mb_obs, mb_returns, vf_coef)
        actual = calc_value_function_loss(critic, mb_obs, mb_returns, vf_coef)
    if (actual - expected).abs() < 1e-4:
        print("All tests in `test_calc_value_function_loss` passed!")
    elif (0.5*actual - expected).abs() < 1e-4:
        raise Exception("Your result was half the expected value. Did you forget to use a factor of 1/2 in the mean squared difference?")
    t.testing.assert_close(actual, expected)

def test_calc_entropy_bonus(calc_entropy_bonus):
    probs = Categorical(logits=t.randn((3, 4)))
    ent_coef = 0.5
    expected = ent_coef * probs.entropy().mean()
    actual = calc_entropy_bonus(probs, ent_coef)
    t.testing.assert_close(expected, actual)

def test_minibatch_indexes(minibatch_indexes):
    for n in range(5):
        frac, minibatch_size = np.random.randint(1, 8, size=(2,))
        batch_size = frac * minibatch_size
        indices = minibatch_indexes(batch_size, minibatch_size)
        assert any([isinstance(indices, list), isinstance(indices, np.ndarray)])
        assert isinstance(indices[0], np.ndarray)
        assert len(indices) == frac
        np.testing.assert_equal(np.sort(np.stack(indices).flatten()), np.arange(batch_size))
    print("All tests in `test_minibatch_indexes` passed.")