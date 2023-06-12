import os
import sys
import gym
import numpy as np
import torch as t
from torch import nn
from torch.distributions.categorical import Categorical
import copy

from part2_dqn.utils import set_global_seeds
device = t.device("cuda" if t.cuda.is_available() else "cpu")
Arr = np.ndarray

from part1_intro_to_rl.utils import make_env
# import part3_ppo.solutions as solutions

def test_get_actor_and_critic(get_actor_and_critic):
    import part3_ppo.solutions as solutions
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", i, i, False, "test-run") for i in range(4)])
    actor, critic = get_actor_and_critic(envs)
    actor_soln, critic_soln = solutions.get_actor_and_critic(envs)
    assert sum(p.numel() for p in actor.parameters()) == sum(p.numel() for p in actor_soln.parameters()) # 4610
    assert sum(p.numel() for p in critic.parameters()) == sum(p.numel() for p in critic_soln.parameters()) # 4545
    for name, param in actor.named_parameters():
        if "bias" in name:
            t.testing.assert_close(param.pow(2).sum().cpu(), t.tensor(0.0))
    for name, param in critic.named_parameters():
        if "bias" in name:
            t.testing.assert_close(param.pow(2).sum().cpu(), t.tensor(0.0))
    print("All tests in `test_agent` passed!")

def test_minibatch_indexes(minibatch_indexes):
    rng = np.random.default_rng(0)
    batch_size = 16
    minibatch_size = 4
    indexes = minibatch_indexes(rng, batch_size, minibatch_size)
    assert np.array(indexes).shape == (batch_size // minibatch_size, minibatch_size)
    assert sorted(np.unique(indexes)) == list(range(batch_size))
    print("All tests in `test_minibatch_indexes` passed!")


def test_compute_advantages_single(compute_advantages, dones_false, single_env):
    import part3_ppo.solutions as solutions
    print("".join([
        "Testing with ",
        "all dones=False" if dones_false else "episode termination",
        ", ",
        "single environment" if single_env else "multiple environments",
        " ... "
    ]))
    t_ = 5
    env_ = 1 if single_env else 12
    next_value = t.randn(env_)
    next_done = t.zeros(env_) if dones_false else t.randint(0, 2, (env_,)) 
    rewards = t.randn(t_, env_)
    values = t.randn(t_, env_)
    dones = t.zeros(t_, env_) if dones_false else t.randint(0, 2, (t_, env_))
    gamma = 0.95
    gae_lambda = 0.9
    args = (next_value, next_done, rewards, values, dones, gamma, gae_lambda)
    actual = compute_advantages(*args)
    expected = solutions.compute_advantages(*args)
    # print(actual, expected)
    t.testing.assert_close(actual, expected)

def test_compute_advantages(compute_advantages):

    for dones_false in [True, False]:
        for single_env in [True, False]:
            test_compute_advantages_single(compute_advantages, dones_false, single_env)
    print("All tests in `test_compute_advantages_single` passed!")



def test_ppo_agent(my_PPOAgent):
    import part3_ppo.solutions as solutions
    
    args = solutions.PPOArgs(use_wandb=False, capture_video=False)
    set_global_seeds(args.seed)
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, i, args.capture_video, None) for i in range(4)])
    agent_solns = solutions.PPOAgent(args, envs)
    for step in range(5):
        infos_solns = agent_solns.play_step()
    
    set_global_seeds(args.seed)
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, i, args.capture_video, None) for i in range(4)])
    agent: solutions.PPOAgent = my_PPOAgent(args, envs)
    agent.critic = copy.deepcopy(agent_solns.critic)
    agent.actor = copy.deepcopy(agent_solns.actor)
    for step in range(5):
        infos = agent.play_step()
    
    assert agent.steps == 20, f"Agent did not take the expected number of steps: expected steps = n_steps*num_envs = 5*4 = 20, got {agent.steps}."

    obs, dones, actions, logprobs, values, rewards = [t.stack(arr).to(device) for arr in zip(*agent.rb.experiences)]
    expected_obs, expected_dones, expected_actions, expected_logprobs, expected_values, expected_rewards = [t.stack(arr).to(device) for arr in zip(*agent.rb.experiences)]

    assert (logprobs <= 0).all(), f"Agent's logprobs are not all negative."
    t.testing.assert_close(actions.cpu(), expected_actions.cpu(), msg="`actions` for agent and agent solns don't match. Make sure you're sampling actions from your actor network's logit distribution (while in inference mode).")
    t.testing.assert_close(values.cpu(), expected_values.cpu(), msg="`values` for agent and agent solns don't match. Make sure you're compute values in inference mode, by passing `self.next_obs` into the critic.")


    print("All tests in `test_agent` passed!")




def test_calc_clipped_surrogate_objective(calc_clipped_surrogate_objective):
    import part3_ppo.solutions as solutions

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
    import part3_ppo.solutions as solutions

    critic = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
    mb_obs = t.randn(5, 3)
    values = critic(mb_obs)
    mb_returns = t.randn(5, 4)
    vf_coef = 0.5
    with t.inference_mode():
        expected = solutions.calc_value_function_loss(values, mb_returns, vf_coef)
        actual = calc_value_function_loss(values, mb_returns, vf_coef)
    if ((actual - expected).abs() > 1e-4) and (0.5*actual - expected).abs() < 1e-4:
        raise Exception("Your result was half the expected value. Did you forget to use a factor of 1/2 in the mean squared difference?")
    t.testing.assert_close(actual, expected)
    print("All tests in `test_calc_value_function_loss` passed!")

def test_calc_entropy_bonus(calc_entropy_bonus):
    probs = Categorical(logits=t.randn((3, 4)))
    ent_coef = 0.5
    expected = ent_coef * probs.entropy().mean()
    actual = calc_entropy_bonus(probs, ent_coef)
    t.testing.assert_close(expected, actual)
    print("All tests in `test_calc_entropy_bonus` passed!")


def test_ppo_scheduler(PPOScheduler):
    import part3_ppo.solutions as solutions

    args = solutions.PPOArgs()
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", i, i, False, "test") for i in range(4)])
    agent = solutions.PPOAgent(args, envs)
    optimizer, scheduler = solutions.make_optimizer(agent, 100, 0.01, 0.5)

    scheduler.step()
    assert (scheduler.n_step_calls == 1)
    assert abs(optimizer.param_groups[0]["lr"] - 0.02)
    print("All tests in `test_ppo_scheduler` passed!")