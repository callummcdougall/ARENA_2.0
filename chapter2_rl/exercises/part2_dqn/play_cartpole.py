import gym
import pygame 
from pygame.locals import *
import time

pygame.init()

env = gym.make("CartPole-v0")
env.reset()
action = 0
running = True
print("CartPole-v0")
print("Left/Right to control Cart")
print("Q to quit, R to reset")
env.metadata['render_fps'] = 30
while running:
    keys = pygame.key.get_pressed()
    if keys[K_LEFT]:
        action = 0
    elif keys[K_RIGHT]:
        action = 1
    elif keys[K_r]:
        env.reset()
    elif keys[K_q]:
        running = False
    else:
        action = 1-action
    _, _, done, _ = env.step(action)
    #if done:
    #    env.reset()
    env.render() 