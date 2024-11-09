import ale_py
from ale_py import ALEInterface
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
ale = ALEInterface()
ale.setInt('random_seed', 0)
gym.register_envs(ale_py)
env = gym.make('BankHeist-v4', render_mode="human")
obs, info= env.reset(seed=10)
print(obs)
for _ in range(10000):
    action_counter = 0
    if action_counter == 10000:
        done = True
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    if done or truncated:
        obs, info = env.reset()

env.close()