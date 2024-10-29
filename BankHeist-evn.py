import ale_py
from ale_py import ALEInterface
import gymnasium as gym
import numpy as np

ale = ALEInterface()

gym.register_envs(ale_py)
env = gym.make('BankHeist-v3', render_mode="human")
obs, info = env.reset()
for _ in range(99):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    if done or truncated:
        obs, info = env.reset()

env.close()