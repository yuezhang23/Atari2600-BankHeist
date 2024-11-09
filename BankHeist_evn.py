import ale_py
from ale_py import ALEInterface
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from Preprocess_env import AtariPreprocessing
import matplotlib.pyplot as plt


ale = ALEInterface()
ale.setInt('random_seed', 0)
gym.register_envs(ale_py)
env = gym.make('BankHeist-v4', frameskip=1)
env = AtariPreprocessing(env,
                    noop_max=30,
                    frame_skip=1,
                    screen_size=84,
                    terminal_on_life_loss=False,
                    grayscale_obs=False,
                    grayscale_newaxis=False,
                    scale_obs=True)

obs, info= env.reset(seed=10)
np.set_printoptions(threshold=np.inf)
print(obs)
for _ in range(1):
    action_counter = 0
    if action_counter == 10000:
        done = True
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    if done or truncated:
        obs, info = env.reset()

env.close()

obs, info = env.reset(seed=10)
plt.imshow(obs, cmap='gray')  # 'gray' colormap for grayscale images
plt.show()