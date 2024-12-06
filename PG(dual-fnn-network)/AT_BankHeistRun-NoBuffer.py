import ale_py
from ale_py import ALEInterface
import gymnasium as gym
import numpy as np
import time
import pickle
import random
import torch
from Preprocess_env import AtariPreprocessing
from torch.distributions import Categorical
from AT import ActorCritic

ale = ALEInterface()
ale.setInt('random_seed', 10)
gym.register_envs(ale_py)
env = gym.make('BankHeist-v4',frameskip=1, render_mode='human')
env = AtariPreprocessing(env,
                    noop_max=30,
                    frame_skip=5,
                    screen_size=84,
                    terminal_on_life_loss=False,
                    # set true
                    grayscale_obs=True,
                    grayscale_newaxis=False,
                    scale_obs=False)

# Define the environment and other parameters
obs, info = env.reset(seed=8)
input_dim = obs.shape
output_dim = 18
discount_factor = 0.99
learning_rate = 1e-3

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_actor_critic(env, episodes, input_dim, output_dim):
    agent = ActorCritic(input_dim, output_dim)
    # agent.net.load_state_dict(torch.load("bank_heist_model_checkpoint_2399.pth", map_location=device))

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.sample_action(state, total_reward)
            state, reward, terminated, truncated, info = env.step(action)

            agent.rewards.append(reward)
            total_reward += reward
            done = terminated or truncated

            env.render()
            done = terminated or truncated
            if done:
              break

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
        agent.update()
          
    return agent

model1 = train_actor_critic(env, episodes=1, input_dim=input_dim, output_dim=output_dim)