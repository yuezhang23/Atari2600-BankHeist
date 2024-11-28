import numpy as np
from PolicyNetwork import *
from PN_Reinforcement import *
from Preprocess_env import AtariPreprocessing
import ale_py
from ale_py import ALEInterface
import gymnasium as gym
import numpy as np
import time


ale = ALEInterface()
ale.setInt('random_seed', 0)
gym.register_envs(ale_py)
env = gym.make('BankHeist-v4',frameskip=1, render_mode='human')
env = AtariPreprocessing(env,
                    noop_max=30,
                    frame_skip=4,
                    screen_size=84,
                    terminal_on_life_loss=False,
                    # set true
                    grayscale_obs=True, 
                    grayscale_newaxis=False,
                    scale_obs=True)

# wrapped env for recording episode statistics
env = gym.wrappers.RecordEpisodeStatistics(env, 50)  
pre_obs, info= env.reset(seed=5)
reward_over_episodes = []


def train_policy_gradient(env, episodes, input_dim, output_dim):   
    agent = REINFORCE(input_dim, output_dim)

    for episode in range(episodes):
        state, reward = env.reset(seed=10)
      
        done = False
        total_reward = 0
        # pre_state = state
        # pre_diff = 0
        start = time.time() 
        while not done:
            action = agent.sample_action(state, total_reward)
            state, reward, terminated, truncated, info = env.step(action)
            num_of_lives = info["lives"]
            
            # 0.1 as a hyperparameter to penalize losing a life
            # print("time diff", time.time() - start)
            # agent.rewards.append(reward)
            agent.rewards.append(reward - 0.1 * (4 - num_of_lives))
            # print("rewards~~~~~~~~~", agent.rewards)
            total_reward += reward
            
            env.render()
            done = terminated or truncated
            if done:
                break

            if episode % 100 == 0:
                torch.save(agent.net.state_dict(), f'bank_heist_model_checkpoint_{episode}.pth')

        reward_over_episodes.append(env.return_queue[-1])
        time_spent = time.time() - start
        print("time spent", time_spent, "number 0f states", len(agent.rewards))
        agent.update()
        print(f"Episode {episode + 1}/{episodes} completed.")
    return agent
   
obs, info = env.reset(seed=10)
input_dim = np.prod(obs.shape)  
output_dim = 18  
model1 = train_policy_gradient(env, episodes=10000, input_dim=input_dim, output_dim=output_dim)