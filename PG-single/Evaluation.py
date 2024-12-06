# from PN_BankHeistRun import env, model1
from PN_Reinforcement import REINFORCE
import numpy as np
from PolicyNetwork import PolicyNetwork
import torch
import numpy as np
from PolicyNetwork import *
from PN_Reinforcement import *
from Preprocess_env import AtariPreprocessing
import ale_py
from ale_py import ALEInterface
import gymnasium as gym
import numpy as np


ale = ALEInterface()
ale.setInt('random_seed', 0)
gym.register_envs(ale_py)
env = gym.make('BankHeist-v4',frameskip=1)
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


def test_policy_gradient(agent, env, episodes, checkpoint_path):
    """
    Test the trained REINFORCE agent on the given environment.

    Parameters:
    - agent: Trained REINFORCE agent.
    - env: The environment to test in.
    - episodes: Number of episodes to test the agent.
    """
    test_rewards = []  # Store total rewards for each episode

    for episode in range(episodes):
        state, info = env.reset(seed=episode + 100)  # Reset environment for a new episode
        agent.reset_hidden_state()  # Reset LSTM hidden state (if using LSTM)
        
        done = False
        total_reward = 0

        while not done:
            # Get action from the trained agent
            action = agent.sample_action(state, total_reward)
            state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            done = terminated or truncated
        
        test_rewards.append(total_reward)  # Store the total reward for this episode
        # print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    # Calculate and print average reward
    avg_reward = np.mean(test_rewards)
    print(f"\nAverage Reward over {episodes} test episodes in {checkpoint_path}: {avg_reward}")
    return test_rewards

# Test the trained agent
# agent = REINFORCE(input_dim, output_dim)
# agent.net = model1

# Create an instance of the PolicyNetwork
obs, info = env.reset(seed=10)
input_dim = np.prod(obs.shape)  
output_dim = 18  
pretrained_model = PolicyNetwork(input_dim, output_dim, use_lstm=False)

dic_path = {100:'bank_heist_model_checkpoint_100.pth', 600:'bank_heist_model_checkpoint_600.pth', 1600:'bank_heist_model_checkpoint_1600.pth',
            2600:'bank_heist_model_checkpoint_2600.pth'}

for (key, value) in dic_path.items():
    # Load the saved parameters
    pretrained_model.load_state_dict(torch.load(value, map_location=torch.device('cpu')))

    # Set the model to evaluation mode (important for inference)
    pretrained_model.eval()

    agent = REINFORCE(input_dim, output_dim)
    agent.net = pretrained_model
    test_rewards = test_policy_gradient(agent, env, 10, value)
