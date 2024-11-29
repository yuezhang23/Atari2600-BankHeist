import random
import ale_py
from ale_py import ALEInterface
import gymnasium as gym
import numpy as np
import torch
from Preprocess_env import AtariPreprocessing
import matplotlib.pyplot as plt
import time
# from replay_buffer import ReplayBuffer
from torch.distributions import Categorical
import torch.nn as nn
from AT_Policy import PolicyNetwork
from AT import ActorCritic

BATCH_SIZE = 32

EPSILON_START = 1.0
EPSILON_DECAY = 0.999 #0.995
MIN_EPSILON = 0.01
GAMMA = 0.99
TARGET_UPDATE_FREQ = 20
MAX_EPISODES = 10000
LEARNING_RATE = 0.0005


from collections import deque
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)


def train(policy_agent, target_agent, replay_buffer, batch_size, device):
    # ensure replay buffer has enough entries
    if len(replay_buffer) < batch_size:
        # can move this check to calling function instead
        print("ERROR ENCOUNTERED")
        return
    
    # sampling from replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    running_g = 0
    discounted_rewards = []  

    # Reverse the numpy array for iteration
    for r in rewards[::-1]:
        running_g = r + policy_agent.gamma * running_g
        discounted_rewards.insert(0, running_g)

    # convert gameplay data to tensors
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # Forward pass through policy and target networks
    action_logits, state_values = policy_agent.net(states, rewards)
    _, target_values = target_agent.net(next_states, rewards)
    rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

    # Compute log probabilities
    distrib = Categorical(logits=action_logits)
    log_probs = distrib.log_prob(actions)

    # Compute TD target
    td_targets = rewards.view(-1, 1) + policy_agent.gamma * target_values * (1 - dones.view(-1, 1))
    td_targets = td_targets.detach()

    # Compute advantage
    advantages = td_targets - state_values

    # Compute losses
    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = nn.functional.mse_loss(state_values, td_targets)
    entropy_bonus = distrib.entropy().mean()

    # Total loss
    total_loss = policy_loss + value_loss - 0.01 * entropy_bonus
    # print("policy_loss", policy_loss, "value_loss", value_loss)

    # Backpropagation
    policy_agent.optimizer.zero_grad()
    total_loss.backward()
    policy_agent.optimizer.step()

def main():
    # Create game environment
    gym_env = gym.make('BankHeist-v4', frameskip=1, render_mode='human')
    env = AtariPreprocessing(gym_env,
                        noop_max=30,
                        frame_skip=5,
                        screen_size=84,
                        terminal_on_life_loss=False,
                        grayscale_obs=False,  # set to True
                        grayscale_newaxis=False,
                        scale_obs=True)

    # Define the environment and other parameters
    obs, info = env.reset(seed=42)
    input_dim = obs.shape
    output_dim = 18
    # discount_factor = 0.99

    # Create Policy and Target Network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # input_dim = (84, 84)
    policy_agent = ActorCritic(input_dim, output_dim)
    target_agent = ActorCritic(input_dim, output_dim)
    policy_agent.net.load_state_dict(torch.load("policy_net_5000.pth", map_location=device))
    target_agent.net.load_state_dict(torch.load("target_net_5000.pth", map_location=device))

    # Create optimizer and replay buffer
    replay_buffer = ReplayBuffer(10000)
    # epsilon = EPSILON_START

    for episode in range(MAX_EPISODES):
        # Print out a progress update every 10 episodes
        if episode % 10 == 0:
            # pass # so as to not print every episode
            print(f"Episode {episode}/{MAX_EPISODES}")

        state, _ = env.reset()
        total_reward = 0
        done = False

        total_steps = 0
        while not done:
            # # Epsilon-greedy choice of action
            # if random.random() < epsilon:
            #     # action = env.action_space.sample()
            #     action = random.choice([0,2,3,4,5]) # only allow movement
            #     # 'no-op' and 'fire' lead to poor training - perhaps after initial training, fire could be introduced
            # else:
            #     with torch.no_grad():
            #         # state_tensor = torch.tensor([state], dtype=torch.float32).to(device) # causes warning
            #         state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device) # faster and warning-free
            #         # random sample for exploratopn
            #         action = policy_nn(state_tensor).argmax().item()
            
            with torch.no_grad():
                # random sample for exploratopn
                # state_tensor = torch.tensor([state], dtype=torch.float32).to(device) # causes warning
                state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device) 
                total_reward_tensor = torch.tensor([total_reward], dtype=torch.float32).to(device)
                action = policy_agent.exploit(state_tensor, total_reward_tensor)

            # Take the chosen action in the environment; update state accordingly
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # Add the state to the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # ADJUST BATCH SIZE AND FREQUENCY OF CALLS TO TRAIN() AS NEEDED
            if total_steps > 0 and total_steps % BATCH_SIZE == 0:
                train(policy_agent, target_agent, replay_buffer, BATCH_SIZE, device)
            total_steps += 1

            env.render()
        # Epsilon decacys over time
        # epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        print(f"Episode {episode}/{MAX_EPISODES}, Total Reward: {total_reward}")

        # Update the target network less frequently than the policy network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_agent.net.load_state_dict(policy_agent.net.state_dict())
        
        if episode % 5000 == 0:  
            torch.save(policy_agent.net.state_dict(), f"policy_net_{episode}.pth")
            torch.save(target_agent.net.state_dict(), f"target_net_{episode}.pth")
    print("Model saved as policy_net.pth and target_net.pth")
    env.close()

if __name__ == "__main__":
    main()