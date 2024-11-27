import random
import ale_py
from ale_py import ALEInterface
import gymnasium as gym
import numpy as np
import torch
from Preprocess_env import AtariPreprocessing
from cnn import cnn as CNN
import matplotlib.pyplot as plt
import time
# from replay_buffer import ReplayBuffer
import torch.nn as nn

BATCH_SIZE = 32

EPSILON_START = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
GAMMA = 0.99
TARGET_UPDATE_FREQ = 1000
MAX_EPISODES = 200 #500
LEARNING_RATE = 0.001

from collections import deque
class ReplayBuffer:
    '''
    This class stores gameplay states for training a CNN
    (Consider moving to its own file for a final implementation)
    '''
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

def train(policy_nn, target_nn, replay_buffer, optimizer, batch_size, gamma, device):
    '''
    Train a pair of CNNs using (1) game states from a replay_buffer and (2) an optimizer
    Models are trained to choose Q values that maximize gameplay reward
    Implemented using MSE Loss
    '''
    # ensure replay buffer has enough entries
    if len(replay_buffer) < batch_size:
        # can move this check to calling function instead
        return
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # convert gameplay data to tensors
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # Compute Q-values
    q_values = policy_nn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Compute target Q-values
    with torch.no_grad():
        max_next_q_values = target_nn(next_states).max(1)[0]
        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    # Compute loss
    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    '''
    Initialize a gym environment of the "Bank Heist" Atari game
    Create policy and target CNNs to learn to play the game
    Train the networks on episodes equal in number to constant MAX_EPISODES
    '''
    # Create game environment
    gym_env = gym.make('BankHeist-v4', frameskip=1)
    env = AtariPreprocessing(gym_env,
                        noop_max=30,
                        frame_skip=4,
                        screen_size=84,
                        terminal_on_life_loss=False,
                        grayscale_obs=False,
                        grayscale_newaxis=False,
                        scale_obs=True)

    # Create CNNS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_nn = CNN(in_channels=3, num_actions=env.action_space.n).to(device)
    target_nn = CNN(in_channels=3, num_actions=env.action_space.n).to(device)

    # Create optimizer and replay buffer
    optimizer = torch.optim.Adam(policy_nn.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(10000)

    epsilon = EPSILON_START

    # Train over a defined number of gameplay episodes
    for episode in range(MAX_EPISODES):
        # Print out a progress update every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode}/{MAX_EPISODES}")

        state, _ = env.reset()

        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy choice of action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    # state_tensor = torch.tensor([state], dtype=torch.float32).to(device) # causes warning
                    state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device) # faster and warning-free
                    action = policy_nn(state_tensor).argmax().item()

            # Take the chosen action in the environment; update state accordingly
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # Add the state to the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # I DON'T THINK THIS IS IN THE RIGHT PLACE RELATIVE TO
            # HOW MANY ACTIONS ARE CHOSEN
            train(policy_nn, target_nn, replay_buffer, optimizer, BATCH_SIZE, GAMMA, device)

        # Epsilon decacys over time
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        # Update the target network less frequently than the policy network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_nn.load_state_dict(policy_nn.state_dict())

    # After training episodes are complete, save the trained CNNs
    torch.save(policy_nn.state_dict(), "policy_nn.pth")
    torch.save(target_nn.state_dict(), "target_nn.pth")
    print("Model saved as policy_nn.pth and target_nn.pth")

    env.close()

if __name__ == "__main__":
    main()
