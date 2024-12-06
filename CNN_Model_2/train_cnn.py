import random
import ale_py
from ale_py import ALEInterface
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
import numpy as np
import torch
from Preprocess_env import AtariPreprocessing
from cnn import cnn as CNN
import matplotlib.pyplot as plt
import time
# from replay_buffer import ReplayBuffer
import torch.nn as nn
from collections import deque

BATCH_SIZE = 32 #32

EPSILON_START = 1.0     #1.0
EPSILON_DECAY = 0.9998  #0.9995
MIN_EPSILON = 0.01      #0.01
GAMMA = 0.99            #0.99
TARGET_UPDATE_FREQ = 20 #20
MAX_EPISODES = 10000     #2000
LEARNING_RATE = 0.0001  #0.0001

SEED = 42 #42

RANDOMIZE_START = True

param_str = f"BS={BATCH_SIZE} ES={EPSILON_START} ED={EPSILON_DECAY} EM={MIN_EPSILON} G={GAMMA} TUF={TARGET_UPDATE_FREQ} ME={MAX_EPISODES} LR={LEARNING_RATE} SEED={SEED} RS={RANDOMIZE_START}"

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
        print("ERROR ENCOUNTERED")
        return
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # convert gameplay data to tensors
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # print(states.shape)
    # states = torch.tensor(states, dtype=torch.float32).permute(0, 1, 4, 2, 3)  # [B, 4, 84, 84, 3] -> [B, 4, 3, 84, 84]
    # next_states = torch.tensor(next_states, dtype=torch.float32).permute(0, 1, 4, 2, 3)
    # print(states.shape)

    # merge 
    states = states.reshape(states.size(0), -1, 84, 84).to(device)  # [B, 4, 3, 84, 84] -> [B, 12, 84, 84]
    next_states = next_states.reshape(next_states.size(0), -1, 84, 84).to(device)

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

    return loss

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
    
    num_frames = 4
    env = FrameStackObservation(env, stack_size=num_frames)

    # Create CNNS
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA available! Training on GPU.", flush=True)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS available! Training on GPU.", flush=True)
    else:
        device = torch.device('cpu')
        print("CUDA NOT available... Training on CPU.", flush=True)

    policy_nn = CNN(in_channels=12, num_actions=env.action_space.n).to(device)
    target_nn = CNN(in_channels=12, num_actions=env.action_space.n).to(device)

    # Create optimizer and replay buffer
    optimizer = torch.optim.Adam(policy_nn.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(10000)

    epsilon = EPSILON_START
    max_reward = 0
    loss = 0

    # Train over a defined number of gameplay episodes
    for episode in range(MAX_EPISODES):
        # Print out a progress update every 10 episodes
        if episode % 10 == 0:
            # pass # so as to not print every episode
            print(f"Episode {episode}/{MAX_EPISODES} --- Epsilon={epsilon:.3f} --- Loss={loss} --- Max Reward: {max_reward}")

        state, _ = env.reset(seed=SEED)

        # print(state.shape)

        total_reward = 0
        done = False
        recent_moves = deque(maxlen=10)

        if RANDOMIZE_START:
            moves = 0
            choice_up_down = random.random() > 0.5

            # randomize start location
            while moves < 100:

                if moves < 10:
                    state, reward, done, truncated, _ = env.step(3) # right

                elif moves < 20:
                    if choice_up_down:
                        state, reward, done, truncated, _ = env.step(2) # up
                    else:
                        state, reward, done, truncated, _= env.step(5) # down
                
                else:
                    action = random.choice([0,2,3,5]) # allow no-op; do NOT allow movement back to left
                    state, reward, done, truncated, _ = env.step(action)
                
                moves += 1

        frame_skip = 1
        frame_counter = 0

        total_steps = 0
        while not done:
            if frame_counter % frame_skip == 0:
                if len(recent_moves) < 10 or len(set(recent_moves)) > 1:
                    # Epsilon-greedy choice of action
                    if random.random() < epsilon:
                        # action = env.action_space.sample()
                        # action = random.choice([0,2,3,4,5])
                        action = random.choice([0,2,3,4,5,6,7,8,9]) # only allow movement
                        # 'no-op' and 'fire' lead to poor training - perhaps after initial training, fire could be introduced
                    else:
                        with torch.no_grad():
                            # state_tensor = torch.tensor([state], dtype=torch.float32).to(device) # causes warning
                            state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device) # faster and warning-free
                            state_tensor = state_tensor.reshape(1, -1, 84, 84)
                            action = policy_nn(state_tensor).argmax().item()
                else:
                    # If the last 10 moves are the same, choose a different action
                    valid_actions = [0, 2, 3, 4, 5]
                    if recent_moves[-1] in valid_actions:
                        last_move = recent_moves[-1]
                        valid_actions.remove(last_move)  # Remove the last move to prevent repeating
                    action = random.choice(valid_actions)  # Pick a different valid action

                recent_moves.append(action)

            # Take the chosen action in the environment; update state accordingly
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # Add the state to the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # ADJUST BATCH SIZE AND FREQUENCY OF CALLS TO TRAIN() AS NEEDED
            if total_steps > 0 and total_steps % BATCH_SIZE == 0:
                loss = train(policy_nn, target_nn, replay_buffer, optimizer, BATCH_SIZE, GAMMA, device)
            total_steps += 1
            frame_counter += 1

        # Epsilon decacys over time
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if total_reward > max_reward:
            max_reward = total_reward

        # Update the target network less frequently than the policy network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_nn.load_state_dict(policy_nn.state_dict())

    # After training episodes are complete, save the trained CNNs
    folder = "nn_weights"

    torch.save(policy_nn.state_dict(), f"./{folder}/policy_nn_{param_str}.pth")
    torch.save(target_nn.state_dict(), f"./{folder}/target_nn_{param_str}.pth")
    print(f"Model saved as:\n"
          f"./{folder}/policy_nn_{param_str}.pth\n"
          f"./{folder}/target_nn_{param_str}.pth")

    env.close()

if __name__ == "__main__":
    main()
