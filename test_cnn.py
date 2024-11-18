import ale_py
from ale_py import ALEInterface
import gymnasium as gym
import numpy as np
import torch
from Preprocess_env import AtariPreprocessing
from cnn import cnn as CNN
import matplotlib.pyplot as plt
import time


BATCH_SIZE = 32

env = gym.make('BankHeist-v4', frameskip=1)
env = AtariPreprocessing(env,
                    noop_max=30,
                    frame_skip=4,
                    screen_size=84,
                    terminal_on_life_loss=False,
                    grayscale_obs=False,
                    grayscale_newaxis=False,
                    scale_obs=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn = CNN(in_channels=3, num_actions=env.action_space.n).to(device)

# Test loop
state_batch = []
done = False
state, _ = env.reset()

while not done:
    # Collect states until we have a full batch
    while len(state_batch) < BATCH_SIZE and not done:
        state_batch.append(state)
        action = env.action_space.sample()
        state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
    
    if len(state_batch) > 0:
        # Process batch
        states_tensor = torch.FloatTensor(np.array(state_batch))
        if len(states_tensor.shape) == 4:
            states_tensor = states_tensor.permute(0, 3, 1, 2)
        states_tensor = states_tensor.to(device)
        
        with torch.no_grad():
            q_values_batch = cnn(states_tensor)
        
        # Clear the current figure
        plt.clf()
        
        # Plot the first state
        plt.subplot(1, 2, 1)
        plt.imshow(state_batch[0])
        plt.title(f'Current Frame\nChosen Action: {q_values_batch[0].argmax().item()}')
        
        # Plot Q-values
        plt.subplot(1, 2, 2)
        q_values = q_values_batch[0].cpu().numpy()
        plt.bar(range(env.action_space.n), q_values)
        plt.title('Q-values for each action')
        
        # Update the display
        plt.draw()
        plt.pause(0.1)
        
        # Clear batch
        state_batch = []

env.close()
