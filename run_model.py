import ale_py
from ale_py import ALEInterface
import gymnasium as gym
import numpy as np
import torch
from Preprocess_env import AtariPreprocessing
from cnn import cnn as CNN
import matplotlib.pyplot as plt

actions_dict = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE"
}

BATCH_SIZE = 32

def main():
    '''
    Initialize a gym environment of the "Bank Heist" Atari game
    Load CNN parameters from a previous training session
    Display gameplay by the CNN-powered agent
    '''
    # Create game environment
    env = gym.make('BankHeist-v4', frameskip=1)
    env = AtariPreprocessing(env,
                            noop_max=30,
                            frame_skip=4,
                            screen_size=84,
                            terminal_on_life_loss=False,
                            grayscale_obs=False,
                            grayscale_newaxis=False,
                            scale_obs=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load saved networks
    policy_nn = CNN(in_channels=3, num_actions=env.action_space.n).to(device)

    # CHOOSE policy_nn.pth OR target_nn.pth (ENSURE CORRECT CHOICE)
    policy_nn.load_state_dict(torch.load("policy_nn.pth", map_location=device))
    policy_nn.eval()  # Set to evaluation mode

    # Initialize game
    done = False
    state, _ = env.reset()

    # Gameplay loop
    while not done:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)
            action_choices = policy_nn(state_tensor)
            action = action_choices.argmax().item()
        state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        #env.render()  # consider moving everything below to a helper function

        # Clear the current figure
        plt.clf()

        # Plot game state
        plt.subplot(1, 2, 1)
        state_to_plot = np.transpose(state, (1, 2, 0))
        plt.imshow(state_to_plot)
        plt.axis("off")
        plt.title(f'Current Frame\nChosen Action: {actions_dict[action]}')

        # Plot Q-values
        plt.subplot(1, 2, 2)
        q_values = action_choices[0].cpu().numpy()
        plt.bar(range(env.action_space.n), q_values)
        plt.bar(range(len(q_values)), q_values, tick_label=list(actions_dict.values()))
        plt.xticks(rotation=90)
        plt.xlabel("Actions")
        plt.ylabel("Q-values")
        plt.title('Q-values for each action')

        # Update the display
        plt.draw()
        plt.pause(0.1)

    env.close()

if __name__ == "__main__":
    main()