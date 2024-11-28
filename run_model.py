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

def run_agent(training_file, gui=False, ):
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
    
    num_frames = 4
    env = FrameStackObservation(env, stack_size=num_frames)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load saved networks
    policy_nn = CNN(in_channels=12, num_actions=env.action_space.n).to(device)

    # CHOOSE policy_nn.pth OR target_nn.pth (ENSURE CORRECT CHOICE)
    policy_nn.load_state_dict(torch.load(training_file, map_location=device))
    # policy_nn.load_state_dict(torch.load("target_nn.pth", map_location=device))
    policy_nn.eval()  # Set to evaluation mode

    # Initialize game
    done = False
    state, _ = env.reset()
    total_reward = 0

    # Gameplay loop
    while not done:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)
            state_tensor = state_tensor.reshape(state_tensor.size(0), -1, 84, 84).to(device)

            action_choices = policy_nn(state_tensor)
            action = action_choices.argmax().item()
        state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        #env.render()  # consider moving everything below to a helper function

        total_reward += reward

        if gui:
            # Clear the current figure
            plt.clf()

            # Plot game state
            plt.subplot(1, 2, 1)
            # print(state[-1].shape)
            state_to_plot = state[-1] # np.transpose(state[-1]) #, (1, 2, 0))
            plt.imshow(state_to_plot)
            plt.axis("off")
            plt.title(f'Current Frame\nChosen Action: {actions_dict[action]}')
            plt.text(
                0.5, -0.1,  # Coordinates relative to the subplot
                f"Reward: ${total_reward:.0f}", # Text
                fontsize=12, ha="center", va="center", # Formatting
                transform=plt.gca().transAxes  # Anchor text to the axis
            )

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
    return total_reward

def main():
    results = []
    for i in range(20):
        start_time = time.perf_counter()
        reward = run_agent("target_nn_BS=32 ES=1.0 ED=0.9995 EM=0.01 G=0.99 TUF=20 ME=2000 LR=0.0001 SEED=42.pth", gui=False)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        results.append({"reward": reward, "runtime": runtime})
        print(f"Reward: {reward} | Runtime: {runtime}")
    
    print("Finished")
    print(results)

if __name__ == "__main__":
    main()