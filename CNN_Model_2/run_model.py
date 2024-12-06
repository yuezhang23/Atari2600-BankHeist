# import Atari environment
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
import ale_py
from ale_py import ALEInterface

# import project files
from Preprocess_env import AtariPreprocessing
from cnn import cnn as CNN

# import nn dependencies
import numpy as np
import torch
import matplotlib.pyplot as plt

# general imports
import time
import random

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

def run_agent(training_file, gui=False, random_start=False):
    '''
    Initialize a gym environment of the "Bank Heist" Atari game
    Load CNN parameters from a previous training session
    Display gameplay by the CNN-powered agent (if GUI is enabled)
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

    moves=0

    frame_skip = 1
    frame_counter = 0 

    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})  # 3:1 width ratio
    fig.set_size_inches(8,4)

    # Gameplay loop
    while not done:
        if frame_counter % frame_skip == 0:
            # randomize start
            if moves < 100 and random_start:
                action_choices = None

                # first, move right, away from the start location
                if moves < 10:
                    action = 3
                    state, reward, done, truncated, _ = env.step(3) # right

                # then, move up or down, further away from the start location
                elif moves < 20:
                    if random.random() > 0.5:
                        action = 2
                        state, reward, done, truncated, _ = env.step(2) # up
                    else:
                        action = 5
                        state, reward, done, truncated, _= env.step(5) # down
                
                # move randomly (not left) into the maze
                else:
                    action = random.choice([0,2,3,5])
                    state, reward, done, truncated, _ = env.step(action)
                
                moves += 1
            # CNN chooses moves based on current state
            else:
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
            # plt.clf()
            axs[0].cla()
            axs[1].cla()

            # Plot game state
            state_to_plot = state[-1] # np.transpose(state[-1]) #, (1, 2, 0))
            axs[0].imshow(state_to_plot)
            axs[0].axis("off")
            axs[0].set_title(f'Current Frame\nChosen Action: {actions_dict[action]}')
            axs[0].text(
                0.5, -0.1,  # Coordinates relative to the subplot
                f"Reward: ${total_reward:.0f}", # Text
                fontsize=12, ha="center", va="center", # Formatting
                transform=axs[0].transAxes  # Anchor text to the axis
            )

            # Plot Q-values
            if action_choices is not None:
                q_values = action_choices[0].cpu().numpy()
                axs[1].bar(range(env.action_space.n), q_values)
                axs[1].bar(range(len(q_values)), q_values, tick_label=list(actions_dict.values()))
                axs[1].set_xticks(range(len(q_values)))
                axs[1].set_xticklabels(list(actions_dict.values()), rotation=90, fontsize=6)
                axs[1].set_xlabel("Actions")
                axs[1].set_ylabel("Q-values")
                axs[1].set_title('Q-values for each action')

            # Update the display
            plt.draw()
            plt.savefig(f"./anim/temp_frame_{frame_counter:04}.png")  # Save the current frame as a temporary image
            plt.pause(0.1)

        frame_counter += 1

        # shorten run for gif generation
        if total_reward > 100:
            pass
            # done = True  # enable to limit run length for image/gif generation

    env.close()
    return total_reward

def run_episodes(run_count, gui, filename, random_start=False):
    '''
    Run one or more episodes of a trained agent in the environment
    Load trained agent from filename
    Can run with/without GUI for display or speed, respectively
    Can run 1 or more episodes, and collect reward and runtime on each episode
    '''
    # remember episode outcomes
    results = []

    # run multiple episodes
    for i in range(run_count):
        start_time = time.perf_counter() # track episode runtime
        reward = run_agent(filename, gui=gui, random_start=random_start) # run episode

        # calculate runtime
        end_time = time.perf_counter()
        runtime = end_time - start_time

        # record results
        results.append({"reward": reward, "runtime": runtime})
        print(f"Reward: {reward} | Runtime: {runtime}")
    
    print("Finished")
    print(results)

def main():
    # filename = "./nn_weights/target_nn_BS=32 ES=1.0 ED=0.9995 EM=0.01 G=0.99 TUF=20 ME=2000 LR=0.0001 SEED=42.pth" # well-trained; exploit
    # filename = "./nn_weights/policy_nn_BS=32 ES=1.0 ED=0.9995 EM=0.01 G=0.99 TUF=20 ME=800 LR=0.0001 SEED=42 RS=True.pth" # random start try #1 (didn't work)
    # filename = "./nn_weights/target_nn_BS=32 ES=1.0 ED=0.9999 EM=0.01 G=0.99 TUF=20 ME=1000 LR=0.0001 SEED=42 RS=True.pth" # random start try #2 (disallow inf. repetition)
    # filename = "./nn_weights/policy_nn_BS=32 ES=1.0 ED=0.9995 EM=0.01 G=0.99 TUF=20 ME=200 LR=0.0001 SEED=42 RS=True.pth" # all the bells and whistles (frame skip, actually fixed repetition)
    # filename = "./nn_weights/policy_nn_BS=32 ES=1.0 ED=0.997 EM=0.01 G=0.99 TUF=20 ME=1000 LR=0.0001 SEED=42 RS=True.pth"
    filename = "./nn_weights/target_nn_BS=32 ES=1.0 ED=0.9998 EM=0.01 G=0.99 TUF=20 ME=10000 LR=0.0001 SEED=42 RS=True.pth"
    # run_episodes(20, False, filename, random_start=False) # collect data on trials
    run_episodes(1, True, filename, random_start=True)   # visualize actions

if __name__ == "__main__":
    main()