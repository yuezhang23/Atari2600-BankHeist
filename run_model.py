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

    # Gameplay loop
    while not done:
        if frame_counter % frame_skip == 0:
            if moves < 100 and random_start:
                action_choices = None
                # action_choices = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

                if moves < 10:
                    action = 3
                    state, reward, done, truncated, _ = env.step(3) # right

                elif moves < 20:
                    if random.random() > 0.5:
                        action = 2
                        state, reward, done, truncated, _ = env.step(2) # up
                    else:
                        action = 5
                        state, reward, done, truncated, _= env.step(5) # down
                
                else:
                    action = random.choice([0,2,3,5])
                    state, reward, done, truncated, _ = env.step(action)
                
                moves += 1

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
            if action_choices is not None:
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

        frame_counter += 1

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
    filename = "./nn_weights/policy_nn_BS=32 ES=1.0 ED=0.997 EM=0.01 G=0.99 TUF=20 ME=1000 LR=0.0001 SEED=42 RS=True.pth"
    run_episodes(20, False, filename, random_start=False) # collect data on trials
    # run_episodes(1, True, filename, random_start=False)   # visualize actions

if __name__ == "__main__":
    main()