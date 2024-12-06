import ale_py
from ale_py import ALEInterface
import gymnasium as gym
import numpy as np
from Preprocess_env import AtariPreprocessing
import matplotlib.pyplot as plt
from pynput import keyboard  # keyboard listening package
import time
import torch
import torch.nn as nn
import torch.optim as optim


# to store user input action
action = 0
running = True

# # # define key press action
def on_press(key):
    global action, running
    try:
        if key.char == 'w':  # going up
            action = 2
        elif key.char == 's':  # going down
            action = 5
        elif key.char == 'a':  # going left
            action = 4
        elif key.char == 'd':  # going right
            action = 3
        elif key.char == 'k':  # upfire
            action = 10
        elif key.char == 'l':  # leftfire
            action = 12
        elif key.char == 'j':  # rightfire
            action = 11
        elif key.char == 'i':  # downfire
            action = 13
        elif key.char == 'q':  # quit
            running = False
    except AttributeError:
        pass

# create keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

ale = ALEInterface()
ale.setInt('random_seed', 0)
gym.register_envs(ale_py)
env = gym.make('BankHeist-v4',frameskip=1)
env = AtariPreprocessing(env,
                    noop_max=30,
                    frame_skip=10,
                    screen_size=84,
                    terminal_on_life_loss=False,
                    # set true
                    grayscale_obs=True,
                    grayscale_newaxis=False,
                    scale_obs=True)


pre_obs, info= env.reset(seed=5)
cnt = 0

while running:  # determine whether the game will continue
    obs, reward, done, truncated, info = env.step(action)  # use user input action
    cnt += 1
    diff = obs - pre_obs
    # changed_positions = np.argwhere(diff != 0)
    # changes = [(tuple(pos), diff[tuple(pos)]) for pos in changed_positions]

    pre_obs = obs
    env.render()  # render game image
    time.sleep(0.1)  # control game flow speed
    print(cnt)

    if done or truncated:
        obs, info = env.reset()

listener.stop()  # stop keyboard listener
print("Game Over", cnt)

env.close()