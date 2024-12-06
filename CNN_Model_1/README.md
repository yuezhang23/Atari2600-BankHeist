# CNN Model 1
This folder contains CNN Model 1, which is a Deep Q Learning Convolutional Neural Network (DQN CNN) trained on the Atari Bank Heist game available in the Gymnasium/ALE environment. This was an attempt to recreate a model from the paper "Human level control through deep reinforcement learning" by Mnih et al. This particular model was trained on up to 2 million frames of play in the game environment, incorporating 1-channel grayscale input, 4-frame stacking, SmoothL1Loss function, and full action space.  The latest trained version of the model achieves an average reward of ~$40-60 using a partially successful exploration method which attempted to fully explore the environment. This was useful to understand how the agent was training to optimize in the second model.

## Environment
* BankHeist_evn.py - implements the Atari Bank Heist game using a Gymnasium/ALE environment
* Preprocess_env.py - wrapper for Bank Heist environment to allow perception by/interaction with neural networks
* test_cnn.py - an early demonstration of a CNN interacting with the environment - retained for reference only

## CNN Training & Running
* cnn.py - implementation of the DQN CNN
* train_cnn.py - training program for the CNN: produces .pth files in "/nn_weights"
* run_model.py - runs a trained CNN in the env: loads CNN from .pth files in "/nn_weights" returns a gif of the run
* /nn_weights - directory of .pth files produced over various training iterations
  (training hyperparameters are generally in filenames, save for earliest models)

## Data & Outputs
* gifmaker.py - auxiliary script used to compile .png frames in non-Git "/anim" folder into animated .gif

