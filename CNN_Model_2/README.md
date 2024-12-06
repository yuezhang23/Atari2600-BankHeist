# CNN Model 2
This folder contains CNN Model 2, which is a Deep Q Learning Convolutional Neural Network (DQN CNN) trained on the Atari Bank Heist game available in the Gymnasium/ALE environment.

# Environment
BankHeist_evn.py - implements the Atari Bank Heist game using a Gymnasium/ALE environment
Preprocess_env.py - wrapper for Bank Heist environment to allow perception by/interaction with neural networks
test_cnn.py - an early demonstration of a CNN interacting with the environment - retained for reference only

# CNN Training & Running
cnn.py - implementation of the DQN CNN
train_cnn.py - training program for the CNN: produces .pth files in /nn_weights
run_model.py - runs a trained CNN in the env: loads CNN from .pth files in /nn_weights
/nn_weights - directory of .pth files produced over various training iterations
  (training hyperparameters are generally in filenames, save for earliest models)

# Data & Outputs
gifmaker.py - auxiliary script used to compile .png frames in non-Git /anim folder into animated .gif
anim_long.gif - animation of a run of the fully trained AI agent (produced previously by gifmaker.py)
run_data.txt - selected rewards/runtimes recorded from several model iterations produced by run_model.py
