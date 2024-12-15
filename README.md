# Atari Bank Heist AI Models
This folder contains three AI models, detailed below, that were created and trained to play the Atari "Bank Heist" game in the Gymnasium/ALE environment.

## CNN Model 1
* This folder contains CNN Model 1, which is a Deep Q Learning Convolutional Neural Network (DQN CNN) trained on the Atari Bank Heist game available in the Gymnasium/ALE environment. This was an attempt to recreate a model from the paper "Human level control through deep reinforcement learning" by Mnih et al. This particular model was trained on up to 2 million frames of play in the game environment, incorporating 1-channel grayscale input, 4-frame stacking, SmoothL1Loss function, and full action space.  The latest trained version of the model achieves an average reward of ~$40-60 using a partially successful exploration method which attempted to fully explore the environment. This was useful to understand how the agent was training to optimize in the second model.

## CNN Model 2
* This folder contains CNN Model 2, which is a Deep Q Learning Convolutional Neural Network (DQN CNN). This particular model was trained on up to 10,000 episodes of play in the game environment, incorporating 3-channel RGB input, 4-frame stacking, forced initial exploration of the environment, and infinite-impasse avoidance. The latest trained version of the model achieves an average reward of ~$1650 using a semi-exploitative but highly effective/optimized strategy of "warping" through the tunnel linking the left and right sides of the game, acquiring rewards from two bank locations that appear regularly near the tunnel on the right side of the screen.

## Gradient-Based Model
* This folder implements gradient-based methods to train RL agents, with a focus on Proximal Policy Optimization (PPO) as the most effective approach. While simpler methods like the REINFORCE algorithm (PG-single) and a dual-network actor-critic model (PG-dual-fnn-a2c) show limited success, PPO achieves meaningful results through techniques like clipping and entropy regularization. Extensive hyperparameter tuning was conducted based on PPO.
# BankHeist
