# Atari Bank Heist AI Models
This folder contains three AI models, detailed below, that were created and trained to play the Atari "Bank Heist" game in the Gymnasium/ALE environment.

## CNN Model 1
* 

## CNN Model 2
* This folder contains CNN Model 2, which is a Deep Q Learning Convolutional Neural Network (DQN CNN). This particular model was trained on up to 10,000 episodes of play in the game environment, incorporating 3-channel RGB input, 4-frame stacking, forced initial exploration of the environment, and infinite-impasse avoidance. The latest trained version of the model achieves an average reward of ~$1650 using a semi-exploitative but highly effective/optimized strategy of "warping" through the tunnel linking the left and right sides of the game, acquiring rewards from two bank locations that appear regularly near the tunnel on the right side of the screen.

## Gradient-Based Model
* This folder implements policy gradient methods to train reinforcement learning agents in the BankHeist environment, with a focus on Proximal Policy Optimization (PPO) as the most effective approach. While simpler methods like the REINFORCE algorithm (PG-single) and a dual-network actor-critic model (PG-dual-fnn-a2c) show limited success, PPO achieves meaningful results through techniques like clipping and entropy regularization. Extensive hyperparameter tuning was conducted based on PPO to compare performance.
