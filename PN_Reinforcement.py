import torch
from PolicyNetwork import *  
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical


class REINFORCE:

    def __init__(self, obs_space_dims: tuple, action_space_dims: int):

        self.learning_rate = 1e-3  
        self.gamma = 0.99  
        self.eps = 1e-6  

        self.probs = []  
        self.rewards = []  

        # Update: Initialize the PolicyNetwork with CNN input dimensions
        self.net = PolicyNetwork(obs_space_dims, action_space_dims, use_lstm=False)  
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

        # LSTM hidden state
        self.hidden_state = None
    
    def reset_hidden_state(self):
        self.hidden_state = None

    def sample_action(self, state: np.ndarray, score: int) -> int:

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)

        # Forward pass through the policy network
        action_logits, self.hidden_state = self.net(state, score, self.hidden_state)  

        # Create a categorical distribution over actions
        distrib = Categorical(logits=action_logits)
        # Sample an action from the distribution
        action = distrib.sample()  
        prob = distrib.log_prob(action)  

        self.probs.append(prob)

        return action.item()  

    def update(self):
        running_g = 0
        gs = []

        # Compute discounted returns (backwards)
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs, dtype=torch.float32)
        log_probs = torch.stack(self.probs)

        # Compute the total loss
        loss = -torch.sum(log_probs * deltas)  

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty variables
        self.probs = []
        self.rewards = []
        self.reset_hidden_state() 
