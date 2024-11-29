import torch
from PolicyNetwork import *  
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical


class REINFORCE:

    def __init__(self, obs_space_dims: tuple, action_space_dims: int):

        self.learning_rate = 1e-3  
        self.gamma = 0.95 
        self.eps = 1e-6  

        self.probs = []  
        self.rewards = []  

        # Update: Initialize the PolicyNetwork with CNN input dimensions
        self.net = PolicyNetwork(obs_space_dims, action_space_dims)  
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray, score: int) -> int:

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  

        action_logits = self.net(state, score)  


        distrib = Categorical(logits=action_logits)
        action = distrib.sample()  
        prob = distrib.log_prob(action)  

        self.probs.append(prob)

        return action.item()  

    def update(self):
        running_g = 0
        gs = []

        # discount rewards
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs, dtype=torch.float32)
        log_probs = torch.stack(self.probs)

        loss = -torch.sum(log_probs * deltas)  

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty variables
        self.probs = []
        self.rewards = []
