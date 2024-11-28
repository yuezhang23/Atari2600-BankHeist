import torch
import torch.nn as nn
from torch.distributions import Categorical
from AT_Policy import PolicyNetwork
import numpy as np
import torch.nn.functional as F

class ActorCritic:
    def __init__(self, obs_space_dims: tuple, action_space_dims: int):
        self.gamma = 0.95
        self.learning_rate = 1e-3

        self.net = PolicyNetwork(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
            
    def sample_action(self, state: torch.Tensor, score: torch.Tensor) -> int:
        # state = state.flatten() if len(state.shape) > 1 else state
        action_logits, state_value = self.net(state, score)
        distrib = Categorical(logits=action_logits)
        action = distrib.sample()
        return action.item()

