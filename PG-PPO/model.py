import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
     # Build a feedforward neural network.
    # print("sizes", sizes)    
    if isinstance(sizes[0], Box):
        obs_dim = np.prod(sizes[0].shape)
        sizes[0] = obs_dim
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        if len(obs.shape) > 2:
            obs = torch.flatten(obs)
        # print("obs", obs.shape)
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        # print("act", act)
        if len(act.shape) > 1:
            return pi.log_prob(act[:, 0])
        else:
            return pi.log_prob(act)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        if len(obs.shape) > 2:
            obs = torch.flatten(obs)
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim =84*84*3
        self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, score):
        with torch.no_grad():
            # Get the logits for the policy distribution
            logits = self.pi._distribution(obs).logits  

            # Apply masking based on the score condition
            if score < 10:
                mask = torch.ones_like(logits) * -float('inf')  # Initialize a mask with very low values
                allowed_actions = torch.tensor([2, 3, 4, 5])   # Define allowed actions
                mask[allowed_actions] = 0                      # Set logits for allowed actions to 0 (no penalty)
                logits += mask                                 # Adjust logits with the mask

            # Create a new distribution with the masked logits
            pi = Categorical(logits=logits)


            # pi = self.pi._distribution(obs)
            # action with pure random policy
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]