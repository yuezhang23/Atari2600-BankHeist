import torch
import torch.nn as nn
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, obs_space_dims: tuple, action_space_dims: int):
        super().__init__()

        input_dim = np.prod(obs_space_dims)
        # print("Input dim: ", input_dim)
        # Shared feature extractor (Fully Connected layers)
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Policy head (actor)
        self.policy_logits_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_dims)
        )

        # Value head (critic)
        self.value_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor, score: torch.Tensor):
        # Normalize input if needed (optional)
        x = x.flatten(start_dim=1)
        features = self.shared_net(x)

        # Actor and Critic outputs
        action_logits = self.policy_logits_net(features)

        # print("Score: ", score.shape)
        for i in range(score.shape[0]):  # Iterate over each example in the batch
            if score[i] < 10:
                # Apply a mask to the action logits for this specific example
                mask = torch.full(action_logits[i].shape, -1e9, dtype=action_logits.dtype, device=action_logits.device)
                mask[2:6] = 0
                action_logits[i] = action_logits[i] + mask

        state_value = self.value_net(features)
        return action_logits, state_value
