import torch
import torch.nn as nn
from cnn import CNN

class PolicyNetwork(nn.Module):
    def __init__(self, obs_space_dims: tuple, action_space_dims: int):
        super().__init__()
        
        # Shared feature extractor (CNN layers)
        # self.shared_net = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )

        # Shared feature extractor (CNN)
        self.shared_net = CNN(3, 18)
        
        # Define the size of the flattened features
        flattened_size = 64 * 7 * 7  # Change based on input dimensions
        
        # Policy head (actor)
        self.policy_logits_net = nn.Sequential(
            nn.Linear(flattened_size, action_space_dims)
        )
        
        # Value head (critic)
        self.value_net = nn.Sequential(
            nn.Linear(flattened_size, 1)
        )

    def forward(self, x: torch.Tensor, score: int):
        x = x / 255.0  # Normalize input
        x = x.unsqueeze(1)  # Add channel dimension (N, 1, H, W)
        
        features = self.shared_net(x)
        
        # Actor and Critic outputs
        action_logits = self.policy_logits_net(features)
        if score < 10:
            # Mask logits to only allow actions 2 to 5
            mask = torch.full(action_logits.shape, -1e9, dtype=action_logits.dtype, device=action_logits.device)
            mask[:, 2:6] = 0  
            action_logits = action_logits + mask
            
        state_value = self.value_net(features)
        
        return action_logits, state_value
