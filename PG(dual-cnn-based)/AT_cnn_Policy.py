import torch
import torch.nn as nn
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, obs_space_dims: tuple, action_space_dims: int):
        super().__init__()

        # Initialize CNN for feature extraction
        # print("Input dim: ", obs_space_dims[2])
        # obs_space_dims = np.transpose(obs_space_dims, (2, 0, 1))
        self.shared_net = cnn(in_channels=3, num_actions=18)

        # Policy head (actor)
        self.policy_logits_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(18, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_dims)
        )

        # Value head (critic)
        self.value_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(18, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def forward(self, x: torch.Tensor, score: torch.Tensor):

        # Feature extraction
        x = x.permute(0, 3, 1, 2)
        features = self.shared_net(x)

        # Actor and Critic outputs
        action_logits = self.policy_logits_net(features)

        # Mask action logits based on score
        for i in range(score.shape[0]):  # Iterate over each example in the batch
            if score[i] < 10:
                # Apply a mask to the action logits for this specific example
                mask = torch.full(action_logits[i].shape, -1e9, dtype=action_logits.dtype, device=action_logits.device)
                mask[2:6] = 0
                action_logits[i] = action_logits[i] + mask

        state_value = self.value_net(features)
        return action_logits, state_value



class cnn(nn.Module):
    def __init__(self, in_channels=3, num_actions=18):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        def conv2d_size_out(size, kernel_size, stride, padding):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        self.fc = nn.Linear(3136, num_actions)
        
        conv1_size = conv2d_size_out(84, 8, 4, 2)
        conv2_size = conv2d_size_out(conv1_size, 4, 2, 1)
        conv3_size = conv2d_size_out(conv2_size, 3, 1, 1)
        
        linear_input_size = conv3_size * conv3_size * 64
        self.fc = nn.Linear(linear_input_size, num_actions) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.fc(x.reshape(x.size(0), -1))
        return x