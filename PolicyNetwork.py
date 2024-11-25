import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, obs_space_dims: tuple, action_space_dims: int, use_lstm: bool = False):
        super().__init__()
        
        # Define CNN layers for spatial feature extraction
        self.shared_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),  # (84, 84) -> (20, 20)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  # (20, 20) -> (9, 9)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # (9, 9) -> (7, 7)
            nn.ReLU(),
            nn.Flatten(),  # Flatten spatial dimensions into a single vector
        )
        
        # Define the size of the flattened features
        flattened_size = 64 * 7 * 7  # From the CNN layer output
        
        # LSTM for temporal dependencies (optional)
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(flattened_size, 256, batch_first=True)
            feature_size = 256  # LSTM output size
        else:
            feature_size = flattened_size

        # Policy logits specific Linear Layer
        self.policy_logits_net = nn.Sequential(
            nn.Linear(feature_size, action_space_dims)  # Map features to action logits
        )

    def forward(self, x: torch.Tensor, score: int, hidden_state=None):
        x = x / 255.0  # Normalize pixel values (assuming input is in [0, 255])
        x = x.unsqueeze(1)  # Add channel dimension for grayscale images: (N, 1, 84, 84)
        
        # Pass through CNN layers
        shared_features = self.shared_net(x)
        
        # if self.use_lstm:
        #     # LSTM expects input of shape (batch, seq_len, feature_size), adding seq_len=1
        #     shared_features, hidden_state = self.lstm(shared_features.unsqueeze(1), hidden_state)
        #     shared_features = shared_features.squeeze(1)  # Remove seq_len dimension
        
        # Output logits for each action
        action_logits = self.policy_logits_net(shared_features)
        
        if score < 10:
            # Mask logits to only allow actions 2 to 5
            mask = torch.full(action_logits.shape, -1e9, dtype=action_logits.dtype, device=action_logits.device)
            mask[:, 2:6] = 0  
            action_logits = action_logits + mask
        
        return action_logits, hidden_state if self.use_lstm else None
