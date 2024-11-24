import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int, use_lstm: bool = False):
        super().__init__()

        hidden_space1 = 512 
        hidden_space2 = 256   

        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
        )


        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(hidden_space2, hidden_space2, batch_first=True)

        # Policy logits specific Linear Layer
        self.policy_logits_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)  
        )


    def forward(self, x: torch.Tensor, score: int, hidden_state=None) -> torch.Tensor:
        x = x / 255.0

        x = x.view(x.size(0), -1)  # Flatten the input
        shared_features = self.shared_net(x.float())

        if self.use_lstm:
            shared_features, hidden_state = self.lstm(shared_features.unsqueeze(1), hidden_state)
            shared_features = shared_features.squeeze(1) 

        # Output logits for each action
        action_logits = self.policy_logits_net(shared_features)

        if score < 10:
            # Mask logits to only allow actions 2 to 5
            mask = torch.full(action_logits.shape, -1e9, dtype=action_logits.dtype, device=action_logits.device)
            mask[:, 2:6] = 0  # Keep logits for actions 2 to 5 unmodified
            action_logits = action_logits + mask

        return action_logits, hidden_state if self.use_lstm else None
