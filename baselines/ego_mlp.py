import torch
import torch.nn as nn

class EgoStateMLPBaseline(nn.Module):
    """
    Ego-State MLP baseline trajectory predictor.
    Takes [v_x, yaw_rate, acceleration] as input and regresses the 8 future waypoints.
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_waypoints=8):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2)
        )

    def forward(self, ego_state):
        """
        Args:
            ego_state: torch.Tensor of shape (B, 3) representing [v_x, yaw_rate, acceleration].
        Returns:
            waypoints: torch.Tensor of shape (B, T, 2) representing (x, y) coords in ego frame.
        """
        B = ego_state.shape[0]
        out = self.net(ego_state)
        # Reshape to (B, T, 2)
        return out.view(B, self.num_waypoints, 2)
