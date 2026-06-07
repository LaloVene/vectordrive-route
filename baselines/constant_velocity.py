import torch
import torch.nn as nn

class ConstantVelocityBaseline(nn.Module):
    """
    Constant Velocity and Yaw Rate baseline trajectory predictor.
    Propagates ego vehicle state using forward velocity vx and yaw rate omega.
    """
    def __init__(self, timestamps=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]):
        super().__init__()
        self.register_buffer("timestamps", torch.tensor(timestamps, dtype=torch.float32))

    def forward(self, vx, yaw_rate):
        """
        Args:
            vx: torch.Tensor of shape (B,) or (B, 1) representing forward velocity.
            yaw_rate: torch.Tensor of shape (B,) or (B, 1) representing yaw rate (rad/s).
        Returns:
            waypoints: torch.Tensor of shape (B, T, 2) representing (x, y) coords in ego frame.
        """
        # Ensure tensor properties and shapes (B, 1)
        if vx.ndim == 1:
            vx = vx.unsqueeze(-1)
        if yaw_rate.ndim == 1:
            yaw_rate = yaw_rate.unsqueeze(-1)

        # t shape: (1, T)
        t = self.timestamps.unsqueeze(0)

        # Handle division by zero via masking
        eps = 1e-5
        linear_mask = torch.abs(yaw_rate) < eps
        safe_yaw = torch.where(linear_mask, torch.full_like(yaw_rate, eps), yaw_rate)

        # Curvilinear trajectory
        x_curve = -(vx / safe_yaw) * (1.0 - torch.cos(safe_yaw * t))
        y_curve = (vx / safe_yaw) * torch.sin(safe_yaw * t)

        # Purely linear trajectory (when yaw_rate ~ 0)
        x_linear = torch.zeros_like(vx) * t
        y_linear = vx * t

        # Select based on mask
        x = torch.where(linear_mask, x_linear, x_curve)
        y = torch.where(linear_mask, y_linear, y_curve)

        # shape: (B, T, 2)
        return torch.stack([x, y], dim=-1)
