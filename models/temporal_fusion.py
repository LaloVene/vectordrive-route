import torch
import torch.nn as nn
import kornia

def compute_warp_matrix(ego_state, dt=0.5):
    """
    Computes the 2x3 pixel-space warp matrix to realign the previous BEV grid 
    with the current ego vehicle frame.
    
    Args:
        ego_state: torch.Tensor of shape (B, 3) containing [vx, yaw_rate, acc].
        dt: float, time difference between frames (default 0.5s).
    Returns:
        M_warp: torch.Tensor of shape (B, 2, 3) representing the affine warp.
    """
    B = ego_state.shape[0]
    device = ego_state.device
    
    vx = ego_state[:, 0]
    yaw_rate = ego_state[:, 1]
    
    # Calculate angular and longitudinal change
    dtheta = yaw_rate * dt
    dy = vx * dt
    
    cos = torch.cos(dtheta)
    sin = torch.sin(dtheta)
    
    # Analytical coefficients mapping target (current t) back to source (previous t-1)
    M_warp = torch.zeros((B, 2, 3), device=device, dtype=ego_state.dtype)
    M_warp[:, 0, 0] = cos
    M_warp[:, 0, 1] = sin
    M_warp[:, 0, 2] = -50.0 * cos + 50.0
    M_warp[:, 1, 0] = -sin
    M_warp[:, 1, 1] = cos
    M_warp[:, 1, 2] = 50.0 * sin - 2.5 * dy
    
    return M_warp

class ConvGRUCell(nn.Module):
    """
    Standard ConvGRU cell for recurrent spatial-temporal updates.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2
        
        self.gates_conv = nn.Conv2d(
            input_dim + hidden_dim, 
            2 * hidden_dim, 
            kernel_size=kernel_size, 
            padding=self.padding
        )
        
        self.candidate_conv = nn.Conv2d(
            input_dim + hidden_dim, 
            hidden_dim, 
            kernel_size=kernel_size, 
            padding=self.padding
        )

    def forward(self, x, h_prev):
        """
        Args:
            x: torch.Tensor of shape (B, input_dim, H, W)
            h_prev: torch.Tensor of shape (B, hidden_dim, H, W)
        """
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates_conv(combined)
        z, r = torch.split(gates, self.hidden_dim, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        
        combined_candidate = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.candidate_conv(combined_candidate))
        
        h_next = (1.0 - z) * h_prev + z * h_tilde
        return h_next

class TemporalFusion(nn.Module):
    """
    Performs Ego-Motion alignment on the previous hidden state
    and fuses it with the current observation using a ConvGRU cell.
    """
    def __init__(self, input_dim=80, hidden_dim=80, kernel_size=3, dt=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.conv_gru = ConvGRUCell(input_dim, hidden_dim, kernel_size=kernel_size)

    def forward(self, x, h_prev, ego_state):
        """
        Args:
            x: torch.Tensor of shape (B, input_dim, H, W)
            h_prev: torch.Tensor of shape (B, hidden_dim, H, W) or None
            ego_state: torch.Tensor of shape (B, 3) [vx, yaw_rate, acc]
        Returns:
            h_next: torch.Tensor of shape (B, hidden_dim, H, W)
        """
        B, _, H, W = x.shape
        device = x.device
        
        # 1. Initialize hidden state to zero if not provided
        if h_prev is None:
            h_prev = torch.zeros((B, self.hidden_dim, H, W), device=device, dtype=x.dtype)
        else:
            # 2. Realign (warp) previous hidden state to current ego vehicle frame
            M_warp = compute_warp_matrix(ego_state, dt=self.dt)
            h_prev = kornia.geometry.transform.warp_affine(
                h_prev, M_warp, dsize=(H, W), mode='bilinear', padding_mode='zeros'
            )
            
        # 3. Recurrent fusion step
        h_next = self.conv_gru(x, h_prev)
        return h_next
