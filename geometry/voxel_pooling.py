import numpy as np
import torch
import torch.nn as nn

class VoxelPooling(nn.Module):
    """
    Vectorized Voxel Pooling (Splat stage) with precomputed Bilinear Soft-Splatting.
    Projects 3D frustum features onto a top-down BEV grid using a precomputed lookup map,
    fractionally distributing each ray's features to its 4 nearest neighboring BEV cells.
    """
    def __init__(self, H_feat=32, W_feat=96, D_bins=40, 
                 H_bev=100, W_bev=100, 
                 horizontal_fov=180.0, vertical_fov=40.0,
                 x_range=(-20.0, 20.0), y_range=(0.0, 40.0)):
        super().__init__()
        self.H_feat = H_feat
        self.W_feat = W_feat
        self.D_bins = D_bins
        self.H_bev = H_bev
        self.W_bev = W_bev
        self.horizontal_fov = np.radians(horizontal_fov)
        self.vertical_fov = np.radians(vertical_fov)
        self.x_range = x_range
        self.y_range = y_range
        
        # 1. Precompute depth values
        depth_values = torch.linspace(2.0, 42.0, D_bins)
        
        # 2. Precompute feature grid pixel locations
        u_feat, v_feat = torch.meshgrid(
            torch.arange(W_feat, dtype=torch.float32),
            torch.arange(H_feat, dtype=torch.float32),
            indexing='xy'
        )
        
        # 3. Calculate yaw (theta) and pitch (phi) angles
        theta = self.horizontal_fov / 2.0 - (u_feat / W_feat) * self.horizontal_fov
        f_virtual_feat = H_feat / (2.0 * np.tan(self.vertical_fov / 2.0))
        phi = torch.atan((H_feat / 2.0 - v_feat) / f_virtual_feat)
        
        # 4. Generate 3D direction vectors in local Ego Frame
        # NuScenes Ego Frame: X: Forward, Y: Left, Z: Up
        x_dir = torch.cos(theta) * torch.cos(phi)
        y_dir = torch.sin(theta) * torch.cos(phi)
        z_dir = torch.sin(phi)
        directions = torch.stack([x_dir, y_dir, z_dir], dim=0) # (3, H_feat, W_feat)
        
        # 5. Multiply directions by depth values to form 3D Frustum Points
        # Shape: (3, D_bins, H_feat, W_feat)
        points_ego = directions.unsqueeze(1) * depth_values.view(1, D_bins, 1, 1)
        
        # Grid X (lateral) maps to -Y_ego (since +Y_ego is left, so -Y_ego is right)
        # Grid Y (longitudinal) maps to X_ego (forward)
        X = -points_ego[1]
        Y = points_ego[0]
        
        # 6. Map 3D points to float BEV grid cell coordinates
        x_cell = (X - x_range[0]) / (x_range[1] - x_range[0]) * W_bev
        y_cell = (y_range[1] - Y) / (y_range[1] - y_range[0]) * H_bev
        
        # 7. Precompute bilinear interpolation corners and weights
        # Shift coordinate center by 0.5 to align with grid cell centers
        x_floor = torch.floor(x_cell - 0.5).long()
        y_floor = torch.floor(y_cell - 0.5).long()
        
        w_x = x_cell - 0.5 - x_floor.float()
        w_y = y_cell - 0.5 - y_floor.float()
        
        corners = [
            (x_floor, y_floor, (1.0 - w_x) * (1.0 - w_y)),       # Top-Left
            (x_floor + 1, y_floor, w_x * (1.0 - w_y)),           # Top-Right
            (x_floor, y_floor + 1, (1.0 - w_x) * w_y),           # Bottom-Left
            (x_floor + 1, y_floor + 1, w_x * w_y)                # Bottom-Right
        ]
        
        flat_idx_list = []
        weight_list = []
        dummy_idx = H_bev * W_bev
        
        for ix, iy, weight in corners:
            # Check boundary validity for this corner
            valid = (ix >= 0) & (ix < W_bev) & (iy >= 0) & (iy < H_bev)
            
            # Map invalid to dummy_idx
            idx = iy * W_bev + ix
            idx = torch.where(valid, idx, torch.tensor(dummy_idx, dtype=torch.long))
            
            # Zero-out weights of invalid corners to prevent features from leaking to dummy bin
            weight = torch.where(valid, weight, torch.tensor(0.0, dtype=torch.float32))
            
            flat_idx_list.append(idx.view(-1))
            weight_list.append(weight.view(-1))
            
        # Register buffers for all 4 corners and weights
        for i in range(4):
            self.register_buffer(f"flat_idx_{i}", flat_idx_list[i])
            self.register_buffer(f"weight_{i}", weight_list[i])

    def forward(self, frustum_tensor):
        """
        Args:
            frustum_tensor: torch.Tensor of shape (B, C_context, D_bins, H_feat, W_feat)
        Returns:
            bev_features: torch.Tensor of shape (B, C_context, H_bev, W_bev)
        """
        B, C, D, H, W = frustum_tensor.shape
        device = frustum_tensor.device
        
        # Flatten the frustum tensor spatially: (B, C, D * H * W)
        frustum_flat = frustum_tensor.reshape(B, C, -1)
        
        # Prepare output buffer with extra index for the dummy bin
        out_size = self.H_bev * self.W_bev + 1
        output = torch.zeros((B, C, out_size), device=device, dtype=frustum_tensor.dtype)
        
        # Accumulate frustum features into grid using scatter_add for all 4 corners
        for i in range(4):
            idx = getattr(self, f"flat_idx_{i}").unsqueeze(0).unsqueeze(0).expand(B, C, -1)
            weight = getattr(self, f"weight_{i}").unsqueeze(0).unsqueeze(0) # (1, 1, D * H * W)
            output.scatter_add_(2, idx, frustum_flat * weight)
            
        # Discard the dummy bin and reshape back to BEV grid
        output_bev = output[:, :, :self.H_bev * self.W_bev]
        output_bev = output_bev.view(B, C, self.H_bev, self.W_bev)
        
        return output_bev
