import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.projection import CylindricalProjection
from models.backbone import ImageBackbone, LSSLiftStage
from geometry.voxel_pooling import VoxelPooling
from models.temporal_fusion import TemporalFusion
from models.planning_head import PlanningHead

class BEVResBlock(nn.Module):
    """
    Standard ResNet basic block applied on the 2D BEV grid to smooth out frustum ray streaks.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class VectorDriveRouteModel(nn.Module):
    """
    Complete end-to-end VectorDrive-Route network.
    Stitches camera feeds into a cylindrical canvas, extracts BEV spatial-temporal features,
    and runs a cross-attention motion planner.
    """
    def __init__(self, anchors, C_feat=256, C_context=80, D_bins=40, H_bev=100, W_bev=100, K_anchors=16):
        super().__init__()
        self.projection = CylindricalProjection()
        self.backbone = ImageBackbone(C_feat=C_feat)
        self.lift = LSSLiftStage(C_feat=C_feat, C_context=C_context, D_bins=D_bins)
        self.pool = VoxelPooling(D_bins=D_bins)
        self.temporal_fusion = TemporalFusion(input_dim=C_context, hidden_dim=C_context)
        self.bev_smooth = nn.Sequential(
            BEVResBlock(C_context),
            BEVResBlock(C_context)
        )
        self.planning_head = PlanningHead(anchors=anchors, C_bev=C_context, C_trans=256, K_anchors=K_anchors)

        # Precompute BEV grid ground-plane points for validity mask projection
        x_range = self.pool.x_range
        y_range = self.pool.y_range
        dx = (x_range[1] - x_range[0]) / W_bev
        dy = (y_range[1] - y_range[0]) / H_bev
        
        xs = torch.linspace(x_range[0] + dx/2, x_range[1] - dx/2, W_bev)
        ys = torch.linspace(y_range[1] - dy/2, y_range[0] + dy/2, H_bev)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')
        
        X_ego = y_grid
        Y_ego = -x_grid
        Z_ego = torch.full_like(x_grid, -1.5) # Ground plane relative to camera height (1.5m)
        
        # Shape: (4, H_bev * W_bev)
        points_ego = torch.stack([X_ego, Y_ego, Z_ego], dim=0).view(3, -1)
        ones = torch.ones((1, points_ego.shape[1]))
        points_ego_hom = torch.cat([points_ego, ones], dim=0)
        self.register_buffer("bev_points_ego_hom", points_ego_hom)

    def forward(self, images, intrinsics, extrinsics, ego_state, h_prev=None):
        """
        Args:
            images: torch.Tensor (B, 3, 3, H_in, W_in)
            intrinsics: torch.Tensor (B, 3, 3, 3)
            extrinsics: torch.Tensor (B, 3, 3, 4)
            ego_state: torch.Tensor (B, 3) [vx, yaw_rate, acc]
            h_prev: torch.Tensor (B, C_context, H_bev, W_bev) or None
        Returns:
            dict containing logits, trajectories, drivable_preds, occupancy_preds, depth_probs, and h_next.
        """
        # 1. Stitch camera views into cylindrical canvas panorama
        images_cyl, _, _, validity_mask = self.projection(images, intrinsics, extrinsics)
        
        # 2. Extract perspective features using ResNet-18 backbone
        feat = self.backbone(images_cyl)
        
        # Downsample the Stage 1 Validity Mask to match your Stride-8 feature map (32x96)
        mask_downscale = F.interpolate(validity_mask, size=(32, 96), mode='nearest')
        
        # Element-wise multiply to kill any features bleeding into the vehicle's blind spots
        feat = feat * mask_downscale
        
        # 3. Lift perspective features to 3D frustum using predicted categorical depth probabilities
        frustum, depth_probs = self.lift(feat)
        
        # 4. Splat frustum features to 2D BEV grid space using vectorized voxel pooling
        x_bev = self.pool(frustum)
        
        # Apply BEV lateral smoothing convolutions to erase frustum streaks
        x_bev = self.bev_smooth(x_bev)
        
        # 5. Temporal recurrent alignment and fusion of BEV features
        h_next = self.temporal_fusion(x_bev, h_prev, ego_state)
        
        # 6. Query scene features with intent anchors via cross-attention and decode outputs
        logits, trajectories, drivable_preds, occupancy_preds = self.planning_head(h_next)
        
        # 7. Apply ground-plane BEV validity mask to erase camera blind spot predictions
        B = images.shape[0]
        device = images.device
        H_in, W_in = images.shape[3], images.shape[4]
        
        points_ego_hom = self.bev_points_ego_hom.unsqueeze(0).expand(B, -1, -1)
        any_valid = torch.zeros((B, points_ego_hom.shape[2]), dtype=torch.bool, device=device)
        
        for c in range(3):
            ext_c = extrinsics[:, c]
            int_c = intrinsics[:, c]
            
            P_cam = torch.bmm(ext_c, points_ego_hom)
            P_pix = torch.bmm(int_c, P_cam)
            
            u_pix = P_pix[:, 0] / torch.clamp(P_pix[:, 2], min=1e-5)
            v_pix = P_pix[:, 1] / torch.clamp(P_pix[:, 2], min=1e-5)
            
            z_valid = P_cam[:, 2] > 0.1
            u_valid = (u_pix >= 0) & (u_pix <= W_in - 1)
            v_valid = (v_pix >= 0) & (v_pix <= H_in - 1)
            
            valid_c = z_valid & u_valid & v_valid
            any_valid = any_valid | valid_c
            
        bev_validity_mask = any_valid.view(B, 1, self.pool.H_bev, self.pool.W_bev).float()
        
        drivable_preds = drivable_preds * bev_validity_mask
        occupancy_preds = occupancy_preds * bev_validity_mask
        
        return {
            'logits': logits,
            'trajectories': trajectories,
            'drivable_preds': drivable_preds,
            'occupancy_preds': occupancy_preds,
            'depth_probs': depth_probs,
            'h_next': h_next
        }

    def load_state_dict(self, state_dict, strict=True):
        # We pop precomputed buffers from the state_dict to prevent older checkpoints 
        # from overwriting the updated horizontal FOV and voxel pooling geometry setups.
        for key in list(state_dict.keys()):
            if 'projection.rays_ego' in key or 'pool.flat_idx' in key or 'pool.weight' in key or 'bev_points_ego_hom' in key:
                state_dict.pop(key)
        return super().load_state_dict(state_dict, strict=False)
