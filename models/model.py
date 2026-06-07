import torch
import torch.nn as nn

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
        images_cyl, _, _, _ = self.projection(images, intrinsics, extrinsics)
        
        # 2. Extract perspective features using ResNet-18 backbone
        feat = self.backbone(images_cyl)
        
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
        
        return {
            'logits': logits,
            'trajectories': trajectories,
            'drivable_preds': drivable_preds,
            'occupancy_preds': occupancy_preds,
            'depth_probs': depth_probs,
            'h_next': h_next
        }
