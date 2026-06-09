import torch
import torch.nn as nn
import torch.nn.functional as F

def binary_focal_loss_with_logits(inputs, targets, alpha=0.25, gamma=2.0, weight=None):
    """
    Binary Focal Loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    probs = torch.sigmoid(inputs)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal_weight = (1.0 - p_t) ** gamma
    loss = focal_weight * bce_loss
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if weight is not None:
        loss = loss * weight
    return loss.mean()

class MultiTaskLoss(nn.Module):
    """
    Joint Multi-Task Loss Engine.
    Combines depth estimation, path classification, waypoint regression,
    safety collision penalty, and auxiliary grid segmentation.
    """
    def __init__(self, w_depth=4.0, w_cls=0.1, w_reg=2.0, w_safety=1.0, 
                 w_drivable=2.0, w_occupancy=2.0, sigma=3.5):
        super().__init__()
        self.w_depth = w_depth
        self.w_cls = w_cls
        self.w_reg = w_reg
        self.w_safety = w_safety
        self.w_drivable = w_drivable
        self.w_occupancy = w_occupancy
        self.sigma = sigma

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict containing:
                depth_probs: (B, D_bins, H_feat, W_feat)
                logits: (B, K_anchors)
                trajectories: (B, K_anchors, 8, 2)
                drivable_preds: (B, 1, 100, 100)
                occupancy_preds: (B, 1, 100, 100)
            targets: dict containing:
                depth_target: (B, 1, H_feat, W_feat)
                depth_mask: (B, 1, H_feat, W_feat)
                future_traj: (B, 8, 2)
                drivable_target: (B, 1, 100, 100)
                occupancy_target: (B, 1, 100, 100)
        Returns:
            loss_dict: dict of individual loss components and total loss.
        """
        B = targets['future_traj'].shape[0]
        device = targets['future_traj'].device
        
        # 1. Auxiliary Depth Loss (Focal Depth Loss)
        depth_probs = predictions['depth_probs']
        depth_target = targets['depth_target']
        depth_mask = targets['depth_mask']
        
        # Map target depth to discrete bin indices [0, 39]
        bin_indices = torch.clamp(
            torch.round((depth_target - 2.0) / (40.0 / 39.0)).long(),
            min=0, max=39
        )
        # Extract predicted probability at target bin index
        pred_prob = torch.gather(depth_probs, dim=1, index=bin_indices)
        
        # Compute focal depth loss
        loss_depth_grid = -0.25 * ((1.0 - pred_prob) ** 2) * torch.log(torch.clamp(pred_prob, min=1e-6))
        loss_depth = torch.sum(loss_depth_grid * depth_mask) / torch.clamp(depth_mask.sum(), min=1.0)
        
        # 2. Anchor Classification KL Loss
        logits = predictions['logits']
        trajectories = predictions['trajectories']
        future_traj = targets['future_traj']
        
        # Extract static anchors from trajectories (before offsets) or from registered buffer
        # Let's compute weights based on the distance between candidate refined trajectories and GT expert trajectory
        # trajectories shape: (B, K, 8, 2), future_traj: (B, 8, 2)
        dist_sq = torch.sum((trajectories - future_traj.unsqueeze(1)) ** 2, dim=(-2, -1)) # (B, K)
        
        # Compute target distribution P_star using softmax-like exponentiation
        W = torch.exp(-dist_sq / (2.0 * (self.sigma ** 2)))
        P_star = W / torch.clamp(torch.sum(W, dim=-1, keepdim=True), min=1e-6)
        P_star = P_star.detach()
        
        # KL Divergence classification loss
        log_preds = F.log_softmax(logits, dim=-1)
        loss_cls = F.kl_div(log_preds, P_star, reduction='batchmean')
        
        # 3. Waypoint Regression Loss
        # Regress offsets only for the best-matching anchor
        k_star = torch.argmax(P_star, dim=-1) # (B,)
        batch_idx = torch.arange(B, device=device)
        refined_traj_best = trajectories[batch_idx, k_star] # (B, 8, 2)
        
        loss_reg = F.smooth_l1_loss(refined_traj_best, future_traj, reduction='mean')
        
        # 4. Safety Regularization Loss
        occupancy_logits = predictions['occupancy_preds']
        occupancy_probs = torch.sigmoid(occupancy_logits) # Convert logits to occupancy probabilities
        
        # Extract waypoint coordinates for all 16 trajectories
        X = trajectories[..., 0] # (B, K, 8)
        Y = trajectories[..., 1] # (B, K, 8)
        
        # Normalize to [-1, 1] for grid_sample, clamped to [-0.99, 0.99] to prevent boundary gradient NaNs
        u_norm = torch.clamp(X / 20.0, min=-0.99, max=0.99)
        v_norm = torch.clamp(1.0 - Y / 20.0, min=-0.99, max=0.99)
        
        # Grid points for bilinear sampling: shape (B, K * 8, 1, 2)
        grid_points = torch.stack([u_norm, v_norm], dim=-1).view(B, -1, 1, 2)
        
        # Sample occupancy values differentiably
        sampled_occupancy = F.grid_sample(
            occupancy_probs, grid_points, mode='bilinear', padding_mode='zeros', align_corners=False
        ).view(B, trajectories.shape[1], 8) # (B, K, 8)
        
        # Average occupancy along trajectory waypoints
        collision_prob = torch.mean(sampled_occupancy, dim=-1) # (B, K)
        
        # Penalty is weighted by prediction probabilities of each trajectory anchor
        pred_probs = torch.softmax(logits, dim=-1) # (B, K)
        loss_safety = torch.sum(pred_probs * collision_prob) / B
        
        # 5. Auxiliary Grid Decoders Losses
        drivable_preds = predictions['drivable_preds']
        drivable_target = targets['drivable_target']
        loss_drivable = binary_focal_loss_with_logits(drivable_preds, drivable_target, alpha=0.5, gamma=2.0)
        
        # Create a distance-weighted map matching the 100x100 BEV grid size.
        # NuScenes longitudinal range is 0.0 to 40.0m. Row 0 is 40.0m, row 99 is 0.0m.
        # Therefore, y_coords must be 40.0 at index 0 and 0.0 at index 99.
        y_coords = torch.linspace(40.0, 0.0, 100, device=device).view(100, 1).repeat(1, 100)
        distance_weights = 1.0 / (y_coords + 1.0)
        distance_weights = distance_weights / distance_weights.mean() # Normalize weights to preserve overall loss scale
        
        # Broadcast weight from (100, 100) to (1, 1, 100, 100)
        distance_weights = distance_weights.unsqueeze(0).unsqueeze(0)
        
        loss_occupancy = binary_focal_loss_with_logits(
            occupancy_logits, targets['occupancy_target'], alpha=0.5, gamma=2.0, weight=distance_weights
        )
        
        # Joint total loss
        total_loss = (self.w_depth * loss_depth + 
                      self.w_cls * loss_cls + 
                      self.w_reg * loss_reg + 
                      self.w_safety * loss_safety + 
                      self.w_drivable * loss_drivable + 
                      self.w_occupancy * loss_occupancy)
        
        return {
            'loss': total_loss,
            'loss_depth': loss_depth,
            'loss_cls': loss_cls,
            'loss_reg': loss_reg,
            'loss_safety': loss_safety,
            'loss_drivable': loss_drivable,
            'loss_occupancy': loss_occupancy
        }
