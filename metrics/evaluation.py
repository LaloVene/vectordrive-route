import torch
import numpy as np

def compute_trajectory_jerk(traj, dt=0.5):
    """
    Computes the maximum jerk along the trajectory.
    Args:
        traj: torch.Tensor or np.ndarray of shape (..., 8, 2)
    Returns:
        max_jerk: tensor/array representing the maximum jerk along each trajectory.
    """
    is_numpy = isinstance(traj, np.ndarray)
    if is_numpy:
        traj = torch.from_numpy(traj)
        
    # Velocity: shape (..., 7, 2)
    vel = (traj[..., 1:, :] - traj[..., :-1, :]) / dt
    
    # Acceleration: shape (..., 6, 2)
    acc = (vel[..., 1:, :] - vel[..., :-1, :]) / dt
    
    # Jerk: shape (..., 5, 2)
    jerk = (acc[..., 1:, :] - acc[..., :-1, :]) / dt
    
    # Magnitudes
    jerk_mag = torch.sqrt(torch.sum(jerk**2, dim=-1)) # shape (..., 5)
    max_jerk = torch.max(jerk_mag, dim=-1)[0] # shape (...)
    
    if is_numpy:
        return max_jerk.numpy()
    return max_jerk

def evaluate_metrics(pred_traj, gt_traj, occupancy_grid, x_range=(-20.0, 20.0), y_range=(0.0, 40.0), dt=0.5):
    """
    Evaluates imitation, safety, and comfort metrics for a batch.
    Args:
        pred_traj: torch.Tensor of shape (B, 8, 2) - final chosen trajectories.
        gt_traj: torch.Tensor of shape (B, 8, 2) - expert trajectories.
        occupancy_grid: torch.Tensor of shape (B, 1, 100, 100) - ground-truth occupancy grid.
    Returns:
        metrics: dict containing l2_error, collision_rate, and max_jerk.
    """
    B = pred_traj.shape[0]
    device = pred_traj.device
    
    # 1. Imitation L2 Error
    dist = torch.sqrt(torch.sum((pred_traj - gt_traj) ** 2, dim=-1)) # (B, 8)
    l2_error = torch.mean(dist).item()
    
    # 2. Collision Rate
    # Convert metric coordinates to grid cell indices
    W_bev, H_bev = occupancy_grid.shape[-1], occupancy_grid.shape[-2]
    
    X = pred_traj[..., 0] # (B, 8)
    Y = pred_traj[..., 1] # (B, 8)
    
    dx = (x_range[1] - x_range[0]) / W_bev
    dy = (y_range[1] - y_range[0]) / H_bev
    
    idx_x = torch.floor((X - x_range[0]) / dx).long()
    idx_y = torch.floor((Y - y_range[0]) / dy).long()
    
    # Boundary check
    in_bounds = (idx_x >= 0) & (idx_x < W_bev) & (idx_y >= 0) & (idx_y < H_bev)
    
    # Check if any waypoint in the trajectory collides
    collisions = torch.zeros(B, dtype=torch.bool, device=device)
    for b in range(B):
        for t in range(8):
            if in_bounds[b, t]:
                ix = idx_x[b, t]
                iy = idx_y[b, t]
                if occupancy_grid[b, 0, iy, ix] >= 0.5:
                    collisions[b] = True
                    break
                
    collision_rate = torch.sum(collisions).float().item() / B
    
    # 3. Jerk Rate (Comfort)
    jerk = compute_trajectory_jerk(pred_traj, dt=dt)
    mean_max_jerk = torch.mean(jerk).item()
    
    return {
        'l2_error': l2_error,
        'collision_rate': collision_rate,
        'max_jerk': mean_max_jerk
    }
