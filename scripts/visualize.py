import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from datasets.nuscenes_dataset import NuscenesDataset
from models.model import VectorDriveRouteModel

def get_args():
    parser = argparse.ArgumentParser(description="VectorDrive-Route Visualization Diagnostics")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="NuScenes dataset version")
    parser.add_argument("--dataroot", type=str, default="data/nuscenes", help="Path to NuScenes dataset root")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--save_path", type=str, default="checkpoints/diagnostics.png", help="Path to save visual output")
    parser.add_argument("--downscale_factor", type=int, default=4, help="Image downscaling factor")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running visualization on device: {device}")
    
    # Load dataset
    dataset = NuscenesDataset(
        version=args.version, 
        dataroot=args.dataroot, 
        split='val', 
        downscale_factor=args.downscale_factor
    )
    
    # Get a sample
    sample = dataset[0]
    
    # Load model
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        anchors = checkpoint['anchors']
        model = VectorDriveRouteModel(anchors=anchors).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: checkpoint not found. Visualizing randomly initialized model.")
        anchors = np.zeros((16, 8, 2))
        model = VectorDriveRouteModel(anchors=anchors).to(device)
        
    model.eval()
    
    # Pack sample into batch shape
    images = sample['images'].unsqueeze(0).to(device)
    intrinsics = sample['intrinsics'].unsqueeze(0).to(device)
    extrinsics = sample['extrinsics'].unsqueeze(0).to(device)
    ego_state = sample['ego_state'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(images, intrinsics, extrinsics, ego_state)
        
        # 1. Cylinder panorama
        images_cyl, _, _, _ = model.projection(images, intrinsics, extrinsics)
        panorama = images_cyl[0].permute(1, 2, 0).cpu().numpy()
        
        # 2. Depth predictions (probs -> expected depth)
        depth_probs = outputs['depth_probs'][0] # (40, H_feat, W_feat)
        depth_bins = torch.linspace(2.0, 42.0, 40, device=device).view(40, 1, 1)
        expected_depth = torch.sum(depth_probs * depth_bins, dim=0).cpu().numpy()
        
        # 3. Trajectories and logits
        logits = outputs['logits'][0] # (16,)
        trajectories = outputs['trajectories'][0].cpu().numpy() # (16, 8, 2)
        best_idx = torch.argmax(logits).item()
        
        # 4. Decoded grids
        drivable_pred = torch.sigmoid(outputs['drivable_preds'][0, 0]).cpu().numpy()
        occupancy_pred = torch.sigmoid(outputs['occupancy_preds'][0, 0]).cpu().numpy()
        
    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle("VectorDrive-Route Diagnostic Panel", fontsize=20, fontweight='bold')
    
    # Panel 1: Panorama
    axes[0, 0].imshow(panorama)
    axes[0, 0].set_title("Stitched Cylindrical Camera Panorama", fontsize=14)
    axes[0, 0].axis('off')
    
    # Panel 2: Depth Target vs Expected Depth
    gt_depth = sample['depth_target'][0].numpy()
    depth_mask = sample['depth_mask'][0].numpy()
    gt_depth_masked = np.where(depth_mask > 0.5, gt_depth, np.nan)
    
    axes[0, 1].imshow(expected_depth, cmap='inferno', vmin=2.0, vmax=42.0)
    axes[0, 1].set_title("Predicted Expected Depth Map", fontsize=14)
    # Scatter plot ground-truth points on top to show alignment
    y_indices, x_indices = np.where(depth_mask > 0.5)
    if len(x_indices) > 0:
        axes[0, 1].scatter(x_indices, y_indices, c=gt_depth[y_indices, x_indices], cmap='inferno', s=3, edgecolors='white', linewidths=0.2)
        
    # Panel 3: Ground Truth Drivable Area & Occupancy
    gt_drivable = sample['drivable_target'][0].numpy()
    gt_occupancy = sample['occupancy_target'][0].numpy()
    
    # Construct a colored map: Green for drivable, Red for occupancy
    gt_map = np.zeros((100, 100, 3))
    gt_map[gt_drivable > 0.5] = [0.0, 0.8, 0.0] # Green
    gt_map[gt_occupancy > 0.5] = [1.0, 0.0, 0.0] # Red
    
    axes[1, 0].imshow(gt_map)
    axes[1, 0].set_title("GT Map (Green=Drivable, Red=Obstacles)", fontsize=14)
    axes[1, 0].invert_yaxis()
    
    # Panel 4: Predicted Drivable Area & Occupancy
    pred_map = np.zeros((100, 100, 3))
    pred_map[drivable_pred > 0.45] = [0.0, 0.8, 0.0]
    pred_map[occupancy_pred > 0.45] = [1.0, 0.0, 0.0]
    
    axes[1, 1].imshow(pred_map)
    axes[1, 1].set_title("Predicted Map (Threshold = 0.45)", fontsize=14)
    axes[1, 1].invert_yaxis()
    
    # Panel 5: Trajectory Candidates on Occupancy Grid
    axes[2, 0].imshow(occupancy_pred, cmap='gray', origin='lower')
    # Plot candidate trajectories
    for k in range(16):
        # Convert metric coordinates (X: [-20, 20], Y: [0, 40]) to grid cells (100x100)
        grid_x = (trajectories[k, :, 0] + 20.0) / 40.0 * 100.0
        grid_y = trajectories[k, :, 1] / 40.0 * 100.0
        if k == best_idx:
            axes[2, 0].plot(grid_x, grid_y, color='cyan', linewidth=3, marker='o', label='Chosen trajectory')
        else:
            axes[2, 0].plot(grid_x, grid_y, color='blue', alpha=0.3)
            
    # Plot GT trajectory
    gt_traj = sample['future_traj'].numpy()
    gt_grid_x = (gt_traj[:, 0] + 20.0) / 40.0 * 100.0
    gt_grid_y = gt_traj[:, 1] / 40.0 * 100.0
    axes[2, 0].plot(gt_grid_x, gt_grid_y, color='gold', linewidth=3, linestyle='--', marker='x', label='GT trajectory')
    axes[2, 0].set_title("Trajectory Candidates on Occupancy", fontsize=14)
    axes[2, 0].legend()
    axes[2, 0].set_xlim(0, 100)
    axes[2, 0].set_ylim(0, 100)
    
    # Panel 6: Trajectories on Drivable Area
    axes[2, 1].imshow(drivable_pred, cmap='gray', origin='lower')
    for k in range(16):
        grid_x = (trajectories[k, :, 0] + 20.0) / 40.0 * 100.0
        grid_y = trajectories[k, :, 1] / 40.0 * 100.0
        if k == best_idx:
            axes[2, 1].plot(grid_x, grid_y, color='cyan', linewidth=3, marker='o', label='Chosen trajectory')
        else:
            axes[2, 1].plot(grid_x, grid_y, color='blue', alpha=0.3)
            
    axes[2, 1].plot(gt_grid_x, gt_grid_y, color='gold', linewidth=3, linestyle='--', marker='x', label='GT trajectory')
    axes[2, 1].set_title("Trajectory Candidates on Drivable Area", fontsize=14)
    axes[2, 1].legend()
    axes[2, 1].set_xlim(0, 100)
    axes[2, 1].set_ylim(0, 100)
    
    # Save the plot
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.save_path, bbox_inches='tight', dpi=150)
    print(f"Diagnostic panel saved to {args.save_path}")

if __name__ == "__main__":
    main()
