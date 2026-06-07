import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.nuscenes_dataset import NuscenesDataset
from models.model import VectorDriveRouteModel
from baselines.constant_velocity import ConstantVelocityBaseline
from baselines.ego_mlp import EgoStateMLPBaseline
from metrics.evaluation import evaluate_metrics

def generate_diagnostics_self_contained(model, sample, save_path, device):
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
    
    axes[0, 1].imshow(expected_depth, cmap='inferno', vmin=2.0, vmax=42.0)
    axes[0, 1].set_title("Predicted Expected Depth Map", fontsize=14)
    y_indices, x_indices = np.where(depth_mask > 0.5)
    if len(x_indices) > 0:
        axes[0, 1].scatter(x_indices, y_indices, c=gt_depth[y_indices, x_indices], cmap='inferno', s=3, edgecolors='white', linewidths=0.2)
        
    # Panel 3: Ground Truth Drivable Area & Occupancy
    gt_drivable = sample['drivable_target'][0].numpy()
    gt_occupancy = sample['occupancy_target'][0].numpy()
    
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
    for k in range(16):
        grid_x = (trajectories[k, :, 0] + 20.0) / 40.0 * 100.0
        grid_y = trajectories[k, :, 1] / 40.0 * 100.0
        if k == best_idx:
            axes[2, 0].plot(grid_x, grid_y, color='cyan', linewidth=3, marker='o', label='Chosen trajectory')
        else:
            axes[2, 0].plot(grid_x, grid_y, color='blue', alpha=0.3)
            
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
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved diagnostic panel to {save_path}")

def get_args():
    parser = argparse.ArgumentParser(description="VectorDrive-Route Model Evaluation")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="NuScenes dataset version")
    parser.add_argument("--dataroot", type=str, default="data/nuscenes", help="Path to NuScenes dataset root")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--downscale_factor", type=int, default=2, help="Image downscaling factor")
    return parser.parse_args()

def identify_scenario(gt_traj, ego_state):
    """
    Classifies a sample into a scenario: Deceleration, Sharp Turn, or Cruising.
    gt_traj: (8, 2)
    ego_state: (3,) [vx, yaw_rate, acc]
    """
    vx, yaw_rate, acc = ego_state
    
    # Deceleration scenario
    if acc < -1.0:
        return 'deceleration'
        
    # Sharp Turn scenario (based on lateral deviation of the final waypoints)
    max_lat = np.abs(gt_traj[:, 0]).max()
    if max_lat > 3.0 or np.abs(yaw_rate) > 0.15:
        return 'sharp_turn'
        
    return 'cruising'

def run_evaluation(model, dataloader, device, perturbation=None):
    """
    Runs evaluation over the dataloader under optional input perturbations.
    """
    model.eval()
    
    all_pred = []
    all_gt = []
    all_occupancy = []
    all_scenarios = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            ego_state = batch['ego_state'].to(device)
            
            # Apply Perturbations
            if perturbation == 'image_noise':
                # Add Gaussian noise to images
                images = images + torch.randn_like(images) * 0.1
                images = torch.clamp(images, 0.0, 1.0)
            elif perturbation == 'calibration_drift':
                # Add slight rotational noise to extrinsics (yaw shift)
                yaw_noise = torch.randn(extrinsics.shape[0], 3, device=device) * 0.05
                for b in range(extrinsics.shape[0]):
                    for c in range(3):
                        # Construct 3x3 rotation matrix for noise yaw
                        theta = yaw_noise[b, c]
                        cos, sin = torch.cos(theta), torch.sin(theta)
                        R_noise = torch.tensor([
                            [cos, -sin, 0.0],
                            [sin, cos, 0.0],
                            [0.0, 0.0, 1.0]
                        ], device=device, dtype=extrinsics.dtype)
                        extrinsics[b, c, :3, :3] = R_noise @ extrinsics[b, c, :3, :3]
            
            outputs = model(images, intrinsics, extrinsics, ego_state)
            
            # Extract chosen refined trajectory for each batch element
            logits = outputs['logits'] # (B, 16)
            trajectories = outputs['trajectories'] # (B, 16, 8, 2)
            
            best_idx = torch.argmax(logits, dim=-1)
            batch_idx = torch.arange(logits.shape[0], device=device)
            pred_traj = trajectories[batch_idx, best_idx] # (B, 8, 2)
            
            all_pred.append(pred_traj.cpu())
            all_gt.append(batch['future_traj'])
            all_occupancy.append(batch['occupancy_target'])
            
            # Identify scenarios for each batch element
            for b in range(logits.shape[0]):
                scen = identify_scenario(batch['future_traj'][b].numpy(), batch['ego_state'][b].numpy())
                all_scenarios.append(scen)
                
    # Concatenate all
    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    all_occupancy = torch.cat(all_occupancy, dim=0)
    
    # Calculate overall metrics
    overall = evaluate_metrics(all_pred, all_gt, all_occupancy)
    
    # Calculate scenario-based metrics
    scenarios_data = {'deceleration': [], 'sharp_turn': [], 'cruising': []}
    for i, scen in enumerate(all_scenarios):
        scenarios_data[scen].append((all_pred[i:i+1], all_gt[i:i+1], all_occupancy[i:i+1]))
        
    scenarios_results = {}
    for name, list_data in scenarios_data.items():
        if len(list_data) > 0:
            pred_s = torch.cat([item[0] for item in list_data], dim=0)
            gt_s = torch.cat([item[1] for item in list_data], dim=0)
            occ_s = torch.cat([item[2] for item in list_data], dim=0)
            scenarios_results[name] = evaluate_metrics(pred_s, gt_s, occ_s)
        else:
            scenarios_results[name] = {'l2_error': 0.0, 'collision_rate': 0.0, 'max_jerk': 0.0}
            
    return overall, scenarios_results

def evaluate_baseline(baseline_model, dataloader):
    """
    Evaluates a non-visual baseline model.
    """
    all_pred = []
    all_gt = []
    all_occupancy = []
    
    for batch in dataloader:
        ego_state = batch['ego_state']
        
        # Forward pass on baseline
        if isinstance(baseline_model, ConstantVelocityBaseline):
            vx = ego_state[:, 0]
            yaw_rate = ego_state[:, 1]
            pred_traj = baseline_model(vx, yaw_rate)
        else:
            pred_traj = baseline_model(ego_state)
        
        all_pred.append(pred_traj)
        all_gt.append(batch['future_traj'])
        all_occupancy.append(batch['occupancy_target'])
        
    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    all_occupancy = torch.cat(all_occupancy, dim=0)
    
    return evaluate_metrics(all_pred, all_gt, all_occupancy)

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    val_dataset = NuscenesDataset(
        version=args.version, 
        dataroot=args.dataroot, 
        split='val', 
        downscale_factor=args.downscale_factor
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Load anchors & checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        anchors = checkpoint['anchors']
        model = VectorDriveRouteModel(anchors=anchors).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: checkpoint not found. Evaluating randomly initialized model.")
        # Fallback to zeros/random anchors
        anchors = np.zeros((16, 8, 2))
        model = VectorDriveRouteModel(anchors=anchors).to(device)
        
    # 1. Evaluate Model (Clean)
    print("\n--- Running Clean Evaluation ---")
    overall, scenarios = run_evaluation(model, val_loader, device)
    
    print("Overall Performance:")
    print(f"  L2 Imitation Error: {overall['l2_error']:.4f} m")
    print(f"  Collision Rate:     {overall['collision_rate'] * 100:.2f} %")
    print(f"  Max Jerk (Comfort): {overall['max_jerk']:.4f} m/s^3")
    
    print("\nScenario Performance:")
    for scen, metrics in scenarios.items():
        print(f"  [{scen.upper()}]:")
        print(f"    L2 Error:       {metrics['l2_error']:.4f} m")
        print(f"    Collision Rate: {metrics['collision_rate'] * 100:.2f} %")
        print(f"    Max Jerk:       {metrics['max_jerk']:.4f} m/s^3")
        
    # 2. Perturbation Testing Suite
    print("\n--- Running Perturbation Tests ---")
    
    # Noise on images
    noise_overall, _ = run_evaluation(model, val_loader, device, perturbation='image_noise')
    print("Image Noise Perturbation:")
    print(f"  L2 Error:       {noise_overall['l2_error']:.4f} m")
    print(f"  Collision Rate: {noise_overall['collision_rate'] * 100:.2f} %")
    
    # Calibration Drift
    drift_overall, _ = run_evaluation(model, val_loader, device, perturbation='calibration_drift')
    print("Calibration Drift Perturbation:")
    print(f"  L2 Error:       {drift_overall['l2_error']:.4f} m")
    print(f"  Collision Rate: {drift_overall['collision_rate'] * 100:.2f} %")
    
    # 3. Evaluate Baselines
    print("\n--- Evaluating Baselines ---")
    
    # Constant Velocity Baseline
    cv_baseline = ConstantVelocityBaseline()
    cv_metrics = evaluate_baseline(cv_baseline, val_loader)
    print("Constant Velocity Baseline:")
    print(f"  L2 Error:       {cv_metrics['l2_error']:.4f} m")
    print(f"  Collision Rate: {cv_metrics['collision_rate'] * 100:.2f} %")
    print(f"  Max Jerk:       {cv_metrics['max_jerk']:.4f} m/s^3")
    
    # Ego MLP Baseline (train it quickly in-memory for 500 epochs)
    print("Training Ego-State MLP baseline (in-memory)...")
    train_dataset = NuscenesDataset(
        version=args.version, 
        dataroot=args.dataroot, 
        split='train', 
        downscale_factor=args.downscale_factor
    )
    
    # Extract training inputs and targets directly from metadata in RAM
    x_train = []
    y_train = []
    for meta in train_dataset.samples_list:
        x_train.append([meta['v_x'], meta['yaw_rate'], meta['acceleration']])
        y_train.append(meta['future_traj'])
        
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    
    ego_mlp = EgoStateMLPBaseline()
    ego_mlp.train()
    optimizer = torch.optim.Adam(ego_mlp.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(500):
        optimizer.zero_grad()
        pred = ego_mlp(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
            
    ego_mlp.eval()
    mlp_metrics = evaluate_baseline(ego_mlp, val_loader)
    print("Ego-State MLP Baseline:")
    print(f"  L2 Error:       {mlp_metrics['l2_error']:.4f} m")
    print(f"  Collision Rate: {mlp_metrics['collision_rate'] * 100:.2f} %")
    print(f"  Max Jerk:       {mlp_metrics['max_jerk']:.4f} m/s^3")
    
    # 4. Scenario Extraction and Image Generation
    print("\n--- Exporting Scenario Diagnostic Plots ---")
    idx_straight = None
    idx_turn = None
    idx_stop = None
    
    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        ego_state_val = sample['ego_state'].numpy()
        vx = ego_state_val[0]
        yaw_rate = np.abs(ego_state_val[1])
        
        # Complete Stop
        if vx < 0.2 and idx_stop is None:
            idx_stop = idx
            
        # Active Turn
        elif yaw_rate > 0.15 and idx_turn is None:
            idx_turn = idx
            
        # Straight Cruising
        elif vx > 3.0 and yaw_rate < 0.03 and idx_straight is None:
            idx_straight = idx
            
        if idx_straight is not None and idx_turn is not None and idx_stop is not None:
            break
            
    # Fallback search if conditions were too strict
    if idx_straight is None:
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            ego_state_val = sample['ego_state'].numpy()
            if np.abs(ego_state_val[1]) < 0.05:
                idx_straight = idx
                break
    if idx_turn is None:
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            if np.abs(sample['ego_state'][1]) > 0.08:
                idx_turn = idx
                break
    if idx_stop is None:
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            if sample['ego_state'][0] < 2.0:
                idx_stop = idx
                break

    print(f"Selected indices: Straight={idx_straight}, Turn={idx_turn}, Stop={idx_stop}")
    
    scenarios = {
        'straight': idx_straight,
        'turn': idx_turn,
        'stop': idx_stop
    }
    
    os.makedirs("checkpoints", exist_ok=True)
    # Always generate and export checkpoints/diagnostics.png (using sample 0)
    print("Exporting standard checkpoints/diagnostics.png...")
    sample_zero = val_dataset[0]
    generate_diagnostics_self_contained(model, sample_zero, "checkpoints/diagnostics.png", device)
    conv_folder = "/home/lalo/.gemini/antigravity/brain/ac300a5a-0b6b-4327-929f-c3fca0185cc4"
    if os.path.exists(conv_folder):
        os.system(f"cp checkpoints/diagnostics.png {os.path.join(conv_folder, 'diagnostics.png')}")

    for name, idx in scenarios.items():
        if idx is not None:
            sample = val_dataset[idx]
            save_path = f"checkpoints/diagnostics_{name}.png"
            generate_diagnostics_self_contained(model, sample, save_path, device)
            # Copy to conversation folder for artifact safety
            if os.path.exists(conv_folder):
                os.system(f"cp {save_path} {os.path.join(conv_folder, f'diagnostics_{name}.png')}")
                
    # 5. Write results.md Report
    report_content = f"""# VectorDrive-Route Evaluation Results

This report compiles the quantitative benchmark metrics and scenario diagnostic panels generated during the evaluation loop over the validation split.

---

## 📊 Trajectory Planner Benchmark

| Metric Category | Specific Evaluation Metric | Your Model's Score | Constant Velocity Baseline | Ego-State MLP Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **Imitation Performance** | minADE (Trajectory L2 Error) (m) | **{overall['l2_error']:.4f}** | {cv_metrics['l2_error']:.4f} | {mlp_metrics['l2_error']:.4f} |
| **Safety Compliance** | Drivable Area Compliance Rate (%) | **{(1.0 - overall['collision_rate']) * 100:.2f}%** | {(1.0 - cv_metrics['collision_rate']) * 100:.2f}% | {(1.0 - mlp_metrics['collision_rate']) * 100:.2f}% |
| | Collision Rate (%) | **{overall['collision_rate'] * 100:.2f}%** | {cv_metrics['collision_rate'] * 100:.2f}% | {mlp_metrics['collision_rate'] * 100:.2f}% |
| **Comfort & Kinematics** | Mean Trajectory Jerk ($m/s^3$) | **{overall['max_jerk']:.4f}** | {cv_metrics['max_jerk']:.4f} | {mlp_metrics['max_jerk']:.4f} |

---

## 📊 Perturbation Robustness Performance

| Perturbation Suite | minADE (Trajectory L2 Error) (m) | Collision Rate (%) |
| :--- | :---: | :---: |
| **Clean Baseline** | **{overall['l2_error']:.4f}** | **{overall['collision_rate'] * 100:.2f}%** |
| **Image Noise (Gaussian $\sigma=0.1$)** | **{noise_overall['l2_error']:.4f}** | **{noise_overall['collision_rate'] * 100:.2f}%** |
| **Calibration Drift (Yaw Shift $\pm 0.05$ rad)** | **{drift_overall['l2_error']:.4f}** | **{drift_overall['collision_rate'] * 100:.2f}%** |

---

## 🖼️ Exported Scenario Diagnostics

### Default Diagnostic Panel
![Default Diagnostics](checkpoints/diagnostics.png)

### Scenario 1: Straight Cruising
![Straight Cruising](checkpoints/diagnostics_straight.png)

### Scenario 2: Active Turn
![Active Turn](checkpoints/diagnostics_turn.png)

### Scenario 3: Complete Stop
![Complete Stop](checkpoints/diagnostics_stop.png)
"""
    
    with open("results.md", "w") as f:
        f.write(report_content)
    print("Evaluation report generated successfully at results.md")

if __name__ == "__main__":
    main()
