import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

from datasets.nuscenes_dataset import NuscenesDataset
from models.model import VectorDriveRouteModel
from utils.kmeans_anchors import compute_kmeans_anchors
from losses.multi_task_loss import MultiTaskLoss

def get_args():
    parser = argparse.ArgumentParser(description="VectorDrive-Route Model Training")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="NuScenes dataset version")
    parser.add_argument("--dataroot", type=str, default="data/nuscenes", help="Path to NuScenes dataset root")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--downscale_factor", type=int, default=2, help="Image downscaling factor")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. Datasets & Dataloaders
    print("Initializing datasets...")
    train_dataset = NuscenesDataset(
        version=args.version, 
        dataroot=args.dataroot, 
        split='train', 
        downscale_factor=args.downscale_factor
    )
    val_dataset = NuscenesDataset(
        version=args.version, 
        dataroot=args.dataroot, 
        split='val', 
        downscale_factor=args.downscale_factor
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}, batches: {len(train_loader)}")
    print(f"Val samples: {len(val_dataset)}, batches: {len(val_loader)}")
    
    # 3. K-Means Anchors
    anchors = compute_kmeans_anchors(train_dataset, num_anchors=16)
    
    # 4. Instantiate Model & Loss
    print("Building model...")
    model = VectorDriveRouteModel(anchors=anchors).to(device)
    loss_fn = MultiTaskLoss().to(device)
    
    # 5. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 15
    history = {
        'train_losses': [],
        'val_losses': [],
        'epochs': []
    }
    
    # 6. Training Loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        model.train()
        train_loss_epoch = 0.0
        train_loss_details = {}
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Send inputs to device
            images = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            ego_state = batch['ego_state'].to(device)
            
            targets = {
                'depth_target': batch['depth_target'].to(device),
                'depth_mask': batch['depth_mask'].to(device),
                'future_traj': batch['future_traj'].to(device),
                'drivable_target': batch['drivable_target'].to(device),
                'occupancy_target': batch['occupancy_target'].to(device)
            }
            
            # Forward with mixed-precision
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(images, intrinsics, extrinsics, ego_state, h_prev=None)
                losses = loss_fn(outputs, targets)
                loss = losses['loss']
                
            # Backward
            scaler.scale(loss).backward()
            
            # Gradient clipping to stabilize training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_epoch += loss.item()
            for k, v in losses.items():
                train_loss_details[k] = train_loss_details.get(k, 0.0) + v.item()
                
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Total Loss: {loss.item():.4f} | Reg: {losses['loss_reg'].item():.4f} | Safety: {losses['loss_safety'].item():.4f}")
                
        # Calculate training epoch averages
        train_loss_epoch /= len(train_loader)
        for k in train_loss_details:
            train_loss_details[k] /= len(train_loader)
            
        # 7. Validation Loop
        model.eval()
        val_loss_epoch = 0.0
        val_loss_details = {}
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                intrinsics = batch['intrinsics'].to(device)
                extrinsics = batch['extrinsics'].to(device)
                ego_state = batch['ego_state'].to(device)
                
                targets = {
                    'depth_target': batch['depth_target'].to(device),
                    'depth_mask': batch['depth_mask'].to(device),
                    'future_traj': batch['future_traj'].to(device),
                    'drivable_target': batch['drivable_target'].to(device),
                    'occupancy_target': batch['occupancy_target'].to(device)
                }
                
                outputs = model(images, intrinsics, extrinsics, ego_state, h_prev=None)
                losses = loss_fn(outputs, targets)
                
                val_loss_epoch += losses['loss'].item()
                for k, v in losses.items():
                    val_loss_details[k] = val_loss_details.get(k, 0.0) + v.item()
                    
        val_loss_epoch /= len(val_loader)
        for k in val_loss_details:
            val_loss_details[k] /= len(val_loader)
            
        scheduler.step()
        
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss_epoch:.4f} (Depth: {train_loss_details['loss_depth']:.4f}, Cls: {train_loss_details['loss_cls']:.4f}, Reg: {train_loss_details['loss_reg']:.4f})")
        print(f"  Val Loss:   {val_loss_epoch:.4f} (Depth: {val_loss_details['loss_depth']:.4f}, Cls: {val_loss_details['loss_cls']:.4f}, Reg: {val_loss_details['loss_reg']:.4f})")
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'anchors': anchors,
            'train_loss': train_loss_epoch,
            'val_loss': val_loss_epoch
        }
        
        torch.save(checkpoint, os.path.join(args.save_dir, "latest_model.pth"))
        
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(checkpoint, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  [New Best Model Saved with Val Loss {best_val_loss:.4f}]")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  [Early stopping counter: {epochs_no_improve}/{patience}]")
            
        history['train_losses'].append(train_loss_epoch)
        history['val_losses'].append(val_loss_epoch)
        history['epochs'].append(epoch)
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered. Training stopped after {epoch} epochs.")
            break
        
    # Save history logs
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    # Always plot losses after training
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(history['epochs'], history['train_losses'], label='Train Loss', marker='o')
        plt.plot(history['epochs'], history['val_losses'], label='Val Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(args.save_dir, "loss_plot.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Loss plot saved to {plot_path}")
    except Exception as e:
        print(f"Error plotting training history: {e}")
        
    print("\nTraining completed successfully! Best model saved.")

if __name__ == "__main__":
    main()
