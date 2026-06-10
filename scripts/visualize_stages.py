import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import kornia

from datasets.nuscenes_dataset import NuscenesDataset
from models.model import VectorDriveRouteModel


def get_args():
    parser = argparse.ArgumentParser(
        description="Export per-stage visualization images"
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--dataroot", type=str, default="data/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--downscale_factor", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="checkpoints/stages")
    return parser.parse_args()


def save_image(arr, path, cmap=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    if cmap is None:
        plt.imshow(arr)
    else:
        plt.imshow(arr, cmap=cmap)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = NuscenesDataset(
        version=args.version,
        dataroot=args.dataroot,
        split="val",
        downscale_factor=args.downscale_factor,
    )
    sample = dataset[0]

    # Load model
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )
        anchors = checkpoint.get("anchors", None)
        model = VectorDriveRouteModel(anchors=anchors).to(device)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except Exception:
            print("Warning: failed strict load_state_dict; using non-strict load.")
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        anchors = np.zeros((16, 8, 2))
        model = VectorDriveRouteModel(anchors=anchors).to(device)

    model.eval()

    # Prepare inputs
    images = sample["images"].unsqueeze(0).to(device)
    intrinsics = sample["intrinsics"].unsqueeze(0).to(device)
    extrinsics = sample["extrinsics"].unsqueeze(0).to(device)
    ego_state = sample["ego_state"].unsqueeze(0).to(device)

    with torch.no_grad():
        # Stage 1: Panorama
        images_cyl, _, _, validity_mask = model.projection(
            images, intrinsics, extrinsics
        )
        panorama = images_cyl[0].permute(1, 2, 0).cpu().numpy()
        save_image(panorama, os.path.join(args.out_dir, "stage1_panorama.png"))

        # Stage 2: Backbone features (mean activation map)
        feat = model.backbone(images_cyl)  # (B, C, Hf, Wf)
        feat_mean = feat[0].mean(dim=0).cpu().numpy()
        save_image(
            feat_mean,
            os.path.join(args.out_dir, "stage2_backbone_mean.png"),
            cmap="viridis",
        )

        # Stage 3: LSS depth expected
        frustum, depth_probs = model.lift(feat)
        depth_bins = torch.linspace(
            2.0, 42.0, depth_probs.shape[1], device=device
        ).view(depth_probs.shape[1], 1, 1)
        expected_depth = torch.sum(depth_probs[0] * depth_bins, dim=0).cpu().numpy()
        save_image(
            expected_depth,
            os.path.join(args.out_dir, "stage3_expected_depth.png"),
            cmap="inferno",
        )

        # Stage 4: Raw BEV (before smoothing)
        x_bev_raw = model.pool(frustum)
        bev_raw_mean = x_bev_raw[0].mean(dim=0).cpu().numpy()
        save_image(
            bev_raw_mean, os.path.join(args.out_dir, "stage4_bev_raw.png"), cmap="magma"
        )

        # Stage 5: BEV after smoothing
        bev_smoothed = model.bev_smooth(x_bev_raw)
        bev_smooth_mean = bev_smoothed[0].mean(dim=0).cpu().numpy()
        save_image(
            bev_smooth_mean,
            os.path.join(args.out_dir, "stage5_bev_smoothed.png"),
            cmap="magma",
        )

        # Stage 6: Warped previous hidden (simulate h_prev for visualization)
        # Create a fake previous hidden by shifting the smoothed BEV slightly
        h_prev = bev_smoothed.clone()
        # Apply an affine warp using the temporal fusion warp matrix
        from models.temporal_fusion import compute_warp_matrix

        M = compute_warp_matrix(ego_state, dt=0.5)
        h_prev_warped = kornia.geometry.transform.warp_affine(
            h_prev,
            M,
            dsize=(bev_smoothed.shape[2], bev_smoothed.shape[3]),
            mode="bilinear",
            padding_mode="zeros",
        )
        h_prev_warped_mean = h_prev_warped[0].mean(dim=0).cpu().numpy()
        save_image(
            h_prev_warped_mean,
            os.path.join(args.out_dir, "stage6_prev_warped.png"),
            cmap="magma",
        )

        # Stage 7: Trajectories (overlay candidate and chosen)
        h_next = model.temporal_fusion(bev_smoothed, None, ego_state)
        logits, trajectories, drivable_preds, occupancy_preds = model.planning_head(
            h_next
        )
        logits = logits[0]
        trajectories = trajectories[0].cpu().numpy()
        best_idx = torch.argmax(logits).item()

        # Plot trajectories on top of predicted drivable map
        drivable = torch.sigmoid(drivable_preds[0, 0]).cpu().numpy()
        traj_fig = plt.figure(figsize=(6, 6))
        plt.imshow(drivable, cmap="gray", origin="upper", extent=[-20, 20, 0, 40])
        for k in range(trajectories.shape[0]):
            gx = trajectories[k, :, 0]
            gy = trajectories[k, :, 1]
            if k == best_idx:
                plt.plot(gx, gy, color="cyan", linewidth=3, marker="o")
            else:
                plt.plot(gx, gy, color="blue", alpha=0.3)
        gt_traj = sample["future_traj"].numpy()
        plt.plot(
            gt_traj[:, 0],
            gt_traj[:, 1],
            color="gold",
            linewidth=3,
            linestyle="--",
            marker="x",
        )
        plt.axis("off")
        traj_path = os.path.join(args.out_dir, "stage7_trajectories.png")
        traj_fig.savefig(traj_path, bbox_inches="tight", dpi=150)
        plt.close(traj_fig)

        # Stage 8: Decoders side-by-side
        drivable_pred = torch.sigmoid(drivable_preds[0, 0]).cpu().numpy()
        occupancy_pred = torch.sigmoid(occupancy_preds[0, 0]).cpu().numpy()
        combined = np.concatenate(
            [
                np.expand_dims(drivable_pred, -1),
                np.expand_dims(occupancy_pred, -1),
                np.zeros_like(np.expand_dims(drivable_pred, -1)),
            ],
            axis=2,
        )
        save_image(combined, os.path.join(args.out_dir, "stage8_decoders_rgb.png"))

    print(f"Saved stage images to {args.out_dir}")


if __name__ == "__main__":
    main()
