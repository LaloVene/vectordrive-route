import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CylindricalProjection(nn.Module):
    """
    Unified Cylindrical Canvas Projection.
    Projects three front-facing cameras into a single panorama.
    """

    def __init__(
        self,
        H_out=256,
        W_out=768,
        H_in=900,
        W_in=1600,
        horizontal_fov=180.0,
        vertical_fov=40.0,
        default_depth=30.0,
    ):
        super().__init__()
        self.H_out = H_out
        self.W_out = W_out
        self.H_in = H_in
        self.W_in = W_in
        self.horizontal_fov = np.radians(horizontal_fov)
        self.vertical_fov = np.radians(vertical_fov)
        self.default_depth = default_depth

        # Precompute the cylindrical ray direction map
        # Create grids with shape (H_out, W_out): v_cyl rows, u_cyl columns
        v_cyl, u_cyl = torch.meshgrid(
            torch.arange(H_out, dtype=torch.float32),
            torch.arange(W_out, dtype=torch.float32),
            indexing="ij",
        )

        # Map pixel column to angle theta (yaw)
        theta = self.horizontal_fov / 2.0 - (u_cyl / W_out) * self.horizontal_fov

        # Map pixel row to angle phi (pitch)
        f_virtual = H_out / (2.0 * np.tan(self.vertical_fov / 2.0))
        phi = torch.atan((H_out / 2.0 - v_cyl) / f_virtual)

        # 3D direction unit vectors in local Ego Frame using cylindrical mapping
        # NuScenes Ego Frame: X: Forward, Y: Left, Z: Up
        cos_phi = torch.cos(phi)
        x_ego = torch.cos(theta) * cos_phi
        y_ego = torch.sin(theta) * cos_phi
        z_ego = torch.sin(phi)

        # Shape: (3, H_out, W_out)
        self.register_buffer("rays_ego", torch.stack([x_ego, y_ego, z_ego], dim=0))

        # Grid cache to avoid recomputations when intrinsics/extrinsics are constant
        self.cached_grid = None
        self.cached_camera_mask = None
        self.cached_seam_mask = None
        self.cached_validity_mask = None
        self.cached_calib_hash = None

    def forward(self, images, intrinsics, extrinsics):
        """
        Args:
            images: torch.Tensor of shape (B, 3, 3, H_in, W_in) representing 3 camera streams.
            intrinsics: torch.Tensor of shape (B, 3, 3, 3) representing intrinsic matrices.
            extrinsics: torch.Tensor of shape (B, 3, 3, 4) representing extrinsic matrices [R | t].
        Returns:
            images_cyl: torch.Tensor of shape (B, 3, H_out, W_out).
            camera_mask: torch.Tensor of shape (B, 3, H_out, W_out).
            seam_mask: torch.Tensor of shape (B, 1, H_out, W_out).
            validity_mask: torch.Tensor of shape (B, 1, H_out, W_out).
        """
        B = images.shape[0]
        device = images.device
        H_in_actual = images.shape[3]
        W_in_actual = images.shape[4]

        # Optimization: use cached grid if calibration and batch sizes match
        calib_hash = (
            float(intrinsics.sum().cpu().item()),
            float(extrinsics.sum().cpu().item()),
            B,
            str(device),
            H_in_actual,
            W_in_actual,
        )
        if self.cached_calib_hash == calib_hash and self.cached_grid is not None:
            grid = self.cached_grid
            camera_mask = self.cached_camera_mask
            seam_mask = self.cached_seam_mask
            validity_mask = self.cached_validity_mask
        else:
            grid, camera_mask, seam_mask, validity_mask = self._compute_grids(
                B, intrinsics, extrinsics, device, H_in_actual, W_in_actual
            )
            self.cached_grid = grid
            self.cached_camera_mask = camera_mask
            self.cached_seam_mask = seam_mask
            self.cached_validity_mask = validity_mask
            self.cached_calib_hash = calib_hash

        # Perform bilinear sampling from the three physical cameras and blend
        output_images = torch.zeros(
            (B, 3, self.H_out, self.W_out), device=device, dtype=images.dtype
        )
        for c in range(3):
            img_c = images[:, c]  # shape: (B, 3, H_in, W_in)
            grid_c = grid[:, c]  # shape: (B, H_out, W_out, 2)

            # grid_sample samples features from input based on normalized coordinates in grid
            sampled = F.grid_sample(
                img_c,
                grid_c,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            # Blend based on selection mask
            mask_c = camera_mask[:, c : c + 1]
            output_images = output_images + sampled * mask_c

        return output_images, camera_mask, seam_mask, validity_mask

    def _compute_grids(self, B, intrinsics, extrinsics, device, H_in, W_in):
        # 3D points in Ego Frame (3, H_out, W_out)
        points_ego = self.rays_ego * self.default_depth
        points_ego_flat = points_ego.view(3, -1)  # (3, N)

        # Convert to homogeneous coords (4, N)
        ones = torch.ones((1, points_ego_flat.shape[1]), device=device)
        points_ego_hom = torch.cat([points_ego_flat, ones], dim=0)  # (4, N)
        points_ego_hom = points_ego_hom.unsqueeze(0).expand(B, -1, -1)  # (B, 4, N)

        coords_x = []
        coords_y = []
        valid_masks = []
        dist_to_center = []

        for c in range(3):
            ext_c = extrinsics[:, c]  # (B, 3, 4)
            int_c = intrinsics[:, c]  # (B, 3, 3)

            # Project to Camera frame: P_cam = ext @ P_ego_hom
            # P_cam shape: (B, 3, N)
            P_cam = torch.bmm(ext_c, points_ego_hom)

            # Project to Pixel frame: P_pix = intrinsics @ P_cam
            P_pix = torch.bmm(int_c, P_cam)

            # Normalized column (u) and row (v) pixel index
            u_pix = P_pix[:, 0] / torch.clamp(P_pix[:, 2], min=1e-5)
            v_pix = P_pix[:, 1] / torch.clamp(P_pix[:, 2], min=1e-5)

            # Validate boundaries and check if depth is in front of camera
            z_valid = P_cam[:, 2] > 0.1
            u_valid = (u_pix >= 0) & (u_pix <= W_in - 1)
            v_valid = (v_pix >= 0) & (v_pix <= H_in - 1)
            valid_c = z_valid & u_valid & v_valid  # (B, N)

            # Distance from optical center (cx, cy) to select best camera in overlap
            cx = int_c[:, 0, 2].unsqueeze(-1)
            cy = int_c[:, 1, 2].unsqueeze(-1)
            dist_c = torch.sqrt((u_pix - cx) ** 2 + (v_pix - cy) ** 2)

            coords_x.append(u_pix)
            coords_y.append(v_pix)
            valid_masks.append(valid_c)
            dist_to_center.append(dist_c)

        coords_x = torch.stack(coords_x, dim=1)  # (B, 3, N)
        coords_y = torch.stack(coords_y, dim=1)  # (B, 3, N)
        valid_masks = torch.stack(valid_masks, dim=1)  # (B, 3, N)
        dist_to_center = torch.stack(dist_to_center, dim=1)  # (B, 3, N)

        # Penalize invalid cameras with huge distance for argmin
        large_val = 1e9
        adjusted_dist = torch.where(
            valid_masks, dist_to_center, torch.full_like(dist_to_center, large_val)
        )
        best_cam_idx = torch.argmin(adjusted_dist, dim=1)  # (B, N)

        # Construct camera mask representation
        camera_mask_flat = (
            F.one_hot(best_cam_idx, num_classes=3).permute(0, 2, 1).to(device)
        )
        any_valid = torch.any(valid_masks, dim=1)  # (B, N)
        camera_mask_flat = camera_mask_flat * any_valid.unsqueeze(1)

        camera_mask = camera_mask_flat.view(B, 3, self.H_out, self.W_out).float()
        validity_mask = any_valid.view(B, 1, self.H_out, self.W_out).float()

        # Seam mask highlights overlapping boundaries (multiple cameras valid)
        num_valid_cams = valid_masks.sum(dim=1)
        seam_mask = (num_valid_cams > 1).view(B, 1, self.H_out, self.W_out).float()

        # Generate sampling grid normalized to [-1, 1]
        grid = torch.zeros((B, 3, self.H_out, self.W_out, 2), device=device)
        # Normalize sampling coordinates for F.grid_sample with align_corners=False
        # For align_corners=False use: x_norm = 2*(x + 0.5)/W - 1
        for c in range(3):
            u_norm = 2.0 * (coords_x[:, c] + 0.5) / float(W_in) - 1.0
            v_norm = 2.0 * (coords_y[:, c] + 0.5) / float(H_in) - 1.0

            grid_c = torch.stack([u_norm, v_norm], dim=-1)
            grid[:, c] = grid_c.view(B, self.H_out, self.W_out, 2)

        return grid, camera_mask, seam_mask, validity_mask
