import torch
import torch.nn as nn
import numpy as np


class PlanningHead(nn.Module):
    """
    Coarse-to-Fine Planning Head.
    Interacts 16 static intent anchors with spatial BEV tokens via cross-attention.
    Decodes intent probability scores, trajectory adjustments, and auxiliary grid maps.
    """

    def __init__(self, anchors=None, C_bev=80, C_trans=256, K_anchors=16):
        super().__init__()
        self.K_anchors = K_anchors
        self.C_trans = C_trans

        # 1. Register intent anchors library
        if anchors is None:
            # If no anchors provided, create small learnable anchor embeddings
            self.intent_anchors = nn.Parameter(torch.randn(K_anchors, 8, 2) * 0.01)
        else:
            if isinstance(anchors, np.ndarray):
                anchors = torch.from_numpy(anchors).float()
            # register static anchors as buffers so they move with the module
            self.register_buffer("intent_anchors", anchors)

        # 2. BEV compression layer: (B, 80, 100, 100) -> (B, 256, 8, 8)
        self.compress = nn.Sequential(
            nn.Conv2d(C_bev, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, C_trans, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_trans),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # 3. Anchor projection MLP: maps flattened anchor (16) to embedding dimension (C_trans)
        self.anchor_proj = nn.Sequential(
            nn.Linear(16, C_trans), nn.ReLU(inplace=True), nn.Linear(C_trans, C_trans)
        )

        # LayerNorms for stability around attention
        self.layernorm_scene = nn.LayerNorm(C_trans)
        self.layernorm_query = nn.LayerNorm(C_trans)

        # 4. Multi-head cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=C_trans, num_heads=8, batch_first=True, dropout=0.1
        )

        # 5. Output sub-heads
        # Classification head: scores each intent anchor
        self.class_head = nn.Linear(C_trans, 1)

        # Regression head: predicts (dx, dy) adjustments for 8 waypoints (16 values)
        self.reg_head = nn.Linear(C_trans, 16)

        # 6. Auxiliary grid decoders: (B, 80, 100, 100) -> (B, 1, 100, 100)
        self.drivable_decoder = nn.Sequential(
            nn.Conv2d(C_bev, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self.occupancy_decoder = nn.Sequential(
            nn.Conv2d(C_bev, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, h_bev):
        """
        Args:
            h_bev: torch.Tensor of shape (B, C_bev, H_bev, W_bev) - latent temporal features.
        Returns:
            logits: torch.Tensor of shape (B, K_anchors) - intent probability logits.
            trajectories: torch.Tensor of shape (B, K_anchors, 8, 2) - refined waypoint coordinates.
            drivable_preds: torch.Tensor of shape (B, 1, 100, 100) - auxiliary drivable area logits.
            occupancy_preds: torch.Tensor of shape (B, 1, 100, 100) - auxiliary occupancy logits.
        """
        B = h_bev.shape[0]
        device = h_bev.device

        # 1. Compress BEV features to extract scene tokens: shape (B, C_trans, 8, 8)
        scene_feat = self.compress(h_bev)

        # Reshape to sequence of tokens: shape (B, 64, C_trans)
        scene_tokens = scene_feat.view(B, self.C_trans, -1).permute(0, 2, 1)

        # 2. Project static intent anchors into query tokens: shape (B, K_anchors, C_trans)
        flat_anchors = self.intent_anchors.view(self.K_anchors, -1)
        q = self.anchor_proj(flat_anchors)
        q = q.unsqueeze(0).expand(B, -1, -1)

        # 3. Perform cross-attention between queries (anchors) and keys/values (scene tokens)
        # attn_out shape: (B, K_anchors, C_trans)
        # Normalize tokens before attention
        scene_tokens = self.layernorm_scene(scene_tokens)
        q = self.layernorm_query(q)

        attn_out, _ = self.cross_attn(query=q, key=scene_tokens, value=scene_tokens)

        # Residual connection: add query embedding back and renormalize
        attn_out = attn_out + q
        attn_out = self.layernorm_query(attn_out)

        # 4. Map attention representation to macro-intent classification logits: shape (B, K_anchors)
        logits = self.class_head(attn_out).squeeze(-1)

        # 5. Map attention representation to continuous trajectory offsets: shape (B, K_anchors, 8, 2)
        reg_offsets = self.reg_head(attn_out).view(B, self.K_anchors, 8, 2)

        # Refine trajectory by adding offsets to static intent anchors
        trajectories = self.intent_anchors.unsqueeze(0) + reg_offsets

        # 6. Decode auxiliary drivable area and obstacle occupancy grid maps
        drivable_preds = self.drivable_decoder(h_bev)
        occupancy_preds = self.occupancy_decoder(h_bev)

        return logits, trajectories, drivable_preds, occupancy_preds
