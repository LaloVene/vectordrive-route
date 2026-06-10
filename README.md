# VectorDrive-Route

> [!WARNING]
> This repository is a research prototype. Code, notebooks, and checkpoints change frequently.

VectorDrive-Route maps multi-camera inputs to a Bird's-Eye-View (BEV) representation and predicts future ego trajectories. The code trains and evaluates models on the nuScenes dataset.

## Quick Start

- Clone and install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Download `nuScenes-mini` and place it under `data/nuscenes/`.

Run training, evaluation, or visualization scripts from the project root. Examples:

Train:

```bash
PYTHONPATH=. .venv/bin/python scripts/train.py --epochs 50 --batch_size 4 --lr 2e-4 --save_dir checkpoints
```

Evaluate:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

## Project Structure

- `baselines/` — simple baselines (constant velocity, ego-mlp)
- `checkpoints/` — model checkpoints and diagnostics
- `data/nuscenes/` — dataset (put nuScenes-mini here)
- `datasets/` — dataset loader and grid targets
- `geometry/` — projection and voxel pooling code
- `losses/` — multi-task losses
- `metrics/` — evaluation metrics
- `models/` — model components and planning head
- `scripts/` — train/evaluate runners
- `utils/` — utilities (e.g., k-means anchors)

## Architecture

```mermaid
graph TD
    CAM[Multi-Camera Front Feeds] -->|Cylindrical Projection| PAN[Panoramic Canvas]
    PAN -->|ResNet-18 Backbone| FEAT[Perspective Features]
    FEAT -->|LSS Lift Stage| FRUS[3D Frustum Space]
    FRUS -->|Voxel Pooling Splat| BEV[Current BEV Feature Map]
    BEV -->|Ego-Motion Warping + ConvGRU| FUSE[Fused Recurrent BEV State]
    FUSE -->|Cross-Attention| TRAJ[Trajectory Planner Head]
    TRAJ -->|Trajectory Offset Regressor| WAY[Refined Waypoints]
    TRAJ -->|Anchor Classifier| PROB[Path Probabilities]
    FUSE -->|Auxiliary Decoders| MAPS[Drivable & Occupancy Grids]

    %% Styling Theme
    classDef sensor fill:#3B82F6,stroke:#1D4ED8,stroke-width:2px,color:#FFFFFF;
    classDef feature2d fill:#F0FDF4,stroke:#16A34A,stroke-width:1px,color:#0F172A;
    classDef feature3d fill:#FFFBEB,stroke:#D97706,stroke-width:1px,color:#0F172A;
    classDef temporal fill:#EFF6FF,stroke:#2563EB,stroke-width:1px,color:#0F172A;
    classDef outputs fill:#F3E8FF,stroke:#9333EA,stroke-width:1px,color:#0F172A;

    class CAM sensor;
    class PAN,FEAT feature2d;
    class FRUS,BEV feature3d;
    class FUSE temporal;
    class TRAJ,WAY,PROB,MAPS outputs;
```

**Key components:**

- Cylindrical projection: stitches front cameras into a panorama and reprojects pixels for sampling.
- Backbone + LSS: extracts perspective features and lifts them into depth-conditioned frustums.
- Voxel pooling: aggregates frustum features into a top-down BEV grid (100×100, X∈[-20,20]m, Y∈[0,40]m).
- Temporal fusion: warps past BEV states to the current frame and fuses them with a ConvGRU.
- Planning head: queries compressed BEV tokens with K-Means anchors (K=16) to classify and regress candidate trajectories.
- Auxiliary decoders: predict drivable area and occupancy grids used for safety checks.

## Stages

This section explains the eight visualization stages.

1. Stage 1 — Cylindrical Stitching
   - Merge three front cameras into a single cylindrical panorama.
   - Remove overlap and create a continuous sampling canvas for projection.

   ![Stage1 Panorama](checkpoints/stages/stage1_panorama.png)

2. Stage 2 — Backbone Features
   - Run a ResNet-18 backbone on the panorama to get dense features.

   ![Stage2 Backbone](checkpoints/stages/stage2_backbone_mean.png)

3. Stage 3 — LSS Depth Lift
   - Predict categorical depth probabilities and lift 2D features into a 3D frustum.
   - Provide depth-conditioned features for BEV pooling.

   ![Stage3 Depth](checkpoints/stages/stage3_expected_depth.png)

4. Stage 4 — Voxel Pooling & Smoothing
   - Project frustum features into a BEV grid and apply lateral smoothing (BEVResBlocks).
   - Gather scene context and reduce radial projection streaks.

   ![Stage4 BEV Raw](checkpoints/stages/stage4_bev_raw.png)

5. Stage 5 — Warping & Temporal Fusion
   - Warp the previous BEV to the current frame using ego motion, then fuse via ConvGRU.
   - Build a temporally consistent BEV state.

   ![Stage5 BEV Smoothed](checkpoints/stages/stage5_bev_smoothed.png)

6. Stage 6 — Trajectory Classification
   - Compress BEV into scene tokens and score K=16 anchors with cross-attention.
   - Rank candidate paths.

7. Stage 7 — Trajectory Regression
   - Regress offsets for selected anchors to produce trajectories the vehicle can follow.

   ![Stage7 Trajectories](checkpoints/stages/stage7_trajectories.png)

8. Stage 8 — Auxiliary Decoders & Safety Metrics
   - Predict drivable and occupancy grids.
     ![Stage8 Decoders](checkpoints/stages/stage8_decoders_rgb.png)

Open the notebook to see the figures and the code that produces them: [visualizations.ipynb](visualizations.ipynb).
