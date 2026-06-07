# VectorDrive-Route Evaluation Results

This report compiles the quantitative benchmark metrics and scenario diagnostic panels generated during the evaluation loop over the validation split.

---

## 📊 Trajectory Planner Benchmark

| Metric Category | Specific Evaluation Metric | Your Model's Score | Constant Velocity Baseline | Ego-State MLP Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **Imitation Performance** | minADE (Trajectory L2 Error) (m) | **10.3774** | 0.5834 | 2.9092 |
| **Safety Compliance** | Drivable Area Compliance Rate (%) | **89.23%** | 94.62% | 93.85% |
| | Collision Rate (%) | **10.77%** | 5.38% | 6.15% |
| **Comfort & Kinematics** | Mean Trajectory Jerk ($m/s^3$) | **1.7900** | 0.0343 | 3.8550 |

---

## 📊 Perturbation Robustness Performance

| Perturbation Suite | minADE (Trajectory L2 Error) (m) | Collision Rate (%) |
| :--- | :---: | :---: |
| **Clean Baseline** | **10.3774** | **10.77%** |
| **Image Noise (Gaussian $\sigma=0.1$)** | **10.4071** | **8.46%** |
| **Calibration Drift (Yaw Shift $\pm 0.05$ rad)** | **10.5008** | **6.15%** |

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
