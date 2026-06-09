# VectorDrive-Route Evaluation Results

This report compiles the quantitative benchmark metrics and scenario diagnostic panels generated during the evaluation loop over the validation split.

---

## 📊 Trajectory Planner Benchmark

| Metric Category | Specific Evaluation Metric | Your Model's Score | Constant Velocity Baseline | Ego-State MLP Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **Imitation Performance** | minADE (Trajectory L2 Error) (m) | **14.7322** | 0.5834 | 2.0909 |
| **Safety Compliance** | Drivable Area Compliance Rate (%) | **94.62%** | 94.62% | 94.62% |
| | Collision Rate (%) | **5.38%** | 5.38% | 5.38% |
| **Comfort & Kinematics** | Mean Trajectory Jerk ($m/s^3$) | **3.1772** | 0.0343 | 7.5471 |

---

## 📊 Perturbation Robustness Performance

| Perturbation Suite | minADE (Trajectory L2 Error) (m) | Collision Rate (%) |
| :--- | :---: | :---: |
| **Clean Baseline** | **14.7322** | **5.38%** |
| **Image Noise (Gaussian $\sigma=0.1$)** | **14.3969** | **6.92%** |
| **Calibration Drift (Yaw Shift $\pm 0.05$ rad)** | **14.5563** | **5.38%** |

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
