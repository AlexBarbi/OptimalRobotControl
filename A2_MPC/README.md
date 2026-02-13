# ORC Homework A2: Model Predictive Control for UR5

This repository contains the solutions for the Optimal Robot Control (ORC) Homework A2. The project implements Optimal Control Problems (OCP) using CasADi to control a UR5 robot arm, making its end-effector follow an infinity-shaped reference path.

## Project Overview

The main objective is to compute optimal trajectories for a UR5 manipulator to track a geometric path (infinity shape) while minimizing various cost functions such as joint velocities, torques, and path deviation. The implementation uses the **ADAM** library for rigid body dynamics and **CasADi** for numerical optimization (Ipopt solver).

## Dependencies

The project relies on the following Python libraries:

- `numpy`
- `matplotlib`
- `casadi`
- `adam-robotics` (imported as `adam`)
- `example-robot_data`
- `pinocchio` (implied by robot wrappers)
- `orc` (Course-specific utility package for simulation and visualization)

Make sure you have these installed or available in your Python environment.

## File Structure & Description

The homework is divided into multiple questions, each addressing a different formulation of the optimal control problem:

### 1. `A2_Q1.py` - Basic Path Following
- **Goal:** Solve a fixed-time OCP to track the path.
- **Cost Function:** Minimizes joint velocities (`w_v`), control efforts/torques (`w_a`), and path speed variations (`w_w`).
- **Constraints:** Kinematics, dynamics, and joint/torque limits.

### 2. `A2_Q2.py` - Cyclic Motion
- **Extension of Q1.**
- **Goal:** Enforce a "cyclic" behavior where the robot returns to its initial configuration at the end of the trajectory.
- **Key Change:** Adds a terminal penalty (`w_final`) on the difference between the final and initial states ($X_N - X_0$).

### 3. `A2_Q3.py` - Trajectory Tracking
- **Goal:** Track a specific trajectory (path + timing) rather than just the geometric path.
- **Key Change:**
  - Constrains the path progression speed $W_k$ to be constant ($1/N$).
  - Introduces a Cartesian tracking error cost term (`w_p`) to penalize deviation from the reference path.

### 4. Time Optimization
These scripts define the time step `dt` as a decision variable to optimize the total execution time or adjust constraints dynamically.

- **`A2_Q4_P.py`**: Solves the OCP with variable `dt`, balancing energetic costs with time-related costs.
- **`A2_Q4_T.py`**: Similar to the above, likely serving as a variation for specific time-optimal or trajectory-optimal tuning.

## Usage

To run any of the solutions, simply execute the corresponding Python script. Ensure that a viewer (like `gepetto-gui` or `Meshcat`, depending on the `orc` configuration) is running if you want to see the 3D simulation.

```bash
# Run Question 1
python3 A2_Q1.py

# Run Question 2
python3 A2_Q2.py

# ... and so on
```

## Visualization

The scripts use `RobotSimulator` to visualize the robot's motion.
- **Reference Path:** Visualized as a series of red spheres.
- **plots:** Matplotlib figures will pop up after the solver finishes, showing:
  - Path progression variable $s(t)$
  - End-effector position (Reference vs. Actual)
  - Joint velocities, positions, and torques.
