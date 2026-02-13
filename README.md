# ORC Homework Repository

This repository contains homework assignments and the final project for the **Optimal Robot Control (ORC)** course.
AUTHORS: **Alejandro Enrique Barbi** & **Alessandro Moscatelli**

The repository is structured into three main assignments, each focusing on a different aspect of robot control.

---

## 📂 Repository Structure

### 1. [Assignment 1: Task Space Inverse Dynamics (TSID)](./A1_TSID)
**Directory:** `A1_TSID/`

This assignment focuses on **Task Space Inverse Dynamics** for bipedal robots (e.g., Romeo, Talos).
*   **Goal:** Simulate bipedal locomotion, maintain balance, and track Center of Mass (CoM) trajectories.
*   **Key Scripts:**
    *   `ex_1_biped.py`: Basic standing and CoM tracking.
    *   `ex_2_biped_walking.py`: Advanced walking simulation using pre-computed trajectories.
*   **Dependencies:** `pinocchio`, `tsid`, `example-robot-data`.

### 2. [Assignment 2: Model Predictive Control (MPC)](./A2_MPC)
**Directory:** `A2_MPC/`

This assignment implements **Model Predictive Control (MPC)** problems for a **UR5 robot arm** using **CasADi**.
*   **Goal:** Compute optimal trajectories for the UR5 end-effector to follow geometric paths (e.g., infinity shape) while minimizing costs (velocity, torque, etc.).
*   **Key Scripts:**
    *   `A2_Q1.py`: Basic path following OCP.
    *   `A2_Q2.py`: Cyclic motion constraints.
    *   `A2_Q3.py`: Trajectory tracking with time constraints.
    *   `A2_Q4_*.py`: Time optimization strategies.
*   **Dependencies:** `casadi`, `adam-robotics`, `pinocchio`.

### 3. [Assignment 3: Value Function Approximation (RL/OCP)](./A3_RL)
**Directory:** `A3_RL/`

This constitutes the **Final Project**. It implements a pipeline to approximate the **Value Function** of an Optimal Control Problem (OCP) using **Deep Learning**.
*   **Goal:** Train a Neural Network to regress the optimal cost from an initial state, allowing it to be used as a terminal cost in an MPC formulation.
*   **Workflow:**
    1.  **Data Generation:** Solving OCPs for a dense grid of initial states using CasADi/IPOPT.
    2.  **Training:** Supervised learning with a PyTorch Neural Network.
    3.  **Simulation:** Verifying performance on Single and Double Pendulum systems.
*   **Key Scripts:** `main.py` (entry point for data gen, training, and sim).
*   **Dependencies:** `pytorch`, `casadi`.

---

## 🚀 Getting Started

Each directory contains its own `README.md` with detailed instructions on how to install dependencies and run the specific scripts.

1.  **Navigate directly to the assignment folder:**

2.  **Follow the specific instructions** in that folder's `README.md`.