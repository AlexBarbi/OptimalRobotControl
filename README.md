# ORC_homework
Barbi and Moscatelli homeworks for ORC course

---

# Optimal Control Value Function Approximation

This project implements a pipeline to approximate the **Value Function ** of an Optimal Control Problem (OCP) using Deep Learning. It generates a dataset of optimal costs using trajectory optimization and trains a Neural Network to regress the cost from the initial state.

## Project Workflow

The process is divided into two main stages, automated within `main.py`:

### 1. Data Generation (OCP Solver)

* **System:** Double Pendulum / UR5 (configurable).
* **Method:** Solves the Optimal Control Problem for a dense grid of initial states  using **CasADi** and **IPOPT**.
* **Sampling:** Uses **Grid Generation** to cover the state space uniformly (positions along the full unit circle  and velocities).
* **Performance:** Utilizes Python `multiprocessing` to solve thousands of OCPs in parallel on all available CPU cores.
* **Output:** A dataset of state-cost pairs .

### 2. Neural Network Training (Supervised Learning)

* **Model:** A Feedforward Neural Network (MLP) implemented in **PyTorch**.
* **Architecture:** Input layer  Hidden Layers (Tanh activation)  Output (Scalar Cost).
* **Scaling:** Incorporates an output scaling factor (`ub`) based on the maximum cost in the dataset to stabilize training.


* **Training:** Minimizes the **Mean Squared Error (MSE)** between the predicted cost and the ground truth optimal cost calculated by the solver.
* **Validation:** Includes automatic train/test splitting and visualizes convergence (Loss vs. Epochs) and prediction accuracy (Ground Truth vs. Prediction).

## File Structure

* `main.py`: The entry point. Handles grid generation, parallel OCP solving, data processing, and triggers the training loop.
* `neural_network.py`: Defines the PyTorch `NeuralNetwork` class and the `train_network` function.
* `model.pt`: The saved trained model weights.
* `value_function_data.npz`: The generated dataset (saved for reproducibility).