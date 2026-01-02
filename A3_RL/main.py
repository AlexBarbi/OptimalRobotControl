#!/usr/bin/env python3

import numpy as np
import casadi as cs
import multiprocessing
import time
from time import sleep
import os
import matplotlib.pyplot as plt
import argparse
import csv
from pathlib import Path
from datetime import datetime
import random

from adam.casadi.computations import KinDynComputations
from example_robot_data.robots_loader import load
import orc.optimal_control.casadi_adam.conf_ur5 as conf_ur5
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.viz_utils import addViewerSphere, applyViewerConfiguration

from neural_network import train_network, NeuralNetwork
import torch
import l4casadi as l4

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ====================== Configuration ======================
# Robot dimensions
from pendulum import Pendulum

# Robot dimensions
# robot = load("double_pendulum_simple")
robot = Pendulum(2, open_viewer=False)
joints_name_list = [s for s in robot.model.names[1:]]
nq = len(joints_name_list)
nx = 2 * nq
nu = nq             # Control dimension (ddq - joint acceleration)

# OCP Parameters
N = 100              # Horizon length
dt = 0.02            # Time step
NUM_SAMPLES = 10000  # Total number of OCPs to solve
NUM_CORES = multiprocessing.cpu_count() # Use all available cores

# Cost Weights (Regulation Task)
W_Q = 10.0          # Weight for joint position error
W_V = 1.0           # Weight for joint velocity error
W_U = 0.1           # Weight for control effort (acceleration)

def solve_single_ocp(x_init, N = 100):
    opti = cs.Opti()
    X = [opti.variable(nx) for _ in range(N + 1)]
    U = [opti.variable(nu) for _ in range(N)]
    x_target = np.zeros(nx)
    cost = 0.0

    for k in range(N):
        q_err = X[k][:nq] - x_target[:nq]
        v_err = X[k][nq:] - x_target[nq:]
        cost += W_Q * cs.sumsqr(q_err) 
        cost += W_V * cs.sumsqr(v_err)
        cost += W_U * cs.sumsqr(U[k])
        
        q_next_euler  = X[k][:nq] + dt * X[k][nq:]
        dq_next_euler = X[k][nq:] + dt * U[k]
        x_next_euler = cs.vertcat(q_next_euler, dq_next_euler)
        opti.subject_to(X[k+1] == x_next_euler)

    opti.subject_to(X[0] == x_init)
    opti.minimize(cost)

    opts = {
        "ipopt.print_level": 0, 
        "print_time": 0, 
        "ipopt.sb": "yes", 
        "ipopt.tol": 1e-4
    }
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
        return (x_init, sol.value(cost))
    except Exception:
        return None

def solve_single_ocp_with_terminal(x_init, N=100, terminal_model=None):
    opti = cs.Opti()
    X = [opti.variable(nx) for _ in range(N + 1)]
    U = [opti.variable(nu) for _ in range(N)]
    x_target = np.zeros(nx)
    cost = 0.0

    for k in range(N):
        q_err = X[k][:nq] - x_target[:nq]
        v_err = X[k][nq:] - x_target[nq:]
        cost += W_Q * cs.sumsqr(q_err) 
        cost += W_V * cs.sumsqr(v_err)
        cost += W_U * cs.sumsqr(U[k])
        
        q_next_euler  = X[k][:nq] + dt * X[k][nq:]
        dq_next_euler = X[k][nq:] + dt * U[k]
        x_next_euler = cs.vertcat(q_next_euler, dq_next_euler)
        opti.subject_to(X[k+1] == x_next_euler)

    opti.subject_to(X[0] == x_init)

    # Neural Network terminal
    if terminal_model is not None:
        # Evaluate terminal cost
        # The model expects a tensor/array of shape (1, nx) or (nx,)
        # l4casadi handles casadi symbols.
        # Ensure that the input to the model matches what it expects (e.g. 2*nq)
        
        # l4casadi needs a symbolic call
        term_cost = terminal_model(X[N])
        cost += term_cost

    opti.minimize(cost)

    opts = {
        "ipopt.print_level": 0, 
        "print_time": 0, 
        "ipopt.sb": "yes", 
        "ipopt.tol": 1e-4
    }
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
        return (x_init, sol.value(cost))
    except Exception:
        return None


def solve_single_ocp_return_terminal(x_init, N = 100):
    """Solve the unconstrained OCP and return the final state as well."""
    opti = cs.Opti()
    X = [opti.variable(nx) for _ in range(N + 1)]
    U = [opti.variable(nu) for _ in range(N)]
    x_target = np.zeros(nx)
    cost = 0.0

    for k in range(N):
        q_err = X[k][:nq] - x_target[:nq]
        v_err = X[k][nq:] - x_target[nq:]
        cost += W_Q * cs.sumsqr(q_err)
        cost += W_V * cs.sumsqr(v_err)
        cost += W_U * cs.sumsqr(U[k])

        q_next_euler  = X[k][:nq] + dt * X[k][nq:]
        dq_next_euler = X[k][nq:] + dt * U[k]
        x_next_euler = cs.vertcat(q_next_euler, dq_next_euler)
        opti.subject_to(X[k+1] == x_next_euler)

    opti.subject_to(X[0] == x_init)
    opti.minimize(cost)

    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
        "ipopt.tol": 1e-4
    }
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
        xN = sol.value(X[N])
        return (x_init, sol.value(cost), np.array(xN).reshape(-1))
    except Exception:
        return None


def solve_single_ocp_with_terminal_return_terminal(x_init, N=100, terminal_model=None):
    """Like solve_single_ocp_with_terminal but also returns x_N."""
    opti = cs.Opti()
    X = [opti.variable(nx) for _ in range(N + 1)]
    U = [opti.variable(nu) for _ in range(N)]
    x_target = np.zeros(nx)
    cost = 0.0

    for k in range(N):
        q_err = X[k][:nq] - x_target[:nq]
        v_err = X[k][nq:] - x_target[nq:]
        cost += W_Q * cs.sumsqr(q_err)
        cost += W_V * cs.sumsqr(v_err)
        cost += W_U * cs.sumsqr(U[k])

        q_next_euler  = X[k][:nq] + dt * X[k][nq:]
        dq_next_euler = X[k][nq:] + dt * U[k]
        x_next_euler = cs.vertcat(q_next_euler, dq_next_euler)
        opti.subject_to(X[k+1] == x_next_euler)

    opti.subject_to(X[0] == x_init)

    if terminal_model is not None:
        term_cost = terminal_model(X[N])
        cost += term_cost

    opti.minimize(cost)

    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
        "ipopt.tol": 1e-4
    }
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
        xN = sol.value(X[N])
        return (x_init, sol.value(cost), np.array(xN).reshape(-1))
    except Exception:
        return None

def generate_random_state():
    q_min, q_max = -np.pi, np.pi
    dq_min, dq_max = -10.0, 10.0
    q_rand = np.random.uniform(q_min, q_max, nq)
    dq_rand = np.random.uniform(dq_min, dq_max, nq)
    return np.concatenate([q_rand, dq_rand])

def generate_grid_states(num_samples, nq, q_lims, dq_lims):
    """
    Genera una griglia uniforme di stati iniziali (q, dq) invece che casuali.
    Cerca di adattare il numero di punti per dimensione per avvicinarsi a num_samples.
    """
    # Dimensione totale dello stato
    nx = 2 * nq
    
    # Calcoliamo quanti punti per dimensione servono: N^(1/nx)
    points_per_dim = int(num_samples ** (1/nx))
    
    print(f"Generazione Griglia: {points_per_dim} punti per ogni dimensione di stato ({nx} dims).")
    
    # Creiamo i linspace per ogni dimensione
    # q tra -pi e pi (la circonferenza completa)
    q_ranges = [np.linspace(q_lims[0], q_lims[1], points_per_dim) for _ in range(nq)]
    # dq tra i limiti di velocità
    dq_ranges = [np.linspace(dq_lims[0], dq_lims[1], points_per_dim) for _ in range(nq)]
    
    # Uniamo tutti i range
    all_ranges = q_ranges + dq_ranges
    
    # Creiamo la meshgrid (prodotto cartesiano di tutti i range)
    mesh = np.meshgrid(*all_ranges)
    
    states_grid = np.vstack([m.flatten() for m in mesh]).T
    
    return states_grid


def compute_VN_pair(args):
    x, Nloc = args
    return solve_single_ocp(x, N=Nloc)


def collect_terminal_dataset(num_samples, M, N_long, grid=False, seed=None, processes=None):
    """Collect terminal states x_M from M-only solves and compute true V_N(x_M).
    Returns arrays (X_boot, Y_boot) where Y_boot = V_N(x_M).
    """
    if seed is not None:
        np.random.seed(seed)

    print(f"Collecting {num_samples} terminal states with M={M} and computing V_N with N={N_long}...")
    # Generate initial states
    if not grid:
        initial_states = [generate_random_state() for _ in range(num_samples)]
    else:
        # fallback: generate smaller grid if requested
        initial_states = [generate_random_state() for _ in range(num_samples)]

    # Compute xM using multiprocessing
    procs = processes or multiprocessing.cpu_count()
    print(f"Using {procs} processes to compute M-horizon terminals...")
    with multiprocessing.Pool(processes=procs) as pool:
        results = pool.map(solve_single_ocp_return_terminal, initial_states)

    valid = [r for r in results if r is not None]
    xM_list = [r[2] for r in valid]

    print(f"Collected {len(xM_list)} terminal states. Now computing true V_N for each xM (this will solve {len(xM_list)} OCPs of horizon N).")
    # Compute true V_N for each xM (can parallelize)
    with multiprocessing.Pool(processes=procs) as pool:
        VN_results = pool.map(compute_VN_pair, [(x, N_long) for x in xM_list])

    # Filter and assemble final dataset
    X_boot = []
    Y_boot = []
    for x, res in zip(xM_list, VN_results):
        if res is not None:
            X_boot.append(x)
            Y_boot.append(res[1])

    X_boot = np.array(X_boot)
    Y_boot = np.array(Y_boot)

    print(f"Final bootstrap dataset size: {len(Y_boot)}")
    return X_boot, Y_boot


def fine_tune_model(model, X_boot, Y_boot, epochs=200, batch_size=64, lr=1e-4, save_path='model_finetuned.pt'):
    """Fine-tune the given model on bootstrap (x, V) data.
    Saves new model to save_path and returns the fine-tuned model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    X = torch.tensor(X_boot, dtype=torch.float32)
    Y = torch.tensor(Y_boot, dtype=torch.float32).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        running = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            count += xb.size(0)
        epoch_loss = running / count
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = { 'model': model.state_dict(), 'ub': model.ub }
        if (epoch+1) % 50 == 0:
            print(f"Finetune Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    # Save best
    torch.save(best_state, save_path)
    print(f"Fine-tuned model saved to {save_path} with best loss {best_loss:.4f}")
    # Load state into model and return
    model.load_state_dict(best_state['model'])
    model.to(device)
    model.eval()
    return model


def solve_single_ocp_get_first_control(x_init, N=100, terminal_model=None):
    """Solve OCP and return the first control input (u0) and the predicted state trajectory.
    Returns (u0, predicted_traj) where predicted_traj is an array of shape (N+1, nx).
    On failure, returns (None, None)."""
    opti = cs.Opti()
    X = [opti.variable(nx) for _ in range(N + 1)]
    U = [opti.variable(nu) for _ in range(N)]
    x_target = np.zeros(nx)
    cost = 0.0

    for k in range(N):
        q_err = X[k][:nq] - x_target[:nq]
        v_err = X[k][nq:] - x_target[nq:]
        cost += W_Q * cs.sumsqr(q_err)
        cost += W_V * cs.sumsqr(v_err)
        cost += W_U * cs.sumsqr(U[k])

        q_next_euler = X[k][:nq] + dt * X[k][nq:]
        dq_next_euler = X[k][nq:] + dt * U[k]
        x_next_euler = cs.vertcat(q_next_euler, dq_next_euler)
        opti.subject_to(X[k + 1] == x_next_euler)

    opti.subject_to(X[0] == x_init)

    if terminal_model is not None:
        term_cost = terminal_model(X[N])
        cost += term_cost

    opti.minimize(cost)

    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
        "ipopt.tol": 1e-4,
    }
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
        u0 = np.array(sol.value(U[0])).reshape(-1)
        # Extract predicted state trajectory
        pred_traj = []
        for k in range(N + 1):
            try:
                xk = np.array(sol.value(X[k])).reshape(-1)
            except Exception:
                xk = np.zeros(nx)
            pred_traj.append(xk)
        pred_traj = np.array(pred_traj)
        # Extract predicted control sequence
        pred_us = []
        for k in range(N):
            try:
                uk = np.array(sol.value(U[k])).reshape(-1)
            except Exception:
                uk = np.zeros(nu)
            pred_us.append(uk)
        pred_us = np.array(pred_us)
        return u0, pred_traj, pred_us
    except Exception:
        return None, None, None


def simulate_mpc(x0, controller, tcost_model=None, M=20, N_long=100, T=None, tol=1e-3, verbose=False, env=None):
    """Simulate an MPC controller in closed-loop.

    controller: 'M' | 'M_term' | 'N+M'
    tcost_model: the pytorch model (needed only for 'M_term', as its l4casadi compiled function)
    T: number of simulation steps (default N_long + M)
    env: optional environment (e.g., Pendulum instance). If provided, its dynamics() will be used for stepping.

    Returns: dict with 'total_cost', 'trajectory' (list of states), 'controls' (list),
             'predicted_trajs' (list of predicted trajectories per step), 'predicted_us' (list of predicted control sequences),
             'reference_traj' (array of predicted next states corresponding to applied controls)
    """
    if T is None:
        T = N_long + M

    traj = [np.array(x0).reshape(-1)]
    u_list = []
    total_cost = 0.0

    # precompile l4 function when needed
    l4_term = None
    if controller == 'M_term':
        if tcost_model is None:
            raise ValueError('tcost_model required for M_term controller')
        l4_term = tcost_model.create_casadi_function()

    # if using env, reset its state
    if env is not None:
        try:
            env.reset(np.array(x0).reshape(-1))
            x = env.obs(env.x)
            # env.obs may return same shape; ensure 1-d
            x = np.array(env.x).reshape(-1)
        except Exception:
            x = np.array(x0).reshape(-1)
        # ensure env uses our dt if available
        try:
            env.DT = dt
        except Exception:
            pass
    else:
        x = np.array(x0).reshape(-1)

    start_time = time.time()
    predicted_trajs = []
    predicted_us = []
    for t in range(T):
        if controller == 'M':
            u0, pred, pred_u = solve_single_ocp_get_first_control(x, N=M, terminal_model=None)
        elif controller == 'M_term':
            u0, pred, pred_u = solve_single_ocp_get_first_control(x, N=M, terminal_model=l4_term)
        elif controller == 'N+M':
            u0, pred, pred_u = solve_single_ocp_get_first_control(x, N=N_long + M, terminal_model=None)
        else:
            raise ValueError('Unknown controller type')

        if u0 is None:
            if verbose:
                print(f"Step {t}: solver failed for controller {controller}; terminating simulation.")
            break

        # store predicted trajectory and predicted control sequence for logging
        predicted_trajs.append(pred)
        predicted_us.append(pred_u)

        # compute step cost and apply dynamics
        if env is not None:
            # env.dynamics modifies state in place and returns (x, reward)
            try:
                x_before = np.array(env.x).reshape(-1)
                _, r = env.dynamics(env.x, u0)
                # dynamics returns (x, -cost) so step_cost = -r
                step_cost = -r
                x = np.array(env.x).reshape(-1)
            except Exception as e:
                # fallback to Euler if env fails
                q = x[:nq]
                dq = x[nq:]
                step_cost = W_Q * np.sum(q ** 2) + W_V * np.sum(dq ** 2) + W_U * np.sum(u0 ** 2)
                q_next = q + dt * dq
                dq_next = dq + dt * u0
                x = np.concatenate([q_next, dq_next])
        else:
            q = x[:nq]
            dq = x[nq:]
            step_cost = W_Q * np.sum(q ** 2) + W_V * np.sum(dq ** 2) + W_U * np.sum(u0 ** 2)
            # apply dynamics (simple Euler discretization as used in OCP)
            q_next = q + dt * dq
            dq_next = dq + dt * u0
            x = np.concatenate([q_next, dq_next])

        total_cost += step_cost

        traj.append(x.copy())
        u_list.append(u0.copy())

        if np.linalg.norm(x) < tol:
            if verbose:
                print(f"Terminated at step {t} because state norm {np.linalg.norm(x):.4e} < tol")
            break
    end_time = time.time()
    print(f"Computed control with {controller} in {end_time - start_time:.4f} seconds.")

    # Build a reference trajectory as the predicted next-state at each step
    reference = [traj[0]]
    for i, pred in enumerate(predicted_trajs):
        if pred is not None and pred.shape[0] >= 2:
            reference.append(pred[1])
        else:
            # if missing, repeat last actual state
            reference.append(traj[min(i + 1, len(traj) - 1)])
    reference = np.array(reference)

    return {
        'total_cost': total_cost,
        'trajectory': np.array(traj),
        'controls': np.array(u_list),
        'predicted_trajs': predicted_trajs,
        'predicted_us': predicted_us,
        'reference_traj': reference,
    }


def simulate_batch(test_states, tcost_model, M=20, N_long=100, T=None, verbose=False, save_dir=None):
    """Run closed-loop simulation for each controller and each test state. Saves CSV if save_dir provided."""
    controllers = ['M', 'M_term', 'N+M']
    results = {c: [] for c in controllers}

    for idx, x0 in enumerate(test_states):
        if verbose:
            print(f"Simulating sample {idx+1}/{len(test_states)}")
        for c in controllers:
            res = simulate_mpc(x0, controller=c, tcost_model=tcost_model if c == 'M_term' else None, M=M, 
                               N_long=N_long, T=T, verbose=False)
            results[c].append(res['total_cost'])

    # convert to arrays
    for c in controllers:
        results[c] = np.array(results[c])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, 'closed_loop_comparison.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'cost_M', 'cost_M_term', 'cost_NM'])
            for i in range(len(test_states)):
                writer.writerow([i, results['M'][i], results['M_term'][i], results['N+M'][i]])
        print(f"Saved closed-loop comparison to {csv_path}")

    return results


def compare_mpcs(test_states, tcost_model, M=20, N_long=100, verbose=True, save_dir=None, diagnostic=False):
    """Run comparison of the three MPC formulations over a set of test states.
    If diagnostic=True, save per-sample terminal-state diagnostics (NN pred vs true V_N).
    Returns dict with arrays and relative errors. Optionally saves plots and CSV to save_dir.
    """
    # Compile l4 casadi terminal model for use inside the OCP
    l4_tcost = tcost_model.create_casadi_function()

    V_M = []
    V_M_term = []
    V_NM = []

    # For diagnostics
    diag_rows = []

    # device for pytorch model evaluation
    try:
        device = next(tcost_model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    for idx, s in enumerate(test_states):
        if verbose:
            print(f"  [{idx+1}/{len(test_states)}] solving...")

        if diagnostic:
            r1 = solve_single_ocp_return_terminal(s, N=M)
            r2 = solve_single_ocp_with_terminal_return_terminal(s, N=M, terminal_model=l4_tcost)
            r3 = solve_single_ocp_return_terminal(s, N=N_long + M)
        else:
            r1 = solve_single_ocp(s, N=M)
            r2 = solve_single_ocp_with_terminal(s, N=M, terminal_model=l4_tcost)
            r3 = solve_single_ocp(s, N=N_long + M)

        if r1 is None or r2 is None or r3 is None:
            if verbose:
                print("    One or more solves failed; skipping this sample.")
            continue

        if diagnostic:
            V_M.append(r1[1])
            V_M_term.append(r2[1])
            V_NM.append(r3[1])

            xM = r1[2]
            xM_term = r2[2]
            # Evaluate NN prediction at xM (use pytorch model)
            xt = torch.tensor(xM, dtype=torch.float32).to(device).unsqueeze(0)
            with torch.no_grad():
                nn_pred = float(tcost_model(xt).cpu().numpy().reshape(-1)[0])

            # Compute true N-long value starting from xM (this is costly but diagnostic)
            true_VN_from_xM = None
            r_true = solve_single_ocp(xM, N=N_long)
            if r_true is not None:
                true_VN_from_xM = r_true[1]

            diag_rows.append({
                'V_M': r1[1],
                'V_M_term': r2[1],
                'V_NM': r3[1],
                'xM': xM,
                'xM_term': xM_term,
                'nn_pred_xM': nn_pred,
                'true_VN_xM': true_VN_from_xM,
                'nn_err_at_xM': None if true_VN_from_xM is None else (nn_pred - true_VN_from_xM)
            })
        else:
            V_M.append(r1[1])
            V_M_term.append(r2[1])
            V_NM.append(r3[1])

    if len(V_M) == 0:
        print("No valid comparisons collected.")
        return None

    V_M = np.array(V_M)
    V_M_term = np.array(V_M_term)
    V_NM = np.array(V_NM)

    rel_err = np.abs(V_M_term - V_NM) / V_NM * 100.0

    print("\nBatch Comparison Results:")
    print(f"  Samples: {len(V_NM)}")
    print(f"  Mean rel err (V_M_term vs V_NM): {np.mean(rel_err):.2f}%")
    print(f"  Median rel err: {np.median(rel_err):.2f}%")
    within5 = np.mean(rel_err < 5.0) * 100.0
    print(f"  % within 5%: {within5:.1f}%")

    # Plot comparisons
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(V_NM, V_M, alpha=0.6, label='M vs N+M')
    ax1.scatter(V_NM, V_M_term, alpha=0.6, label='M+term vs N+M')
    mn = min(np.min(V_NM), np.min(V_M), np.min(V_M_term))
    mx = max(np.max(V_NM), np.max(V_M), np.max(V_M_term))
    ax1.plot([mn, mx], [mn, mx], 'k--')
    ax1.set_xlabel('V_{N+M}')
    ax1.set_ylabel('Other')
    ax1.legend()
    ax1.set_title('Scatter vs Reference V_{N+M}')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(rel_err, bins=30)
    ax2.set_xlabel('Relative Error (%)')
    ax2.set_title('Relative Error: V_M_term vs V_{N+M}')

    plt.tight_layout()

    # Save results if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        scatter_path = os.path.join(save_dir, 'scatter_vs_ref.png')
        fig.savefig(scatter_path)

        # Save data CSV
        csv_path = os.path.join(save_dir, 'comparison_results.csv')
        with open(csv_path, 'w', newline='') as _f:
            writer = csv.writer(_f)
            writer.writerow(['V_NM', 'V_M', 'V_M_term', 'rel_err'])
            for a, b, c, d in zip(V_NM, V_M, V_M_term, rel_err):
                writer.writerow([a, b, c, d])

        # Save arrays
        np.savez(os.path.join(save_dir, 'arrays.npz'), V_NM=V_NM, V_M=V_M, V_M_term=V_M_term, rel_err=rel_err)

        print(f"Saved results to {save_dir}")

    # Save diagnostics if requested
    if diagnostic and save_dir is not None and len(diag_rows) > 0:
        diag_csv = os.path.join(save_dir, 'diagnostics.csv')
        with open(diag_csv, 'w', newline='') as _f:
            writer = csv.writer(_f)
            writer.writerow(['V_M','V_M_term','V_NM','nn_pred_xM','true_VN_xM','nn_err_at_xM','xM','xM_term'])
            for row in diag_rows:
                writer.writerow([
                    row['V_M'], row['V_M_term'], row['V_NM'], row['nn_pred_xM'], row['true_VN_xM'], row['nn_err_at_xM'],
                    ' '.join(map(str, row['xM'])), ' '.join(map(str, row['xM_term']))
                ])
        print(f"Saved diagnostics to {diag_csv}")

    plt.show()

    return {'V_M': V_M, 'V_M_term': V_M_term, 'V_NM': V_NM, 'rel_err': rel_err, 'diag_rows': diag_rows}


def main(LOAD_DATA_PATH = None, LOAD_MODEL_PATH = None, GRID_SAMPLE = True, num_tests=100, M=20, N_long=100, save_dir=None, 
         seed=None, diagnostic=False, simulate=True, sim_tests=20, sim_T=None, sim_save_dir=None):
    
    x_data = None
    y_data = None
    if LOAD_DATA_PATH == None:
        print(f"Starting data generation with {NUM_SAMPLES} samples on {NUM_CORES} cores.")
        if not GRID_SAMPLE:
            # 1. Generate random initial states
            initial_states = [generate_random_state() for i in range(NUM_SAMPLES)]

        else:
            q_min, q_max = -np.pi, np.pi
            dq_min, dq_max = -10.0, 10.0
            
            # 1. Generate grid search
            initial_states_array = generate_grid_states(NUM_SAMPLES, nq, (q_min, q_max), (dq_min, dq_max))
            
            initial_states = [row for row in initial_states_array]

        # 2. Parallel Processing
        start_time = time.time()
        with multiprocessing.Pool(processes=NUM_CORES) as pool:
            results = pool.map(solve_single_ocp, initial_states)
        end_time = time.time()

        # 3. Filter valid results
        valid_data = [res for res in results if res is not None]
        
        if len(valid_data) == 0:
            print("Nessuna soluzione valida trovata. Controlla il solver o i vincoli.")
            return

        x_data = np.array([res[0] for res in valid_data])
        y_data = np.array([res[1] for res in valid_data])

        print(f'The dataset generation took {end_time - start_time:.2f} [s], with {len(valid_data)} valid solutions')
        
        # Salve dataset
        np.savez('model/value_function_data_grid.npz', x_init=x_data, V_opt=y_data)
    else:
        data = np.load(LOAD_DATA_PATH)
        x_data = data['x_init']
        y_data = data['V_opt']
    
    # Simple visualization of Cost distribution
    # try:
    #     plt.figure(figsize=(10, 5))
    #     plt.hist(y_data, bins=50)
    #     plt.title("Distribution of Optimal Costs V*(x)")
    #     plt.xlabel("Cost")
    #     plt.ylabel("Frequency")
    #     plt.grid(True, alpha=0.3)
    #     plt.show()
    # except ImportError:
    #     pass
    
    tcost_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = nx
    output_dim = 1

    # Set seed if provided
    if seed is not None:
        print(f"Setting random seed to {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    
    # Check if model exists
    if LOAD_MODEL_PATH is None:
        # Prefer model/model.pt but keep backward compatibility with model.pt
        if os.path.exists(os.path.join('model','model.pt')):
            LOAD_MODEL_PATH = os.path.join('model','model.pt')
        elif os.path.exists('model.pt'):
            LOAD_MODEL_PATH = 'model.pt'

    if LOAD_MODEL_PATH == None:
        print("Starting training...")
        tcost_model = train_network(x_data, y_data, lr=1e-3)
    else:
        print(f"Loading model from {LOAD_MODEL_PATH}")
        checkpoint = torch.load(LOAD_MODEL_PATH)
        model_state_dict = checkpoint['model']
        ub_val = checkpoint['ub']

        tcost_model = NeuralNetwork(input_dim, 64, output_dim, ub=ub_val).to(device)
        tcost_model.load_state_dict(model_state_dict)
        
    # --- COMPARISON LOGIC ---
    print("\n" + "="*30)
    print("RUNNING MPC COMPARISON")
    print("="*30)
    
    # M (short horizon) and N_long (long horizon used for training) are taken from function arguments
    
    # Create CasADi function for the network
    # We need to compile the l4casadi function
    
    l4_tcost = tcost_model.create_casadi_function()
    
    # If simulate-only mode: skip one-shot open-loop solves and batch comparisons
    if simulate:
        T_sim = sim_T if (sim_T is not None) else (N_long + M)
        test_state = generate_random_state()
        print(f"Simulate-only mode: running single closed-loop simulation from a random state (len={len(test_state)})")
        print(f"Test State: q={test_state[:nq]}, dq={test_state[nq:]}")
        controllers = ['M', 'M_term', 'N+M']
        sim_results = {}
        # directory to save plots/data
        save_dir_sim = sim_save_dir or save_dir
        if save_dir_sim:
            os.makedirs(save_dir_sim, exist_ok=True)
        for c in controllers:
            print(f"\nSimulating controller: {c}")
            # create env instance with realistic dynamics
            env = Pendulum(nq, open_viewer=False)
            env.DT = dt
            res = simulate_mpc(test_state, controller=c, tcost_model=tcost_model if c == 'M_term' else None, M=M, N_long=N_long, T=T_sim, verbose=True, env=env)
            if res is None:
                sim_results[c] = float('nan')
                print(f"  Solver failed for controller {c}; recorded NaN.")
                continue
            sim_results[c] = res['total_cost'] if res is not None else float('nan')
            print(f"  Total closed-loop cost ({c}): {sim_results[c]:.4f}")

            # Save trajectory, controls, predicted trajectories and predicted controls
            traj = np.array(res['trajectory'])  # shape (T_steps+1, nx)
            controls = np.array(res['controls'])  # shape (T_steps, nu)
            predicted_trajs = res.get('predicted_trajs', [])
            predicted_us = res.get('predicted_us', [])
            reference = res.get('reference_traj', None)

            t_states = np.arange(traj.shape[0]) * dt
            t_controls = np.arange(controls.shape[0]) * dt if controls.size else np.array([])

            if save_dir_sim:
                npz_path = os.path.join(save_dir_sim, f'closed_loop_{c}_data.npz')
                np.savez(npz_path, trajectory=traj, controls=controls, t_states=t_states, t_controls=t_controls, predicted_trajs=predicted_trajs, predicted_us=predicted_us, reference_traj=reference)
                print(f"  Saved data to {npz_path}")

                # Plot position, velocity, torque and reference
                try:
                    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
                    # Positions (actual vs reference)
                    for j in range(nq):
                        axs[0].plot(t_states, traj[:, j], label=f'q{j} (actual)')
                        if reference is not None:
                            axs[0].plot(t_states, reference[:, j], '--', label=f'q{j} (ref)')
                    axs[0].set_ylabel('position')
                    axs[0].legend(loc='best')
                    axs[0].grid(True)

                    # Velocities (actual vs reference)
                    for j in range(nq):
                        axs[1].plot(t_states, traj[:, nq + j], label=f'dq{j} (actual)')
                        if reference is not None:
                            axs[1].plot(t_states, reference[:, nq + j], '--', label=f'dq{j} (ref)')
                    axs[1].set_ylabel('velocity')
                    axs[1].legend(loc='best')
                    axs[1].grid(True)

                    # Torques (controls) and predicted first-step control
                    if controls.size:
                        for j in range(nu):
                            axs[2].plot(t_controls, controls[:, j], label=f'u{j} (applied)')
                    # predicted first-step control sequence
                    pred_u0_seq = []
                    for pu in predicted_us:
                        if pu is not None and pu.shape[0] > 0:
                            pred_u0_seq.append(pu[0])
                        else:
                            pred_u0_seq.append(np.zeros(nu))
                    pred_u0_seq = np.array(pred_u0_seq)
                    if pred_u0_seq.size:
                        t_pred = np.arange(pred_u0_seq.shape[0]) * dt
                        for j in range(nu):
                            axs[2].plot(t_pred, pred_u0_seq[:, j], '--', label=f'u{j} (pred u0)')

                    axs[2].set_ylabel('torque')
                    axs[2].set_xlabel('time [s]')
                    axs[2].legend(loc='best')
                    axs[2].grid(True)

                    fig.suptitle(f'Closed-loop {c}  Total cost: {sim_results[c]:.4f}')
                    fig.tight_layout(rect=[0, 0, 1, 0.96])
                    png_path = os.path.join(save_dir_sim, f'closed_loop_{c}_traj.png')
                    fig.savefig(png_path)
                    plt.close(fig)
                    print(f"  Saved plot to {png_path}")
                except Exception as e:
                    print(f"  Plotting failed for controller {c}: {e}")

        # summary
        print("\nSingle-state Closed-loop Summary:")
        for c in controllers:
            print(f"  {c:8s}: {sim_results.get(c, float('nan')):.4f}")
        # save summary CSV if requested
        if save_dir_sim:
            csv_path = os.path.join(save_dir_sim, 'closed_loop_single_state.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['controller', 'total_cost'])
                for c in controllers:
                    writer.writerow([c, sim_results.get(c, float('nan'))])
            print(f"Saved single-state closed-loop results to {csv_path}")
        return
    
    # # Pick a random state for comparison
    # test_state = generate_random_state()
    # # Print size and split into positions (q) and velocities (dq)
    # print(f"Test State (len={len(test_state)}): q={test_state[:nq]}, dq={test_state[nq:]}")
    
    # # 1. Horizon M without terminal cost
    # print(f"\n1. Horizon M={M} without terminal cost")
    # res1 = solve_single_ocp(test_state, N=M)
    # cost1 = res1[1] if res1 else float('nan')
    # print(f"   Cost: {cost1}")
    
    # # 2. Horizon M with NN terminal cost
    # print(f"\n2. Horizon M={M} with NN terminal cost")
    # res2 = solve_single_ocp_with_terminal(test_state, N=M, terminal_model=l4_tcost)
    # cost2 = res2[1] if res2 else float('nan')
    # print(f"   Cost: {cost2}")
    
    # # 3. Horizon N + M without terminal cost
    # NM = N_long + M
    # print(f"\n3. Horizon N+M={NM} without terminal cost")
    # res3 = solve_single_ocp(test_state, N=NM)
    # cost3 = res3[1] if res3 else float('nan')
    # print(f"   Cost: {cost3}")
    
    # print("\nComparison Summary:")
    # print(f"V_M       : {cost1:.4f}")
    # print(f"V_M_term  : {cost2:.4f}")
    # print(f"V_N+M     : {cost3:.4f}")
    
    # if cost3 != 0:
    #     print(f"Rel Error (V_M_term vs V_N+M): {abs(cost2 - cost3)/cost3 * 100:.2f}%")
    # else:
    #     print("Rel Error: reference cost is zero.")

    # --- Batch comparison (optional, may take long) ---
    if num_tests and num_tests > 0:
        print(f"\nRunning batch comparison on {num_tests} random states (this may take a while)...")
        test_states = [generate_random_state() for _ in range(num_tests)]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = save_dir or f'results_{timestamp}'
        batch_res = compare_mpcs(test_states, tcost_model, M=M, N_long=N_long, verbose=True, save_dir=results_dir, diagnostic=diagnostic)
        if batch_res is not None:
            mean_rel = np.mean(batch_res['rel_err'])
            print(f"Batch Mean Rel Error: {mean_rel:.2f}%")
            if diagnostic and len(batch_res.get('diag_rows', [])) > 0:
                # Quick aggregated diagnostic statistics
                errs = [r['nn_err_at_xM'] for r in batch_res['diag_rows'] if r['nn_err_at_xM'] is not None]
                errs = np.array(errs)
                print(f"Diagnostic on NN error at terminal states (nn_pred - true_VN): mean={np.mean(errs):.3f}, std={np.std(errs):.3f}")
    else:
        print("Skipping batch comparison (num_tests=0)")

    # --- Closed-loop simulation (optional) ---
    # if simulate:
    #     sim_n = sim_tests if (sim_tests and sim_tests > 0) else 20
    #     print(f"\nRunning closed-loop simulations on {sim_n} random states...")
    #     sim_states = [generate_random_state() for _ in range(sim_n)]
    #     timestamp_sim = datetime.now().strftime('%Y%m%d_%H%M%S')
    #     sim_save = sim_save_dir or save_dir or f'sim_results_{timestamp_sim}'
    #     sim_res = simulate_batch(sim_states, tcost_model, M=M, N_long=N_long, T=sim_T, verbose=True, save_dir=sim_save)
    #     # Print simple summary
    #     print('\nClosed-loop Summary (mean costs):')
    #     print(f"  M only mean:      {np.mean(sim_res['M']):.4f}")
    #     print(f"  M + terminal mean:{np.mean(sim_res['M_term']):.4f}")
    #     print(f"  N + M mean:       {np.mean(sim_res['N+M']):.4f}")
    # else:
    #     print('Skipping closed-loop simulations (use --simulate)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MPC comparison experiments')
    parser.add_argument('--load-data', type=str, default=None, help='Path to dataset (.npz)')
    parser.add_argument('--load-model', type=str, default=None, help='Path to model (.pt)')
    parser.add_argument('--no-grid', action='store_true', help='Disable grid dataset generation (use random)')
    parser.add_argument('--num-tests', type=int, default=0, help='Number of test states for batch comparison')
    parser.add_argument('--M', type=int, default=10, help='Short horizon M')
    parser.add_argument('--N-long', type=int, default=100, help='Long horizon N used in training')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save comparison results')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--diag', action='store_true', help='Run diagnostic (save terminal-state diagnostics)')
    parser.add_argument('--simulate', action='store_true', help='Run closed-loop MPC simulations')
    parser.add_argument('--sim-tests', type=int, default=20, help='Number of initial states for closed-loop simulation')
    parser.add_argument('--sim-T', type=int, default=None, help='Number of closed-loop steps (default N_long+M)')
    parser.add_argument('--sim-save-dir', type=str, default=None, help='Directory to save closed-loop simulation results')
    args = parser.parse_args()

    LOAD_DATA_PATH = args.load_data if args.load_data else ('model/value_function_data_grid.npz' if os.path.exists('model/value_function_data_grid.npz') else None)
    LOAD_MODEL_PATH = args.load_model if args.load_model else (
        (os.path.join('model','model.pt') if os.path.exists(os.path.join('model','model.pt')) else (
            'model.pt' if os.path.exists('model.pt') else None
        ))
    )
    DATASET_GENERATION_GRID = not args.no_grid

    main(LOAD_DATA_PATH, LOAD_MODEL_PATH, DATASET_GENERATION_GRID, num_tests=args.num_tests, M=args.M, N_long=args.N_long, 
         save_dir=args.save_dir, seed=args.seed, diagnostic=args.diag, simulate=args.simulate, sim_tests=args.sim_tests, 
         sim_T=args.sim_T, sim_save_dir=args.sim_save_dir)