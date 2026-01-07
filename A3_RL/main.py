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

# Simulation now uses joint_space_mpc.py; pendulum.py is deprecated.

# Allow selecting robot type via CLI flag '--robot-type' or the ROBOT_TYPE env var.
# If provided on the command line, set it early so `config.py` can pick it up when imported.
import sys
if 'ROBOT_TYPE' not in os.environ:
    for i, arg in enumerate(sys.argv):
        if arg == '--robot-type' and i + 1 < len(sys.argv):
            os.environ['ROBOT_TYPE'] = sys.argv[i + 1]
        elif arg.startswith('--robot-type='):
            os.environ['ROBOT_TYPE'] = arg.split('=', 1)[1]

# Configuration and constants have been moved to `config.py`
from config import ROBOT, NQ, NX, NU, TORQUE_LIMIT, ACTUATED_INDICES, N, DT, NUM_SAMPLES, NUM_CORES, W_Q, W_V, W_U
# OCP solvers are consolidated in `ocp.py` and simulation helpers in `simulation.py`
# Dataset helpers are in `data.py`
def solve_single_ocp(x_init, N = N):
    """Thin wrapper delegating to `ocp.solve_single_ocp`."""
    from ocp import solve_single_ocp as _solve
    return _solve(x_init, N=N)

def solve_single_ocp_with_terminal(x_init, N=100, terminal_model=None):
    """Wrapper delegating to `ocp.solve_single_ocp` with a terminal model."""
    from ocp import solve_ocp
    return solve_ocp(x_init, N=N, terminal_model=terminal_model)


def solve_single_ocp_return_terminal(x_init, N = N):
    from ocp import solve_single_ocp_return_terminal as _impl
    return _impl(x_init, N=N)


def solve_single_ocp_with_terminal_return_terminal(x_init, N=N, terminal_model=None):
    from ocp import solve_single_ocp_with_terminal_return_terminal as _impl
    return _impl(x_init, N=N, terminal_model=terminal_model)

def generate_random_state():
    from data import generate_random_state as _impl
    return _impl()

def generate_grid_states(num_samples, nq, q_lims, dq_lims):
    from data import generate_grid_states as _impl
    return _impl(num_samples, nq, q_lims, dq_lims)


def compute_VN_pair(args):
    x, Nloc = args
    return solve_single_ocp(x, N=Nloc)


def collect_terminal_dataset(num_samples, M, N_long, grid=False, seed=None, processes=None):
    from data import collect_terminal_dataset as _impl
    return _impl(num_samples, M, N_long, grid=grid, seed=seed, processes=processes)


def fine_tune_model(model, X_boot, Y_boot, epochs=200, batch_size=64, lr=1e-4, save_path='model_finetuned.pt'):
    from data import fine_tune_model as _impl
    return _impl(model, X_boot, Y_boot, epochs=epochs, batch_size=batch_size, lr=lr, save_path=save_path)


def solve_single_ocp_get_first_control(x_init, N=100, terminal_model=None):
    from ocp import solve_single_ocp_get_first_control as _impl
    return _impl(x_init, N=N, terminal_model=terminal_model)


def simulate_mpc(*args, **kwargs):
    from simulation import simulate_mpc as _impl
    return _impl(*args, **kwargs)


def simulate_batch(*args, **kwargs):
    # Not implemented in joint_space_mpc; raise if used
    raise NotImplementedError('simulate_batch is not implemented; use joint_space_mpc or write a wrapper')


def compare_mpcs(*args, **kwargs):
    # Not implemented in joint_space_mpc; raise if used
    raise NotImplementedError('compare_mpcs is not implemented; use joint_space_mpc or write a wrapper')

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
            dq_min, dq_max = -8.0, 8.0
            
            # 1. Generate grid search
            initial_states_array = generate_grid_states(NUM_SAMPLES, NQ, (q_min, q_max), (dq_min, dq_max))
            
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
        print(f'First 50 points of the dataset (x_init, V*):')
        for i in range(min(50, len(valid_data))):
            print(f"  x_init[{i}]: {x_data[i]}, V*[{i}]: {y_data[i]}")
        # Salve dataset
        np.savez('model/value_function_data_grid.npz', x_init=x_data, V_opt=y_data)
    else:
        data = np.load(LOAD_DATA_PATH)
        x_data = data['x_init']
        y_data = data['V_opt']
    
    tcost_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = NX
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
        tcost_model = train_network(x_data, y_data)
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
        print(f"Test State: q={test_state[:NQ]}, dq={test_state[NQ:]}")
        controllers = ['M', 'M_term', 'N+M']
        sim_results = {}
        # directory to save plots/data
        save_dir_sim = sim_save_dir or save_dir
        if save_dir_sim:
            os.makedirs(save_dir_sim, exist_ok=True)
        for c in controllers:
            print(f"\nSimulating controller: {c}")
            # create env instance with realistic dynamics
            # env = Pendulum(nq, open_viewer=False)
            # env.DT = dt
            # res = simulate_mpc(test_state, controller=c, tcost_model=tcost_model if c == 'M_term' else None, M=M, N_long=N_long, T=T_sim, verbose=True, env=env)
            res = simulate_mpc(test_state, controller=c, tcost_model=tcost_model if c == 'M_term' else None, M=M, N_long=N_long, T=T_sim, verbose=True)
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

            t_states = np.arange(traj.shape[0]) * DT
            t_controls = np.arange(controls.shape[0]) * DT if controls.size else np.array([])

            if save_dir_sim:
                npz_path = os.path.join(save_dir_sim, f'closed_loop_{c}_data.npz')
                np.savez(npz_path, trajectory=traj, controls=controls, t_states=t_states, t_controls=t_controls, predicted_trajs=predicted_trajs, predicted_us=predicted_us, reference_traj=reference)
                print(f"  Saved data to {npz_path}")

                # Plot position, velocity, torque and reference
                try:
                    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
                    # Positions (actual vs reference)
                    for j in range(NQ):
                        axs[0].plot(t_states, traj[:, j], label=f'q{j} (actual)')
                        if reference is not None:
                            axs[0].plot(t_states, reference[:, j], '--', label=f'q{j} (ref)')
                    axs[0].set_ylabel('position')
                    axs[0].legend(loc='best')
                    axs[0].grid(True)

                    # Velocities (actual vs reference)
                    for j in range(NQ):
                        axs[1].plot(t_states, traj[:, NQ + j], label=f'dq{j} (actual)')
                        if reference is not None:
                            axs[1].plot(t_states, reference[:, NQ + j], '--', label=f'dq{j} (ref)')
                    axs[1].set_ylabel('velocity')
                    axs[1].legend(loc='best')
                    axs[1].grid(True)

                    # Torques (controls) and predicted first-step control
                    if controls.size:
                        for j in range(NU):
                            axs[2].plot(t_controls, controls[:, j], label=f'u{j} (applied)')
                    # predicted first-step control sequence
                    pred_u0_seq = []
                    for pu in predicted_us:
                        if pu is not None and pu.shape[0] > 0:
                            pred_u0_seq.append(pu[0])
                        else:
                            pred_u0_seq.append(np.zeros(NU))
                    pred_u0_seq = np.array(pred_u0_seq)
                    if pred_u0_seq.size:
                        t_pred = np.arange(pred_u0_seq.shape[0]) * DT
                        for j in range(NU):
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
    parser.add_argument('--actuated-indices', type=str, default=None, help='Comma-separated list of actuated joint indices (e.g. 0 or 0,1)')
    parser.add_argument('--torque-limit', type=float, default=None, help='Per-joint torque limit (overrides pendulum.umax)')
    # parser.add_argument('--robot-type', type=str, default=None, help='Robot selection: pendulum (default) or double_pendulum')
    args = parser.parse_args()
    
    # LOAD_DATA_PATH = args.load_data if args.load_data else ('model/value_function_data_grid.npz' if os.path.exists('model/value_function_data_grid.npz') else None)

    # # If provided via CLI, set the environment variable so config.py can be consistent
    # if args.robot_type is not None:
    #     os.environ['ROBOT_TYPE'] = args.robot_type
    #     LOAD_MODEL_PATH = args.load_model if args.load_model else (
    #         (os.path.join('model','model.pt') if os.path.exists(os.path.join('model','model.pt')) else (
    #             'model.pt' if os.path.exists('model.pt') else None
    #         ))
    #     )
    # DATASET_GENERATION_GRID = not args.no_grid
    
    LOAD_DATA_PATH = args.load_data if args.load_data else ('model/value_function_data_grid.npz' if os.path.exists('model/value_function_data_grid.npz') else None)
    LOAD_MODEL_PATH = args.load_model if args.load_model else (
        (os.path.join('model','model.pt') if os.path.exists(os.path.join('model','model.pt')) else (
            'model.pt' if os.path.exists('model.pt') else None
        ))
    )
    DATASET_GENERATION_GRID = not args.no_grid

    # Override actuation settings if provided via CLI
    if args.actuated_indices is not None:
        try:
            ACTUATED_INDICES = [int(x.strip()) for x in args.actuated_indices.split(',') if x.strip()!='']
            # Validate indices
            ACTUATED_INDICES = [i for i in ACTUATED_INDICES if 0 <= i < NU]
            if len(ACTUATED_INDICES) == 0:
                raise ValueError('No valid actuated indices provided')
            print(f"Using actuated indices from CLI: {ACTUATED_INDICES}")
        except Exception as e:
            print(f"Failed to parse --actuated-indices; using default from config. Error: {e}")

    if args.torque_limit is not None:
        try:
            TORQUE_LIMIT = float(args.torque_limit)
            print(f"Using torque limit from CLI: {TORQUE_LIMIT}")
        except Exception:
            print("Failed to parse --torque-limit; using default from config")

    main(LOAD_DATA_PATH, LOAD_MODEL_PATH, DATASET_GENERATION_GRID, num_tests=args.num_tests, M=args.M, N_long=args.N_long, 
         save_dir=args.save_dir, seed=args.seed, diagnostic=args.diag, simulate=args.simulate, sim_tests=args.sim_tests, 
         sim_T=args.sim_T, sim_save_dir=args.sim_save_dir)