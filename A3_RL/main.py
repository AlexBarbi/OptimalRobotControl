#!/usr/bin/env python3

import numpy as np
import multiprocessing
import time
import os
import matplotlib.pyplot as plt
import argparse
import csv
import random
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from neural_network import train_network, NeuralNetwork


def run_simulation_instance(args):
    """
    Worker function to run a single simulation instance in parallel.
    
    This function initializes a random seed, generates a random initial state,
    creates a CasADi function for the terminal cost (if applicable), and runs
    simulations for different controllers:
    - 'M': MPC with short horizon M.
    - 'M_term': MPC with short horizon M and learned terminal cost.
    - 'N+M': MPC with long horizon (baseline).

    Args:
        args: A tuple containing:
            idx (int): Index of the simulation instance.
            seed (int): Base seed for random number generation.
            tcost_model (NeuralNetwork): The trained neural network for terminal cost.
            M (int): Short prediction horizon.
            N_long (int): Long prediction horizon (for baseline).
            T (float): Simulation duration.

    Returns:
        tuple: (idx, test_state, results) where results is a dictionary of simulation outcomes.
    """
    idx, seed, tcost_model, M, N_long, T = args
    # Reseed to ensure different random states across workers
    pid = os.getpid()
    if seed is not None:
        np.random.seed(seed + idx)
        random.seed(seed + idx)
        torch.manual_seed(seed + idx)
    else:
        s = int(time.time()*1000) % 100000 + pid
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)

    test_state = generate_random_state()
    
    # Create a unique name for the CasADi function to avoid collisions in parallel execution
    unique_name = f"term_cost_{pid}_{idx}"
    l4_term = None
    try:
        tcost_model.cpu() 
        # Create CasADi function from PyTorch model for efficient evaluation in OCP
        l4_term = tcost_model.create_casadi_function(name=unique_name)
    except Exception as e:
        print(f"Worker {pid}: Failed to create l4_model: {e}")

    controllers = ['M', 'M_term', 'N+M']
    results = {}
    
    for c in controllers:
        term_fn = None
        if c == 'M_term':
            if l4_term is None:
                res = None # Fail gracefully if terminal model creation failed
            else:
                term_fn = l4_term
        
        try:
            # delegated to simulation.simulate_mpc
            res = simulate_mpc(test_state, controller=c, terminal_cost_fn=term_fn, M=M, N_long=N_long, T=T, verbose=False)
        except Exception as e:
            res = None
        results[c] = res

    return idx, test_state, results

# Configuration and constants have been moved to `config.py`
from config import NQ, NX, NU, N, DT, NUM_SAMPLES, NUM_CORES, T, PENDULUM

# Determine model directory based on robot type
if PENDULUM == 'single_pendulum':
    MODEL_DIR = 'model_single'
else:
    MODEL_DIR = 'model_double'

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
    """
    Solves OCP and returns the optimal cost (Value function).
    Wrapper for `ocp.solve_single_ocp_return_terminal`.
    """
    from ocp import solve_single_ocp_return_terminal as _impl
    return _impl(x_init, N=N)


def solve_single_ocp_with_terminal_return_terminal(x_init, N=N, terminal_model=None):
    """
    Solves OCP with a terminal cost model and returns the optimal cost.
    Wrapper for `ocp.solve_single_ocp_with_terminal_return_terminal`.
    """
    from ocp import solve_single_ocp_with_terminal_return_terminal as _impl
    return _impl(x_init, N=N, terminal_model=terminal_model)

def generate_random_state():
    """Generates a random initial state. Wrapper for `data.generate_random_state`."""
    from data import generate_random_state as _impl
    return _impl()


def simulate_mpc(*args, **kwargs):
    """Runs an MPC simulation. Wrapper for `simulation.simulate_mpc`."""
    from simulation import simulate_mpc as _impl
    return _impl(*args, **kwargs)


def main(LOAD_DATA_PATH = None, LOAD_MODEL_PATH = None, M=20, N_long=100, 
         seed=None, simulate=True, sim_tests=20, sim_T=None, save=None):
    """
    Main function to orchestrate data generation, model training, and MPC simulation comparisons.

    Args:
        LOAD_DATA_PATH (str, optional): Path to existing dataset .npz file. If None, generates new data.
        LOAD_MODEL_PATH (str, optional): Path to existing model .pt file. If None, trains a new model.
        M (int): Prediction horizon for the short-horizon MPC.
        N (int): Prediction horizon for the long-horizon MPC (used as ground truth/baseline).
        seed (int, optional): Random seed for reproducibility.
        sim (bool): Whether to run closed-loop MPC simulations.
        Nsim (int): Number of simulation instances to run.
        save (str, optional): Directory to save simulation outcomes and plots.
    """
    
    x_data = None
    y_data = None
    if LOAD_DATA_PATH == None:
        print(f"Starting data generation with {NUM_SAMPLES} samples on {NUM_CORES} cores.")

        # 1. Generate random initial states
        initial_states = [generate_random_state() for i in range(NUM_SAMPLES)]
            
        # 2. Parallel Processing to solve OCP for each initial state (Value Function generation)
        start_time = time.time()
        with multiprocessing.Pool(processes=NUM_CORES) as pool:
            # Solves optimal control problem for each state to get V*(x)
            results = pool.map(solve_single_ocp, initial_states)
        end_time = time.time()

        # 3. Filter valid results (where solver converged)
        valid_data = [res for res in results if res is not None]
        
        if len(valid_data) == 0:
            print("Nessuna soluzione valida trovata. Controlla il solver o i vincoli.")
            return

        x_data = np.array([res[0] for res in valid_data]) # Initial states
        y_data = np.array([res[1] for res in valid_data]) # Optimal costs (Value function)

        print(f'The dataset generation took {end_time - start_time:.2f} [s], with {len(valid_data)} valid solutions')
        print(f'First 50 points of the dataset (x_init, V*):')
        for i in range(min(50, len(valid_data))):
            print(f"  x_init[{i}]: {x_data[i]}, V*[{i}]: {y_data[i]}")
        # Save dataset for future use
        os.makedirs(MODEL_DIR, exist_ok=True)
        np.savez(os.path.join(MODEL_DIR, 'value_function_data_grid.npz'), x_init=x_data, V_opt=y_data)
    else:
        # Load existing dataset
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
    
    # Check if model exists or needs to be trained
    if LOAD_MODEL_PATH is None:
        # Prefer MODEL_DIR/model.pt but keep backward compatibility with model.pt
        if os.path.exists(os.path.join(MODEL_DIR,'model.pt')):
            LOAD_MODEL_PATH = os.path.join(MODEL_DIR,'model.pt')
        elif os.path.exists('model.pt'):
            LOAD_MODEL_PATH = 'model.pt'

    if LOAD_MODEL_PATH == None:
        print("Starting training...")
        # Train neural network to approximate the Value function (Terminal Cost)
        tcost_model = train_network(x_data, y_data, save_dir=MODEL_DIR)
    else:
        print(f"Loading model from {LOAD_MODEL_PATH}")
        checkpoint = torch.load(LOAD_MODEL_PATH)
        model_state_dict = checkpoint['model']
        ub_val = checkpoint['ub'] # Normalization/scaling constant if used

        tcost_model = NeuralNetwork(input_dim, 64, output_dim, ub=ub_val).to(device)
        tcost_model.load_state_dict(model_state_dict)
        
    # --- COMPARISON LOGIC ---
    print("\n" + "="*30)
    print("RUNNING MPC COMPARISON")
    print("="*30)
    
    # If simulate-only mode: skip one-shot open-loop solves and batch comparisons
    if simulate:
        print(f"Simulate-only mode: running {sim_tests} closed-loop simulation(s).")
        controllers = ['M', 'M_term', 'N+M']
        
        sim_results = {c: [] for c in controllers}
        sim_exec_times = {c: [] for c in controllers}
        
        # directory to save plots/data
        if save:
            os.makedirs(save, exist_ok=True)
            
        import copy
        # Ensure model is on CPU for multiprocessing sharing (pickling)
        tcost_cpu = copy.deepcopy(tcost_model).cpu()
        
        sim_args = []
        T_val = sim_T if sim_T else T
        
        # Prepare arguments for parallel workers
        for i in range(sim_tests):
             seed_i = seed if seed is not None else None
             sim_args.append((i, seed_i, tcost_cpu, M, N_long, T_val))
        
        print(f"Starting parallel simulations ({sim_tests}) on {NUM_CORES} cores...")
        start_time_all = time.time()
        
        # Execute simulations in parallel
        with multiprocessing.Pool(processes=NUM_CORES) as pool:
             results_list = pool.map(run_simulation_instance, sim_args)
        
        print(f"Simulations completed in {time.time() - start_time_all:.2f}s.")
        
        # Process results
        for i, (_, _, res_dict) in enumerate(results_list):
            is_last = (i == len(results_list) - 1)
            
            for c in controllers:
                res = res_dict.get(c)
                if res is None:
                    # Record failure (NaN)
                    sim_results[c].append(float('nan'))
                    sim_exec_times[c].append(float('nan'))
                    continue
                
                total_cost = res.get('total_cost', float('nan'))
                exec_time_val = np.sum(res['exec_time']) if (res is not None and 'exec_time' in res) else float('nan')
                
                sim_results[c].append(total_cost)
                sim_exec_times[c].append(exec_time_val)
                
                # Only save detailed trajectory data for the LAST simulation to avoid clutter
                if is_last:
                     # Save trajectory, controls, etc.
                     traj = np.array(res['trajectory'])
                     controls = np.array(res['controls'])
                     predicted_trajs = res.get('predicted_trajs', [])
                     predicted_us = res.get('predicted_us', [])
                     reference = res.get('reference_traj', None)
                     
                     if traj.ndim == 1: traj = traj.reshape(1, -1) if len(traj.shape)>0 else traj 
                     if controls.ndim == 1: controls = controls.reshape(-1, 1)

                     t_states = np.arange(traj.shape[0]) * DT
                     t_controls = np.arange(controls.shape[0]) * DT if controls.size else np.array([])
                     
                     if save:
                        npz_path = os.path.join(save, f'closed_loop_{c}_data_last.npz')
                        np.savez(npz_path, trajectory=traj, controls=controls, t_states=t_states, t_controls=t_controls, predicted_trajs=predicted_trajs, predicted_us=predicted_us, reference_traj=reference)
                        print(f"  Saved last simulation data to {npz_path}")

                        try:
                            # Generate comparison plots for positions, velocities, and torques
                            fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
                            # Positions
                            for j in range(NQ):
                                axs[0].plot(t_states, traj[:, j], label=f'q{j} (actual)')
                                if reference is not None:
                                    axs[0].plot(t_states, reference[:, j], '--', label=f'q{j} (ref)')
                            axs[0].set_ylabel('position')
                            axs[0].legend(loc='best')
                            axs[0].grid(True)
                            # Velocities
                            for j in range(NQ):
                                axs[1].plot(t_states, traj[:, NQ + j], label=f'dq{j} (actual)')
                                if reference is not None:
                                    axs[1].plot(t_states, reference[:, NQ + j], '--', label=f'dq{j} (ref)')
                            axs[1].set_ylabel('velocity')
                            axs[1].legend(loc='best')
                            axs[1].grid(True)
                            # Torques
                            if controls.size:
                                for j in range(NU):
                                    axs[2].plot(t_controls, controls[:, j], label=f'u{j} (applied)')
                            axs[2].set_ylabel('torque')
                            axs[2].set_xlabel('time [s]')
                            axs[2].legend(loc='best')
                            axs[2].grid(True)

                            fig.suptitle(f'Closed-loop {c} (Last Sim) Total cost: {total_cost:.4f}')
                            fig.tight_layout(rect=[0, 0, 1, 0.96])
                            png_path = os.path.join(save, f'closed_loop_{c}_traj_last.png')
                            fig.savefig(png_path)
                            plt.close(fig)
                            print(f"  Saved plot to {png_path}")
                        except Exception as e:
                            print(f"  Plotting failed for controller {c}: {e}")

        # Summary Statistics
        print("\n" + "="*50)
        print("Closed-loop Simulation Summary")
        print("="*50)
        
        if save:
            # Save summary CSV with success rates, mean costs, and execution times
            summary_csv_path = os.path.join(save, 'closed_loop_summary.csv')
            with open(summary_csv_path, 'w', newline='') as f_csv:
                csv_writer = csv.writer(f_csv)
                csv_writer.writerow(['controller', 'success_rate', 'mean_cost', 'std_cost', 'mean_exec_time'])
                
                for c in controllers:
                    costs = [val for val in sim_results[c] if not np.isnan(val)]
                    times = [val for val in sim_exec_times[c] if not np.isnan(val)]
                    success_rate = len(costs) / sim_tests * 100.0
                    
                    mean_cost = np.mean(costs) if costs else float('nan')
                    std_cost = np.std(costs) if costs else float('nan')
                    mean_time = np.mean(times) if times else float('nan')
                    
                    print(f"Controller: {c}")
                    print(f"  Success Rate:     {success_rate:.1f}% ({len(costs)}/{sim_tests})")
                    print(f"  Mean Cost:        {mean_cost:.4f}")
                    print(f"  Std Cost:         {std_cost:.4f}")
                    print(f"  Mean Exec Time:   {mean_time:.4f} s")
                    print("-" * 30)
                    
                    csv_writer.writerow([c, success_rate, mean_cost, std_cost, mean_time])
            print(f"Summary saved to {summary_csv_path}")
        return
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MPC comparison experiments')
    parser.add_argument('--load-data', type=str, default=None, help='Path to dataset (.npz)')
    parser.add_argument('--load-model', type=str, default=None, help='Path to model (.pt)')
    parser.add_argument('--M', type=int, default=10, help='Short horizon M')
    parser.add_argument('--N', type=int, default=100, help='Long horizon N used in training')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--sim', action='store_true', help='Run closed-loop MPC simulations')
    parser.add_argument('--Nsim', type=int, default=10, help='Number of initial states for closed-loop simulation')
    parser.add_argument('--save', type=str, default=None, help='Directory to save closed-loop simulation results')
    parser.add_argument('--robot', type=str, default=None, help='Robot selection: double_pendulum or single_pendulum')
    args = parser.parse_args()
    
    LOAD_DATA_PATH = args.load_data if args.load_data else (os.path.join(MODEL_DIR, 'value_function_data_grid.npz') if os.path.exists(os.path.join(MODEL_DIR, 'value_function_data_grid.npz')) else None)
    LOAD_MODEL_PATH = args.load_model if args.load_model else (
        (os.path.join(MODEL_DIR,'model.pt') if os.path.exists(os.path.join(MODEL_DIR,'model.pt')) else (
            'model.pt' if os.path.exists('model.pt') else None
        ))
    )

    main(LOAD_DATA_PATH, LOAD_MODEL_PATH, M=args.M, N_long=args.N, seed=args.seed, 
         simulate=args.sim, sim_tests=args.Nsim, save=args.save)