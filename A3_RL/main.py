#!/usr/bin/env python3

import copy
import config

import numpy as np
import multiprocessing
import time
import os
import matplotlib.pyplot as plt
import argparse
import csv
import random
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from neural_network import train_network, NeuralNetwork
from config import NQ, NX, NU, N, DT, NUM_SAMPLES, NUM_CORES, ROBOT, T, PENDULUM, M, SEED, VELOCITY_LIMIT

# Determine model directory based on robot type
if PENDULUM == 'single_pendulum':
    MODEL_DIR = 'model_single'
else:
    MODEL_DIR = 'model_double'

def run_simulation_instance(args):
    if len(args) == 4:
        idx, tcost_model, seed, record_video = args
    else:
        idx, tcost_model, seed = args
        record_video = False

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
    
    unique_name = f"term_cost_{pid}_{idx}"
    l4_term = None
    try:
        tcost_model.cpu() 
        l4_term = tcost_model.create_casadi_function(name=unique_name)
    except Exception as e:
        print(f"Worker {pid}: Failed to create l4_model: {e}")

    controllers = {
        'M': {'horizon': M, 'terminal_cost': None},
        'M_term': {'horizon': M, 'terminal_cost': l4_term},
        'N+M': {'horizon': M+N, 'terminal_cost': None}
    }
    results = {}
    
    for name, params in controllers.items():
        try:
            res = simulate_mpc(test_state, params['horizon'], params['terminal_cost'], record_video=record_video)
        except Exception as e:
            res = None

        results[name] = res
        time.sleep(1)

    return idx, test_state, results

def solve_single_ocp(x_init):
    from ocp import solve_single_ocp as _solve
    return _solve(x_init)

def generate_random_state():
    from data import generate_random_state as _impl
    return _impl(q_min=[-np.pi] * NQ, q_max=[np.pi] * NQ, dq_max=VELOCITY_LIMIT)

def simulate_mpc(*args, **kwargs):
    from simulation import simulate_mpc as _impl
    return _impl(*args, **kwargs)

def main(LOAD_DATA_PATH = None, LOAD_MODEL_PATH = None, sim_tests=10, save=None):
    x_data = None
    y_data = None

    if LOAD_DATA_PATH == None:
        print(f"Starting data generation with {NUM_SAMPLES} samples on {NUM_CORES} cores.")

        # 1. Generate random initial states
        initial_states = [generate_random_state() for _ in range(NUM_SAMPLES)]
            
        # 2. Parallel Processing to solve OCP for each initial state
        start_time = time.time()
        with multiprocessing.Pool(processes=NUM_CORES) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        solve_single_ocp,
                        initial_states,
                        chunksize=16
                    ),
                    total=len(initial_states),
                    desc="Generating Data"
                )
            )
        end_time = time.time()

        # 3. Filter valid results (where solver converged)
        valid_data = [res for res in results if res is not None]
        
        if len(valid_data) == 0:
            print("No valid solution found.")
            return

        x_data = np.array([res[0] for res in valid_data]) # Initial states
        y_data = np.array([res[1] for res in valid_data]) # Optimal costs

        print(f'The dataset generation took {end_time - start_time:.2f} [s], with {len(valid_data)} valid solutions')

        os.makedirs(MODEL_DIR, exist_ok=True)
        np.savez(os.path.join(MODEL_DIR, f'dataset_{end_time - start_time:.2f}.npz'), x_init=x_data, V_opt=y_data)
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
    if SEED is not None:
        print(f"Setting random seed to {SEED}")
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        random.seed(SEED)
    
    # Check if model exists or needs to be trained
    if LOAD_MODEL_PATH is None:
        if os.path.exists(os.path.join(MODEL_DIR,'model.pt')):
            LOAD_MODEL_PATH = os.path.join(MODEL_DIR,'model.pt')
        elif os.path.exists('model.pt'):
            LOAD_MODEL_PATH = 'model.pt'

    if LOAD_MODEL_PATH == None:
        # Train neural network
        print("Starting training...")
        tcost_model = train_network(
            x_data, y_data, 
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            lr=config.LR,
            patience=config.PATIENCE,
            save_dir=MODEL_DIR
        )
    else:
        # Load Neural Network
        print(f"Loading model from {LOAD_MODEL_PATH}")
        checkpoint = torch.load(LOAD_MODEL_PATH, weights_only=True)
        model_state_dict = checkpoint['model']
        ub_val = checkpoint['ub']

        tcost_model = NeuralNetwork(input_dim, config.HIDDEN_SIZE, output_dim, ub=ub_val).to(device)
        tcost_model.load_state_dict(model_state_dict)
        
    # Simulation tests
    if sim_tests > 0:
        print("\n" + "="*30)
        print("RUNNING MPC SIMULATIONS")
        print("="*30)
        
        controllers = ['M', 'M_term', 'N+M']
        
        sim_results = {c: [] for c in controllers}
        sim_exec_times = {c: [] for c in controllers}
        
        # Directory to save plots/data
        if save:
            os.makedirs(save, exist_ok=True)
        
        # Ensure model is on CPU for multiprocessing sharing
        tcost_cpu = copy.deepcopy(tcost_model).cpu()
        
        sim_args = []
        for i in range(sim_tests):
            sim_args.append((i, tcost_cpu, SEED))
        
        print(f"Starting parallel simulations ({sim_tests}) on {NUM_CORES} cores...")
        
        # Execute simulations in parallel
        with multiprocessing.Pool(processes=NUM_CORES) as pool:
            results_list = list(tqdm(pool.imap(run_simulation_instance, sim_args), total=len(sim_args), desc="Running Simulations"))

        # Process results
        for i, (_, _, res_dict) in enumerate(results_list):
            
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
                
            try:
                # Create one figure with 9 subplots (3 rows x 3 cols assuming 3 controllers)
                # Rows: Position, Velocity, Acceleration
                # Cols: Controllers
                num_ctrls = len(controllers)
                fig, axs = plt.subplots(4, num_ctrls, figsize=(5 * num_ctrls, 10), sharex=True)
                
                # Ensure axs is 2D array even if num_ctrls=1
                if num_ctrls == 1:
                    axs = axs.reshape(4, 1)
                for idx, c in enumerate(controllers):
                    res = res_dict.get(c)
                    if res is None:
                            continue
                    
                    traj = np.array(res['trajectory'])
                    controls = np.array(res['controls'])
                    reference = res.get('reference_traj', None)
                    total_cost_plot = res.get('total_cost', float('nan'))

                    if traj.ndim == 1: traj = traj.reshape(1, -1) if len(traj.shape)>0 else traj 
                    if controls.ndim == 1: controls = controls.reshape(-1, 1)
                    
                    t_states = np.arange(traj.shape[0]) * DT
                    t_controls = np.arange(controls.shape[0]) * DT if controls.size else np.array([])
                    
                    # Row 0: Positions
                    for j in range(NQ):
                        axs[0, idx].plot(t_states, traj[:, j], label=f'q{j}')
                        if reference is not None:
                            axs[0, idx].plot(t_states, reference[:, j], '--', label=f'q{j} ref')
                    axs[0, idx].set_title(f'{c}\nCost: {float(total_cost_plot):.2f} - Time: {float(sim_exec_times[c][i]):.2f}s')
                    axs[0, idx].grid(True)
                    if idx == 0: axs[0, idx].set_ylabel('Position [rad]')
                    axs[0, idx].legend(loc='best', fontsize='small')

                    # Row 1: Velocities
                    for j in range(NQ):
                        axs[1, idx].plot(t_states, traj[:, NQ + j], label=f'dq{j}')
                        if reference is not None:
                            axs[1, idx].plot(t_states, reference[:, NQ + j], '--', label=f'dq{j} ref')
                    axs[1, idx].grid(True)
                    if idx == 0: axs[1, idx].set_ylabel('Velocity [rad/s]')
                    
                    # Row 2: Accelerations
                    if controls.size:
                        for j in range(NU):
                            axs[2, idx].plot(t_controls, controls[:, j], label=f'u{j}')
                    axs[2, idx].grid(True)
                    axs[2, idx].set_xlabel('Time [s]')
                    if idx == 0: axs[2, idx].set_ylabel('Acceleration [rad/s²]')
                    
                    # Raw 3: Torques (if available)
                    for j in range(NU):
                        if 'applied_torques' in res:
                            torques = np.array(res['applied_torques'])
                            if torques.ndim == 1:
                                torques = torques.reshape(-1, 1)
                            t_torques = np.arange(torques.shape[0]) * DT
                            axs[3, idx].plot(t_torques, torques[:, j], label=f'tau{j}')
                    axs[3, idx].grid(True)
                    axs[3, idx].set_xlabel('Time [s]')
                    if idx == 0: axs[3, idx].set_ylabel('Torque [Nm]')

                fig.suptitle(f'Comparison of Controllers at {i}-th iteration', fontsize=16)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                png_path = os.path.join(save, f'controllers_comparison_{i}.png')
                fig.savefig(png_path)
                plt.close(fig)
                # print(f"  Saved comparison plot to {png_path}")
            except Exception as e:
                print(f"  Plotting failed: {e}")

        # Summary Statistics
        print("\n" + "="*50)
        print("Closed-loop Simulation Summary")
        print("="*50)

        # Print summary statistics to console
        summary_stats_rows = []
        for c in controllers:
            costs = [val for val in sim_results[c] if not np.isnan(val)]
            times = [val for val in sim_exec_times[c] if not np.isnan(val)]
            success_rate = len(costs) / sim_tests * 100.0
            
            mean_cost = np.mean(costs) if costs else float('nan')
            std_cost = np.std(costs) if costs else float('nan')
            mean_time = np.mean(times) if times else float('nan')
            
            # Collect for saving later
            summary_stats_rows.append([c, success_rate, mean_cost, std_cost, mean_time])
            
            print(f"Controller: {c}")
            print(f"  Success Rate:     {success_rate:.1f}% ({len(costs)}/{sim_tests})")
            print(f"  Mean Cost:        {mean_cost:.4f}")
            print(f"  Std Cost:         {std_cost:.4f}")
            print(f"  Mean Exec Time:   {mean_time:.4f} s")
            print("-" * 30)
    
        if save:
            # Save summary CSV with detailed results for every simulation
            summary_csv_path = os.path.join(save, 'simulations_results.csv')
            with open(summary_csv_path, 'w', newline='') as f_csv:
                csv_writer = csv.writer(f_csv)
                csv_writer.writerow(['simulation_id', 'controller', 'cost', 'exec_time', 'success'])

                for i in range(sim_tests):
                    for c in controllers:
                        cost = sim_results[c][i]
                        exec_time = sim_exec_times[c][i]
                        success = not np.isnan(cost)
                        csv_writer.writerow([i, c, cost, exec_time, success])

            # Save summary statistics to CSV
            stats_csv_path = os.path.join(save, 'statistics_summary.csv')
            try:
                with open(stats_csv_path, 'w', newline='') as f_stats:
                    stats_writer = csv.writer(f_stats)
                    stats_writer.writerow(['controller', 'success_rate', 'mean_cost', 'std_cost', 'mean_time'])
                    stats_writer.writerows(summary_stats_rows)
            except Exception as e:
                print(f"Failed to save stats CSV: {e}")

            # Save summary statistics to PNG
            try:
                # Prepare data for formatting
                table_cell_text = []
                col_labels = ['Controller', 'Success Rate', 'Mean Cost', 'Std Cost', 'Mean Time']
                
                for row in summary_stats_rows:
                    c, success_rate, mean_cost, std_cost, mean_time = row
                    formatted_row = [
                        str(c),
                        f"{success_rate:.1f}%",
                        f"{mean_cost:.4f}",
                        f"{std_cost:.4f}",
                        f"{mean_time:.4f} s"
                    ]
                    table_cell_text.append(formatted_row)

                fig, ax = plt.subplots(figsize=(8, 2 + len(controllers)*0.5))
                ax.axis('off')
                ax.axis('tight')
                
                table = ax.table(cellText=table_cell_text, colLabels=col_labels, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.2)
                
                png_path = os.path.join(save, 'statistics_summary.png')
                plt.savefig(png_path, bbox_inches='tight', dpi=300)
                plt.close(fig)
                print(f"Summary PNG saved to {png_path}")
                
            except Exception as e:
                print(f"Could not save summary PNG: {e}")

            print(f"Summary saved to {summary_csv_path} and {stats_csv_path}")
        return
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MPC comparison experiments')
    parser.add_argument('--load-data', type=str, default=None, help='Path to dataset (.npz)')
    parser.add_argument('--load-model', type=str, default=None, help='Path to model (.pt)')
    parser.add_argument('--sim', type=int, default=0, help='Number of initial states for closed-loop simulation')
    parser.add_argument('--save', type=str, default=None, help='Directory to save closed-loop simulation results')
    args = parser.parse_args()
    
    LOAD_DATA_PATH = args.load_data
    if LOAD_DATA_PATH is None:
        if os.path.exists(MODEL_DIR):
            all_files = os.listdir(MODEL_DIR)
            npz_files = [f for f in all_files if f.endswith('.npz')]
            if npz_files:
                npz_files.sort()
                LOAD_DATA_PATH = os.path.join(MODEL_DIR, npz_files[-1])
                print(f"Auto-detected dataset: {LOAD_DATA_PATH}")

    LOAD_MODEL_PATH = args.load_model if args.load_model else (
        (os.path.join(MODEL_DIR,'model.pt') if os.path.exists(os.path.join(MODEL_DIR,'model.pt')) else (
            'model.pt' if os.path.exists('model.pt') else None
        ))
    )

    main(LOAD_DATA_PATH, LOAD_MODEL_PATH, sim_tests=args.sim, save=args.save)