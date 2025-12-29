#!/usr/bin/env python3

import numpy as np
import casadi as cs
import multiprocessing
import time
from time import sleep
import os
import matplotlib.pyplot as plt

from adam.casadi.computations import KinDynComputations
from example_robot_data.robots_loader import load
import orc.optimal_control.casadi_adam.conf_ur5 as conf_ur5
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.viz_utils import addViewerSphere, applyViewerConfiguration

from neural_network import train_network, NeuralNetwork
import torch
import l4casadi as l4

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

def generate_random_state():
    q_min, q_max = -np.pi, np.pi
    dq_min, dq_max = -2.0, 2.0
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

def main(LOAD_DATA_PATH = None, LOAD_MODEL_PATH = None, GRID_SAMPLE = True):
    
    x_data = None
    y_data = None
    if LOAD_DATA_PATH == None:
        print(f"Starting data generation with {NUM_SAMPLES} samples on {NUM_CORES} cores.")
        if not GRID_SAMPLE:
            # 1. Generate random initial states
            initial_states = [generate_random_state() for i in range(NUM_SAMPLES)]

        else:
            q_min, q_max = -np.pi, np.pi
            dq_min, dq_max = -2.0, 2.0
            
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
        np.savez('value_function_data_grid.npz', x_init=x_data, V_opt=y_data)
    else:
        data = np.load(LOAD_DATA_PATH)
        x_data = data['x_init']
        y_data = data['V_opt']
    
    # Simple visualization of Cost distribution
    try:
        plt.figure(figsize=(10, 5))
        plt.hist(y_data, bins=50)
        plt.title("Distribution of Optimal Costs V*(x)")
        plt.xlabel("Cost")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.show()
    except ImportError:
        pass
    
    tcost_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = nx
    output_dim = 1
    
    # Check if model exists
    if LOAD_MODEL_PATH is None and os.path.exists('model.pt'):
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
    
    # M (short horizon)
    M = 20
    # N (long horizon used for training) - from config N=100
    N_long = 100
    
    # Create CasADi function for the network
    # We need to compile the l4casadi function
    
    l4_tcost = tcost_model.create_casadi_function()
    
    # Pick a random state for comparison
    test_state = generate_random_state()
    print(f"Test State: {test_state}")
    
    # 1. Horizon M without terminal cost
    print(f"\n1. Horizon M={M} without terminal cost")
    res1 = solve_single_ocp(test_state, N=M)
    cost1 = res1[1] if res1 else float('nan')
    print(f"   Cost: {cost1}")
    
    # 2. Horizon M with NN terminal cost
    print(f"\n2. Horizon M={M} with NN terminal cost")
    res2 = solve_single_ocp_with_terminal(test_state, N=M, terminal_model=l4_tcost)
    cost2 = res2[1] if res2 else float('nan')
    print(f"   Cost: {cost2}")
    
    # 3. Horizon N + M without terminal cost
    NM = N_long + M
    print(f"\n3. Horizon N+M={NM} without terminal cost")
    res3 = solve_single_ocp(test_state, N=NM)
    cost3 = res3[1] if res3 else float('nan')
    print(f"   Cost: {cost3}")
    
    print("\nComparison Summary:")
    print(f"V_M       : {cost1:.4f}")
    print(f"V_M_term  : {cost2:.4f}")
    print(f"V_N+M     : {cost3:.4f}")
    
    print(f"Rel Error (V_M_term vs V_N+M): {abs(cost2 - cost3)/cost3 * 100:.2f}%")

if __name__ == "__main__":
    LOAD_DATA_PATH = 'value_function_data_grid.npz' if os.path.exists('value_function_data_grid.npz') else None
    LOAD_MODEL_PATH = 'model.pt' if os.path.exists('model.pt') else None
    
    DATASET_GENERATION_GRID = True
    main(LOAD_DATA_PATH, LOAD_MODEL_PATH, DATASET_GENERATION_GRID)