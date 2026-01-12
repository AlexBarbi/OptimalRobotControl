"""Simulation helpers: closed-loop simulation and batch comparison.

This module provides the core `simulate_mpc` function, which implements the
Model Predictive Control (MPC) loop. It sets up the optimization problem
(using CasADi), interfaces with the robot simulator (Pinocchio or MuJoCo),
and runs the closed-loop simulation.
"""
import numpy as np
import os
import casadi as cs
from time import time as clock
from termcolor import colored
from example_robot_data.robots_loader import load
from adam.casadi.computations import KinDynComputations
import pinocchio as pin

from config import DT, W_Q, W_V, W_U, T, PENDULUM
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.robot_wrapper import RobotWrapper

# Import constraint helper if available in your local ocp.py
try:
    from ocp import _enforce_actuation
except ImportError:
    def _enforce_actuation(opti, u):
        pass

def simulate_mpc(x0, controller, tcost_model=None, terminal_cost_fn=None, M=20, N_long=100, T=T, tol=1e-3, verbose=False, steady_time=0.5, steady_error_tol=1e-2):
    """
    Runs a closed-loop MPC simulation for a given controller configuration.

    This function constructs and solves an Optimal Control Problem (OCP) at each
    time step, applies the first control action, steps the simulator, and repeats.
    It supports three controller modes:
    - 'M': Standard MPC with a short horizon M.
    - 'M_term': MPC with a short horizon M and a learned terminal cost function.
    - 'N+M' (or other): Baseline MPC with a long horizon (N_long + M).

    Args:
        x0 (np.ndarray): Initial state vector [q, dq].
        controller (str): Controller type ('M', 'M_term', or 'N+M').
        tcost_model (NeuralNetwork, optional): PyTorch model for terminal cost (used if controller='M_term').
        terminal_cost_fn (casadi.Function, optional): Pre-compiled CasADi function for terminal cost.
        M (int, optional): Short prediction horizon length. Defaults to 20.
        N_long (int, optional): Long prediction horizon length (for baseline). Defaults to 100.
        T (int, optional): Total simulation duration in time steps. Defaults to global T.
        tol (float, optional): Solver tolerance. Defaults to 1e-3.
        verbose (bool, optional): Whether to print debug info. Defaults to False.
        steady_time (float, optional): Time (in seconds) to stay within error tolerance to trigger early stop.
        steady_error_tol (float, optional): Error tolerance for early stopping.

    Returns:
        dict: A dictionary containing simulation results:
            - 'total_cost': Sum of running costs.
            - 'trajectory': Array of visited states (T+1 x NX).
            - 'controls': Array of applied controls (T x NU).
            - 'predicted_trajs': List of predicted trajectories at each step.
            - 'predicted_us': List of predicted control sequences at each step.
            - 'reference_traj': Reference trajectory (constructed from predictions).
            - 'stopped_early': Boolean indicating if simulation stopped due to convergence.
            - 'stop_time': Time at which simulation stopped (if stopped_early).
            - 'exec_time': List of solver execution times per step.
    """
    print("Load robot model")
    # Load the double pendulum
    if PENDULUM == 'single_pendulum':
        urdf_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(urdf_dir, 'single_pendulum_description/urdf/single_pendulum.urdf')
        robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[urdf_dir])
        robot.urdf = urdf_path
    else:
        robot = load(PENDULUM)

    print("Create KinDynComputations object")
    joints_name_list = [s for s in robot.model.names[1:]] # skip the first name because it is "universe"
    nq = len(joints_name_list)  # number of joints
    nx = 2*nq # size of the state variable
    nu = nq  # size of the control input
    kinDyn = KinDynComputations(robot.urdf, joints_name_list)

    ADD_SPHERE = 0
    SPHERE_POS = np.array([0.2, -0.10, 0.5])
    SPHERE_SIZE = np.ones(3)*0.1
    SPHERE_RGBA = np.array([1, 0, 0, 1.])

    DO_WARM_START = True
    SOLVER_TOLERANCE = 1e-4
    SOLVER_MAX_ITER = 1000

    SIMULATOR = "pinocchio" #"mujoco" or "pinocchio" or "ideal"
    POS_BOUNDS_SCALING_FACTOR = 0.2
    qMin = POS_BOUNDS_SCALING_FACTOR * robot.model.lowerPositionLimit
    qMax = POS_BOUNDS_SCALING_FACTOR * robot.model.upperPositionLimit
    dt_sim = DT
    # N_sim = N
    q0 = x0[:nq]  # initial joint configuration
    dq0= x0[nq:]  # initial joint velocities

    # MPC parameters
    # q_des = np.array([0.0, np.pi])
    if nq == 1:
        q_des = np.array([0.0])
    else:
        q_des = np.array([0.0, np.pi])

    J = 1
    # Check if J is within bounds for this robot (Double pendulum has nq=2)
    if J < nq:
        q_des[J] = qMin[J] + 0.01*(qMax[J] - qMin[J])
        
    w_p = W_Q   # position weight
    w_v = W_V  # velocity weight
    w_a = W_U  # acceleration weight
    
    # Initialize Simulator
    if(SIMULATOR=="mujoco"):
        from orc.utils.mujoco_simulator import MujocoSimulator
        print("Creating simulator...")
        simu = MujocoSimulator("ur5", dt_sim) 
        simu.set_state(q0, dq0)
    else:
        r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
        
        # --- FIX: Create a COMPREHENSIVE dummy configuration object ---
        class DummyConf:
            q0 = np.zeros(nq)          # Correct size for Double Pendulum (2)
            dt = dt_sim                # Time step
            
            # Physics / Simulation flags
            randomize_robot_model = False
            model_variation = 0.0
            simulation_type = 'euler' # 'timestepping' or 'euler'
            
            # Friction parameters
            tau_coulomb_max = np.zeros(nq) 
            tau_viscous = np.zeros(nq)     
            
            # Viewer / Visualization flags
            use_viewer = False          # Enable viewer
            which_viewer = 'meshcat'   # 'gepetto' or 'meshcat'
            viewer_name = "robot_simulator"
            show_floor = False         # Often checked in viewer init
            DISPLAY_T = dt_sim         # Refresh period for viewer
            frame_name = "ee_link"     # Frame to track (usually end-effector)
            
        simu = RobotSimulator(DummyConf, r)
        simu.init(q0, dq0)
        simu.display(q0)
        
    print("Create optimization parameters")
    opti = cs.Opti()
    param_x_init = opti.parameter(nx) # Initial state parameter (updated every MPC step)
    param_q_des = opti.parameter(nq)  # Desired position parameter
    cost = 0

    # ---------------------------------------------------------
    # System Dynamics & Integration
    # ---------------------------------------------------------
    # create the dynamics function
    q   = cs.SX.sym('q', nq)
    dq  = cs.SX.sym('dq', nq)
    ddq = cs.SX.sym('ddq', nq)
    state = cs.vertcat(q, dq)
    rhs    = cs.vertcat(dq, ddq)
    f = cs.Function('f', [state, ddq], [rhs]) # Simple integrator f(x, u) -> x_next

    # create a Casadi inverse dynamics function (Pinocchio interface)
    H_b = cs.SX.eye(4)     # base configuration (identity for fixed base)
    v_b = cs.SX.zeros(6)   # base velocity (zero for fixed base)
    bias_forces = kinDyn.bias_force_fun()
    mass_matrix = kinDyn.mass_matrix_fun()
    
    # discard the first 6 elements because they are associated to the robot base (which is fixed)
    h = bias_forces(H_b, q, v_b, dq)[6:]
    MM = mass_matrix(H_b, q)[6:,6:]
    tau = MM @ ddq + h
    inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

    # ---------------------------------------------------------
    # Decision Variables
    # ---------------------------------------------------------
    X, U = [], []
    X += [opti.variable(nx)] # do not apply pos/vel bounds on initial state
    
    # Determine the horizon length based on the controller type
    if controller == 'M':
        horizon = M
    elif controller == 'M_term':
        horizon = M
    else:
        horizon = N_long + M
    
    # Create variables for the prediction horizon
    # --- FIX: These loops were commented out, causing X and U to be too short/empty ---
    for k in range(1, horizon+1): 
        X += [opti.variable(nx)]
    for k in range(horizon): 
        U += [opti.variable(nq)]
    # ---------------------------------------------------------------------------------

    print("Add initial conditions")
    opti.subject_to(X[0] == param_x_init)
    
    # Cost function and Dynamics constraints
    for k in range(horizon):     
        # Quadratic Running Cost
        cost += w_p * (X[k][:nq] - param_q_des).T @ (X[k][:nq] - param_q_des)
        cost += w_v * X[k][nq:].T @ X[k][nq:]
        cost += w_a * U[k].T @ U[k]

        # Dynamics constraint: x_{k+1} = x_k + dt * f(x_k, u_k)
        # Note: U[k] here represents acceleration (ddq), not torque. 
        # The torque limit is enforced via _enforce_actuation which likely maps ddq -> tau
        opti.subject_to(X[k+1] == X[k] + DT * f(X[k], U[k]))
        
        # Actuation limits (torque limits)
        _enforce_actuation(opti, U[k])

    # Optional terminal cost
    if controller == 'M_term':
        # Add learned terminal cost V(x_N)
        if terminal_cost_fn is not None:
             cost += terminal_cost_fn(X[horizon])
        elif tcost_model is not None:
             l4_term = tcost_model.create_casadi_function()
             cost += l4_term(X[horizon])
        else:
            raise ValueError('tcost_model or terminal_cost_fn required for M_term controller')

    opti.minimize(cost)

    print("Create the optimization problem")
    opts = {
        "error_on_fail": False,
        "ipopt.print_level": 0,
        "ipopt.tol": SOLVER_TOLERANCE,
        "ipopt.constr_viol_tol": SOLVER_TOLERANCE,
        "ipopt.compl_inf_tol": SOLVER_TOLERANCE,
        "print_time": 0,                # print information about execution time
        "detect_simple_bounds": True,
        "ipopt.max_iter": 1000
    }
    opti.solver("ipopt", opts)

    # set up simulation environment
    if(SIMULATOR=="mujoco" and ADD_SPHERE):
        simu.add_sphere(pos=SPHERE_POS, size=SPHERE_SIZE, rgba=SPHERE_RGBA)

    # Solve the problem to convergence the first time
    x = np.concatenate([q0, dq0])
    opti.set_value(param_q_des, q_des)
    opti.set_value(param_x_init, x)
    sol = opti.solve()
    opts["ipopt.max_iter"] = SOLVER_MAX_ITER
    opti.solver("ipopt", opts)

    # ---------------------------------------------------------
    # Initialize Data Logging
    # ---------------------------------------------------------
        
    traj = [x.copy()]   # Store initial state
    u_list = []
    predicted_trajs = []
    predicted_us = []
    total_cost = 0.0
    
    exec_time = []

    # Early stop if tracking error remains below steady_error_tol for steady_time seconds
    stopped_early = False
    stop_iter = None
    steady_counter = 0
    steady_steps_required = max(1, int(np.ceil(steady_time / DT))) if steady_time and steady_time > 0 else None

    print("Start the MPC loop")
    for i in range(T):
        # print("\n--- MPC Iteration %d ---"%i)
        start_time = clock()

        if(DO_WARM_START):
            # use current solution as initial guess for next problem
            for t in range(horizon):
                opti.set_initial(X[t], sol.value(X[t+1]))
            for t in range(horizon-1):
                opti.set_initial(U[t], sol.value(U[t+1]))
            opti.set_initial(X[horizon], sol.value(X[horizon]))
            opti.set_initial(U[horizon-1], sol.value(U[horizon-1]))
            # initialize dual variables
            lam_g0 = sol.value(opti.lam_g)
            opti.set_initial(opti.lam_g, lam_g0)
        
        # print("Time step", i, "State", x)
        opti.set_value(param_x_init, x)
        try:
            sol = opti.solve()
        except:
            sol = opti.debug
        
        end_time = clock()
        exec_time.append(end_time - start_time)
        
        # ---------------------------------------------------------
        # Extract Predictions and Controls
        # ---------------------------------------------------------
        # Extract predicted trajectory (Horizon+1 x State Dim)
        pred_x = np.array([sol.value(X[k]) for k in range(horizon+1)])
        # Extract predicted controls (Horizon x Control Dim)
        pred_u = np.array([sol.value(U[k]) for k in range(horizon)])
        
        predicted_trajs.append(pred_x)
        predicted_us.append(pred_u)
        
        # Get control input to apply (first element of solution)
        u_applied = pred_u[0]
        u_list.append(u_applied)

        # ---------------------------------------------------------
        # Compute Running Cost
        # ---------------------------------------------------------
        curr_q = x[:nq]
        curr_v = x[nq:]
        step_cost = w_p * np.sum((curr_q - q_des)**2) + \
                    w_v * np.sum(curr_v**2) + \
                    w_a * np.sum(u_applied**2)
        total_cost += step_cost

        # Check for sustained near-zero tracking error and stop simulation if met
        error_norm = np.linalg.norm(curr_q - q_des)
        if steady_steps_required is not None:
            if error_norm < steady_error_tol:
                steady_counter += 1
            else:
                steady_counter = 0
            if steady_counter >= steady_steps_required:
                stopped_early = True
                stop_iter = i
                # If we have a predicted next state, append it so trajectory and reference lengths match
                try:
                    if 'pred_x' in locals() and pred_x is not None and pred_x.shape[0] >= 2:
                        next_state = pred_x[1]
                        traj.append(next_state.copy())
                except Exception:
                    pass
                if verbose:
                    print(colored(f"Stopping simulation at step {i} (t={i*DT:.3f}s): error {error_norm:.3e} < {steady_error_tol} for {steady_time}s", "green"))
                break

        # ---------------------------------------------------------
        # Step Simulator
        # ---------------------------------------------------------
        tau = inv_dyn(sol.value(X[0]), sol.value(U[0])).toarray().squeeze()
        
        if(SIMULATOR=="mujoco"):
            simu.step(tau, DT)
            x = np.concatenate([simu.data.qpos, simu.data.qvel])
        elif(SIMULATOR=="pinocchio"):
            simu.simulate(tau, DT, int(DT/dt_sim))
            x = np.concatenate([simu.q, simu.v])
        elif(SIMULATOR=="ideal"):
            x = sol.value(X[1])

        # Store new state
        traj.append(x.copy())
    
    # ---------------------------------------------------------
    # Construct Reference Trajectory for Return
    # ---------------------------------------------------------
    # Often helpful to see what the MPC planned to do at the next step relative to where we actually went
    reference = [traj[0]]
    for k, pred in enumerate(predicted_trajs):
        if pred is not None and pred.shape[0] >= 2:
            reference.append(pred[1])
        else:
            reference.append(traj[min(k + 1, len(traj) - 1)])
    reference = np.array(reference)

    return {
        'total_cost': total_cost,
        'trajectory': np.array(traj),
        'controls': np.array(u_list),
        'predicted_trajs': predicted_trajs,
        'predicted_us': predicted_us,
        'reference_traj': reference,
        'stopped_early': stopped_early,
        'stop_time': (stop_iter*DT) if stop_iter is not None else None,
        'exec_time': exec_time
    }

__all__ = ['simulate_mpc']