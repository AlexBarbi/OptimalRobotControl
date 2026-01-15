"""Optimal control problem helpers (centralized solver implementations).

This module consolidates the multiple similar solver functions into a single
flexible solver with small wrappers for legacy call sites.
"""
import casadi as cs
import numpy as np
from config import NQ, NX, NU, W_P, W_V, W_A, VELOCITY_LIMIT, TORQUE_LIMIT, KINDYN, ROBOT, DT, N

def _enforce_actuation(opti, U_k):
    """
    Applies torque limits to the control variable U_k in the optimization problem.

    Args:
        opti (casadi.Opti): The optimization problem instance.
        U_k (casadi.MX): The control variable at step k.
    """

    try:
        for j in range(NU):
            opti.subject_to(U_k[j] <= 9.81)
            opti.subject_to(U_k[j] >= - 9.81)
    except Exception:
        pass

def solve_ocp(x_init, terminal_model=None, return_xN=False, return_first_control=False):
    """
    Solves the Optimal Control Problem (OCP) for a given initial state.

    This function configures and solves a trajectory optimization problem using CasADi and IPOPT.
    It minimizes a cost function composed of quadratic state and control costs, and optionally
    a learned terminal cost.

    Args:
        x_init (np.ndarray): The initial state vector [q, dq].
        horizon (int, optional): The prediction horizon. Defaults to 100.
        terminal_model (casadi.Function, optional): A CasADi function representing the terminal cost V(x_N).
        return_xN (bool, optional): If True, returns the final state x_N. Used for bootstrapping.
        return_first_control (bool, optional): If True, returns the first control action u_0 and predicted trajectories. Used for MPC.

    Returns:
        tuple or None:
            - If return_first_control is True: (u_0, predicted_trajectory, predicted_controls)
            - If return_xN is True: (x_init, optimal_cost, x_N)
            - Otherwise: (x_init, optimal_cost)
            - Returns None if the solver fails.
    """

    SOLVER_TOLERANCE = 1e-4
    SOLVER_MAX_ITER = 1000

    # SIMULATOR = "pinocchio" #"mujoco" or "pinocchio" or "ideal"
    POS_BOUNDS_SCALING_FACTOR = 0.2
    # VEL_BOUNDS_SCALING_FACTOR = 2.0
    qMin = POS_BOUNDS_SCALING_FACTOR * ROBOT.model.lowerPositionLimit
    qMax = POS_BOUNDS_SCALING_FACTOR * ROBOT.model.upperPositionLimit
    # vMax = VEL_BOUNDS_SCALING_FACTOR * ROBOT.model.velocityLimit
    # dt_sim = DT
    # N_sim = horizon
    # initial joint configuration and velocities (x_init is [q, dq])
    q0 = x_init[:NQ]  # initial joint configuration
    dq0 = x_init[NQ:]  # initial joint velocities

    # dt = 0.010 # time step MPC
    # horizon = horizon  # time horizon MPC
    # q_des = np.array([0, -1.57, 0, 0, 0, 0]) # desired joint configuration
    if NQ == 1:
        q_des = np.array([0.0])
    else:
        q_des = np.array([0.0, np.pi])
    J = 1
    # Check if J is within bounds for this robot (Double pendulum has nq=2)
    if J < NQ:
        q_des[J] = qMin[J] + 0.01*(qMax[J] - qMin[J])

    # print("Create optimization parameters")
    ''' The parameters P contain:
        - the initial state (first 12 values)
        - the target configuration (last 6 values)
    '''
    opti = cs.Opti()
    param_x_init = opti.parameter(NX)
    param_q_des  = opti.parameter(NQ)
    cost = 0

    # ---------------------------------------------------------
    # System Dynamics (Pinocchio + CasADi AD)
    # ---------------------------------------------------------
    # Create symbolic variables for the dynamics function
    q   = cs.SX.sym('q', NQ)
    dq  = cs.SX.sym('dq', NQ)
    ddq = cs.SX.sym('ddq', NQ)
    state = cs.vertcat(q, dq)
    rhs   = cs.vertcat(dq, ddq)
    f = cs.Function('f', [state, ddq], [rhs])

    # Robot Physical Parameters and Dynamics
    H_b = cs.SX.eye(4)     # base configuration (identity)
    v_b = cs.SX.zeros(6)   # base velocity (zero)
    bias_forces = KINDYN.bias_force_fun()
    mass_matrix = KINDYN.mass_matrix_fun()
    
    # Calculate terms for the equation of motion: M(q)ddq + h(q, dq) = tau
    # We discard the first 6 elements as they correspond to the fixed base
    h = bias_forces(H_b, q, v_b, dq)[6:]
    MM = mass_matrix(H_b, q)[6:,6:]
    tau = MM @ ddq + h
    inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])
    
    horizon = N - 1
    # ---------------------------------------------------------
    # Decision Variables
    # ---------------------------------------------------------
    X, U = [], []
    X += [opti.variable(NX)]
    for k in range(1, horizon+1): 
        X += [opti.variable(NX)]
        opti.subject_to( opti.bounded(-VELOCITY_LIMIT, X[-1][NQ:], VELOCITY_LIMIT) )
    for k in range(horizon): 
        U += [opti.variable(NU)]

    # print("Add initial conditions")
    opti.subject_to(X[0] == param_x_init)

    # ---------------------------------------------------------
    # Cost & Constraints Loop
    # ---------------------------------------------------------
    for k in range(horizon - 1):     
        # Running cost
        cost += W_P * cs.sumsqr(X[k][:NQ] - param_q_des)
        cost += W_V * cs.sumsqr(X[k][NQ:])
        cost += W_A * cs.sumsqr(U[k])

        # Dynamics constraints (Simple Euler integration)
        opti.subject_to(X[k+1] == X[k] + DT * f(X[k], U[k]))
        
        # Torque limits
        opti.subject_to( opti.bounded(-TORQUE_LIMIT, inv_dyn(X[k], U[k]), TORQUE_LIMIT) )
   
    # Learned Terminal Cost (Value Function Approximation)
    # If provided, this allows the short-horizon OCP to approximate an infinite-horizon problem
    if terminal_model is not None:
        cost += terminal_model(X[horizon])

    opti.minimize(cost)

    # print("Create the optimization problem")
    opts = {
        "error_on_fail": False,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.tol": SOLVER_TOLERANCE,
        "ipopt.constr_viol_tol": SOLVER_TOLERANCE,
        "ipopt.compl_inf_tol": SOLVER_TOLERANCE,
        "print_time": 0,                # print information about execution time
        "detect_simple_bounds": True,
        "ipopt.max_iter": 1000
    }

    opti.solver("ipopt", opts)

    # Solve the problem to convergence the first time
    x = np.concatenate([q0, dq0])
    opti.set_value(param_q_des, q_des)
    opti.set_value(param_x_init, x)

    # First solve attempt (if it fails we'll try again with different settings)
    try:
        sol = opti.solve()
    except Exception as e:
        print("Initial solver attempt failed:\horizon", e)
        print("Attempting a second solve with adjusted options...")

    opts["ipopt.max_iter"] = SOLVER_MAX_ITER
    opti.solver("ipopt", opts)
    
    try:
        sol = opti.solve()
        if return_first_control:
            try:
                u0 = np.array(sol.value(U[0])).reshape(-1)
            except Exception:
                u0 = np.zeros(NU)
            pred_traj = []
            for k in range(horizon + 1):
                try:
                    pred_traj.append(np.array(sol.value(X[k])).reshape(-1))
                except Exception:
                    pred_traj.append(np.zeros(NX))
            pred_traj = np.array(pred_traj)
            pred_us = []
            for k in range(horizon):
                try:
                    pred_us.append(np.array(sol.value(U[k])).reshape(-1))
                except Exception:
                    pred_us.append(np.zeros(NU))
            pred_us = np.array(pred_us)
            return u0, pred_traj, pred_us

        if return_xN:
            xN = sol.value(X[horizon])
            return (x_init, sol.value(cost), np.array(xN).reshape(-1))

        return (x_init, sol.value(cost))
    except Exception:
        return None


# Backwards-compatible wrappers

def solve_single_ocp(x_init):
    """
    Solves OCP and returns (x_init, optimal_cost).
    Used for generating the Value Function dataset.
    """
    return solve_ocp(x_init, terminal_model=None, return_xN=False, return_first_control=False)