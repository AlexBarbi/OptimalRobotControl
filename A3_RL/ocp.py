"""Optimal control problem helpers (centralized solver implementations).

This module consolidates the multiple similar solver functions into a single
flexible solver with small wrappers for legacy call sites.
"""
import casadi as cs
import numpy as np
from config import NQ, NX, NU, W_Q, W_V, W_U, TORQUE_LIMIT, KINDYN, ROBOT


def _enforce_actuation(opti, U_k):
    try:
        for j in range(NU):
            opti.subject_to(U_k[j] <= TORQUE_LIMIT)
            opti.subject_to(U_k[j] >= -TORQUE_LIMIT)
    except Exception:
        pass


def solve_ocp(x_init, N=100, terminal_model=None, return_xN=False, return_first_control=False):

    SOLVER_TOLERANCE = 1e-4
    SOLVER_MAX_ITER = 1000

    # SIMULATOR = "pinocchio" #"mujoco" or "pinocchio" or "ideal"
    POS_BOUNDS_SCALING_FACTOR = 0.2
    # VEL_BOUNDS_SCALING_FACTOR = 2.0
    qMin = POS_BOUNDS_SCALING_FACTOR * ROBOT.model.lowerPositionLimit
    qMax = POS_BOUNDS_SCALING_FACTOR * ROBOT.model.upperPositionLimit
    # vMax = VEL_BOUNDS_SCALING_FACTOR * ROBOT.model.velocityLimit
    # dt_sim = DT
    # N_sim = N
    # initial joint configuration and velocities (x_init is [q, dq])
    q0 = x_init[:NQ]  # initial joint configuration
    dq0 = x_init[NQ:]  # initial joint velocities

    # dt = 0.010 # time step MPC
    N = N  # time horizon MPC
    # q_des = np.array([0, -1.57, 0, 0, 0, 0]) # desired joint configuration
    if NQ == 1:
        q_des = np.array([0.0])
    else:
        q_des = np.array([0.0, np.pi])
    J = 1
    # Check if J is within bounds for this robot (Double pendulum has nq=2)
    if J < NQ:
        q_des[J] = qMin[J] + 0.01*(qMax[J] - qMin[J])
    w_p = W_Q   # position weight
    w_v = W_V  # velocity weight
    w_a = W_U  # acceleration weight

    print("Create optimization parameters")
    ''' The parameters P contain:
        - the initial state (first 12 values)
        - the target configuration (last 6 values)
    '''
    opti = cs.Opti()
    param_x_init = opti.parameter(NX)
    param_q_des = opti.parameter(NQ)
    cost = 0

    # create the dynamics function: inputs are state and torque (tau)
    q   = cs.SX.sym('q', NQ)
    dq  = cs.SX.sym('dq', NQ)
    tau_sym = cs.SX.sym('tau', NQ)
    state = cs.vertcat(q, dq)

    # create a Casadi inverse dynamics building blocks
    H_b = cs.SX.eye(4)     # base configuration
    v_b = cs.SX.zeros(6)   # base velocity
    bias_forces = KINDYN.bias_force_fun()
    mass_matrix = KINDYN.mass_matrix_fun()
    # discard the first 6 elements because they are associated to the robot base
    h = bias_forces(H_b, q, v_b, dq)[6:]
    MM = mass_matrix(H_b, q)[6:,6:]

    # acceleration from torque: ddq = MM^{-1} * (tau - h)
    ddq_from_tau = cs.inv(MM) @ (tau_sym - h)
    rhs = cs.vertcat(dq, ddq_from_tau)

    # dynamics function mapping (x, tau) -> xdot
    f = cs.Function('f', [state, tau_sym], [rhs])

    # create all the decision variables
    X, U = [], []
    X += [opti.variable(NX)] # do not apply pos/vel bounds on initial state
    for k in range(1, N+1): 
        X += [opti.variable(NX)]
        # opti.subject_to( opti.bounded(lbx, X[-1], ubx) )
    for k in range(N): 
        U += [opti.variable(NU)]

    print("Add initial conditions")
    opti.subject_to(X[0] == param_x_init)

    for k in range(N):     
        # print("Compute cost function")
        cost += w_p * (X[k][:NQ] - param_q_des).T @ (X[k][:NQ] - param_q_des)
        cost += w_v * X[k][NQ:].T @ X[k][NQ:]
        cost += w_a * U[k].T @ U[k]

        opti.subject_to(X[k+1] == f(X[k], U[k]))

        # torque bounds: controls are torques, so bound U directly
        _enforce_actuation(opti, U[k])
    
    if terminal_model is not None:
        cost += terminal_model(X[N])

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

    # Solve the problem to convergence the first time
    x = np.concatenate([q0, dq0])
    opti.set_value(param_q_des, q_des)
    opti.set_value(param_x_init, x)

    # First solve attempt (if it fails we'll try again with different settings)
    try:
        sol = opti.solve()
    except Exception as e:
        print("Initial solver attempt failed:\n", e)
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
            for k in range(N + 1):
                try:
                    pred_traj.append(np.array(sol.value(X[k])).reshape(-1))
                except Exception:
                    pred_traj.append(np.zeros(NX))
            pred_traj = np.array(pred_traj)
            pred_us = []
            for k in range(N):
                try:
                    pred_us.append(np.array(sol.value(U[k])).reshape(-1))
                except Exception:
                    pred_us.append(np.zeros(NU))
            pred_us = np.array(pred_us)
            return u0, pred_traj, pred_us

        if return_xN:
            xN = sol.value(X[N])
            return (x_init, sol.value(cost), np.array(xN).reshape(-1))

        return (x_init, sol.value(cost))
    except Exception:
        return None


# Backwards-compatible wrappers
def solve_single_ocp(x_init, N=100):
    return solve_ocp(x_init, N=N, terminal_model=None, return_xN=False, return_first_control=False)


def solve_single_ocp_with_terminal(x_init, N=100, terminal_model=None):
    return solve_ocp(x_init, N=N, terminal_model=terminal_model, return_xN=False, return_first_control=False)


def solve_single_ocp_return_terminal(x_init, N=100):
    return solve_ocp(x_init, N=N, terminal_model=None, return_xN=True)


def solve_single_ocp_with_terminal_return_terminal(x_init, N=100, terminal_model=None):
    return solve_ocp(x_init, N=N, terminal_model=terminal_model, return_xN=True)


def solve_single_ocp_get_first_control(x_init, N=100, terminal_model=None):
    res = solve_ocp(x_init, N=N, terminal_model=terminal_model, return_first_control=True)
    if res is None:
        return None, None, None
    return res
