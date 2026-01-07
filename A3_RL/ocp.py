"""Optimal control problem helpers (centralized solver implementations).

This module consolidates the multiple similar solver functions into a single
flexible solver with small wrappers for legacy call sites.
"""
import casadi as cs
import numpy as np
from config import NQ, NX, NU, DT, W_Q, W_V, W_U, TORQUE_LIMIT, ACTUATED_INDICES, KINDYN, ROBOT

from example_robot_data.robots_loader import load
from adam.casadi.computations import KinDynComputations
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.robot_loaders import loadUR
from orc.utils.robot_wrapper import RobotWrapper
import orc.optimal_control.casadi_adam.conf_ur5 as conf_ur5




def _enforce_actuation(opti, U_k):
    try:
        for j in range(NU):
            if j not in ACTUATED_INDICES:
                opti.subject_to(U_k[j] == 0)
            else:
                opti.subject_to(U_k[j] <= TORQUE_LIMIT)
                opti.subject_to(U_k[j] >= -TORQUE_LIMIT)
    except Exception:
        pass


def solve_ocp(x_init, N=100, terminal_model=None, return_xN=False, return_first_control=False):
    # print("Load robot model")
    # robot = load("double_pendulum")

    # print("Create KinDynComputations object")
    # joints_name_list = [s for s in robot.model.names[1:]] # skip the first name because it is "universe"
    # nq = len(joints_name_list)  # number of joints
    # nx = 2*nq # size of the state variable
    # kinDyn = KinDynComputations(robot.urdf, joints_name_list)

    ADD_SPHERE = 0
    SPHERE_POS = np.array([0.2, -0.10, 0.5])
    SPHERE_SIZE = np.ones(3)*0.1
    SPHERE_RGBA = np.array([1, 0, 0, 1.])

    # WITH THIS CONFIGURATION THE SOLVER ENDS UP VIOLATING THE JOINT LIMITS
    # ADDING THE TERMINAL CONSTRAINT FIXES EVERYTHING!
    # BUT SO DOES:
    # - DECREASING THE POSITION WEIGHT IN THE COST
    # - INCREASING THE ACCELERATION WEIGHT IN THE COST
    # - INCREASING THE MAX NUMBER OF ITERATIONS OF THE SOLVER
    DO_WARM_START = True
    SOLVER_TOLERANCE = 1e-4
    SOLVER_MAX_ITER = 1000

    SIMULATOR = "pinocchio" #"mujoco" or "pinocchio" or "ideal"
    POS_BOUNDS_SCALING_FACTOR = 0.2
    VEL_BOUNDS_SCALING_FACTOR = 2.0
    qMin = POS_BOUNDS_SCALING_FACTOR * ROBOT.model.lowerPositionLimit
    qMax = POS_BOUNDS_SCALING_FACTOR * ROBOT.model.upperPositionLimit
    vMax = VEL_BOUNDS_SCALING_FACTOR * ROBOT.model.velocityLimit
    dt_sim = DT
    N_sim = N
    # initial joint configuration and velocities (x_init is [q, dq])
    q0 = x_init[:NQ]  # initial joint configuration
    dq0 = x_init[NQ:]  # initial joint velocities

    dt = 0.010 # time step MPC
    N = N  # time horizon MPC
    # q_des = np.array([0, -1.57, 0, 0, 0, 0]) # desired joint configuration
    q_des = np.zeros(NQ)
    J = 1
    q_des[J] = qMin[J] + 0.01*(qMax[J] - qMin[J])
    w_p = W_Q   # position weight
    w_v = W_V  # velocity weight
    w_a = W_U  # acceleration weight
    w_final_v = 0e0 # final velocity cost weight
    USE_TERMINAL_CONSTRAINT = 0


    # if(SIMULATOR=="mujoco"):
    #     from orc.utils.mujoco_simulator import MujocoSimulator
    #     print("Creating simulator...")
    #     simu = MujocoSimulator("ur5", dt_sim)
    #     simu.set_state(q0, dq0)
    # else:
    #     r = RobotWrapper(ROBOT.model, ROBOT.collision_model, ROBOT.visual_model)
    #     simu = RobotSimulator(conf_ur5, r)
    #     simu.init(q0, dq0)
    #     simu.display(q0)
        

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

    # inverse dynamics (for reference): tau = MM * ddq + h
    ddq = cs.SX.sym('ddq', NQ)
    tau_expr = MM @ ddq + h
    inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau_expr])

    # pre-compute state and torque bounds
    lbx = qMin.tolist() + (-vMax).tolist()
    ubx = qMax.tolist() + vMax.tolist()
    tau_min = (-ROBOT.model.effortLimit).tolist()
    tau_max = ROBOT.model.effortLimit.tolist()

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

    # warm-start (initial guess) to improve feasibility
    # for k in range(N+1):
    #     try:
    #         opti.set_initial(X[k], np.concatenate([q0, dq0]))
    #     except Exception:
    #         pass
    # for k in range(N):
    #     try:
    #         opti.set_initial(U[k], np.zeros(NU))
    #     except Exception:
    #         pass

    for k in range(N):     
        # print("Compute cost function")
        cost += w_p * (X[k][:NQ] - param_q_des).T @ (X[k][:NQ] - param_q_des)
        cost += w_v * X[k][NQ:].T @ X[k][NQ:]
        cost += w_a * U[k].T @ U[k]

        # dynamics (state update using torque -> ddq mapping)
        opti.subject_to(X[k+1] == X[k] + DT * f(X[k], U[k]))

        # torque bounds: controls are torques, so bound U directly
        # opti.subject_to( opti.bounded(tau_min, U[k], tau_max))

        # enforce actuation pattern (zero torque on unactuated joints)
        _enforce_actuation(opti, U[k])

    # add the final cost
    # cost += w_final_v * X[-1][nq:].T @ X[-1][nq:]

    # if(USE_TERMINAL_CONSTRAINT):
    #     opti.subject_to(X[-1][nq:] == 0.0)
    
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

    # set up simulation environment
    # if(SIMULATOR=="mujoco" and ADD_SPHERE):
    #     simu.add_sphere(pos=SPHERE_POS, size=SPHERE_SIZE, rgba=SPHERE_RGBA)

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



# def solve_ocp1(x_init, N=100, terminal_model=None, return_xN=False, return_first_control=False):
#     """Generic OCP solver.

#     Args:
#         x_init: initial state (array-like)
#         N: horizon length
#         terminal_model: casadi callable for terminal cost (optional)
#         return_xN: if True return final state along with cost
#         return_first_control: if True return first control and predicted traj

#     Returns:
#         By default: (x_init, cost) or None on failure.
#         If return_xN: (x_init, cost, xN)
#         If return_first_control: (u0, pred_traj, pred_us) or (None, None, None)
#     """
#     opti = cs.Opti()
#     X = [opti.variable(nx) for _ in range(N+1)]
#     U = [opti.variable(nu) for _ in range(N)]
#     # x_target = np.zeros(nx)
#     cost = 0.0

#     for k in range(N):
#         # q_err = X[k][:nq] - x_target[:nq]
#         # v_err = X[k][nq:] - x_target[nq:]
#         cost += W_Q * cs.sumsqr(X[k][:nq])
#         cost += W_V * cs.sumsqr(X[k][nq:])
#         cost += W_U * cs.sumsqr(U[k])

#         _enforce_actuation(opti, U[k])

#         q_next_euler = X[k][:nq] + dt * X[k][nq:]
#         dq_next_euler = X[k][nq:] + dt * U[k]
#         x_next_euler = cs.vertcat(q_next_euler, dq_next_euler)
#         opti.subject_to(X[k + 1] == x_next_euler)

#     opti.subject_to(X[0] == x_init)
        
#     if terminal_model is not None:
#         cost += terminal_model(X[N])

#     opti.minimize(cost)

#     opts = {
#         "ipopt.print_level": 0,
#         "print_time": 0,
#         "ipopt.sb": "yes",
#         "ipopt.tol": 1e-4,
#     }
#     opti.solver("ipopt", opts)

#     try:
#         sol = opti.solve()
#         if return_first_control:
#             try:
#                 u0 = np.array(sol.value(U[0])).reshape(-1)
#             except Exception:
#                 u0 = np.zeros(nu)
#             pred_traj = []
#             for k in range(N + 1):
#                 try:
#                     pred_traj.append(np.array(sol.value(X[k])).reshape(-1))
#                 except Exception:
#                     pred_traj.append(np.zeros(nx))
#             pred_traj = np.array(pred_traj)
#             pred_us = []
#             for k in range(N):
#                 try:
#                     pred_us.append(np.array(sol.value(U[k])).reshape(-1))
#                 except Exception:
#                     pred_us.append(np.zeros(nu))
#             pred_us = np.array(pred_us)
#             return u0, pred_traj, pred_us

#         if return_xN:
#             xN = sol.value(X[N])
#             return (x_init, sol.value(cost), np.array(xN).reshape(-1))

#         return (x_init, sol.value(cost))
#     except Exception:
#         return None


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
