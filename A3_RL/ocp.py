"""Optimal control problem helpers (centralized solver implementations).

This module consolidates the multiple similar solver functions into a single
flexible solver with small wrappers for legacy call sites.
"""
import casadi as cs
import numpy as np
from config import NQ, NX, NU, W_P, W_V, W_T, VELOCITY_LIMIT, ACCEL_LIMIT, TORQUE_LIMIT, KINDYN, ROBOT, DT, N, SOLVER_TOLERANCE, SOLVER_MAX_ITER

def create_ocp(horizon, terminal_cost_fn=None):
    # print("Create optimization parameters")
    ''' The parameters P contain:
        - the initial state (first 12 values)
        - the target configuration (last 6 values)
    '''
    opti = cs.Opti()
    param_x_init = opti.parameter(NX)
    param_q_des  = opti.parameter(NQ)
    cost = 0

    # System Dynamics (Pinocchio + CasADi AD)
    q   = cs.SX.sym('q', NQ)
    dq  = cs.SX.sym('dq', NQ)
    ddq = cs.SX.sym('ddq', NQ)
    state = cs.vertcat(q, dq)
    rhs   = cs.vertcat(dq, ddq)
    f = cs.Function('f', [state, ddq], [rhs])

    # Robot Physical Parameters and Dynamics
    H_b = cs.SX.eye(4)
    v_b = cs.SX.zeros(6)
    bias_forces = KINDYN.bias_force_fun()
    mass_matrix = KINDYN.mass_matrix_fun()
    
    # Calculate terms for the equation of motion: M(q)ddq + h(q, dq) = tau
    # We discard the first 6 elements as they correspond to the fixed base
    h = bias_forces(H_b, q, v_b, dq)[6:]
    MM = mass_matrix(H_b, q)[6:,6:]
    tau = MM @ ddq + h
    inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])
    
    horizon -= 1

    # Decision Variables
    X, U = [], []
    X += [opti.variable(NX)]
    for k in range(1, horizon + 1): 
        X += [opti.variable(NX)]
        opti.subject_to( opti.bounded(-VELOCITY_LIMIT, X[-1][NQ:], VELOCITY_LIMIT) )
    for k in range(horizon): 
        U += [opti.variable(NU)]
        opti.subject_to( opti.bounded(-ACCEL_LIMIT, U[-1], ACCEL_LIMIT) )

    # print("Add initial conditions")
    opti.subject_to(X[0] == param_x_init)

    # Cost & Constraints Loop
    for k in range(horizon):     
        tau = inv_dyn(X[k], U[k])
        # Running cost
        cost += W_P * cs.sumsqr(X[k][:NQ] - param_q_des)
        cost += W_V * cs.sumsqr(X[k][NQ:])
        # cost += W_A * cs.sumsqr(U[k])
        cost += W_T * cs.sumsqr(tau)

        # Dynamics constraints (Simple Euler integration)
        opti.subject_to(X[k+1] == X[k] + DT * f(X[k], U[k]))
        
        # Torque limits
        opti.subject_to( opti.bounded(-TORQUE_LIMIT, tau, TORQUE_LIMIT) )
   
    if terminal_cost_fn is not None:
        cost += terminal_cost_fn(X[horizon])

    opti.minimize(cost)

    opts = {
        "error_on_fail": False,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.tol": SOLVER_TOLERANCE,
        "ipopt.constr_viol_tol": SOLVER_TOLERANCE,
        "ipopt.compl_inf_tol": SOLVER_TOLERANCE,
        "print_time": 0,                # print information about execution time
        "detect_simple_bounds": True,
        "ipopt.max_iter": SOLVER_MAX_ITER
    }

    opti.solver("ipopt", opts)

    return (opti, param_x_init, param_q_des, X, U, cost, inv_dyn)

def solve_single_ocp(x_init, q_des = [0.0] * NQ):
    q0  = x_init[:NQ]
    dq0 = x_init[NQ:]
    
    opti, opti_x_init, opti_q_des, _, _, cost, _ = create_ocp(N)

    x = np.concatenate([q0, dq0])
    opti.set_value(opti_x_init, x)
    opti.set_value(opti_q_des, q_des)
    
    try:
        sol = opti.solve()
        return (x_init, sol.value(cost))
    except Exception:
        return None