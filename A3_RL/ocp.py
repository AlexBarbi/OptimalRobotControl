"""Optimal control problem helpers (centralized solver implementations).

This module consolidates the multiple similar solver functions into a single
flexible solver with small wrappers for legacy call sites.
"""
import casadi as cs
import numpy as np
from config import nq, nx, nu, dt, W_Q, W_V, W_U, TORQUE_LIMIT, ACTUATED_INDICES


def _enforce_actuation(opti, U_k):
    try:
        for j in range(nu):
            if j not in ACTUATED_INDICES:
                opti.subject_to(U_k[j] == 0)
            else:
                opti.subject_to(U_k[j] <= TORQUE_LIMIT)
                opti.subject_to(U_k[j] >= -TORQUE_LIMIT)
    except Exception:
        pass


def solve_ocp(x_init, N=100, terminal_model=None, return_xN=False, return_first_control=False):
    """Generic OCP solver.

    Args:
        x_init: initial state (array-like)
        N: horizon length
        terminal_model: casadi callable for terminal cost (optional)
        return_xN: if True return final state along with cost
        return_first_control: if True return first control and predicted traj

    Returns:
        By default: (x_init, cost) or None on failure.
        If return_xN: (x_init, cost, xN)
        If return_first_control: (u0, pred_traj, pred_us) or (None, None, None)
    """
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

        _enforce_actuation(opti, U[k])

        q_next_euler = X[k][:nq] + dt * X[k][nq:]
        dq_next_euler = X[k][nq:] + dt * U[k]
        x_next_euler = cs.vertcat(q_next_euler, dq_next_euler)
        opti.subject_to(X[k + 1] == x_next_euler)

    opti.subject_to(X[0] == x_init)

    if terminal_model is not None:
        cost += terminal_model(X[N])

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
        if return_first_control:
            try:
                u0 = np.array(sol.value(U[0])).reshape(-1)
            except Exception:
                u0 = np.zeros(nu)
            pred_traj = []
            for k in range(N + 1):
                try:
                    pred_traj.append(np.array(sol.value(X[k])).reshape(-1))
                except Exception:
                    pred_traj.append(np.zeros(nx))
            pred_traj = np.array(pred_traj)
            pred_us = []
            for k in range(N):
                try:
                    pred_us.append(np.array(sol.value(U[k])).reshape(-1))
                except Exception:
                    pred_us.append(np.zeros(nu))
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
