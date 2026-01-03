"""Simulation helpers: closed-loop simulation and batch comparison.
"""
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch
from ocp import solve_single_ocp_get_first_control, solve_single_ocp, solve_single_ocp_return_terminal, solve_ocp
from config import nq, nu, dt, W_Q, W_V, W_U


def simulate_mpc(x0, controller, tcost_model=None, M=20, N_long=100, T=None, tol=1e-3, verbose=False, env=None):
    if T is None:
        T = N_long + M

    traj = [np.array(x0).reshape(-1)]
    u_list = []
    total_cost = 0.0

    l4_term = None
    if controller == 'M_term':
        if tcost_model is None:
            raise ValueError('tcost_model required for M_term controller')
        l4_term = tcost_model.create_casadi_function()

    if env is not None:
        try:
            env.reset(np.array(x0).reshape(-1))
            x = np.array(env.x).reshape(-1)
        except Exception:
            x = np.array(x0).reshape(-1)
        try:
            env.DT = dt
        except Exception:
            pass
    else:
        x = np.array(x0).reshape(-1)

    start_time = time.time()
    predicted_trajs = []
    predicted_us = []

    for t in range(T):
        if controller == 'M':
            u0, pred, pred_u = solve_single_ocp_get_first_control(x, N=M, terminal_model=None)
        elif controller == 'M_term':
            u0, pred, pred_u = solve_single_ocp_get_first_control(x, N=M, terminal_model=l4_term)
        elif controller == 'N+M':
            u0, pred, pred_u = solve_single_ocp_get_first_control(x, N=N_long + M, terminal_model=None)
        else:
            raise ValueError('Unknown controller type')

        if u0 is None:
            if verbose:
                print(f"Step {t}: solver failed for controller {controller}; terminating simulation.")
            break

        predicted_trajs.append(pred)
        predicted_us.append(pred_u)

        # apply control limits
        try:
            mask = np.zeros(nu)
            for j in range(nu):
                mask[j] = 1
            u_applied = np.array(u0).reshape(-1) * mask
            u_applied = np.clip(u_applied, -1000, 1000)
        except Exception:
            u_applied = np.array(u0).reshape(-1)

        # step dynamics
        if env is not None:
            try:
                _, r = env.dynamics(env.x, u_applied)
                step_cost = -r
                x = np.array(env.x).reshape(-1)
            except Exception:
                q = x[:nq]
                dq = x[nq:]
                step_cost = W_Q * np.sum(q ** 2) + W_V * np.sum(dq ** 2) + W_U * np.sum(u_applied ** 2)
                q_next = q + dt * dq
                dq_next = dq + dt * u_applied
                x = np.concatenate([q_next, dq_next])
        else:
            q = x[:nq]
            dq = x[nq:]
            step_cost = W_Q * np.sum(q ** 2) + W_V * np.sum(dq ** 2) + W_U * np.sum(u_applied ** 2)
            q_next = q + dt * dq
            dq_next = dq + dt * u_applied
            x = np.concatenate([q_next, dq_next])

        total_cost += step_cost
        traj.append(x.copy())
        u_list.append(u_applied.copy())

        if np.linalg.norm(x) < tol:
            if verbose:
                print(f"Terminated at step {t} because state norm {np.linalg.norm(x):.4e} < tol")
            break

    end_time = time.time()
    print(f"Computed control with {controller} in {end_time - start_time:.4f} seconds.")

    reference = [traj[0]]
    for i, pred in enumerate(predicted_trajs):
        if pred is not None and pred.shape[0] >= 2:
            reference.append(pred[1])
        else:
            reference.append(traj[min(i + 1, len(traj) - 1)])
    reference = np.array(reference)

    return {
        'total_cost': total_cost,
        'trajectory': np.array(traj),
        'controls': np.array(u_list),
        'predicted_trajs': predicted_trajs,
        'predicted_us': predicted_us,
        'reference_traj': reference,
    }


# Lightweight wrappers around the earlier compare_mpcs / simulate_batch behavior can be added here on demand.

__all__ = ['simulate_mpc']