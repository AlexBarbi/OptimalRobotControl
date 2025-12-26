#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import casadi as cs
import time
from time import sleep

from adam.casadi.computations import KinDynComputations
from example_robot_data.robots_loader import load
import orc.optimal_control.casadi_adam.conf_ur5 as conf_ur5
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.viz_utils import addViewerSphere, applyViewerConfiguration


# ====================== Simple plotting utility ======================
def plot_infinity(t_init, t_final):
    r = 0.1
    t = np.linspace(t_init, t_final, 300)
    x = r * np.cos(2*np.pi*t)
    y = r * 0.5*np.sin(4*np.pi*t)
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, 'x')
    plt.xlim([-0.11, 0.11])
    plt.ylim([-0.06, 0.06])
    plt.title("Infinity-shaped path")
    plt.grid(True)
    plt.show()


# ====================== Robot and Dynamics Setup ======================
robot = load("ur5")
joints_name_list = [s for s in robot.model.names[1:]]
nq = len(joints_name_list)
nx = 2 * nq

dt_min = 0.001  # 0.001 as default, 0.005 for optimal
dt_max = 0.1
N = 100
q0 = np.zeros(nq)
dq0 = np.zeros(nq)
x_init = np.concatenate([q0, dq0])
w_max = 5
frame_name = "ee_link"
r_path = 0.2

# CasADi symbolic variables
q = cs.SX.sym('q', nq)
dq = cs.SX.sym('dq', nq)
ddq = cs.SX.sym('ddq', nq)
state = cs.vertcat(q, dq)
rhs = cs.vertcat(dq, ddq)
f = cs.Function('f', [state, ddq], [rhs])

# Inverse dynamics
kinDyn = KinDynComputations(robot.urdf, joints_name_list)
H_b = cs.SX.eye(4)
v_b = cs.SX.zeros(6)
bias_forces = kinDyn.bias_force_fun()
mass_matrix = kinDyn.mass_matrix_fun()
h = bias_forces(H_b, q, v_b, dq)[6:]
M = mass_matrix(H_b, q)[6:, 6:]
tau = M @ ddq + h
inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

# Forward kinematics
fk_fun = kinDyn.forward_kinematics_fun(frame_name)
ee_pos = fk_fun(H_b, q)[:3, 3]
fk = cs.Function('fk', [q], [ee_pos])

y = fk(q0)
c_path = np.array([y[0]-r_path, y[1], y[2]]).squeeze()

lbx = robot.model.lowerPositionLimit.tolist() + (-robot.model.velocityLimit).tolist()
ubx = robot.model.upperPositionLimit.tolist() + robot.model.velocityLimit.tolist()
tau_min = (-robot.model.effortLimit).tolist()
tau_max = robot.model.effortLimit.tolist()


# ====================== Optimization Problem ======================
def create_decision_variables(N, nx, nu, lbx, ubx):
    opti = cs.Opti()
    X, U, S, W = [], [], [], []
    for _ in range(N + 1):
        X += [opti.variable(nx)]
        opti.subject_to(opti.bounded(lbx, X[-1], ubx))
        S += [opti.variable(1)]
        opti.subject_to(opti.bounded(0, S[-1], 1))
    for _ in range(N):
        U += [opti.variable(nu)]
        W += [opti.variable(1)]
        
    dt = opti.variable(1)
    opti.subject_to(opti.bounded(dt_min, dt, dt_max))
        
    return opti, X, U, S, W, dt

def define_running_cost_and_dynamics(opti, X, U, S, W, N, dt, x_init,
                                     c_path, r_path, w_v, w_a, w_p, w_time,
                                     tau_min, tau_max):
    
    # Constrain the initial state X[0] to be equal to the initial condition x_init
    opti.subject_to(X[0] == x_init)

    # Initialize the path variable S[0] to 0.0
    opti.subject_to(S[0] == 0.0)

    # Constrain the final path variable S[-1] to be 1.0
    opti.subject_to(S[-1] == 1.0)

    cost = 0.0
    for k in range(N):
        # CHANGED: add control variable contraint
        opti.subject_to(W[k] == 1 / N)

        # Compute the end-effector position using forward kinematics
        ee_pos = fk(X[k][:nq])

        # Constrain ee_pos to lie on the desired path in x, y, z
        path = cs.vertcat(
            c_path[0] + r_path * cs.cos(2 * cs.pi * S[k]),
            c_path[1] + r_path * 0.5 * cs.sin(4 * cs.pi * S[k]),
            c_path[2]
        )

        # CHANGED: add trajectory tracking cost term
        cost += w_p * cs.sumsqr(ee_pos - path)

        # Add velocity tracking cost term
        cost += w_v * cs.sumsqr(X[k][nq:])

        # Add actuation effort cost term
        cost += w_a * cs.sumsqr(U[k])
        
        # Add path progression speed cost term
        cost += 1e-6 * cs.sumsqr(W[k])

        # Add discrete-time dynamics constraint
        opti.subject_to( X[k+1] == X[k] + dt * f(X[k], U[k]) )
        
        # Add path variable dynamics constraint
        opti.subject_to( S[k+1] == S[k] + W[k])

        # Constrain the joint torques to remain within [tau_min, tau_max]
        opti.subject_to( opti.bounded(tau_min, inv_dyn(X[k], U[k]), tau_max))
        
    return cost * dt

def define_terminal_cost_and_constraints(X, S, c_path, r_path, w_p, w_final, dt, w_time):
    # Compute the end-effector position using forward kinematics
    ee_pos = fk(X[-1][:nq])

    # Constrain ee_pos to lie on the desired path in x, y, z
    path = cs.vertcat(
        c_path[0] + r_path * cs.cos(2 * cs.pi * S[-1]),
        c_path[1] + r_path * 0.5 * cs.sin(4 * cs.pi * S[-1]),
        c_path[2]
    )
    
    # CHANGED
    cost = w_p * cs.sumsqr(ee_pos - path) * dt
    
    cost += w_final * cs.sumsqr(X[-1] - X[0])
    
    cost += w_time * dt**2

    return cost


def create_and_solve_ocp(N, nx, nq, lbx, ubx, x_init,
                         c_path, r_path, w_v, w_a, w_final, w_p, w_time,
                         tau_min, tau_max):
    opti, X, U, S, W, dt = create_decision_variables(N, nx, nq, lbx, ubx)
    running_cost = define_running_cost_and_dynamics(opti, X, U, S, W, N, dt, x_init,
                                                    c_path, r_path, w_v, w_a, w_p, w_time,
                                                    tau_min, tau_max)
    terminal_cost = define_terminal_cost_and_constraints(X, S, c_path, r_path, w_p, w_final, dt, w_time)
    opti.minimize(running_cost + terminal_cost)

    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4}
    opti.solver("ipopt", opts)

    t0 = time.time()
    sol = opti.solve()
    print(f"Solver time: {time.time() - t0:.2f}s")
    return sol, X, U, S, W, dt


def extract_solution(sol, X, U, S, W, dt):
    x_sol = np.array([sol.value(X[k]) for k in range(N + 1)]).T
    ddq_sol = np.array([sol.value(U[k]) for k in range(N)]).T
    s_sol = np.array([sol.value(S[k]) for k in range(N + 1)]).T
    q_sol = x_sol[:nq, :]
    dq_sol = x_sol[nq:, :]
    w_sol = np.array([sol.value(W[k]) for k in range(N)]).T
    tau = np.zeros((nq, N))
    for i in range(N):
        tau[:, i] = inv_dyn(x_sol[:, i], ddq_sol[:, i]).toarray().squeeze()
    ee = np.zeros((3, N + 1))
    for i in range(N + 1):
        ee[:, i] = fk(x_sol[:nq, i]).toarray().squeeze()
    ee_des = np.zeros((3, N + 1))
    for i in range(N + 1):
        ee_des[:, i] = np.array([c_path[0] + r_path*np.cos(2*np.pi*s_sol[i]),
                                 c_path[1] + r_path*0.5*np.sin(4*np.pi*s_sol[i]),
                                 c_path[2]])
    dt_sol = sol.value(dt)
    print(dt_sol)
    return q_sol, dq_sol, ddq_sol, tau, ee, ee_des, s_sol, w_sol, dt_sol


# ====================== Simulation and Visualization ======================
r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
simu = RobotSimulator(conf_ur5, r)
simu.init(q0, dq0)
simu.display(q0)

REF_SPHERE_RADIUS = 0.02
EE_REF_SPHERE_COLOR = np.array([1, 0, 0, .5])


def display_motion(q_traj, ee_des_traj):
    for i in range(N + 1):
        t0 = time.time()
        simu.display(q_traj[:, i])
        addViewerSphere(r.viz, f'world/ee_ref_{i}', REF_SPHERE_RADIUS, EE_REF_SPHERE_COLOR)
        applyViewerConfiguration(r.viz, f'world/ee_ref_{i}', ee_des_traj[:, i].tolist() + [0, 0, 0, 1.])
        t1 = time.time()
        if(t1-t0 < dt_sol):
            sleep(dt_sol - (t1-t0))


# ====================== Main Execution ======================
if __name__ == "__main__":
    # print("Plotting reference infinity curve...")
    #plot_infinity(0, 1)

    log_w_v, log_w_a, log_w_w = -6, -6, -6
    log_w_final = -6
    log_w_p = 2
    log_w_t = -1    # -1 as default, 1 for optimal

    sol, X, U, S, W, dt = create_and_solve_ocp(
        N, nx, nq, lbx, ubx, x_init, c_path, r_path,
        10**log_w_v, 10**log_w_a, 10**log_w_final, 10**log_w_p, 10**log_w_t,
        tau_min, tau_max
    )
    q_sol, dq_sol, u_sol, tau, ee, ee_des, s_sol, w_sol, dt_sol = extract_solution(sol, X, U, S, W, dt)

    print("Displaying robot motion...")
    for i in range(3):
        display_motion(q_sol, ee_des)

    # Plot results
    tt = np.linspace(0, (N + 1) * dt_sol, N + 1)
    plt.figure(figsize=(10, 4))
    plt.plot([tt[0], tt[-1]], [0, 1], ':', label='straight line', alpha=0.7)
    plt.plot(tt, s_sol, label='s')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(ee_des[0,:].T, ee_des[1,:].T, 'r x', label='EE des', alpha=0.7)
    plt.plot(ee[0,:].T, ee[1,:].T, 'k o', label='EE', alpha=0.7)
    plt.xlabel('End-effector pos x [m]')
    plt.ylabel('End-effector pos y [m]')
    plt.legend()
    plt.grid(True)
    
    plt.figure(figsize=(10, 4))
    for i in range(3):
        plt.plot(tt, ee_des[i,:].T, ':', label=f'EE des {i}', alpha=0.7)
        plt.plot(tt, ee[i,:].T, label=f'EE {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('End-effector pos [m]')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    for i in range(dq_sol.shape[0]):
        plt.plot(tt, dq_sol[i,:].T, label=f'dq {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint velocity [rad/s]')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    for i in range(q_sol.shape[0]):
        plt.plot([tt[0], tt[-1]], [q_sol[i,0], q_sol[i,0]], ':', label='straight line', alpha=0.7)
        plt.plot(tt, q_sol[i,:].T, label=f'q {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint position [rad]')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    for i in range(tau.shape[0]):
        plt.plot(tt[:-1], tau[i,:].T, label=f'tau {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint torque [Nm]')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    plt.plot(tt[:-1], w_sol.T, label=f'w', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid(True)
    plt.show()