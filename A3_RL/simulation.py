
import numpy as np
import os
import casadi as cs
from time import time as clock
import time
from example_robot_data.robots_loader import load
from adam.casadi.computations import KinDynComputations
import pinocchio as pin

import imageio

from config import NQ, DT, W_P, W_V, W_A, W_T, T, PENDULUM, N, M, VIEWER
from ocp import create_ocp
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.robot_wrapper import RobotWrapper

def simulate_mpc(x0, horizon, terminal_cost_fn=None, record_video=False):
    if PENDULUM == 'single_pendulum':
        urdf_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(urdf_dir, 'single_pendulum_description/urdf/single_pendulum.urdf')
        robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[urdf_dir])
        robot.urdf = urdf_path
    else:
        robot = load(PENDULUM)

    dt_sim = 0.002
    
    q0 = x0[:NQ]
    dq0 = x0[NQ:]
    # dq0 = np.zeros(NQ)
    x0 = np.concatenate([q0, dq0])

    q_des = np.array([0.0] * NQ)

    # Initialize Simulator
    r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
    
    class DummyConf:
        q0 = x0[:NQ]
        v0 = x0[NQ:]
        dt = dt_sim
        
        # Physics / Simulation flags
        randomize_robot_model = False
        model_variation = 0.0
        simulation_type = 'euler'
        
        # Friction parameters
        tau_coulomb_max = np.zeros(NQ) 
        tau_viscous = np.zeros(NQ)     
        
        # Viewer / Visualization flags
        use_viewer = VIEWER   
        which_viewer = 'meshcat'
        viewer_name = "robot_simulator"
        show_floor = False
        DISPLAY_T = dt_sim
        frame_name = "ee_link" 
            
    simu = RobotSimulator(DummyConf, r)
    simu.init(q0, dq0)
    simu.display(q0)
    
    time.sleep(10)  
    
    opti, opti_x_init, opti_q_des, X, U, cost, inv_dyn = create_ocp(horizon, terminal_cost_fn)

    # Solve the problem to convergence the first time
    x = np.concatenate([q0, dq0])
    opti.set_value(opti_x_init, x)
    opti.set_value(opti_q_des, q_des)

    try:
        sol = opti.solve()
    except:
        sol = opti.debug
        
    # Initialize Data Logging
     
    traj = [x.copy()]   # Store initial state
    u_list = []
    predicted_trajs = []
    predicted_us = []
    total_cost = 0.0
    
    exec_time = []

    # Early stop if tracking error remains below steady_error_tol for steady_time seconds
    stopped_early = False
    stop_iter = None
    taus = []

    horizon -= 1
    
    frames = []
    
    for _ in range(T):
        start_time = clock()

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

        opti.set_value(opti_x_init, x)
        try:
            sol = opti.solve()
        except:
            sol = opti.debug
        
        end_time = clock()
        exec_time.append(end_time - start_time)
        
        # Extract Predictions and Controls
        pred_x = np.array([sol.value(X[k]) for k in range(horizon+1)])
        pred_u = np.array([sol.value(U[k]) for k in range(horizon)])
        
        predicted_trajs.append(pred_x)
        predicted_us.append(pred_u)
        
        # Get control input to apply (first element of solution)
        u_applied = pred_u[0]
        u_list.append(u_applied)
        
        # Compute Running Cost
        curr_q = x[:NQ]
        curr_v = x[NQ:]
        
        pos_error = curr_q - q_des
        tau = inv_dyn(sol.value(X[0]), sol.value(U[0])).toarray().squeeze()

        total_cost += W_P * cs.sumsqr(pos_error)
        total_cost += W_V * cs.sumsqr(curr_v)
        # total_cost += W_A * cs.sumsqr(u_applied)
        total_cost += W_T * cs.sumsqr(tau)
        
        # Step Simulator
        simu.simulate(tau, DT, int(DT/dt_sim))
        x = np.concatenate([simu.q, simu.v])

        # Store new state
        traj.append(x.copy())
        taus.append(tau)
    
    # Construct Reference Trajectory for Return
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
        'applied_torques': np.array(taus),
        'stopped_early': stopped_early,
        'stop_time': (stop_iter*DT) if stop_iter is not None else None,
        'exec_time': exec_time,
        'frames': frames
    }

__all__ = ['simulate_mpc']