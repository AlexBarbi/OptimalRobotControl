"""Project configuration and constants used across modules."""
import numpy as np
import os
# from pendulum import Pendulum
import multiprocessing

# Robot init: allow choosing robot via ROBOT_TYPE env var or CLI flag '--robot-type'
# _robot_choice = os.environ.get('ROBOT_TYPE', 'pendulum').lower()
# if _robot_choice in ('double', 'double_pendulum', 'double-pendulum'):
#     try:
#         from example_robot_data.robots_loader import load
#         robot = load('double_pendulum')
#         print(f"Config: using robot 'double_pendulum' (ROBOT_TYPE={_robot_choice})")
#     except Exception as _e:
#         print(f"Warning: failed to load 'double_pendulum' ({_e}); falling back to simple Pendulum")
#         robot = Pendulum(2, open_viewer=False)
# else:
#     robot = Pendulum(2, open_viewer=False)
#     print(f"Config: using simple Pendulum (ROBOT_TYPE={_robot_choice})")

# joints_name_list = [s for s in robot.model.names[1:]]
# nq = len(joints_name_list)
# nx = 2 * nq
# nu = nq
from example_robot_data.robots_loader import load
import pinocchio as pin
from adam.casadi.computations import KinDynComputations

PENDULUM = os.environ.get('ROBOT_TYPE', 'double_pendulum').lower()

if PENDULUM == 'single_pendulum':
    urdf_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(urdf_dir, 'single_pendulum_description/urdf/single_pendulum.urdf')
    ROBOT = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[urdf_dir])
    ROBOT.urdf = urdf_path
else:
    ROBOT = load(PENDULUM)

joints_name_list = [s for s in ROBOT.model.names[1:]] # skip the first name because it is "universe"
NQ = len(joints_name_list)  # number of joints
NX = 2*NQ # size of the state variable
# Controls are torques, one per joint
NU = NQ  # size of the control input
KINDYN = KinDynComputations(ROBOT.urdf, joints_name_list)
# Actuation and limits
TORQUE_LIMIT = getattr(ROBOT, 'umax', getattr(ROBOT, 'torque_limit', 10.0))
# Default actuated indices to all joints if robot doesn't expose them (some wrappers lack `nu`)
ACTUATED_INDICES = list(getattr(ROBOT, 'actuated_indices', list(range(NQ))))
# OCP / simulation parameters
N = 100
DT = 0.01
M = 10

T = 750  # Total simulation time steps

# Dataset / parallelism
NUM_SAMPLES = 10000
NUM_CORES = multiprocessing.cpu_count()

# Cost weights
W_Q = 1000.0
W_V = 1.0
W_U = 0.1

# Convenience
SEED = None

__all__ = [
    'robot', 'joints_name_list', 'nq', 'nx', 'nu', 'TORQUE_LIMIT', 'ACTUATED_INDICES',
    'N', 'M', 'dt', 'NUM_SAMPLES', 'NUM_CORES', 'W_Q', 'W_V', 'W_U', 'SEED'
]