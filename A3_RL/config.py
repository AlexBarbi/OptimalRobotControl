"""Project configuration and constants used across modules."""
import os
import multiprocessing
from example_robot_data.robots_loader import load
import pinocchio as pin
from adam.casadi.computations import KinDynComputations
import numpy as np


ROBOT_TYPE = 'single'  # 'single' or 'double'

PENDULUM = f'{ROBOT_TYPE}_pendulum'
if PENDULUM == 'single_pendulum':
    urdf_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(urdf_dir, 'single_pendulum_description/urdf/single_pendulum.urdf')
    ROBOT = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[urdf_dir])
    ROBOT.urdf = urdf_path
else:
    ROBOT = load(PENDULUM)

joints_name_list = [s for s in ROBOT.model.names[1:]] # skip the first name because it is "universe"
NQ = len(joints_name_list) 
NX = 2 * NQ
NU = NQ
KINDYN = KinDynComputations(ROBOT.urdf, joints_name_list)

VELOCITY_LIMIT = np.where(
    ROBOT.model.velocityLimit != 0,
    ROBOT.model.velocityLimit,
    5.0
)

ACCEL_LIMIT = np.array([9.81 * 2] * NQ)

TORQUE_LIMIT = np.where(
    ROBOT.model.effortLimit != 0,
    ROBOT.model.effortLimit,
    10.0
)

# OCP solver parameters
SOLVER_TOLERANCE = 1e-4
SOLVER_MAX_ITER = 1000

# OCP / simulation parameters
N = 100
DT = 0.02
M = 5
ENFORCE_BOUNDS = False

# Total simulation time steps
T = 750  

# Dataset / parallelism
NUM_SAMPLES = 10000
NUM_CORES = multiprocessing.cpu_count()

# Cost weights for single pendulum
W_P_single = 1e2
W_V_single = 1e1
W_T_single = 1e-1

# Cost weights for double pendulum
W_P_double = 1e2
W_V_double = 1e1
W_T_double = 1e-1

if PENDULUM == 'single_pendulum':
    W_P = W_P_single
    W_V = W_V_single
    W_T = W_T_single
else:
    W_P = W_P_double
    W_V = W_V_double
    W_T = W_T_double

# Neural network
HIDDEN_SIZE = 128
EPOCHS = 10000
BATCH_SIZE = 128
LR = 5e-4
PATIENCE = 100

# Convenience
VIEWER = True
SEED = 43

__all__ = [
    'robot', 'joints_name_list', 'nq', 'nx', 'nu', 'TORQUE_LIMIT', 'ACTUATED_INDICES',
    'N', 'M', 'dt', 'NUM_SAMPLES', 'NUM_CORES', 'W_P', 'W_V', 'W_A', 'W_T', 'SEED'
]