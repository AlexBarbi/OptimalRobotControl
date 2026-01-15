"""Project configuration and constants used across modules."""
import os
import multiprocessing
from example_robot_data.robots_loader import load
import pinocchio as pin
from adam.casadi.computations import KinDynComputations
import numpy as np

PENDULUM = os.environ.get('ROBOT_TYPE', 'double_pendulum').lower()

if PENDULUM == 'single_pendulum':
    urdf_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(urdf_dir, 'single_pendulum_description/urdf/single_pendulum.urdf')
    ROBOT = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[urdf_dir])
    ROBOT.urdf = urdf_path
else:
    ROBOT = load(PENDULUM)
    # urdf_dir = os.path.dirname(os.path.abspath(__file__))
    # urdf_path = os.path.join(urdf_dir, 'double_pendulum_description/urdf/double_pendulum.urdf')
    # ROBOT = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[urdf_dir])
    # ROBOT.urdf = urdf_path

joints_name_list = [s for s in ROBOT.model.names[1:]] # skip the first name because it is "universe"
NQ = len(joints_name_list)  # number of joints
NX = 2 * NQ
NU = NQ
KINDYN = KinDynComputations(ROBOT.urdf, joints_name_list)

VELOCITY_LIMIT = np.where(
    ROBOT.model.velocityLimit != 0,
    ROBOT.model.velocityLimit,
    20.0
)

ACCEL_LIMIT = np.array([9.81 * 2] * NQ)

TORQUE_LIMIT = np.where(
    ROBOT.model.effortLimit != 0,
    ROBOT.model.effortLimit,
    10.0
)

# OCP / simulation parameters
N = 100
DT = 0.02
M = 5

T = 500  # Total simulation time steps

# Dataset / parallelism
NUM_SAMPLES = 5000
NUM_CORES = multiprocessing.cpu_count()

# Cost weights
W_P = 100.0
W_V = 10.0
W_A = 1.0e-3

# Neural network
HIDDEN_SIZE = 128
EPOCHS = 1000
BATCH_SIZE = 128
LR = 5e-4
PATIENCE = 100

# Convenience
VIEWER = False
SEED = 55

__all__ = [
    'robot', 'joints_name_list', 'nq', 'nx', 'nu', 'TORQUE_LIMIT', 'ACTUATED_INDICES',
    'N', 'M', 'dt', 'NUM_SAMPLES', 'NUM_CORES', 'W_Q', 'W_V', 'W_U', 'SEED'
]