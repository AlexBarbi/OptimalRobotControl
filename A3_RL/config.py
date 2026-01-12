"""Project configuration and constants used across modules."""
import os
import multiprocessing
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
TORQUE_LIMIT = getattr(ROBOT, 'umax', getattr(ROBOT, 'torque_limit', 50.0))

# OCP / simulation parameters
N = 50
DT = 0.02
M = 5

T = 500  # Total simulation time steps

# Dataset / parallelism
NUM_SAMPLES = 10000
NUM_CORES = multiprocessing.cpu_count()

# Cost weights
W_Q = 1000
W_V = 0.1
W_U = 1e-4

# Convenience
SEED = 42

__all__ = [
    'robot', 'joints_name_list', 'nq', 'nx', 'nu', 'TORQUE_LIMIT', 'ACTUATED_INDICES',
    'N', 'M', 'dt', 'NUM_SAMPLES', 'NUM_CORES', 'W_Q', 'W_V', 'W_U', 'SEED'
]