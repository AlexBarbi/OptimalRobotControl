"""Project configuration and constants used across modules."""
import numpy as np
from pendulum import Pendulum
import multiprocessing

# Robot init
robot = Pendulum(2, open_viewer=False)
joints_name_list = [s for s in robot.model.names[1:]]
nq = len(joints_name_list)
nx = 2 * nq
nu = nq

# Actuation and limits
TORQUE_LIMIT = getattr(robot, 'umax', 2.0)
ACTUATED_INDICES = getattr(robot, 'actuated_indices', list(range(robot.nu)))

# OCP / simulation parameters
N = 100
dt = 0.02
M = 10

# Dataset / parallelism
NUM_SAMPLES = 5000
NUM_CORES = multiprocessing.cpu_count()

# Cost weights
W_Q = 10.0
W_V = 1.0
W_U = 0.1

# Convenience
SEED = None

__all__ = [
    'robot', 'joints_name_list', 'nq', 'nx', 'nu', 'TORQUE_LIMIT', 'ACTUATED_INDICES',
    'N', 'M', 'dt', 'NUM_SAMPLES', 'NUM_CORES', 'W_Q', 'W_V', 'W_U', 'SEED'
]