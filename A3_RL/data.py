"""Dataset generation, bootstrapping and fine-tuning helpers."""
import numpy as np

from config import NQ


def generate_random_state():
    """
    Generates a random state vector for the robot.
    
    The state consists of joint positions (q) and joint velocities (dq).
    - Positions are sampled uniformly from [-pi, pi].
    - Velocities are sampled uniformly from [-8.0, 8.0].

    Returns:
        np.ndarray: A 1D array of size 2*NQ containing [q_1, ..., q_nq, dq_1, ..., dq_nq].
    """
    q_min, q_max = -np.pi, np.pi
    dq_min, dq_max = -8.0, 8.0
    
    # Random joint positions
    q_rand = np.random.uniform(q_min, q_max, NQ)
    # Random joint velocities
    dq_rand = np.random.uniform(dq_min, dq_max, NQ)
    
    return np.concatenate([q_rand, dq_rand])
