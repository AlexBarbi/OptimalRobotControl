"""Dataset generation, bootstrapping and fine-tuning helpers."""
import numpy as np

from config import NQ, ROBOT


def generate_random_state(q_min, q_max, dq_max):
    """
    Generates a random state vector for the robot.
    """
    q_min = np.asarray(q_min)
    q_max = np.asarray(q_max)
    dq_max = np.asarray(dq_max)

    assert q_min.shape == q_max.shape == dq_max.shape, "q_min, q_max, and dq_max must have the same shape"

    q_rand = np.random.uniform(low=q_min, high=q_max)
    dq_rand = np.random.uniform(low=-dq_max, high=dq_max)

    return np.concatenate([q_rand, dq_rand])
