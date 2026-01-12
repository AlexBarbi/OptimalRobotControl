"""Dataset generation, bootstrapping and fine-tuning helpers."""
import numpy as np

from config import NQ


def generate_random_state():
    q_min, q_max = -np.pi, np.pi
    dq_min, dq_max = -8.0, 8.0
    q_rand = np.random.uniform(q_min, q_max, NQ)
    dq_rand = np.random.uniform(dq_min, dq_max, NQ)
    return np.concatenate([q_rand, dq_rand])