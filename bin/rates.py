from numba import njit
import numpy as np


@njit(nogil=True)
def sigmoid(x):
    return 1 / (1 + np.exp(x)), 1
