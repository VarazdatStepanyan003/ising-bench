from numba import njit
from math import floor
import numpy as np


@njit(nogil=True)
def binary_search(x, arr):
    a = 0
    b = len(arr) - 1
    while a < b:
        m = floor((a + b) / 2)
        if arr[m + 1] <= x:
            a = m + 1
        elif arr[m] > x:
            b = m
        else:
            return m
    return -1


@njit(nogil=True)
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


@njit(nogil=True)
def interpolate(y):
    nans, x = nan_helper(y)
    y_new = np.copy(y)
    y_new[nans] = np.interp(x(nans), x(~nans), y_new[~nans])
    return y_new