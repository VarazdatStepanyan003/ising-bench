from engine import many_simulate, smooth
from bin.helpers import interpolate
from numba import njit


@njit(nogil=True)
def run(n, m):
    n_of_steps = n
    max_time = 250
    n_of_repetitions = m
    dt = 1
    times, observabless = many_simulate(n_of_steps=n_of_steps, max_time=max_time, n_of_repetitions=n_of_repetitions)
