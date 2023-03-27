from numba import njit
import numpy as np
from bin.rates import sigmoid

state_init = np.ones(100)
N = len(state_init)
rate_constant = 0.1
n_of_observables = 2


@njit(nogil=True)
def observables(state, time):
    m = np.mean(state)
    e = 0
    for i in range(N):
        e += state[i] * state[(i + 1) % N]
    return m, e / N


@njit(nogil=True)
def energy(state, time):
    s = 0
    for i in range(N):
        s += -J(time) * state[i] * state[(i + 1) % N] - h(time) * state[i]
    return s


@njit(nogil=True)
def decide(state, time):
    nstate = swap(state, np.random.randint(N))
    r, R = sigmoid(beta(time) * (energy(nstate, time) - energy(state, time)))
    u = 1 - np.random.random()
    dt = (np.log(1 / u) * rate_constant / R) / N
    if np.random.uniform(0, R) <= r:
        return True, nstate, dt
    return False, state, dt


@njit(nogil=True)
def swap(state, index):
    nstate = np.copy(state)
    nstate[index] *= -1
    return nstate


@njit(nogil=True)
def J(time):
    return 1


@njit(nogil=True)
def h(time):
    return 0.1


@njit(nogil=True)
def beta(time):
    return 1