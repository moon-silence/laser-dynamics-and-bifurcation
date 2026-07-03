import numpy as np
from scipy.integrate import odeint
from .models import threshold_rhs, rate_equations_rhs

def simulate_threshold(n0, t, G, N0, k, alpha):
    """
    Simulate the 1D threshold equation from initial photon number n0. [web:84]
    """
    def rhs(n, t_):
        return threshold_rhs(n, t_, G, N0, k, alpha)

    n = odeint(rhs, n0, t).flatten()
    return t, n

def simulate_rate_equations(state0, t, gamma, mu):
    """
    Simulate the 2D laser rate equations from initial state (I0, N0). [web:63]
    """
    def rhs(state, t_):
        return rate_equations_rhs(state, t_, gamma, mu)

    traj = odeint(rhs, state0, t)
    I = traj[:, 0]
    N = traj[:, 1]
    return t, I, N
