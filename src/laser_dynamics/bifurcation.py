import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from .models import threshold_rhs

def sweep_threshold_long_term(G=1.0, k=1.0, alpha=1.0,
                              N0_values=None,
                              t_max=50.0, n0=0.01):
    """
    Sweep over pump values N0 and record the long-term photon number
    for the 1D threshold model.

    Returns (N0_array, n_long_array). [web:84]
    """
    if N0_values is None:
        N0_values = np.linspace(0.0, 3.0, 100)

    t = np.linspace(0.0, t_max, 1000)
    n_long = []

    for N0 in N0_values:
        def rhs(n, t_):
            return threshold_rhs(n, t_, G, N0, k, alpha)
        n = odeint(rhs, n0, t).flatten()
        # Take the last value as "long-term" photon number
        n_long.append(n[-1])

    return np.array(N0_values), np.array(n_long)

def plot_threshold_bifurcation(G=1.0, k=1.0, alpha=1.0,
                               N0_min=0.0, N0_max=3.0, num_points=100,
                               t_max=50.0, n0=0.01,
                               filename=None):
    """
    Produce a simple bifurcation-style plot: pump N0 vs long-term photon number.
    [web:84]
    """
    N0_values = np.linspace(N0_min, N0_max, num_points)
    N0_arr, n_long_arr = sweep_threshold_long_term(G, k, alpha,
                                                   N0_values=N0_values,
                                                   t_max=t_max, n0=n0)

    plt.figure()
    plt.plot(N0_arr, n_long_arr, ".")
    plt.xlabel("Pump N0")
    plt.ylabel("Long-term photon number n∞")
    plt.title("Laser threshold bifurcation-style plot")
    plt.grid(True)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    else:
        plt.show()
