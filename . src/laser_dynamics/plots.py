import matplotlib.pyplot as plt
from .models import jacobian_rate_equations

def plot_threshold_time(t, n, filename=None):
    """
    Plot photon number vs time for the 1D threshold model. [web:84]
    """
    plt.figure()
    plt.plot(t, n)
    plt.xlabel("time")
    plt.ylabel("photon number n(t)")
    plt.title("Laser threshold dynamics (1D model)")
    plt.grid(True)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_rate_time(t, I, N, filename=None):
    """
    Plot intensity and inversion vs time for the 2D rate-equation model. [web:63]
    """
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax[0].plot(t, I)
    ax[0].set_ylabel("Intensity I(t)")
    ax[0].grid(True)
    ax[1].plot(t, N)
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("Inversion N(t)")
    ax[1].grid(True)
    fig.suptitle("Laser rate-equation dynamics (2D model)")
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_phase_I_N(I, N, filename=None):
    """
    Phase plot in (I, N) plane for the 2D rate-equation model. [web:63]
    """
    plt.figure()
    plt.plot(I, N)
    plt.xlabel("Intensity I")
    plt.ylabel("Inversion N")
    plt.title("Phase plot in (I, N) plane")
    plt.grid(True)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def print_eigenvalues_at_equilibrium(I_eq, N_eq, gamma):
    """
    Compute and print eigenvalues at a given equilibrium. [web:63]
    """
    J = jacobian_rate_equations(I_eq, N_eq, gamma)
    eigvals = np.linalg.eigvals(J)
    print("Equilibrium (I, N) =", (I_eq, N_eq))
    print("Jacobian:\n", J)
    print("Eigenvalues:", eigvals)
    return eigvals
