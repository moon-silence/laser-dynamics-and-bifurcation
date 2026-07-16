import numpy as np

def maxwell_bloch_rhs(state, t, kappa, gamma1, gamma2, lam):
    """
    Maxwell–Bloch laser model: [web:84][web:143]

        dE/dt = kappa (P - E)
        dP/dt = gamma1 (E*D - P)
        dD/dt = gamma2 (lam + 1 - D - lam*E*P)

    where:
      E     = electric field amplitude (real)
      P     = polarization
      D     = population inversion
      kappa = cavity decay rate
      gamma1 = polarization decay rate
      gamma2 = inversion decay rate
      lam    = pump parameter
    """
    E, P, D = state
    dE = kappa * (P - E)
    dP = gamma1 * (E * D - P)
    dD = gamma2 * (lam + 1.0 - D - lam * E * P)
    return np.array([dE, dP, dD])

def maxwell_bloch_fixed_points(lam):
    """
    Fixed point for the Maxwell–Bloch laser model (nonlasing state):

      E* = 0, P* = 0, D* = lam + 1. [web:84][web:143]
    """
    E_star = 0.0
    P_star = 0.0
    D_star = lam + 1.0
    return np.array([E_star, P_star, D_star])

def maxwell_bloch_jacobian(E, P, D, kappa, gamma1, gamma2, lam):
    """
    Jacobian matrix of the Maxwell–Bloch system at (E, P, D). [web:143]

      dE/dt = kappa (P - E)
      dP/dt = gamma1 (E*D - P)
      dD/dt = gamma2 (lam + 1 - D - lam*E*P)
    """
    # ∂(dE)/∂E = -kappa,  ∂(dE)/∂P = kappa,  ∂(dE)/∂D = 0
    # ∂(dP)/∂E = gamma1 * D, ∂(dP)/∂P = -gamma1, ∂(dP)/∂D = gamma1 * E
    # ∂(dD)/∂E = -gamma2 * lam * P,
    # ∂(dD)/∂P = -gamma2 * lam * E,
    # ∂(dD)/∂D = -gamma2
    J = np.array([
        [-kappa,           kappa,               0.0],
        [gamma1 * D,      -gamma1,             gamma1 * E],
        [-gamma2 * lam * P, -gamma2 * lam * E, -gamma2]
    ])
    return J

def lorenz_like_transform(state, lam):
    """
    Shift D to D' = D - (lam + 1), as in the standard Lorenz-style laser transformation. [web:84]

    Returns (E, P, D_prime).
    """
    E, P, D = state
    D_prime = D - (lam + 1.0)
    return np.array([E, P, D_prime])