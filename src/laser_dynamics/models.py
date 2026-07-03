import numpy as np

def threshold_rhs(n, t, G, N0, k, alpha):
    """
    1D laser threshold equation:
        dn/dt = (G*N0 - k)*n - alpha*G*n**2

    where:
      n     = photon number
      G     = gain coefficient
      N0    = pump / initial excited atoms
      k     = loss coefficient
      alpha = depletion factor

    This is the standard nonlinear growth law showing laser threshold. [web:84]
    """
    a = G * N0 - k
    b = alpha * G
    return a * n - b * n**2

def rate_equations_rhs(state, t, gamma, mu):
    """
    2D laser rate equations:
        dI/dt = (N - 1)*I
        dN/dt = gamma*(mu - N - N*I)

    where:
      I     = intensity
      N     = inversion
      mu    = pump parameter
      gamma = time-scale ratio

    This model is a standard baseline for local laser stability analysis. [web:63]
    """
    I, N = state
    dI = (N - 1.0) * I
    dN = gamma * (mu - N - N * I)
    return np.array([dI, dN])

def equilibria_rate_equations(gamma, mu):
    """
    Return equilibrium points for the 2D system:

      (I, N) = (0, mu)          nonlasing equilibrium
      (I, N) = (mu - 1, 1)      lasing equilibrium (exists if mu > 1)

    [web:63]
    """
    eq_off = np.array([0.0, mu])
    eq_on = np.array([mu - 1.0, 1.0]) if mu > 1.0 else None
    return eq_off, eq_on

def jacobian_rate_equations(I, N, gamma):
    """
    Jacobian matrix for the 2D rate equations:

        J = [[N - 1,     I],
             [-gamma*N, -gamma*(1 + I)]]

    [web:63]
    """
    J = np.array([[N - 1.0,     I],
                  [-gamma * N, -gamma * (1.0 + I)]])
    return J
