import sys
from pathlib import Path

# Make sure Python can import the src package when running this app
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src.laser_dynamics.simulate import simulate_rate_equations
from src.laser_dynamics.models import equilibria_rate_equations
from src.laser_dynamics.bifurcation import sweep_threshold_long_term
from src.laser_dynamics.advanced_models import (
    maxwell_bloch_rhs,
    maxwell_bloch_fixed_points,
    maxwell_bloch_jacobian,
)

from scipy.integrate import odeint

st.set_page_config(page_title="Laser Dynamics & Bifurcation", layout="wide")

st.title("Laser Dynamics, Bifurcation, and Lorenz-Type Behavior")

st.markdown(
    """
This app is an **interactive visualization** of laser models in nonlinear dynamics:

- A **2D rate-equation** laser model for intensity and inversion.
- A **3D Maxwell–Bloch** laser model that can be transformed into a Lorenz-type system.
- A simple **bifurcation-style view** showing how steady-state intensity depends on a control parameter.

Use the sliders to change parameters and see how dynamics and stability change.
"""
)

# --- Sidebar: explanation of why bifurcations matter ---
with st.sidebar:
    st.header("Why bifurcations matter")
    st.markdown(
        """
A **bifurcation** is a qualitative change in the long-term behavior of a
system when a parameter passes a critical value. [web:163][web:170]

In lasers, this can mean:
- switching from no lasing to steady lasing (threshold),
- changing from monotone approach to equilibrium to oscillations,
- or more complex routes toward chaotic behavior in higher-dimensional models.

Bifurcation diagrams plot steady-state or long-term values against a parameter,
revealing **critical points** and **branches** of behavior. [web:170]
"""
    )

# --- Tabs for different models ---
tab_2d, tab_3d, tab_bif = st.tabs(
    ["2D Rate-Equation Model", "3D Lorenz-Type Laser", "Threshold Bifurcation View"]
)

# --- 2D model tab ---
with tab_2d:
    st.subheader("2D Laser Rate-Equation Model")

    st.markdown(
        r"""
We consider intensity \(I(t)\) and inversion \(N(t)\):

\[
\dot I = (N - 1) I, \qquad
\dot N = \gamma(\mu - N - N I),
\]

with parameters:
- \(\mu\) = pump parameter,
- \(\gamma\) = ratio of time scales. [web:63]

Equilibria:
- Nonlasing: \((I^*, N^*) = (0, \mu)\),
- Lasing: \((I^*, N^*) = (\mu - 1, 1)\) (exists if \(\mu > 1\)).
"""
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        mu = st.slider("Pump parameter μ", 0.0, 3.0, 1.5, 0.1)
        gamma = st.slider("Time-scale ratio γ", 0.1, 5.0, 1.0, 0.1)
        I0 = st.number_input("Initial intensity I(0)", 0.0, 5.0, 0.01, 0.01)
        N0 = st.number_input("Initial inversion N(0)", 0.0, 5.0, mu, 0.1)
        t_max = st.slider("Simulation time", 10.0, 100.0, 50.0, 5.0)
        num_steps = 1000

    t = np.linspace(0.0, t_max, int(num_steps))
    state0 = np.array([I0, N0])

    t, I, N = simulate_rate_equations(state0, t, gamma, mu)
    eq_off, eq_on = equilibria_rate_equations(gamma, mu)

    with col_right:
        st.markdown("**Equilibria and linear stability**")
        st.write(f"Nonlasing equilibrium: I = {eq_off[0]:.3f}, N = {eq_off[1]:.3f}")
        if eq_on is not None:
            st.write(f"Lasing equilibrium: I = {eq_on[0]:.3f}, N = {eq_on[1]:.3f}")
        else:
            st.write("Lasing equilibrium does not exist for μ ≤ 1.")

    # Time series
    st.markdown("### Time series")
    fig_ts, ax_ts = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax_ts[0].plot(t, I)
    ax_ts[0].set_ylabel("Intensity I(t)")
    ax_ts[0].grid(True)
    ax_ts[1].plot(t, N)
    ax_ts[1].set_xlabel("time")
    ax_ts[1].set_ylabel("Inversion N(t)")
    ax_ts[1].grid(True)
    fig_ts.suptitle(f"2D laser dynamics (μ={mu:.2f}, γ={gamma:.2f})")
    st.pyplot(fig_ts)

    # Phase plot
    st.markdown("### Phase plot in (I, N) plane")
    fig_ph, ax_ph = plt.subplots(figsize=(5, 4))
    ax_ph.plot(I, N)
    ax_ph.set_xlabel("Intensity I")
    ax_ph.set_ylabel("Inversion N")
    ax_ph.set_title("Phase plot in (I, N)")
    ax_ph.grid(True)
    st.pyplot(fig_ph)

# --- 3D Lorenz-type laser tab ---
with tab_3d:
    st.subheader("3D Maxwell–Bloch / Lorenz-Type Laser Model")

    st.markdown(
        r"""
We consider a three-dimensional laser model with

- \(E(t)\): electric field amplitude,
- \(P(t)\): polarization,
- \(D(t)\): population inversion,

governed by

\[
\begin{aligned}
\dot E &= \kappa (P - E), \\
\dot P &= \gamma_1 (E D - P), \\
\dot D &= \gamma_2 \bigl(\lambda + 1 - D - \lambda E P\bigr),
\end{aligned}
\]

with decay rates \(\kappa, \gamma_1, \gamma_2 > 0\) and pump parameter \(\lambda\). [web:84][web:143]

The nonlasing equilibrium is
\[
E^* = 0,\quad P^* = 0,\quad D^* = \lambda + 1.
\]

With a suitable shift and rescaling, this system can be mapped to a Lorenz-type
model that exhibits chaotic dynamics for certain parameter values. [web:84][web:142]
"""
    )

    col3_l, col3_r = st.columns([1, 1])
    with col3_l:
        lam = st.slider("Pump λ", 0.0, 5.0, 1.5, 0.1)
        kappa = st.slider("κ (field decay)", 0.1, 5.0, 1.0, 0.1)
        gamma1 = st.slider("γ₁ (polarization decay)", 0.1, 5.0, 1.0, 0.1)
        gamma2 = st.slider("γ₂ (inversion decay)", 0.1, 5.0, 1.0, 0.1)

        E0 = st.number_input("Initial E(0)", -2.0, 2.0, 0.1, 0.1)
        P0 = st.number_input("Initial P(0)", -2.0, 2.0, 0.0, 0.1)
        D0 = st.number_input("Initial D(0)", -2.0, 5.0, 1.0, 0.1)

        t3_max = st.slider("Simulation time (3D)", 10.0, 100.0, 50.0, 5.0)
        n3_steps = 2000

    # Fixed point and Jacobian
    fp = maxwell_bloch_fixed_points(lam)
    J = maxwell_bloch_jacobian(fp[0], fp[1], fp[2], kappa, gamma1, gamma2, lam)
    eigvals = np.linalg.eigvals(J)
    eigvals_str = [f"{ev.real:.3f} + {ev.imag:.3f}i" for ev in eigvals]

    with col3_r:
        st.markdown("**Nonlasing equilibrium and eigenvalues**")
        st.write(f"Equilibrium (E*, P*, D*) = ({fp[0]:.3f}, {fp[1]:.3f}, {fp[2]:.3f})")
        st.write("Eigenvalues of Jacobian at this equilibrium (formatted):")
        for s in eigvals_str:
            st.write(s)

        st.markdown(
            """
Real parts of the eigenvalues indicate how perturbations behave:
- negative real parts → decay (stability),
- positive real parts → growth (instability),
- complex pairs → oscillatory behavior.
"""
        )

    # Time series for 3D system
    st.markdown("### 3D trajectories (E, P, D)")

    t3 = np.linspace(0.0, t3_max, int(n3_steps))
    state0_3d = np.array([E0, P0, D0])

    traj = odeint(
        lambda s, t_: maxwell_bloch_rhs(s, t_, kappa, gamma1, gamma2, lam),
        state0_3d, t3
    )
    E = traj[:, 0]
    P = traj[:, 1]
    D = traj[:, 2]

    fig3, ax3 = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ax3[0].plot(t3, E); ax3[0].set_ylabel("E(t)"); ax3[0].grid(True)
    ax3[1].plot(t3, P); ax3[1].set_ylabel("P(t)"); ax3[1].grid(True)
    ax3[2].plot(t3, D); ax3[2].set_xlabel("time"); ax3[2].set_ylabel("D(t)"); ax3[2].grid(True)
    fig3.suptitle(f"3D Maxwell–Bloch dynamics (λ={lam:.2f})")
    st.pyplot(fig3)

    # 2D projection of trajectory
    st.markdown("### Phase portrait (E vs P)")

    fig_ep, ax_ep = plt.subplots(figsize=(5, 4))
    ax_ep.plot(E, P)
    ax_ep.set_xlabel("E")
    ax_ep.set_ylabel("P")
    ax_ep.set_title("Projection of trajectory in (E, P) plane")
    ax_ep.grid(True)
    st.pyplot(fig_ep)

# --- Bifurcation tab ---
with tab_bif:
    st.subheader("Threshold Bifurcation View (1D Model)")

    st.markdown(
        r"""
To visualize **threshold as a bifurcation**, we use a simple one-dimensional
model for photon number \(n(t)\) and sweep the pump parameter \(N_0\). [web:84][web:170]

The bifurcation-style plot shows **long-term photon number** \(n_\infty\) versus
pump \(N_0\): below threshold, the stable branch is \(n_\infty = 0\), and above
threshold, a positive branch appears.
"""
    )

    G = st.slider("Gain G", 0.5, 2.0, 1.0, 0.1)
    k = st.slider("Loss k", 0.5, 2.0, 1.0, 0.1)
    alpha = st.slider("Nonlinearity α", 0.5, 2.0, 1.0, 0.1)

    st.markdown("### Pump vs long-term intensity")

    N0_values = np.linspace(0.0, 3.0, 120)
    N0_arr, n_long_arr = sweep_threshold_long_term(
        G=G, k=k, alpha=alpha, N0_values=N0_values, t_max=50.0, n0=0.01
    )

    fig_bif, ax_bif = plt.subplots(figsize=(6, 4))
    ax_bif.plot(N0_arr, n_long_arr, ".", markersize=3)
    ax_bif.set_xlabel("Pump N0")
    ax_bif.set_ylabel("Long-term photon number n∞")
    ax_bif.set_title("Threshold bifurcation-style plot")
    ax_bif.grid(True)
    st.pyplot(fig_bif)

    st.markdown(
        """
This kind of diagram is a graphical tool for understanding how steady states
change and where qualitative transitions (bifurcations) occur as parameters
are varied. [web:170]
"""
    )