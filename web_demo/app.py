import numpy as np
import streamlit as st
from src.laser_dynamics.simulate import simulate_rate_equations
from src.laser_dynamics.models import equilibria_rate_equations
import matplotlib.pyplot as plt

st.title("Laser Dynamics: Threshold and Stability")

st.markdown(
    "This demo shows a simple 2D laser rate-equation model. "
    "Move the sliders to change the pump parameter μ and time-scale ratio γ, "
    "then see how the intensity and inversion evolve."
)

mu = st.slider("Pump parameter μ", min_value=0.0, max_value=3.0, value=1.5, step=0.1)
gamma = st.slider("Time-scale ratio γ", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

t = np.linspace(0.0, 50.0, 1000)
state0 = np.array([0.01, mu])  # small initial intensity, inversion at μ

t, I, N = simulate_rate_equations(state0, t, gamma, mu)
eq_off, eq_on = equilibria_rate_equations(gamma, mu)

st.subheader("Time series")
fig_ts, ax_ts = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
ax_ts[0].plot(t, I)
ax_ts[0].set_ylabel("Intensity I(t)")
ax_ts[0].grid(True)
ax_ts[1].plot(t, N)
ax_ts[1].set_xlabel("time")
ax_ts[1].set_ylabel("Inversion N(t)")
ax_ts[1].grid(True)
fig_ts.suptitle("Laser rate-equation dynamics")
st.pyplot(fig_ts)

st.subheader("Phase plot")
fig_ph, ax_ph = plt.subplots(figsize=(5, 4))
ax_ph.plot(I, N)
ax_ph.set_xlabel("Intensity I")
ax_ph.set_ylabel("Inversion N")
ax_ph.set_title("Phase plot in (I, N) plane")
ax_ph.grid(True)
st.pyplot(fig_ph)

st.markdown("**Equilibria**")
st.write(f"Nonlasing equilibrium: I = {eq_off[0]:.3f}, N = {eq_off[1]:.3f}")
if eq_on is not None:
    st.write(f"Lasing equilibrium: I = {eq_on[0]:.3f}, N = {eq_on[1]:.3f}")
else:
    st.write("Lasing equilibrium does not exist for μ ≤ 1.")
