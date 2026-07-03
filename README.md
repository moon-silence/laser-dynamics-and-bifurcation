# Laser Dynamics and Bifurcation

This project studies laser threshold and stability in simple rate-equation models,
then wraps the math in a small interactive demo. It is inspired by laser examples
in nonlinear dynamics texts (e.g., the laser threshold section in Strogatz) and
standard laser rate-equation literature on laser stability and bifurcation. [web:63][web:5]

## Problem

Semiconductor lasers exhibit threshold behavior: below a critical pump level they
do not lase, and above it they sustain a stable optical intensity. Small changes
in parameters can also change the qualitative dynamics — for example, switching
from monotone decay to damped oscillations. This project asks:

> How do simple laser rate-equation models encode threshold and stability,
> and how do parameter changes alter the long-term dynamics?

## Math & methods

I implement two models:

- **1D threshold model** for photon number:
  \[
  \dot n = (G N_0 - k)n - \alpha G n^2,
  \]
  which shows a transition from zero to positive stable intensity as the
  pump parameter \(N_0\) crosses a threshold. [web:84]

- **2D rate-equation model** for intensity \(I\) and inversion \(N\):
  \[
  \dot I = (N - 1)I, \qquad
  \dot N = \gamma(\mu - N - N I),
  \]
  whose equilibria and Jacobian eigenvalues describe nonlasing and lasing
  states and their stability. [web:63]

Numerical integration uses `scipy.integrate.odeint`, and plots are generated
with `matplotlib`. The Streamlit demo lets you vary parameters and immediately
see how trajectories change.

## Results (at a glance)

- The 1D model reproduces a **laser threshold**: below a critical pump value,
  the only stable state is zero intensity; above threshold, a positive intensity
  equilibrium becomes stable. [web:84]

- In the 2D model, the **nonlasing equilibrium** \((I=0, N=\mu)\) is stable for
  \(\mu < 1\) and loses stability for \(\mu > 1\), while the **lasing equilibrium**
  \((I=\mu-1, N=1)\) exists and is stable when \(\mu > 1\). [web:63]

- Time series and phase plots show how changing \(\mu\) and \(\gamma\) alters the
  approach to equilibrium (monotone vs damped oscillations).

> (You can add `figures/` images later and embed them here.)

## Demo

Once dependencies are installed locally:

```bash
pip install -r requirements.txt
streamlit run web_demo/app.py
```

This opens a browser UI where you can:

- Slide the pump parameter \(\mu\) and time-scale ratio \(\gamma\).
- See the intensity \(I(t)\) and inversion \(N(t)\) time series.
- Inspect the phase plot in the \((I, N)\) plane.

## Files

- `src/laser_dynamics/models.py` — threshold and rate-equation models.
- `src/laser_dynamics/simulate.py` — numerical integration helpers.
- `src/laser_dynamics/plots.py` — plotting utilities.
- `web_demo/app.py` — Streamlit demo.
- `report/main.tex` — LaTeX write-up.
- `notebooks/01_threshold_and_stability.ipynb` — analysis notebook (to be added).
