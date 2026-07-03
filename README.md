# laser-dynamics-and-bifurcation
# Laser Dynamics and Bifurcation

This project studies laser threshold and stability in simple rate-equation models,
then wraps the math in a small interactive demo. It is inspired by laser examples
in nonlinear dynamics texts (e.g., the laser threshold section in Strogatz) and
standard laser rate-equation literature. [web:63][web:5]

## Project idea

We start from a one-dimensional laser threshold model for photon number, then
extend to a two-dimensional intensity–inversion rate-equation system. The goals:

- Find equilibria and threshold conditions.
- Analyze local stability via Jacobians and eigenvalues.
- Visualize dynamics with time series and phase plots.
- Provide a simple web UI for exploring parameter effects.

## Code layout

- `src/laser_dynamics/models.py` — laser equations (1D threshold + 2D rate equations).
- `src/laser_dynamics/simulate.py` — numerical integration wrappers (`scipy.integrate`).
- `src/laser_dynamics/plots.py` — plotting utilities (time series, phase plots).
- `web_demo/app.py` — Streamlit app for interactive exploration.
- `notebooks/01_threshold_and_stability.ipynb` — narrative analysis.
- `report/main.tex` — project write-up.

## Getting started (locally)

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the web demo:

   ```bash
   streamlit run web_demo/app.py
   ```

## Mathematical models (high level)

- **1D threshold model**: a nonlinear growth law for photon number showing
  the transition from no lasing to positive stable intensity. [web:84]

- **2D rate-equation model**: coupled equations for intensity and inversion,
  used to study equilibria and local stability via Jacobians and eigenvalues. [web:63]

Details are in the notebook and report.
