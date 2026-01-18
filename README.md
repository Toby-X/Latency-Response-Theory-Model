# Latency Response Theory Model (LART)
This is the codebase for the paper "**La**tency **R**esponse **T**heory Model".

<!-- Thanks for the python-implementation of efficient sampler for truncated normal distributions from [truncated-mvn-sampler](https://github.com/brunzema/truncated-mvn-sampler?tab=readme-ov-file). -->

## Installation
Install the required packages using the following command in your python environment:
```bash
pip install -r requirements.txt
```

## Repository Structure

- `LaRT.py`: The core library containing model definitions and estimation algorithms.
- `utils.py`: The utility functions for sampling and initialization.
- `minimax_tilting_sampler.py`: The minimax tilting sampler for truncated normal distributions, provided by [truncated-mvn-sampler](https://github.com/brunzema/truncated-mvn-sampler?tab=readme-ov-file).
- `simulation/`: The simulation code.
- `application/`: The real data code.
- `requirements.txt`: The required packages.
- `data/`: The data files for LLM generated responses for mathematics questions from multiple benchmark datasets.


## Usage

To run the LART, use the following command in your python environment:
```python
from LaRT import LaRT_SAEM_full
import numpy as np

# Data Preparation (without following the model assumptions)
R = np.random.randint(0, 2, size=(100, 10))
X = np.random.lognormal(size=(100, 5))

# Fit the model
theta_est, tau_est, a_est, b_est, omega_est, phi_est, lam_est, rho_est, n_iter = LaRT_SAEM_full(R, X)

# Get the results
print(theta_est)
print(tau_est)
print(a_est)
print(b_est)
print(omega_est)
print(phi_est)
print(lam_est)
print(rho_est)
print(n_iter)
```