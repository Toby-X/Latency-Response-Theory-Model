"""Synthetic data generator matching Section 6 of the LaRT paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


@dataclass(frozen=True)
class SyntheticLaRTData:
    response: NDArray[np.int64]
    latency: NDArray[np.float64]
    theta: NDArray[np.float64]
    tau: NDArray[np.float64]
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    omega: NDArray[np.float64]
    phi: NDArray[np.float64]
    lam: NDArray[np.float64]
    rho: float


def generate_lart_data(
    n_models: int = 100,
    n_items: int = 50,
    rho: float = -0.8,
    seed: int | None = None,
) -> SyntheticLaRTData:
    """Generate responses and latencies under the paper's assumed model.

    Item parameters follow Section 6: ``a ~ U(0.5, 1)``, ``b ~ N(0, .5)``,
    ``omega ~ N(0, 1)``, ``phi ~ U(0.5, 1.5)``, and ``lambda ~ U(0.5, 2)``.
    """

    if n_models < 2 or n_items < 2:
        raise ValueError("n_models and n_items must both be at least 2")
    if not -1 < rho < 1:
        raise ValueError("rho must lie strictly between -1 and 1")

    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 1.0, n_items)
    b = rng.normal(0.0, 0.5, n_items)
    omega = rng.normal(0.0, 1.0, n_items)
    phi = rng.uniform(0.5, 1.5, n_items)
    lam = rng.uniform(0.5, 2.0, n_items)

    sigma = np.array([[1.0, rho], [rho, 1.0]])
    traits = rng.multivariate_normal(np.zeros(2), sigma, n_models)
    theta, tau = traits[:, 0], traits[:, 1]

    probability = norm.cdf(theta[:, None] * a[None, :] + b[None, :])
    response = rng.binomial(1, probability)
    log_latency = rng.normal(
        omega[None, :] - tau[:, None] * phi[None, :],
        lam[None, :],
    )

    return SyntheticLaRTData(
        response=response,
        latency=np.exp(log_latency),
        theta=theta,
        tau=tau,
        a=a,
        b=b,
        omega=omega,
        phi=phi,
        lam=lam,
        rho=rho,
    )
