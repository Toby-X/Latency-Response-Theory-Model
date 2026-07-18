"""Small, validated public API around the paper's reference implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .estimation import MIRT_SAEM_full, cMIRT_SAEM_full


@dataclass(frozen=True)
class LaRTResult:
    """Estimates returned by the Latency-Response Theory model."""

    theta: NDArray[np.float64]
    tau: NDArray[np.float64]
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    omega: NDArray[np.float64]
    phi: NDArray[np.float64]
    lam: NDArray[np.float64]
    rho: float
    n_iter: int


@dataclass(frozen=True)
class IRTResult:
    """Estimates returned by the normal-ogive IRT baseline."""

    theta: NDArray[np.float64]
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    sigma2: float
    n_iter: int


def _response_matrix(response: ArrayLike) -> NDArray[np.float64]:
    value = np.asarray(response, dtype=float)
    if value.ndim != 2 or value.size == 0:
        raise ValueError("response must be a non-empty N x J matrix")
    if not np.isfinite(value).all() or not np.isin(value, (0.0, 1.0)).all():
        raise ValueError("response must contain only finite 0/1 values")
    return value


def fit_lart(response: ArrayLike, latency: ArrayLike, **kwargs) -> LaRTResult:
    """Fit LaRT to binary responses and strictly positive CoT token counts.

    Parameters are passed through to the reference ``cMIRT_SAEM_full`` routine;
    useful reproducibility options include ``seed``, ``n_samples``, ``eps``, and
    ``max_iter``.
    """

    response_array = _response_matrix(response)
    latency_array = np.asarray(latency, dtype=float)
    if latency_array.shape != response_array.shape:
        raise ValueError("latency must have the same N x J shape as response")
    if not np.isfinite(latency_array).all() or np.any(latency_array <= 0):
        raise ValueError("latency must contain only finite, strictly positive values")

    values = cMIRT_SAEM_full(response_array, latency_array, **kwargs)
    return LaRTResult(*values)


def fit_irt(response: ArrayLike, **kwargs) -> IRTResult:
    """Fit the paper's normal-ogive IRT baseline."""

    values = MIRT_SAEM_full(_response_matrix(response), **kwargs)
    return IRTResult(*values)
