"""Reproduce Section 7.2.1 predictive power on AMC23 + AIME24 + AIME25."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from lart import irt_saem_full, lart_saem_full, update_indi_fixed_all


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "benchmarks"
OUTPUT_DIR = REPO_ROOT / "results" / "applications"


def load_combined_benchmarks() -> tuple[np.ndarray, np.ndarray]:
    """Load the paper's 128-model, 100-item three-benchmark matrix."""
    response = pd.read_csv(DATA_DIR / "correctness_matrix_combined.csv", index_col=0)
    latency = pd.read_csv(DATA_DIR / "cot_length_matrix_combined.csv", index_col=0)
    if response.shape != (128, 100) or latency.shape != response.shape:
        raise ValueError(
            "combined matrices must contain 128 LLMs and the 100 AMC23/AIME24/AIME25 items"
        )
    if not response.index.equals(latency.index):
        raise ValueError("combined response and latency rows do not match")
    return response.to_numpy(), latency.to_numpy(dtype=float) + 1.0


binary_array, cot_array = load_combined_benchmarks()


def training_split(n_models: int, n_train: int = 100, seed: int = 42) -> np.ndarray:
    """Return the fixed training mask used in the manuscript."""
    if not 1 <= n_train < n_models:
        raise ValueError("n_train must leave at least one held-out model")
    rng = np.random.RandomState(seed)
    chosen = rng.choice(n_models, n_train, replace=False)
    mask = np.zeros(n_models, dtype=bool)
    mask[chosen] = True
    return mask


def fit_training_models(
    response: np.ndarray,
    latency: np.ndarray,
    *,
    max_iter: int = 100,
    seed: int = 42,
) -> dict[str, np.ndarray | float]:
    """Fit LaRT and IRT item parameters on the training LLMs."""
    lart = lart_saem_full(response, latency, n_samples=1, seed=seed, max_iter=max_iter)
    irt = irt_saem_full(response, n_samples=1, seed=seed, max_iter=max_iter)
    return {
        "theta_lart_train": lart[0],
        "tau_lart_train": lart[1],
        "a_lart_train": lart[2],
        "b_lart_train": lart[3],
        "omega_lart_train": lart[4],
        "phi_lart_train": lart[5],
        "lam_lart_train": lart[6],
        "rho_lart": float(lart[7]),
        "theta_irt_train": irt[0],
        "a_irt_train": irt[1],
        "b_irt_train": irt[2],
        "sigma_irt_train": float(irt[3]),
    }


def update_irt_theta(
    theta_initial: np.ndarray,
    response: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    sigma2: float = 1.0,
) -> np.ndarray:
    """MAP-update held-out LLM abilities with fixed normal-ogive IRT items."""
    n_models = response.shape[0]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        signed = (2 * response - 1) * (
            theta[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :]
        )
        log_likelihood = np.sum(norm.logcdf(signed)) - np.sum(theta**2) / (2 * sigma2)
        ratio = np.exp(
            np.nan_to_num(norm.logpdf(signed) - norm.logcdf(signed), neginf=0.0)
        )
        gradient = np.sum(
            ratio * (2 * response - 1) * a[np.newaxis, :], axis=1
        ) - theta / sigma2
        return -log_likelihood / n_models, -gradient / n_models

    result = minimize(
        objective,
        theta_initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxls": 50, "ftol": 1e-10, "gtol": 1e-7},
    )
    return result.x


def cross_validated_mae(
    response: np.ndarray,
    latency: np.ndarray,
    parameters: dict[str, np.ndarray | float],
    *,
    n_folds: int = 5,
    seed: int = 1025,
) -> pd.DataFrame:
    """Infer held-out LLM traits and score held-out items by mean absolute error."""
    n_models, n_items = response.shape
    fold = np.arange(n_items) % n_folds
    np.random.RandomState(seed).shuffle(fold)
    rows = []
    for fold_index in range(n_folds):
        test = fold == fold_index
        train = ~test
        theta0 = np.zeros(n_models)
        tau0 = np.zeros(n_models)
        theta_lart, _ = update_indi_fixed_all(
            theta0,
            tau0,
            response[:, train],
            np.log(latency[:, train]),
            np.asarray(parameters["a_lart_train"])[train],
            np.asarray(parameters["b_lart_train"])[train],
            np.asarray(parameters["omega_lart_train"])[train],
            np.asarray(parameters["phi_lart_train"])[train],
            np.asarray(parameters["lam_lart_train"])[train],
            float(parameters["rho_lart"]),
        )
        theta_irt = update_irt_theta(
            theta0,
            response[:, train],
            np.asarray(parameters["a_irt_train"])[train],
            np.asarray(parameters["b_irt_train"])[train],
        )
        lart_probability = norm.cdf(
            np.asarray(parameters["a_lart_train"])[test][np.newaxis, :]
            * theta_lart[:, np.newaxis]
            + np.asarray(parameters["b_lart_train"])[test][np.newaxis, :]
        )
        irt_probability = norm.cdf(
            np.asarray(parameters["a_irt_train"])[test][np.newaxis, :]
            * theta_irt[:, np.newaxis]
            + np.asarray(parameters["b_irt_train"])[test][np.newaxis, :]
        )
        rows.append(
            {
                "fold": fold_index + 1,
                "lart_mae": np.mean(np.abs(response[:, test] - lart_probability)),
                "irt_mae": np.mean(np.abs(response[:, test] - irt_probability)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    train = training_split(binary_array.shape[0])
    parameters = fit_training_models(binary_array[train], cot_array[train])
    scores = cross_validated_mae(binary_array[~train], cot_array[~train], parameters)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_DIR / "rest3_pred_params.npz", **parameters)
    scores.to_csv(OUTPUT_DIR / "predictive_power_mae.csv", index=False)
    print(scores.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Average MAE:", scores[["lart_mae", "irt_mae"]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
