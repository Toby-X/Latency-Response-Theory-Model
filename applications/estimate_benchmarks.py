"""Fit LaRT and normal-ogive IRT separately to the four paper benchmarks.

This reproduces the dataset-specific fits used in Section 7.1.  The four
datasets are never concatenated for this analysis.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import multiprocess as mp
import numpy as np
import pandas as pd

from lart import irt_saem_full, lart_saem_full


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "benchmarks"
OUTPUT_DIR = REPO_ROOT / "results" / "applications"
BENCHMARKS = ("math500", "amc23", "aime24", "aime25")

# These incomplete model/prompt rows are excluded in the paper analysis.
EXCLUDED_ROWS = (
    "meta_llama_Llama_3.2_1B_one_shot",
    "meta_llama_Llama_3.2_1B_zero_shot",
    "meta_llama_Meta_Llama_3_8B_one_shot",
    "meta_llama_Meta_Llama_3_8B_zero_shot",
    "microsoft_phi_4_one_shot",
    "microsoft_phi_4_zero_shot",
    "TinyLlama_TinyLlama_1.1B_Chat_v1.0_zero_shot",
    "TinyLlama_TinyLlama_1.1B_Chat_v1.0_one_shot",
    "google_gemma_3_1b_pt_one_shot",
    "google_gemma_3_1b_pt_zero_shot",
    "google_gemma_7b_it_one_shot",
    "google_gemma_7b_it_zero_shot",
    "google_vaultgemma_1b_one_shot",
    "google_vaultgemma_1b_zero_shot",
    "meta_llama_Llama_3.2_3B_one_shot",
    "meta_llama_Llama_3.2_3B_zero_shot",
    "openai_community_gpt2_one_shot",
    "openai_community_gpt2_zero_shot",
)


def load_benchmark(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one benchmark and apply the paper's row/token preprocessing."""
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; choose from {BENCHMARKS}")
    response = pd.read_csv(
        DATA_DIR / f"correctness_matrix_{name}.csv", index_col=0
    ).drop(list(EXCLUDED_ROWS), errors="ignore")
    latency = pd.read_csv(
        DATA_DIR / f"cot_length_matrix_{name}.csv", index_col=0
    ).drop(list(EXCLUDED_ROWS), errors="ignore")
    if not response.index.equals(latency.index):
        raise ValueError(f"row mismatch in {name}")
    return response.to_numpy(), latency.to_numpy(dtype=float) + 1.0


def run_model_task(
    task: tuple[str, str], max_iter: int = 100, seed: int = 42
) -> dict[str, object]:
    """Fit one of the two paper models to one benchmark."""
    dataset, model = task
    response, latency = load_benchmark(dataset)
    if model == "LaRT":
        theta, tau, a, b, omega, phi, lam, rho, n_iter = lart_saem_full(
            response, latency, n_samples=1, seed=seed, max_iter=max_iter
        )
        return {
            "dataset": dataset,
            "model": model,
            "theta": theta,
            "tau": tau,
            "a": a,
            "b": b,
            "omega": omega,
            "phi": phi,
            "lam": lam,
            "rho": rho,
            "iter": n_iter,
        }
    if model == "IRT":
        theta, a, b, sigma, n_iter = irt_saem_full(
            response, n_samples=1, seed=seed, max_iter=max_iter
        )
        return {
            "dataset": dataset,
            "model": model,
            "theta_irt": theta,
            "a_irt": a,
            "b_irt": b,
            "sigma_irt": sigma,
            "iter_irt": n_iter,
        }
    raise ValueError("model must be 'LaRT' or 'IRT'")


def merge_results(results: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Merge the LaRT and IRT records for each benchmark."""
    merged: dict[str, dict[str, object]] = {}
    for result in results:
        dataset = str(result["dataset"])
        merged.setdefault(dataset, {"dataset": dataset}).update(result)
        merged[dataset].pop("model", None)
    return merged


def save_results(merged: dict[str, dict[str, object]]) -> None:
    """Write files matching the retained Section 7.1 processed artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    math = merged["math500"]
    math_frame = pd.DataFrame(
        {
            "theta_joint": [math["theta"]],
            "tau_joint": [math["tau"]],
            "a_joint": [math["a"]],
            "b_joint": [math["b"]],
            "omega_joint": [math["omega"]],
            "phi_joint": [math["phi"]],
            "lam_joint": [math["lam"]],
            "rho_joint": [math["rho"]],
            "theta_irt": [math["theta_irt"]],
            "a_irt": [math["a_irt"]],
            "b_irt": [math["b_irt"]],
            "sigma2_irt": [math["sigma_irt"]],
            "n_iter_joint": [math["iter"]],
            "n_iter_irt": [math["iter_irt"]],
        }
    )
    math_frame.to_parquet(
        OUTPUT_DIR / "estimated_parameters_math500_all.parquet", index=False
    )

    other = pd.DataFrame([merged[name] for name in ("amc23", "aime24", "aime25")])
    other.to_parquet(OUTPUT_DIR / "estimated_parameters_three_benchmarks.parquet", index=False)


def main() -> None:
    tasks = [(dataset, model) for dataset in BENCHMARKS for model in ("LaRT", "IRT")]
    n_cores = min(len(tasks), int(os.getenv("SLURM_CPUS_PER_TASK", "8")))
    worker = partial(run_model_task, max_iter=100, seed=42)
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(worker, tasks)
    save_results(merge_results(results))
    print(f"Saved four separate benchmark fits in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
