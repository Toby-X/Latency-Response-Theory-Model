"""Compare the packaged estimator with the author's previous working file."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from lart import fit_lart, generate_lart_data


WORKING_ESTIMATOR_SHA256 = "6a70f53fa17af3dd111b168198980ba19a3846d5431e5865d05ee4f7851ebad2"
WORKING_SAMPLER_SHA256 = "5e838f9e959095ed2b8c50fec5167b484c95fda45d6687cd4f8d3e03c499243a"
PARAMETER_NAMES = ("theta", "tau", "a", "b", "omega", "phi", "lam", "rho")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_working_code(working_dir: Path) -> ModuleType:
    estimator_path = working_dir / "cMIRT_EM_c.py"
    sampler_path = working_dir / "minimax_tilting_sampler.py"
    for path, expected in (
        (estimator_path, WORKING_ESTIMATOR_SHA256),
        (sampler_path, WORKING_SAMPLER_SHA256),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"working-folder file not found: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"{path} does not match the tested working code: "
                f"expected SHA-256 {expected}, got {actual}"
            )

    # cMIRT_EM_c.py imports its sibling sampler as a top-level module.
    sys.path.insert(0, str(working_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "lart_previous_working_folder", estimator_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load {estimator_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _as_tuple(result) -> tuple:
    if hasattr(result, "theta"):
        return (
            result.theta,
            result.tau,
            result.a,
            result.b,
            result.omega,
            result.phi,
            result.lam,
            result.rho,
            result.n_iter,
        )
    return tuple(result)


def _rmse(estimate, truth) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(estimate) - truth))))


def _fit(method: str, fit, replicate: int, data, max_iter: int) -> tuple[dict, tuple]:
    started = time.perf_counter()
    result = _as_tuple(
        fit(
            data.response,
            data.latency,
            n_samples=1,
            max_iter=max_iter,
            seed=replicate,
        )
    )
    elapsed = time.perf_counter() - started
    theta, tau, a, b, omega, phi, lam, rho, n_iter = result
    row = {
        "replicate": replicate,
        "method": method,
        "runtime_seconds": elapsed,
        "n_iter": int(n_iter),
        "rmse_theta": _rmse(theta, data.theta),
        "rmse_tau": _rmse(tau, data.tau),
        "rmse_a": _rmse(a, data.a),
        "rmse_b": _rmse(b, data.b),
        "rmse_omega": _rmse(omega, data.omega),
        "rmse_phi": _rmse(phi, data.phi),
        "rmse_lam": _rmse(lam, data.lam),
        "abs_error_rho": abs(float(rho) - data.rho),
        "rho_estimate": float(rho),
    }
    return row, result


def _estimate_differences(replicate: int, current: tuple, working: tuple) -> dict:
    differences = {"replicate": replicate}
    for name, current_value, working_value in zip(
        PARAMETER_NAMES, current[:8], working[:8], strict=True
    ):
        differences[f"max_abs_diff_{name}"] = float(
            np.max(np.abs(np.asarray(current_value) - np.asarray(working_value)))
        )
    differences["same_n_iter"] = bool(current[8] == working[8])
    return differences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--n-models", type=int, default=40)
    parser.add_argument("--n-items", type=int, default=12)
    parser.add_argument("--rho", type=float, default=-0.8)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--data-seed", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("validation/results"))
    args = parser.parse_args()

    working_dir = args.working_dir.resolve()
    working_code = _load_working_code(working_dir)
    rows: list[dict] = []
    differences: list[dict] = []
    for replicate in range(args.replicates):
        data = generate_lart_data(
            n_models=args.n_models,
            n_items=args.n_items,
            rho=args.rho,
            seed=args.data_seed + replicate,
        )
        current_row, current_result = _fit(
            "latest_packaged", fit_lart, replicate, data, args.max_iter
        )
        working_row, working_result = _fit(
            "previous_working_folder",
            working_code.cMIRT_SAEM_full,
            replicate,
            data,
            args.max_iter,
        )
        rows.extend((current_row, working_row))
        differences.append(
            _estimate_differences(replicate, current_result, working_result)
        )

    results = pd.DataFrame(rows)
    difference_frame = pd.DataFrame(differences)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "working_folder_comparison.csv"
    difference_path = args.output_dir / "working_folder_estimate_differences.csv"
    summary_path = args.output_dir / "working_folder_comparison_summary.json"
    results.to_csv(detail_path, index=False)
    difference_frame.to_csv(difference_path, index=False)

    metric_columns = [
        column
        for column in results
        if column.startswith("rmse_") or column == "abs_error_rho"
    ]
    difference_columns = [column for column in difference_frame if column.startswith("max_abs_diff_")]
    summary = {
        "design": {
            "replicates": args.replicates,
            "n_models": args.n_models,
            "n_items": args.n_items,
            "rho": args.rho,
            "n_samples": 1,
            "max_iter": args.max_iter,
            "data_seeds": list(range(args.data_seed, args.data_seed + args.replicates)),
            "fit_seeds": list(range(args.replicates)),
        },
        "previous_working_code": {
            "directory": str(working_dir),
            "entry_point": "cMIRT_EM_c.py:cMIRT_SAEM_full",
            "estimator_sha256": WORKING_ESTIMATOR_SHA256,
            "sampler_sha256": WORKING_SAMPLER_SHA256,
        },
        "methods": {},
        "direct_estimate_comparison": {
            column: float(difference_frame[column].max())
            for column in difference_columns
        },
        "same_iteration_count_in_every_replicate": bool(
            difference_frame["same_n_iter"].all()
        ),
    }
    for method, group in results.groupby("method", sort=False):
        summary["methods"][method] = {
            "successful_fits": int(len(group)),
            "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
            "mean_n_iter": float(group["n_iter"].mean()),
            "mean_metrics": {
                metric: float(group[metric].mean()) for metric in metric_columns
            },
        }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed metrics: {detail_path}")
    print(f"Estimate differences: {difference_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
