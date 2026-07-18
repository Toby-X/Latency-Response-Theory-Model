"""Small, reproducible comparison with the former GitHub implementation.

The previous implementation is loaded from an explicitly supplied checkout so
that the comparison executes that release verbatim.  Errors are recorded as
results; the script never patches or substitutes code in the former release.
"""

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


PREVIOUS_COMMIT = "e09e0f710957edfe33df4a63ada606041f0943ef"
PREVIOUS_LART_SHA256 = "c028e31f48ba0125eb34eb54f41e9197a38aa3ca7392188dfb5b386d4eeb474d"


def _load_previous(previous_dir: Path) -> ModuleType:
    module_path = previous_dir / "LaRT.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"former implementation not found: {module_path}")
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    if digest != PREVIOUS_LART_SHA256:
        raise ValueError(
            "LaRT.py does not match the tested former release: "
            f"expected SHA-256 {PREVIOUS_LART_SHA256}, got {digest}"
        )

    # LaRT.py imports the sibling modules `utils` and
    # `minimax_tilting_sampler` by their top-level names.
    sys.path.insert(0, str(previous_dir))
    try:
        spec = importlib.util.spec_from_file_location("lart_previous_release", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(estimate) - truth))))


def _metrics(method: str, replicate: int, elapsed: float, result, truth) -> dict:
    theta, tau, a, b, omega, phi, lam, rho, n_iter = result

    # Resolve the two harmless factor-sign indeterminacies before measuring
    # recovery.  Both reference implementations normally choose these signs
    # themselves, but explicit alignment makes the metric definition complete.
    theta, tau = np.asarray(theta).copy(), np.asarray(tau).copy()
    a, phi = np.asarray(a).copy(), np.asarray(phi).copy()
    rho = float(rho)
    if np.dot(a, truth.a) < 0:
        theta, a, rho = -theta, -a, -rho
    if np.dot(phi, truth.phi) < 0:
        tau, phi, rho = -tau, -phi, -rho

    return {
        "replicate": replicate,
        "method": method,
        "status": "ok",
        "runtime_seconds": elapsed,
        "n_iter": int(n_iter),
        "rmse_theta": _rmse(theta, truth.theta),
        "rmse_tau": _rmse(tau, truth.tau),
        "rmse_a": _rmse(a, truth.a),
        "rmse_b": _rmse(np.asarray(b), truth.b),
        "rmse_omega": _rmse(np.asarray(omega), truth.omega),
        "rmse_phi": _rmse(phi, truth.phi),
        "rmse_lam": _rmse(np.asarray(lam), truth.lam),
        "abs_error_rho": abs(rho - truth.rho),
        "rho_estimate": rho,
        "error": "",
    }


def _fit_one(method: str, fit, replicate: int, truth, max_iter: int) -> dict:
    started = time.perf_counter()
    try:
        result = fit(
            truth.response,
            truth.latency,
            n_samples=1,
            max_iter=max_iter,
            seed=replicate,
        )
        if hasattr(result, "theta"):
            result = (
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
        return _metrics(method, replicate, time.perf_counter() - started, result, truth)
    except Exception as exc:  # The former release's failures are part of the result.
        return {
            "replicate": replicate,
            "method": method,
            "status": "failed",
            "runtime_seconds": time.perf_counter() - started,
            "n_iter": np.nan,
            "rmse_theta": np.nan,
            "rmse_tau": np.nan,
            "rmse_a": np.nan,
            "rmse_b": np.nan,
            "rmse_omega": np.nan,
            "rmse_phi": np.nan,
            "rmse_lam": np.nan,
            "abs_error_rho": np.nan,
            "rho_estimate": np.nan,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--n-models", type=int, default=40)
    parser.add_argument("--n-items", type=int, default=12)
    parser.add_argument("--rho", type=float, default=-0.8)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--data-seed", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("validation/results"))
    args = parser.parse_args()

    previous = _load_previous(args.previous_dir.resolve())
    rows: list[dict] = []
    for replicate in range(args.replicates):
        truth = generate_lart_data(
            n_models=args.n_models,
            n_items=args.n_items,
            rho=args.rho,
            seed=args.data_seed + replicate,
        )
        rows.append(_fit_one("current", fit_lart, replicate, truth, args.max_iter))
        rows.append(
            _fit_one(
                "previous_release",
                previous.LaRT_SAEM_full,
                replicate,
                truth,
                args.max_iter,
            )
        )

    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "small_scale_comparison.csv"
    summary_path = args.output_dir / "small_scale_comparison_summary.json"
    frame.to_csv(detail_path, index=False)

    metric_columns = [column for column in frame if column.startswith(("rmse_", "abs_error_"))]
    summary: dict = {
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
        "previous_release": {
            "commit": PREVIOUS_COMMIT,
            "lart_py_sha256": PREVIOUS_LART_SHA256,
            "entry_point": "LaRT.py:LaRT_SAEM_full",
        },
        "methods": {},
    }
    for method, group in frame.groupby("method", sort=False):
        successful = group[group["status"] == "ok"]
        summary["methods"][method] = {
            "successful_fits": int(len(successful)),
            "failed_fits": int((group["status"] == "failed").sum()),
            "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
            "mean_metrics_over_successful_fits": {
                metric: (float(successful[metric].mean()) if len(successful) else None)
                for metric in metric_columns
            },
            "errors": sorted(set(group.loc[group["status"] == "failed", "error"])),
        }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results: {detail_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
