"""Run reduced, non-destructive execution checks across the LaRT repository."""

from __future__ import annotations

import argparse
import compileall
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SIMULATIONS = ROOT / "simulations"
for directory in (str(SRC), str(SIMULATIONS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from lart import fit_irt, fit_lart, generate_lart_data  # noqa: E402
from lart.traditional_saem import lart_saem_full as traditional_lart_saem_full  # noqa: E402


class SkipCheck(Exception):
    """Signal that an external dependency or hardware pathway was not executed."""


def _load_file(path: Path, prefix: str = "smoke"):
    name = f"{prefix}_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _subprocess(command: list[str]) -> str:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(SRC), str(SIMULATIONS)))
    process = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}\n"
            f"{process.stdout}\n{process.stderr}"
        )
    return process.stdout.strip()


def _finite_tuple(values: tuple) -> None:
    for value in values:
        if isinstance(value, (int, float, np.number, np.ndarray)):
            if not np.isfinite(np.asarray(value)).all():
                raise AssertionError("estimator returned a non-finite value")


def check_python_compilation() -> str:
    paths = sorted(
        path
        for path in ROOT.rglob("*.py")
        if ".git" not in path.parts and ".egg-info" not in str(path)
    )
    for path in paths:
        if not compileall.compile_file(path, quiet=1, force=True):
            raise SyntaxError(path)
    return f"compiled {len(paths)} Python files"


def check_notebooks() -> str:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    cell_count = 0
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                compile(source, f"{path}:cell {index}", "exec")
                cell_count += 1
    return f"parsed {len(notebooks)} notebooks and compiled {cell_count} code cells"


def check_core_estimators() -> str:
    data = generate_lart_data(n_models=20, n_items=10, seed=11)
    lart_result = fit_lart(data.response, data.latency, seed=11, max_iter=2)
    irt_result = fit_irt(data.response, seed=11, max_iter=2)
    traditional = traditional_lart_saem_full(
        data.response,
        data.latency,
        seed=11,
        max_iter=2,
        n_burn=1,
    )
    _finite_tuple(tuple(lart_result.__dict__.values()))
    _finite_tuple(tuple(irt_result.__dict__.values()))
    _finite_tuple(tuple(traditional))
    return "current LaRT, IRT, and traditional-SAEM fits completed"


SIMULATION_CASES = {
    "main_study": {"N": 20},
    "vary_sample_size": {"N": 20, "J": 12},
    "vary_test_length": {"N": 20, "J": 12},
    "vary_correlation": {"N": 20, "J": 12, "rho": -0.5},
    "misspec_logistic_lognormal": {"N": 20, "J": 12},
    "misspec_logistic_poisson": {"N": 20, "J": 12},
    "misspec_logistic_poisson_shifted": {"N": 20, "J": 12},
    "misspec_probit_poisson": {"N": 20, "J": 12},
    "misspec_probit_poisson_shifted": {"N": 20, "J": 12},
    "traditional_saem_comparison": {"N": 12, "J": 12},
    "correct_logistic_lognormal": {"N": 40, "J": 12},
}


def check_simulation(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if name == "correct_logistic_lognormal" and exc.name == "girth":
            raise SkipCheck('install the optional simulation dependency: pip install -e ".[simulation]"')
        raise
    parameters = SIMULATION_CASES[name].copy()
    if name == "vary_sample_size":
        module.J_FIXED = parameters["J"]
    elif name == "vary_test_length":
        module.N_FIXED = parameters["N"]
    result = module.experiment_fn(3, parameters, max_iter=2)
    if not isinstance(result, dict) or not result:
        raise AssertionError("experiment_fn did not return a result")
    return f"reduced replicate returned {len(result)} result fields"


def check_simulation_output() -> str:
    module = importlib.import_module("_output")
    with tempfile.TemporaryDirectory() as temporary:
        module.OUTPUT_DIR = Path(temporary)
        path = module.save_results(pd.DataFrame({"seed": [1], "value": [0.5]}), "smoke.parquet")
        saved = pd.read_parquet(path)
        if saved.shape != (1, 2):
            raise AssertionError("simulation output round trip changed shape")
    return "Parquet output helper completed a temporary round trip"


def check_benchmark_estimation() -> str:
    module = _load_file(ROOT / "applications" / "estimate_benchmarks.py", "application")
    full_response, full_latency = module.load_benchmark("math500")
    response = np.asarray(full_response[:20, :12])
    latency = np.asarray(full_latency[:20, :12], dtype=float)
    lart_result = module.lart_saem_full(response, latency, seed=5, max_iter=2)
    irt_result = module.irt_saem_full(response, seed=5, max_iter=2)
    _finite_tuple(tuple(lart_result))
    _finite_tuple(tuple(irt_result))
    return f"loaded separate MATH500 matrix {full_response.shape}; reduced fits completed"


def check_predictive_power() -> str:
    module = _load_file(ROOT / "applications" / "predictive_power.py", "prediction")
    train_response = module.binary_array[:20, :12]
    train_latency = module.cot_array[:20, :12]
    parameters = module.fit_training_models(
        train_response, train_latency, max_iter=2, seed=5
    )
    scores = module.cross_validated_mae(
        module.binary_array[20:24, :12],
        module.cot_array[20:24, :12],
        parameters,
        n_folds=3,
        seed=5,
    )
    if scores.shape != (3, 3) or not np.isfinite(scores.to_numpy()).all():
        raise AssertionError(scores)
    return "three-benchmark training fit and held-out item folds completed"


def _application_efficiency(path: Path) -> str:
    module = _load_file(path, "efficiency")
    n_models, n_items, fit_items, initial_items = 4, 6, 12, 4
    response = np.asarray(module.binary_array[:n_models, :n_items])
    latency = np.asarray(module.cot_array[:n_models, :n_items], dtype=float)
    prediction = _load_file(
        ROOT / "applications" / "predictive_power.py", "efficiency_prediction"
    )
    parameters = prediction.fit_training_models(
        module.binary_array[:20, :fit_items],
        module.cot_array[:20, :fit_items],
        max_iter=2,
        seed=5,
    )
    joint = module.step_wise_evaluation_joint(
        response,
        latency,
        parameters["a_lart_train"][:n_items],
        parameters["b_lart_train"][:n_items],
        parameters["omega_lart_train"][:n_items],
        parameters["phi_lart_train"][:n_items],
        parameters["lam_lart_train"][:n_items],
        float(parameters["rho_lart"]),
        num_items=initial_items,
        n_steps=n_items,
    )
    irt = module.step_wise_evaluation_irt(
        response,
        parameters["a_irt_train"][:n_items],
        parameters["b_irt_train"][:n_items],
        num_items=initial_items,
        n_steps=n_items,
    )
    _finite_tuple(tuple(joint) + tuple(irt))
    return "adaptive item-selection paths completed on 4 models and 6 items"


def check_sensitivity() -> str:
    module = _load_file(ROOT / "applications" / "sensitivity_math500.py", "sensitivity")
    full_response = module.binary_array[:20, :12]
    full_latency = module.cot_array[:20, :12]
    questions = module.select_questions(
        full_response, full_latency, num_questions=6, max_iter=2, seed=5
    )
    response = full_response[:, questions]
    latency = full_latency[:, questions]
    lart_result = module.run_model_task(
        (20, "LaRT"), response, latency, max_iter=2, seed=5
    )
    irt_result = module.run_model_task(
        (20, "IRT"), response, latency, max_iter=2, seed=5
    )
    if lart_result.get("status") != "success" or irt_result.get("status") != "success":
        raise AssertionError((lart_result, irt_result))
    return "LaRT and IRT sensitivity workers completed"


def check_validity() -> str:
    module = _load_file(ROOT / "applications" / "validity_math500.py", "validity")
    module.binary_subarrays = [module.binary_array[:20, :12]]
    module.cot_subarrays = [module.cot_array[:20, :12]]
    module.num_subarrays = 1
    lart_result = module.run_model(("LaRT", 0), max_iter=2, seed=5)
    irt_result = module.run_model(("IRT", 0), max_iter=2, seed=5)
    if lart_result.get("model_type") != "LaRT" or irt_result.get("model_type") != "IRT":
        raise AssertionError((lart_result, irt_result))
    return "LaRT and IRT partition-validity workers completed"


def check_matrix_builder() -> str:
    records = []
    for model in ("model-a", "model-b"):
        for item in ("q1", "q2"):
            records.append(
                {
                    "model": model,
                    "prompt": "zero-shot",
                    "id": item,
                    "correct": int(model == "model-a"),
                    "cot_tokens": 7 if item == "q1" else 9,
                }
            )
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "records.jsonl"
        input_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        output_dir = temporary_path / "matrices"
        _subprocess(
            [
                sys.executable,
                "data_generation/build_matrices.py",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--dataset",
                "smoke",
            ]
        )
        accuracy = pd.read_csv(output_dir / "correctness_matrix_smoke.csv", index_col=0)
        latency = pd.read_csv(output_dir / "cot_length_matrix_smoke.csv", index_col=0)
        if accuracy.shape != (2, 2) or latency.shape != (2, 2):
            raise AssertionError("matrix builder returned unexpected shapes")
    return "scored JSONL converted to temporary 2 x 2 LaRT matrices"


def check_generation_helpers() -> str:
    module = _load_file(ROOT / "data_generation" / "generate_responses.py", "generation")
    reasoning, answer = module.split_boxed_answer("Reasoning {x}. Final \\boxed{7}")
    if reasoning != "Reasoning {x}. Final" or answer != "7":
        raise AssertionError((reasoning, answer))
    _subprocess([sys.executable, "data_generation/generate_responses.py", "--help"])
    return "prompt module imports without GPU packages; boxed-answer parser and CLI help passed"


def check_generation_inference() -> str:
    raise SkipCheck("requires a CUDA GPU, a downloaded model, transformers, and vLLM")


def check_previous_comparison(working_dir: Path | None) -> str:
    if working_dir is None or not working_dir.is_dir():
        raise SkipCheck("previous working directory was not supplied or does not exist")
    with tempfile.TemporaryDirectory() as temporary:
        _subprocess(
            [
                sys.executable,
                "validation/compare_working_folder.py",
                "--working-dir",
                str(working_dir),
                "--replicates",
                "1",
                "--n-models",
                "20",
                "--n-items",
                "10",
                "--max-iter",
                "2",
                "--output-dir",
                temporary,
            ]
        )
        summary = json.loads(
            (Path(temporary) / "working_folder_comparison_summary.json").read_text()
        )
        if summary["methods"]["latest_packaged"]["successful_fits"] != 1:
            raise AssertionError(summary)
    return "one paired current/working-folder comparison completed"


def check_data_artifacts() -> str:
    csv_count = 0
    for path in (ROOT / "data" / "benchmarks").rglob("*.csv"):
        pd.read_csv(path, index_col=0)
        csv_count += 1
    for removed in (ROOT / "data" / "processed", ROOT / "results"):
        if removed.exists():
            raise AssertionError(f"committed output directory still exists: {removed}")
    return f"read {csv_count} benchmark CSV files; no saved-fit/output directories present"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("/Users/zhiyuxu/Programmes/LLM_Eval"),
        help="previous working folder used by compare_working_folder.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional path for a JSON report; no report is saved by default",
    )
    args = parser.parse_args()

    checks: list[tuple[str, str, Callable[[], str]]] = [
        ("python-compilation", "repository", check_python_compilation),
        ("notebook-code", "notebooks", check_notebooks),
        ("core-estimators", "src/lart", check_core_estimators),
    ]
    checks.extend(
        (
            f"simulation:{name}",
            f"simulations/{name}.py",
            lambda name=name: check_simulation(name),
        )
        for name in SIMULATION_CASES
    )
    checks.extend(
        [
            ("simulation-output", "simulations/_output.py", check_simulation_output),
            (
                "application:estimate_benchmarks",
                "applications/estimate_benchmarks.py",
                check_benchmark_estimation,
            ),
            (
                "application:predictive_power",
                "applications/predictive_power.py",
                check_predictive_power,
            ),
            (
                "application:item_efficiency",
                "applications/item_efficiency.py",
                lambda: _application_efficiency(
                    ROOT / "applications" / "item_efficiency.py"
                ),
            ),
            (
                "application:sensitivity_math500",
                "applications/sensitivity_math500.py",
                check_sensitivity,
            ),
            (
                "application:validity_math500",
                "applications/validity_math500.py",
                check_validity,
            ),
            ("data-generation:matrix-builder", "data_generation/build_matrices.py", check_matrix_builder),
            (
                "data-generation:helpers",
                "data_generation/generate_responses.py",
                check_generation_helpers,
            ),
            (
                "data-generation:gpu-inference",
                "data_generation/generate_responses.py",
                check_generation_inference,
            ),
            (
                "working-folder-comparison",
                "validation/compare_working_folder.py",
                lambda: check_previous_comparison(args.working_dir),
            ),
            ("data-artifacts", "data/benchmarks/", check_data_artifacts),
        ]
    )

    results = []
    for name, scope, function in checks:
        started = time.perf_counter()
        try:
            detail = function()
            status = "passed"
        except SkipCheck as exc:
            detail = str(exc)
            status = "skipped"
        except Exception as exc:  # Continue so the report contains every failure.
            detail = f"{type(exc).__name__}: {exc}"
            status = "failed"
        elapsed = time.perf_counter() - started
        results.append(
            {
                "name": name,
                "scope": scope,
                "status": status,
                "seconds": elapsed,
                "detail": detail,
            }
        )
        print(f"{status.upper():7} {name:45} {elapsed:7.2f}s  {detail}")

    summary = {
        "design": {
            "simulation_replicates_per_program": 1,
            "simulation_max_iter": 2,
            "application_max_iter": 2,
            "temporary_outputs_only": True,
        },
        "counts": {
            status: sum(result["status"] == status for result in results)
            for status in ("passed", "failed", "skipped")
        },
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nReport: {args.output}")
    if summary["counts"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
