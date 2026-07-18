# Latency-Response Theory Model (LaRT)

Official research code and post-processed data for:

> **[Latency-Response Theory Model: Evaluating Large Language Models via Response Accuracy and Chain-of-Thought Length](https://arxiv.org/abs/2512.07019)**
> Zhiyu Xu, Jia Liu, Yixin Wang, and Yuqi Gu

LaRT jointly models whether an LLM answers an item correctly and how many Chain-of-Thought
(CoT) tokens it uses before the final answer. The joint model estimates latent mathematical
ability, latent speed, item accuracy parameters, item latency parameters, and their correlation.

This repository follows the structure of the current [LaRT paper](https://arxiv.org/abs/2512.07019).

## Quick start

```bash
git clone https://github.com/Toby-X/Latency-Response-Theory-Model.git
cd Latency-Response-Theory-Model
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The concise public API validates its inputs and returns named results:

```python
from lart import fit_lart, generate_lart_data

data = generate_lart_data(n_models=100, n_items=50, rho=-0.8, seed=42)
fit = fit_lart(data.response, data.latency, seed=42)

print(fit.theta)  # latent ability for each LLM
print(fit.tau)    # latent speed for each LLM
print(fit.rho)    # ability-speed correlation
```

`response` must be an `N x J` binary matrix. `latency` must be an equally shaped matrix of
strictly positive token counts. The paper application scripts add one to observed counts before
fitting because some raw generations have zero recorded CoT tokens.

## Repository layout

```text
src/lart/          reference SAEM estimator, validated API, and synthetic generator
simulations/       Section 6 and Appendix E simulation programs
applications/      Section 7 real-data estimation and comparison programs
data/benchmarks/   post-processed 0/1 accuracy and CoT-token matrices
data/processed/    post-processed application data used for paper tables and figures
data_generation/   sample vLLM generation and matrix-building workflow
notebooks/          output-free manuscript analysis notebooks
validation/         working-code comparison and repository-wide smoke test
```

## Paper-to-repository map

| Manuscript component | Code |
|---|---|
| Section 6, Figs. 2-3 | `simulations/main_study.py`, `simulations/vary_sample_size.py` |
| Appendix E.1, Figs. S1-S2 | `simulations/misspec_logistic_poisson.py`, `simulations/correct_logistic_lognormal.py` |
| Appendix E.2, Fig. S3 | `simulations/vary_correlation.py` |
| Appendix E.2, Fig. S4 | `simulations/vary_test_length.py` |
| Appendix E.3, Figs. S5-S6 | `simulations/traditional_saem_comparison.py`, `src/lart/traditional_saem.py` |
| Section 7.1 qualitative fits (four datasets, fitted separately) | `applications/estimate_benchmarks.py` |
| Section 7.2.1 predictive power (AMC23 + AIME24 + AIME25) | `applications/predictive_power.py` |
| Section 7.2.2 item efficiency (AMC23 + AIME24 + AIME25) | `applications/item_efficiency.py` |
| Section 7.2.3 validity (MATH500) | `applications/validity_math500.py` |
| Section 7.2.4 LLM efficiency (MATH500) | `applications/sensitivity_math500.py` |
| Appendix F generation protocol | `data_generation/generate_responses.py` |

All publication-facing functions, scripts, labels, and retained-result filenames use the final
LaRT/IRT terminology.

The exact Section 7 post-processed application data are included under `data/processed/`. Large
Monte Carlo experiment outputs and newly generated fits are not committed.

## Reproducing simulations

The full design uses 200 Monte Carlo replications per setting and is intended for a multi-core
machine or cluster. Scripts use `SLURM_CPUS_PER_TASK` when available:

```bash
SLURM_CPUS_PER_TASK=8 python simulations/vary_correlation.py
SLURM_CPUS_PER_TASK=8 python simulations/vary_test_length.py
```

See `simulations/README.md` for the complete list. The paper's informed spectral initialization
is in `src/lart/estimation.py`; the traditional burn-in SAEM used only for Appendix E.3 is isolated
in `src/lart/traditional_saem.py`.

## Reproducing applications

Application programs load the committed matrices from `data/benchmarks/` and write newly fitted
outputs to `results/applications/`. For example:

```bash
python applications/estimate_benchmarks.py
python applications/predictive_power.py
python applications/validity_math500.py
```

These jobs are computationally intensive. The post-processed application data used in the paper
are included under `data/processed/`; newly fitted parameters are written locally under `results/`,
which is intentionally excluded from version control. See `applications/README.md` and
`data/README.md` for details.

## Response generation

The full raw LLM generations are much larger than the publication repository. The committed data
are the post-processed matrices used for analysis. `data_generation/` supplies a runnable vLLM
example with the exact Appendix F prompts and hyperparameters, plus a converter from verified,
scored JSONL records to LaRT matrices.

<!-- ## Reproducibility notes

- Paper experiments use seed 42 for the application splits and fixed seeds for Monte Carlo runs.
- The estimator is a probit/log-normal joint model; the IRT comparison is normal-ogive IRT.
- The public API is a thin validation layer over the reference `lart_saem_full` and
  `irt_saem_full` routines.
- Output-free notebooks are included for provenance; the clean programs in `simulations/` and
  `applications/` are the preferred entry points. -->

## Citation and license

Paper: [arXiv:2512.07019](https://arxiv.org/abs/2512.07019). Citation metadata is provided in
`CITATION.cff`. The code is released under the MIT License; see `LICENSE`.
