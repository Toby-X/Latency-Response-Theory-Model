# Simulation programs

All generators are included, so a simulation can be run without downloading data. The full paper
settings use 200 replications and can be computationally expensive. Each script respects
`SLURM_CPUS_PER_TASK` when choosing its process count.

| Paper location | Program | Design |
|---|---|---|
| Section 6 | `main_study.py` / `vary_sample_size.py` | J=50, rho=-0.8, N in {100, 200, 500} |
| Appendix E.1 | `misspec_logistic_poisson.py` | logistic response and shifted-Poisson latency fitted with LaRT |
| Appendix E.2, Fig. S3 | `vary_correlation.py` | N=200, J=50, rho in {-0.2, -0.4, -0.6, -0.8} |
| Appendix E.2, Fig. S4 | `vary_test_length.py` | N=200, rho=-0.8, J in {20, 50, 100, 200} |
| Appendix E.3 | `traditional_saem_comparison.py` | informed initialization versus burn-in SAEM |

Additional misspecification programs are retained because they were run during robustness work:
probit-Poisson, logistic-lognormal, and shifted-count variants. `correct_logistic_lognormal.py`
provides the correctly specified logistic comparison.

After installing the package, run a program from the repository root, for example:

```bash
SLURM_CPUS_PER_TASK=8 python simulations/vary_correlation.py
```

The exact post-processed outputs used for manuscript plots are in `results/simulations/`.
