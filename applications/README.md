# Real-data applications

These programs reproduce the Section 7 model fits in the
[LaRT paper](https://arxiv.org/abs/2512.07019). Install the package first. Only inputs under
`data/benchmarks/` are versioned; generated fits are written under ignored `results/applications/`.

| Paper analysis | Program | Dataset |
|---|---|---|
| Joint fits and qualitative parameter analysis (Section 7.1) | `estimate_benchmarks.py` | MATH500, AMC23, AIME24, and AIME25 fitted separately |
| Predictive power (Section 7.2.1) | `predictive_power.py` | AMC23 + AIME24 + AIME25 |
| Item efficiency (Section 7.2.2) | `item_efficiency.py` | AMC23 + AIME24 + AIME25 |
| Validity across five partitions (Section 7.2.3) | `validity_math500.py` | MATH500 |
| LLM efficiency (Section 7.2.4) | `sensitivity_math500.py` | MATH500 |

Only analyses reported in the current manuscript are included. In particular, predictive power
and item efficiency are not run on MATH500, and the four datasets are not pooled for Section 7.1.
The paper uses seed 42 for model splits and fitting (and seed 1025 for predictive item folds).
Full fitting is CPU-intensive and uses multiprocessing where applicable.

The output-free notebooks in `notebooks/` contain the plotting and table calculations used in the
manuscript. They are retained as an auditable analysis record; the programs in this directory are
the preferred entry points for rerunning model fits.
