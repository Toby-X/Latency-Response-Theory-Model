# Real-data applications

These programs reproduce the model fits underlying Section 7. Install the package first. Inputs
are versioned under `data/benchmarks/`; final saved fits are under `data/processed/`.

| Paper analysis | Program |
|---|---|
| Joint fits and qualitative parameter analysis (Section 7.1) | `estimation_all.py`, `estimation_math500_all.py` |
| Predictive power (Section 7.2.1) | `prediction_all.py` |
| Item efficiency (Section 7.2.2) | `efficiency_rest3.py` |
| Validity across five MATH500 partitions (Section 7.2.3) | `validity_math500.py` |
| LLM efficiency (Section 7.2.4) | `sensitivity_math500.py` |

`prediction_math500.py` and `efficiency_math500.py` are included as companion analyses. The
paper uses seed 42 throughout these scripts. Full fitting is CPU-intensive and uses multiprocessing.

The output-free notebooks in `notebooks/` contain the plotting and table calculations used in the
manuscript. They are retained as an auditable analysis record; the programs in this directory are
the preferred entry points for rerunning model fits.
