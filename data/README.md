# Data

This directory contains the benchmark matrices and post-processed application data used in the
[LaRT paper](https://arxiv.org/abs/2512.07019). It does not contain the much larger raw model
generations or the full Monte Carlo experiment outputs.

## Benchmark matrices

For each benchmark, rows are evaluated model/prompt combinations and columns are items.

| Benchmark | Accuracy shape | CoT-length shape |
|---|---:|---:|
| MATH500 | 158 x 500 | 158 x 500 |
| AMC23 | 156 x 40 | 156 x 40 |
| AIME24 | 156 x 30 | 156 x 30 |
| AIME25 | 156 x 30 | 156 x 30 |
| AMC23 + AIME24 + AIME25 after filtering | 128 x 100 | 128 x 100 |

`correctness_matrix_*.csv` contains 0/1 scores. `cot_length_matrix_*.csv` contains token counts
for the generated reasoning before the final boxed answer. Application scripts add one to counts
before taking logarithms, preserving rows where generation yielded a zero count.

## Post-processed application data

`processed/` contains the fitted application parameters, predictive-power results, validity
partitions, item/LLM-efficiency trajectories, and fixed item selection used for the Section 7
tables and figures. These compact publication data are versioned; newly generated runs are not.

The public matrices are post-processed evaluation data. Users regenerating them should inspect
the sample workflow in `data_generation/` and apply a suitable mathematical answer grader.
