# Data

This directory contains the post-processed benchmark inputs used in the
[LaRT paper](https://arxiv.org/abs/2512.07019). It contains neither saved model fits nor the much
larger raw model generations.

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

The public matrices are post-processed evaluation data. Users regenerating them should inspect
the sample workflow in `data_generation/` and apply a suitable mathematical answer grader.
