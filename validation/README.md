# Comparison with the previous working folder

This validation compares the latest packaged estimator with the actual code in
the author's previous working folder. Both implementations receive exactly the
same generated datasets, posterior-sampling settings, and random seeds.

The working-folder baseline is identified as:

- Directory: `/Users/zhiyuxu/Programmes/LLM_Eval`
- Entry point: `cMIRT_EM_c.py:cMIRT_SAEM_full`
- Estimator SHA-256:
  `6a70f53fa17af3dd111b168198980ba19a3846d5431e5865d05ee4f7851ebad2`
- Supporting sampler: `minimax_tilting_sampler.py`
- Sampler SHA-256:
  `5e838f9e959095ed2b8c50fec5167b484c95fda45d6687cd4f8d3e03c499243a`

Reproduce the committed small-scale comparison from the repository root:

```bash
python validation/compare_working_folder.py \
  --working-dir /Users/zhiyuxu/Programmes/LLM_Eval
```

The committed run uses five datasets with 40 models, 12 items, true
`rho = -0.8`, one posterior sample per iteration, and at most 50 iterations.
RMSE is used for vector parameters and absolute error for `rho`, matching the
paper's simulation metrics. This is a smoke-scale check rather than the full
Monte Carlo study.
