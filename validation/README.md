# Validation against the former GitHub release

`compare_previous_release.py` runs the current publication estimator and the
former GitHub implementation on exactly the same generated datasets and fit
seeds. It executes the old code without patches and records failures rather
than replacing them with a corrected implementation.

The former code is identified as:

- Git commit: `e09e0f710957edfe33df4a63ada606041f0943ef`
- Entry point: `LaRT.py:LaRT_SAEM_full`
- Supporting files: `utils.py` and `minimax_tilting_sampler.py`
- SHA-256 of `LaRT.py`:
  `c028e31f48ba0125eb34eb54f41e9197a38aa3ca7392188dfb5b386d4eeb474d`

From the repository root, reproduce the comparison with an isolated checkout:

```bash
git worktree add /tmp/lart-previous e09e0f710957edfe33df4a63ada606041f0943ef
python validation/compare_previous_release.py --previous-dir /tmp/lart-previous
```

The committed results use five datasets with 40 models, 12 items,
`rho = -0.8`, one posterior sample per iteration, and at most 50 iterations.
This is a smoke-scale parameter-recovery check, not the paper's full Monte
Carlo study.
