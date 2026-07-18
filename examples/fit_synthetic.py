"""Minimal end-to-end LaRT example using data generated from the paper model."""

from lart import fit_lart, generate_lart_data


data = generate_lart_data(n_models=40, n_items=20, seed=7)
result = fit_lart(data.response, data.latency, seed=7, max_iter=100)

print(f"estimated rho: {result.rho:.3f} (true rho: {data.rho:.3f})")
print(f"iterations: {result.n_iter}")
print("first five ability estimates:", result.theta[:5])
