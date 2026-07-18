"""Minimal end-to-end LaRT example using data generated from the paper model."""

import argparse

from lart import fit_lart, generate_lart_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-models", type=int, default=40)
    parser.add_argument("--n-items", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data = generate_lart_data(
        n_models=args.n_models,
        n_items=args.n_items,
        seed=args.seed,
    )
    result = fit_lart(
        data.response,
        data.latency,
        seed=args.seed,
        max_iter=args.max_iter,
    )

    print(f"estimated rho: {result.rho:.3f} (true rho: {data.rho:.3f})")
    print(f"iterations: {result.n_iter}")
    print("first five ability estimates:", result.theta[:5])


if __name__ == "__main__":
    main()
