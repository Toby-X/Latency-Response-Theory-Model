import numpy as np
from scipy.stats import norm
from lart import irt_saem_full, lart_saem_full
import pandas as pd
import multiprocess as mp
from tqdm import tqdm
import itertools
import os
from _output import save_results

# ---------------------------------------------------------------------------
# Simulation: Fixed J, Increasing N
# ---------------------------------------------------------------------------
# J is held constant at J_FIXED.  N ranges over N_GRID.
# Each (N, seed) combination is an independent replication.
# ---------------------------------------------------------------------------

J_FIXED = 50
N_GRID  = [100, 200, 500]


def gen_indi_given_all(N, a, b, omega, phi, lam, Sigma, seed=None):
    """
    Generate individual latent traits and responses given population parameters.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    xi = rng.multivariate_normal([0, 0], Sigma, N)
    theta = xi[:, 0]
    tau   = xi[:, 1]

    probit_arg = theta[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :]
    prob = norm.cdf(probit_arg)
    R = rng.binomial(1, prob)
    log_T = rng.normal(
        omega[np.newaxis, :] - phi[np.newaxis, :] * tau[:, np.newaxis],
        lam[np.newaxis, :]
    )
    T = np.exp(log_T)

    return R, T, theta, tau


def gen_data(N, J, seed=None):
    """
    Generate synthetic data with N LLMs and J items.
    """
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    a     = rng.uniform(0.5, 1,   size=J)
    b     = rng.normal(0, 0.5,    size=J)
    omega = rng.normal(0, 1,      size=J)
    phi   = rng.uniform(0.5, 1.5, size=J)
    lam   = rng.uniform(0.5, 2,   size=J)
    Sigma = np.array([[1., -.8], [-.8, 1.]])

    R, T, theta, tau = gen_indi_given_all(N, a, b, omega, phi, lam, Sigma, seed=seed)
    return R, T, theta, tau, a, b, omega, phi, lam, Sigma


def experiment_fn(seed, param_dict, esp=1e-4, max_iter=100):
    """Single experiment: fixed J, draw N from param_dict."""
    try:
        np.random.seed(seed)

        C = 1
        N = param_dict['N']
        J = J_FIXED

        R, T, theta_true, tau_true, a_true, b_true, omega_true, phi_true, lam_true, Sigma_true = \
            gen_data(N, J, seed=seed)

        # --- LaRT ---
        theta_est, tau_est, a_est, b_est, omega_est, phi_est, lam_est, rho_est, iter_jml = \
            lart_saem_full(R, T, n_samples=C, eps=esp, max_iter=max_iter, seed=seed)

        # --- IRT ---
        theta_est_irt, a_est_irt, b_est_irt, _sigma2_irt, iter_irt = \
            irt_saem_full(R, n_samples=C, eps=esp, max_iter=max_iter, seed=seed)

        return {
            'N':              N,
            'J':              J,
            'rmse_theta':     np.sqrt(np.mean((theta_est     - theta_true) ** 2)),
            'rmse_tau':       np.sqrt(np.mean((tau_est       - tau_true)   ** 2)),
            'rmse_a':         np.sqrt(np.mean((a_est         - a_true)     ** 2)),
            'rmse_b':         np.sqrt(np.mean((b_est         - b_true)     ** 2)),
            'rmse_omega':     np.sqrt(np.mean((omega_est     - omega_true) ** 2)),
            'rmse_phi':       np.sqrt(np.mean((phi_est       - phi_true)   ** 2)),
            'rmse_lam':       np.sqrt(np.mean((lam_est       - lam_true)   ** 2)),
            'mae_rho':        np.mean(np.abs(rho_est - Sigma_true[0, 1])),
            'rmse_theta_irt': np.sqrt(np.mean((theta_est_irt - theta_true) ** 2)),
            'rmse_a_irt':     np.sqrt(np.mean((a_est_irt     - a_true)     ** 2)),
            'rmse_b_irt':     np.sqrt(np.mean((b_est_irt     - b_true)     ** 2)),
            'iter_jml':       iter_jml,
            'iter_irt':       iter_irt,
            'seed':           seed,
        }
    except Exception as e:
        print(f"Error in experiment (seed={seed}, N={param_dict.get('N')}): {e}")
        return None


def run_experiment(param_grid, n_exp=200, n_cores=8):
    param_comb = []

    if isinstance(param_grid, list) and all(isinstance(item, dict) for item in param_grid):
        param_comb = param_grid
    else:
        keys   = param_grid.keys()
        values = param_grid.values()
        for combo in itertools.product(*values):
            param_comb.append(dict(zip(keys, combo)))

    all_exp = [
        (exp_id, param_combo)
        for param_combo in param_comb
        for exp_id in range(n_exp)
    ]

    with mp.Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.starmap(experiment_fn, all_exp),
            total=len(all_exp),
            desc=f"Running experiments (fixed J={J_FIXED}, varying N)",
        ))

    results    = [r for r in results if r is not None]
    results_df = pd.DataFrame(results)

    out_file = save_results(results_df, f"LaRT_sim_fixedJ{J_FIXED}_varyN.parquet")
    print(f"Results saved to {out_file}")
    return results_df


if __name__ == "__main__":
    param_grid = {"N": N_GRID}
    n_cores    = int(os.getenv('SLURM_CPUS_PER_TASK', 8))

    results_df = run_experiment(param_grid, n_exp=200, n_cores=n_cores)
    print(results_df.groupby('N')[['rmse_theta', 'rmse_tau', 'rmse_a', 'rmse_b',
                                    'rmse_omega', 'rmse_phi', 'rmse_lam', 'mae_rho']].mean())
