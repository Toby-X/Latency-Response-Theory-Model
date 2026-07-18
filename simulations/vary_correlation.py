import numpy as np
from scipy.stats import norm
from cMIRT_EM_c import cMIRT_SAEM_full, MIRT_SAEM_full
import pandas as pd
import multiprocess as mp
from tqdm import tqdm
import itertools
import os
from _output import save_results

# ---------------------------------------------------------------------------
# Simulation: Fixed N and J, Increasing |rho|
# ---------------------------------------------------------------------------
# N and J are fixed.  The off-diagonal of Sigma (rho) ranges over
# RHO_GRID = [0.2, 0.4, 0.6, 0.8] (negative, matching the model's sign
# convention that faster responders tend to score higher: Sigma[0,1] = -rho).
#
# Accuracy metric: RMSE for continuous parameters, MAE for rho (scalar).
# ---------------------------------------------------------------------------

N_FIXED  = 200
J_FIXED  = 50
RHO_GRID = [-0.2, -0.4, -0.6, -0.8]   # magnitudes; stored as negative in Sigma


def gen_indi_given_all(N, a, b, omega, phi, lam, Sigma, seed=None):
    """
    Generate individual latent traits and responses given population parameters.
    """
    rng = np.random.default_rng(seed)

    xi    = rng.multivariate_normal([0, 0], Sigma, N)
    theta = xi[:, 0]
    tau   = xi[:, 1]

    probit_arg = theta[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :]
    prob = norm.cdf(probit_arg)
    R    = rng.binomial(1, prob)
    log_T = rng.normal(
        omega[np.newaxis, :] - phi[np.newaxis, :] * tau[:, np.newaxis],
        lam[np.newaxis, :]
    )
    T = np.exp(log_T)

    return R, T, theta, tau


def gen_data(N, J, rho, seed=None):
    """
    Generate synthetic data with N subjects, J items, and off-diagonal rho.
    Sigma = [[1, -rho], [-rho, 1]]:  negative off-diagonal so that faster
    (higher tau) individuals tend to have higher theta (accuracy).
    """
    rng = np.random.default_rng(seed)

    a     = rng.uniform(0.5, 1,   size=J)
    b     = rng.normal(0, 0.5,    size=J)
    omega = rng.normal(0, 1,      size=J)
    phi   = rng.uniform(0.5, 1.5, size=J)
    lam   = rng.uniform(0.5, 2,   size=J)
    Sigma = np.array([[1., -rho], [-rho, 1.]])

    R, T, theta, tau = gen_indi_given_all(N, a, b, omega, phi, lam, Sigma, seed=seed)
    return R, T, theta, tau, a, b, omega, phi, lam, Sigma


def experiment_fn(seed, param_dict, esp=1e-4, max_iter=100):
    """Single replicate: probit-log-normal data with varying rho."""
    try:
        np.random.seed(seed)

        C   = 1
        N   = param_dict.get('N',   N_FIXED)
        J   = param_dict.get('J',   J_FIXED)
        rho = param_dict['rho']

        R, T, theta_true, tau_true, a_true, b_true, omega_true, phi_true, lam_true, Sigma_true = \
            gen_data(N, J, rho, seed=seed)

        # --- Fit LaRT ---
        theta_est, tau_est, a_est, b_est, omega_est, phi_est, lam_est, rho_est, iter_jml = \
            cMIRT_SAEM_full(R, T, n_samples=C, eps=esp, max_iter=max_iter, seed=seed)

        # --- Fit IRT ---
        theta_est_irt, a_est_irt, b_est_irt, _sigma2_irt, iter_irt = \
            MIRT_SAEM_full(R, n_samples=C, eps=esp, max_iter=max_iter, seed=seed)

        return {
            'N':              N,
            'J':              J,
            'rho_true':       rho,
            # LaRT RMSE
            'rmse_theta':     np.sqrt(np.mean((theta_est     - theta_true) ** 2)),
            'rmse_tau':       np.sqrt(np.mean((tau_est       - tau_true)   ** 2)),
            'rmse_a':         np.sqrt(np.mean((a_est         - a_true)     ** 2)),
            'rmse_b':         np.sqrt(np.mean((b_est         - b_true)     ** 2)),
            'rmse_omega':     np.sqrt(np.mean((omega_est     - omega_true) ** 2)),
            'rmse_phi':       np.sqrt(np.mean((phi_est       - phi_true)   ** 2)),
            'rmse_lam':       np.sqrt(np.mean((lam_est       - lam_true)   ** 2)),
            # rho: MAE (scalar parameter)
            'mae_rho':        float(np.abs(rho_est - Sigma_true[0, 1])),
            # IRT RMSE
            'rmse_theta_irt': np.sqrt(np.mean((theta_est_irt - theta_true) ** 2)),
            'rmse_a_irt':     np.sqrt(np.mean((a_est_irt     - a_true)     ** 2)),
            'rmse_b_irt':     np.sqrt(np.mean((b_est_irt     - b_true)     ** 2)),
            'iter_jml':       iter_jml,
            'iter_irt':       iter_irt,
            'seed':           seed,
        }
    except Exception as e:
        print(f"Error (seed={seed}, rho={param_dict.get('rho')}): {e}")
        return None


def run_experiment(param_grid, n_exp=200, n_cores=8):
    param_comb = []
    if isinstance(param_grid, list) and all(isinstance(p, dict) for p in param_grid):
        param_comb = param_grid
    else:
        for combo in itertools.product(*param_grid.values()):
            param_comb.append(dict(zip(param_grid.keys(), combo)))

    all_exp = [
        (exp_id, pc)
        for pc in param_comb
        for exp_id in range(n_exp)
    ]

    with mp.Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.starmap(experiment_fn, all_exp),
            total=len(all_exp),
            desc=f"Varying rho (N={N_FIXED}, J={J_FIXED})",
        ))

    results    = [r for r in results if r is not None]
    results_df = pd.DataFrame(results)

    out_file = save_results(results_df, f"cMIRT_sim_varyRho_N{N_FIXED}_J{J_FIXED}.parquet")
    print(f"Results saved to {out_file}")
    return results_df


if __name__ == "__main__":
    param_grid = {
        "N":   [N_FIXED],
        "J":   [J_FIXED],
        "rho": RHO_GRID,   # [0.2, 0.4, 0.6, 0.8]
    }
    n_cores = int(os.getenv('SLURM_CPUS_PER_TASK', 8))

    results_df = run_experiment(param_grid, n_exp=200, n_cores=n_cores)
    rmse_cols  = [c for c in results_df.columns if c.startswith('rmse_')]
    print(results_df.groupby('rho')[rmse_cols + ['mae_rho']].mean().to_string())
