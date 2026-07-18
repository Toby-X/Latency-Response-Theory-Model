import numpy as np
from scipy.stats import norm, spearmanr
from scipy.special import expit          # logistic sigmoid
from cMIRT_EM_c import cMIRT_SAEM_full, MIRT_SAEM_full
import pandas as pd
import multiprocess as mp
from tqdm import tqdm
import itertools
import os
from _output import save_results

# ---------------------------------------------------------------------------
# Misspecification Scenario 1: Logistic-Log-Normal
# ---------------------------------------------------------------------------
# DATA generated with:
#   Response :  P(R=1) = sigmoid(theta * a + b)   [logistic link]
#   Latency  :  log(T) ~ N(omega - phi*tau, lam)  [log-normal, same as model]
#
# MODEL FITTED (LaRT) assumes:
#   Response :  P(R=1) = Phi(theta * a + b)        [probit link]  <-- misspecified
#   Latency  :  log(T) ~ N(omega - phi*tau, lam)   [log-normal]
#
# Accuracy metric: Spearman rank correlation for all parameters except rho,
#                  MAE for rho.
# ---------------------------------------------------------------------------

N_DEFAULT = 200
J_DEFAULT = 50


def gen_indi_given_all_logistic_lognormal(N, a, b, omega, phi, lam, Sigma, seed=None):
    """
    Generate responses via logistic link and latencies via log-normal.
    """
    rng = np.random.default_rng(seed)

    xi    = rng.multivariate_normal([0, 0], Sigma, N)
    theta = xi[:, 0]
    tau   = xi[:, 1]

    # Logistic response (misspecification)
    logit_arg = theta[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :]
    prob = expit(logit_arg)
    R    = rng.binomial(1, prob)

    # Log-normal latency (correct)
    log_T = rng.normal(
        omega[np.newaxis, :] - phi[np.newaxis, :] * tau[:, np.newaxis],
        lam[np.newaxis, :]
    )
    T = np.exp(log_T)

    return R, T, theta, tau


def gen_data(N, J, seed=None):
    """Generate synthetic data with N subjects, J items (logistic-log-normal)."""
    rng = np.random.default_rng(seed)

    a     = rng.uniform(0.5, 1,   size=J)
    b     = rng.normal(0, 0.5,    size=J)
    omega = rng.normal(0, 1,      size=J)
    phi   = rng.uniform(0.5, 1.5, size=J)
    lam   = rng.uniform(0.5, 2,   size=J)
    Sigma = np.array([[1., -.8], [-.8, 1.]])

    R, T, theta, tau = gen_indi_given_all_logistic_lognormal(
        N, a, b, omega, phi, lam, Sigma, seed=seed
    )
    return R, T, theta, tau, a, b, omega, phi, lam, Sigma


def spearman_r(x, y):
    """Return Spearman rank correlation between two 1-D arrays."""
    r, _ = spearmanr(x, y)
    return float(r)


def experiment_fn(seed, param_dict, esp=1e-4, max_iter=100):
    """Single replicate: logistic-log-normal data, probit-log-normal model."""
    try:
        np.random.seed(seed)

        C = 1
        N = param_dict.get('N', N_DEFAULT)
        J = param_dict.get('J', J_DEFAULT)

        R, T, theta_true, tau_true, a_true, b_true, omega_true, phi_true, lam_true, Sigma_true = \
            gen_data(N, J, seed=seed)

        # --- Fit LaRT (probit-log-normal) ---
        theta_est, tau_est, a_est, b_est, omega_est, phi_est, lam_est, rho_est, iter_jml = \
            cMIRT_SAEM_full(R, T, n_samples=C, eps=esp, max_iter=max_iter, seed=seed)

        # --- Fit IRT (probit only) ---
        theta_est_irt, a_est_irt, b_est_irt, _sigma2_irt, iter_irt = \
            MIRT_SAEM_full(R, n_samples=C, eps=esp, max_iter=max_iter, seed=seed)

        return {
            'N':                   N,
            'J':                   J,
            'misspecification':    'logistic_lognormal',
            # LaRT rank correlations
            'spearman_theta':      spearman_r(theta_est,     theta_true),
            'spearman_tau':        spearman_r(tau_est,       tau_true),
            'spearman_a':          spearman_r(a_est,         a_true),
            'spearman_b':          spearman_r(b_est,         b_true),
            'spearman_omega':      spearman_r(omega_est,     omega_true),
            'spearman_phi':        spearman_r(phi_est,       phi_true),
            'spearman_lam':        spearman_r(lam_est,       lam_true),
            # rho: MAE (scalar)
            'mae_rho':             float(np.mean(np.abs(rho_est - Sigma_true[0, 1]))),
            # IRT rank correlations
            'spearman_theta_irt':  spearman_r(theta_est_irt, theta_true),
            'spearman_a_irt':      spearman_r(a_est_irt,     a_true),
            'spearman_b_irt':      spearman_r(b_est_irt,     b_true),
            'iter_jml':            iter_jml,
            'iter_irt':            iter_irt,
            'seed':                seed,
        }
    except Exception as e:
        print(f"Error (seed={seed}, params={param_dict}): {e}")
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
            desc="Misspec: logistic-log-normal",
        ))

    results    = [r for r in results if r is not None]
    results_df = pd.DataFrame(results)

    out_file = save_results(results_df, "cMIRT_misspec_logistic_lognormal.parquet")
    print(f"Results saved to {out_file}")
    return results_df


if __name__ == "__main__":
    param_grid = {
        "N": [100, 200, 500],   # from simulation.py
        "J": [J_DEFAULT],           # fixed at 50
    }
    n_cores = int(os.getenv('SLURM_CPUS_PER_TASK', 8))

    results_df = run_experiment(param_grid, n_exp=200, n_cores=n_cores)
    rank_cols  = [c for c in results_df.columns if c.startswith('spearman_')]
    print(results_df.groupby(['N', 'J'])[rank_cols + ['mae_rho']].mean().to_string())
