import numpy as np
from scipy.stats import norm
from lart.traditional_saem import lart_saem_full, irt_saem_full, fisher_info_theta_lart, fisher_info_theta_irt, post_samp_tau_theta, post_samp_theta
import pandas as pd
import multiprocess as mp
from tqdm import tqdm
import itertools
import os
from _output import save_results

def gen_indi_given_all(N, a, b, omega, phi, lam, Sigma, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    xi = rng.multivariate_normal([0, 0], Sigma, N)
    theta = xi[:, 0]
    tau = xi[:, 1]

    probit_arg = theta[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :]
    prob = norm.cdf(probit_arg)
    # expit_arg = theta[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :]
    # prob = expit(expit_arg)
    R = rng.binomial(1, prob)
    log_T = rng.normal(omega[np.newaxis, :] - phi[np.newaxis, :] * tau[:, np.newaxis], lam[np.newaxis, :])
    T = np.exp(log_T)
    return R, T, theta, tau

def gen_data(N, J, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()
    # a = rng.choice([-0.2, 0.2, -0.4, 0.4, -0.6, 0.6], size=J)
    a = rng.uniform(0.5, 1, size=J)
    # b = rng.choice([-0.2, 0.2, -0.4, 0.4, -0.6, 0.6], size=J)
    b = rng.normal(0, 0.5, J)
    # a = rng.lognormal(0, 0.25, J)
    # b = rng.normal(0, 1, J)
    # omega = rng.normal(0, 1, J)
    # phi = rng.lognormal(0, 1/4, J)
    # omega = rng.choice([-1, 2, -3, 4, -5, 6], size=J)
    omega = rng.normal(0, 1, J)
    # phi = rng.choice([1, -2, 3, -4, -5, -6], size=J)
    phi = rng.uniform(0.5, 1.5, J)
    lam = rng.uniform(0.5, 2, J)
    # Appendix E.3 uses the same rho=-0.8 design as Section 6.
    Sigma = np.array([[1., -.8], [-.8, 1.]])
    R, T, theta, tau = gen_indi_given_all(N, a, b, omega, phi, lam, Sigma, seed=seed)
    return R, T, theta, tau, a, b, omega, phi, lam, Sigma

def experiment_fn(seed, param_dict, esp=1e-4, max_iter=100):
    try:
        np.random.seed(seed+100)
        pid = os.getpid()
        print(f"Worker process {pid} is starting task with seed {seed}.")

        C = 1
        N = param_dict['N']
        J = 50
        R, T, theta_true, tau_true, a_true, b_true, omega_true, phi_true, lam_true, Sigma_true = gen_data(N, J, seed+100)

        # Initialize results storage
        theta_est, tau_est, a_est, b_est, omega_est, phi_est, lam_est, rho_est, iter_jml = lart_saem_full(R, T, n_samples=C, eps=esp, max_iter=max_iter, seed=seed)
        theta_est_irt, a_est_irt, b_est_irt, sigma2_est_irt, iter_irt = irt_saem_full(R, n_samples=C, eps=esp, max_iter=max_iter, seed=seed)

        rmse_theta = np.min([np.sqrt(np.mean((theta_est - theta_true) ** 2)), np.sqrt(np.mean((theta_est + theta_true) ** 2))])
        rmse_tau = np.min([np.sqrt(np.mean((tau_est - tau_true) ** 2)), np.sqrt(np.mean((tau_est + tau_true) ** 2))])
        rmse_a = np.min([np.sqrt(np.mean((a_est - a_true) ** 2)), np.sqrt(np.mean((a_est + a_true) ** 2))])
        rmse_b = np.min([np.sqrt(np.mean((b_est - b_true) ** 2)), np.sqrt(np.mean((b_est + b_true) ** 2))])
        rmse_omega = np.min([np.sqrt(np.mean((omega_est - omega_true) ** 2)), np.sqrt(np.mean((omega_est + omega_true) ** 2))])
        rmse_phi = np.min([np.sqrt(np.mean((phi_est - phi_true) ** 2)), np.sqrt(np.mean((phi_est + phi_true) ** 2))])
        rmse_lam = np.min([np.sqrt(np.mean((lam_est - lam_true) ** 2)), np.sqrt(np.mean((lam_est + lam_true) ** 2))])
        mae_rho = np.min([np.mean(np.abs(rho_est - Sigma_true[0,1])),np.mean(np.abs(rho_est + Sigma_true[0,1]))])

        rmse_theta_irt = np.min([np.sqrt(np.mean((theta_est_irt - theta_true) ** 2)), np.sqrt(np.mean((theta_est_irt + theta_true) ** 2))])
        rmse_a_irt = np.min([np.sqrt(np.mean((a_est_irt - a_true) ** 2)), np.sqrt(np.mean((a_est_irt + a_true) ** 2))])
        rmse_b_irt = np.min([np.sqrt(np.mean((b_est_irt - b_true) ** 2)), np.sqrt(np.mean((b_est_irt + b_true) ** 2))])
        # mae_sigma2_irt = np.min([np.mean(np.abs(sigma2_est_irt**2 - Sigma_true[0, 0])),np.mean(np.abs(sigma2_est_irt**2 + Sigma_true[0, 0]))])

        Sigma_est = np.array([[1., rho_est], [rho_est, 1.]])
        fisher_info_lart_est = fisher_info_theta_lart(a_est, b_est, theta_est, Sigma_est)
        fisher_info_irt_est = fisher_info_theta_irt(a_est_irt, b_est_irt, theta_est_irt, sigma2_est_irt)
        fisher_info_lart_true = fisher_info_theta_lart(a_true, b_true, theta_true, Sigma_true)
        fisher_info_irt_true = fisher_info_theta_irt(a_true, b_true, theta_true, Sigma_true[0, 0])

        theta_samples_irt = post_samp_theta(1000, R, a_est_irt, b_est_irt)
        theta_samples_lart, tau_samples_lart = post_samp_tau_theta(1000, R, np.log(T), a_est, b_est, omega_est, phi_est, lam_est, rho_est)
        sd_theta_samples_irt = np.std(theta_samples_irt, axis=1)
        sd_theta_samples_lart = np.std(theta_samples_lart, axis=1)

        quantile_diff_theta_samples_irt = np.percentile(theta_samples_irt, 97.5, axis=1) - np.percentile(theta_samples_irt, 2.5, axis=1)
        quantile_diff_theta_samples_lart = np.percentile(theta_samples_lart, 97.5, axis=1) - np.percentile(theta_samples_lart, 2.5, axis=1)

        return {
            'N': N,
            'rmse_theta': rmse_theta,
            'rmse_tau': rmse_tau,
            'rmse_a': rmse_a,
            'rmse_b': rmse_b,
            'rmse_omega': rmse_omega,
            'rmse_phi': rmse_phi,
            'rmse_lam': rmse_lam,
            'mae_rho': mae_rho,
            'rmse_theta_irt': rmse_theta_irt,
            'rmse_a_irt': rmse_a_irt,
            'rmse_b_irt': rmse_b_irt,
            # 'mae_sigma2_irt': mae_sigma2_irt,
            'iter_jml': iter_jml,
            'iter_irt': iter_irt,
            'fisher_info_lart_est': fisher_info_lart_est,
            'fisher_info_irt_est': fisher_info_irt_est,
            'fisher_info_lart_true': fisher_info_lart_true,
            'fisher_info_irt_true': fisher_info_irt_true,
            'sd_theta_samples_irt': sd_theta_samples_irt,
            'sd_theta_samples_lart': sd_theta_samples_lart,
            'len_theta_samples_irt': quantile_diff_theta_samples_irt,
            'len_theta_samples_lart': quantile_diff_theta_samples_lart,
            'seed': seed
        }
    except Exception as e:
        print(f"Error occurred in worker process {pid} with seed {seed}: {e}")
        return None

def run_experiment(param_grid, n_exp=100, n_cores=8):
        # param_grid should be a list of dictionaries
    param_comb = []

    if isinstance(param_grid, list) and all(isinstance(item, dict) for item in param_grid):
        param_comb = param_grid
    else:
        keys = param_grid.keys()
        values = param_grid.values()
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            param_comb.append(param_dict)

    all_exp = []
    for param_combo in param_comb:
        for exp_id in range(n_exp):
            all_exp.append((exp_id, param_combo))

    with mp.Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.starmap(experiment_fn, all_exp),
            total=len(all_exp),
            desc="Running experiments",
        ))

    results = [res for res in results if res is not None]
    results_df = pd.DataFrame(results)

    save_results(results_df, "traditional_saem_comparison.parquet")
    return results_df

if __name__ == "__main__":
    param_grid = {
        'N': [100, 200, 500]
    }
    # results_df = run_experiment(param_grid, n_exp=100, n_cores=24)

    n_cores = int(os.getenv('SLURM_CPUS_PER_TASK', 8))

    results_df = run_experiment(param_grid, n_exp=200, n_cores=n_cores)
