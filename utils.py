"""
utils.py: Helper functions, sampling algorithms, and initialization routines for LaRT.
"""

import numpy as np
from scipy.stats import norm
from scipy.sparse.linalg import svds
from typing import Tuple, Optional, Union

# Try import, but allow utils to be loaded for other things even if sampler missing
try:
    from minimax_tilting_sampler import TruncatedMVN
except ImportError:
    TruncatedMVN = None

# --- Constants ---
EPSILON = 1e-12
CLIP_MIN = 1e-9
CLIP_MAX = 1.0 - 1e-9

# --- Basic Math Helpers ---

def safe_log(x: np.ndarray) -> np.ndarray:
    """Computes log with a small epsilon to prevent -inf."""
    return np.log(x + EPSILON)

def probit_log_ratio(val: np.ndarray) -> np.ndarray:
    """Computes exp(log_pdf - log_cdf) safely for probit gradients."""
    log_pdf_val = norm.logpdf(val)
    log_cdf_val = norm.logcdf(val)
    return np.exp(np.nan_to_num(log_pdf_val - log_cdf_val, neginf=0.0))

def compute_probit_args(R: np.ndarray, theta: np.ndarray, 
                        a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Helper to compute (2R-1)*(a*theta+b) handling broadcasting."""
    if theta.ndim == 2 and R.ndim == 2: # (N, J)
        return (2 * R - 1) * (theta[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :])
    elif theta.ndim == 2 and R.ndim == 3: # (N, C, J) for samples
        return (2 * R[:, np.newaxis, :] - 1) * \
               (theta[:, :, np.newaxis] * a[np.newaxis, np.newaxis, :] + 
                b[np.newaxis, np.newaxis, :])
    else:
        return (2 * R - 1) * (theta * a + b)

# --- Spectral Initialization ---

def spectral_init_lart(R: np.ndarray, T: np.ndarray) -> Tuple:
    """Performs Spectral Initialization for LaRT."""
    N, J = R.shape
    log_T = np.log(T + CLIP_MIN)

    # R initialization
    U_R1, S_R1, Vt_R1 = np.linalg.svd(R, full_matrices=False)
    threshold = 1.01 * np.sqrt(N)
    idx = np.where(S_R1 > threshold)[0][-1] if np.any(S_R1 > threshold) else 0
    K_tilde = max(1, idx) + 1

    R_est = U_R1[:, :K_tilde] @ np.diag(S_R1[:K_tilde]) @ Vt_R1[:K_tilde, :]
    R_inv_est = norm.ppf(np.clip(R_est, CLIP_MIN, CLIP_MAX))

    b_ini = np.mean(R_inv_est, axis=0)
    U_R2, S_R2, Vt_R2 = svds(R_inv_est - b_ini[np.newaxis, :], k=1)
    theta_ini = U_R2[:, 0] * np.sqrt(N)
    a_ini = Vt_R2[0, :] * S_R2[0] / np.sqrt(N)

    if np.sum(a_ini) < 0:
        a_ini, theta_ini = -a_ini, -theta_ini

    # T initialization
    omega_ini = np.mean(log_T, axis=0)
    log_T_res = log_T - omega_ini[np.newaxis, :]
    U_T, S_T, Vt_T = svds(log_T_res, k=1)

    tau_ini = U_T[:, 0] * np.sqrt(N)
    phi_ini = -Vt_T[0, :] * S_T[0] / np.sqrt(N)
    
    # Residual variance
    rank_1 = S_T[0] * np.outer(U_T[:, 0], Vt_T[0, :])
    lam_ini = np.sqrt(np.mean((log_T_res - rank_1)**2, axis=0) + 1e-9)

    if np.sum(phi_ini) < 0:
        phi_ini, tau_ini = -phi_ini, -tau_ini

    # Covariance
    indi = np.stack([theta_ini, tau_ini], axis=1)
    rho_est = np.mean(indi[:, :, np.newaxis] @ indi[:, np.newaxis, :], axis=0)[0, 1]

    return theta_ini, tau_ini, a_ini, b_ini, omega_ini, phi_ini, lam_ini, rho_est

def spectral_init_irt(R: np.ndarray) -> Tuple:
    """Performs Spectral Initialization for IRT."""
    N, J = R.shape
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    
    threshold = 1.01 * np.sqrt(N)
    idx = np.where(S > threshold)[0][-1] if np.any(S > threshold) else 0
    K = max(1, idx) + 1

    R_est = U[:, :K] @ np.diag(S[:K]) @ Vt[:K, :]
    R_inv = norm.ppf(np.clip(R_est, CLIP_MIN, CLIP_MAX))

    b_ini = np.mean(R_inv, axis=0)
    U2, S2, Vt2 = svds(R_inv - b_ini[np.newaxis, :], k=1)
    
    theta_ini = U2[:, 0] * np.sqrt(N)
    a_ini = Vt2[0, :] * S2[0] / np.sqrt(N)

    if np.sum(a_ini) < 0:
        a_ini, theta_ini = -a_ini, -theta_ini
        
    return theta_ini, a_ini, b_ini, np.mean(theta_ini**2)

# --- Sampling Logic (SUN / TruncatedMVN) ---

def _sun_sampler_single(n_samples, mu_theta, sigma_theta2, R_i, a, b, seed=None):
    """Internal helper: SUN sampling for one individual."""
    if TruncatedMVN is None:
        raise ImportError("minimax_tilting_sampler is required for sampling.")
        
    if seed is not None: np.random.seed(seed)
    J = a.shape[0]
    
    diag_vals = 2 * R_i - 1
    D1 = (diag_vals * a).reshape(-1, 1)
    D2 = (diag_vals * b).reshape(-1, 1)
    S = np.diag(np.sqrt((D1.flatten()**2) * sigma_theta2 + 1))

    # V1 Sampler setup
    S_inv = np.linalg.inv(S)
    Sigma_V1 = S_inv @ (sigma_theta2 * (D1 @ D1.T) + np.eye(J)) @ S_inv
    lb = (-S_inv @ (D2 + D1 * mu_theta)).flatten()
    
    sampler = TruncatedMVN(np.zeros(J), Sigma_V1, lb, np.inf * np.ones(J), seed=seed)
    samples_V1 = sampler.sample(int(n_samples))

    # Transformation
    tmp_inv = np.linalg.inv(sigma_theta2 * D1 @ D1.T + np.eye(J))
    sigma_0_2 = 1 - sigma_theta2 * D1.T @ tmp_inv @ D1
    
    rng = np.random.default_rng(seed)
    samples_V0 = rng.normal(0, np.sqrt(sigma_0_2.flatten()), n_samples)
    
    sigma_theta = np.sqrt(sigma_theta2)
    V1_trans = sigma_theta * D1.T @ tmp_inv @ S @ samples_V1
    
    return sigma_theta * (samples_V0 + V1_trans.flatten()) + mu_theta

def post_samp_theta_irt(n_samples: int, R: np.ndarray, a: np.ndarray, 
                        b: np.ndarray, seed: int = None) -> np.ndarray:
    """Generates posterior samples for IRT theta."""
    theta_samples = np.zeros((R.shape[0], n_samples))
    for i in range(R.shape[0]):
        theta_samples[i, :] = _sun_sampler_single(n_samples, 0.0, 1.0, R[i], a, b, seed)
    return theta_samples

def post_samp_lart(n_samples: int, R: np.ndarray, log_T: np.ndarray, 
                   params: dict, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generates posterior samples for LaRT theta and tau."""
    a, b, omega, phi = params['a'], params['b'], params['omega'], params['phi']
    lam, rho = params['lam'], params['rho']
    
    N = R.shape[0]
    rng = np.random.default_rng(seed)
    
    sigma_tau2_p = (1 / (1 - rho**2) + np.sum((phi/lam)**2)) ** (-1)
    B = -np.sum((log_T - omega) * (phi/lam**2), axis=1)
    sigma_theta2_p = (1/(1-rho**2) - sigma_tau2_p * rho ** 2 / (1 - rho ** 2)) ** (-1)

    theta_s = np.zeros((N, n_samples))
    tau_s = np.zeros((N, n_samples))

    for i in range(N):
        mu_theta = sigma_theta2_p * sigma_tau2_p * B[i] * rho / (1 - rho**2)
        theta_s[i] = _sun_sampler_single(n_samples, mu_theta, sigma_theta2_p, R[i], a, b, seed)
        
        mu_tau = (B[i] + (theta_s[i] * rho)/(1-rho**2)) * sigma_tau2_p
        tau_s[i] = rng.normal(mu_tau, np.sqrt(sigma_tau2_p), n_samples)
        
    return theta_s, tau_s

# --- Fisher Information ---

def fisher_info_lart(a: np.ndarray, b: np.ndarray, theta: np.ndarray, 
                     Sigma: np.ndarray) -> np.ndarray:
    val = theta[:, None] * a[None, :] + b[None, :]
    log_terms = 2 * norm.logpdf(val) - norm.logcdf(val) - norm.logcdf(-val)
    fi_1 = np.sum(np.exp(np.nan_to_num(log_terms, neginf=0.0)) * a**2, axis=1)
    return fi_1 + np.linalg.inv(Sigma)[0, 0]

def fisher_info_irt(a: np.ndarray, b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    val = theta[:, None] * a[None, :] + b[None, :]
    log_terms = 2 * norm.logpdf(val) - norm.logcdf(val) - norm.logcdf(-val)
    fi_1 = np.sum(np.exp(np.nan_to_num(log_terms, neginf=0.0)) * a**2, axis=1)
    return fi_1 + 1