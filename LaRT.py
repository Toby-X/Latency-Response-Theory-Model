"""
LaRT.py: Main estimation algorithms (SAEM) and model definitions for LaRT and IRT.

This module relies on 'utils.py' for sampling and initialization routines.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Tuple, Optional, Callable, Dict, Any, Union

import utils

# ==============================================================================
# 1. Model Definitions: Log-Likelihoods & Gradients
# ==============================================================================

def _lart_log_like_samp(R: np.ndarray, log_T: np.ndarray, 
                        theta_s: np.ndarray, tau_s: np.ndarray, 
                        a: np.ndarray, b: np.ndarray, 
                        omega: np.ndarray, phi: np.ndarray, 
                        lam: np.ndarray, rho: float) -> float:
    """Calculates LaRT log-likelihood averaged over Monte Carlo samples."""
    N, C = theta_s.shape
    Sigma_inv = np.linalg.inv(np.array([[1., rho], [rho, 1.]]))
    
    # 1. Probit Component
    arg = utils.compute_probit_args(R, theta_s, a, b)
    ll_1 = np.sum(utils.safe_log(norm.cdf(arg))) / C
    
    # 2. Response Time Component
    # log_T: (N, J) -> (N, 1, J) for broadcasting against samples (N, C, 1)
    resid = (log_T[:, np.newaxis, :] - omega[np.newaxis, np.newaxis, :] - 
             tau_s[:, :, np.newaxis] * phi[np.newaxis, np.newaxis, :]) 
    
    ll_2 = -N * np.sum(np.log(lam))
    ll_3 = -np.sum(resid**2 / (lam[np.newaxis, np.newaxis, :]**2)) / 2 / C
    
    # 3. Latent Variable Priors
    xi = np.stack([theta_s, tau_s], axis=1) # Shape (N, 2, C)
    # Quadratic form: sum over N (i) and C (k), matrix dims (p, q)
    quad = np.einsum('ijk,jl,ilk->', xi, Sigma_inv, xi)
    
    ll_4_const = -0.5 * N * np.log(np.linalg.det(np.array([[1., rho], [rho, 1.]])))
    ll_4 = ll_4_const - quad / (2 * C)
    
    return ll_1 + ll_2 + ll_3 + ll_4

def _lart_grad_samp(theta_s: np.ndarray, tau_s: np.ndarray, 
                    R: np.ndarray, log_T: np.ndarray, 
                    a: np.ndarray, b: np.ndarray, 
                    omega: np.ndarray, phi: np.ndarray, 
                    lam: np.ndarray) -> np.ndarray:
    """Calculates gradients for LaRT global parameters averaged over samples."""
    N, C = theta_s.shape
    
    # --- Probit Gradients (a, b) ---
    val = utils.compute_probit_args(R, theta_s, a, b)
    ratio = utils.probit_log_ratio(val) # (N, C, J)
    
    # Sum over N (axis 0) and C (axis 1), then average by C
    term_common = ratio * (2 * R[:, np.newaxis, :] - 1)
    grad_a = np.sum(term_common * theta_s[:, :, np.newaxis], axis=(0, 1)) / C
    grad_b = np.sum(term_common, axis=(0, 1)) / C
    
    # --- Time Gradients (omega, phi, lam) ---
    resid = (log_T[:, np.newaxis, :] - omega[np.newaxis, np.newaxis, :] - 
             tau_s[:, :, np.newaxis] * phi[np.newaxis, np.newaxis, :])
    
    lam_sq = lam[np.newaxis, np.newaxis, :] ** 2
    
    grad_omega = np.sum(resid / lam_sq, axis=(0, 1)) / C
    grad_phi = -np.sum(resid * tau_s[:, :, np.newaxis] / lam_sq, axis=(0, 1)) / C # Note: minus sign from derivative of (-tau*phi)
    
    # d/dlam of -N*log(lam) - sum(resid^2 / 2lam^2)
    # = -N/lam + sum(resid^2)/lam^3
    sum_resid_sq = np.sum(resid**2, axis=(0, 1)) / C
    grad_lam = -N / lam + sum_resid_sq / (lam**3)
    
    return np.concatenate([grad_a, grad_b, grad_omega, grad_phi, grad_lam])

def _irt_log_like_samp(theta_s: np.ndarray, R: np.ndarray, 
                       a: np.ndarray, b: np.ndarray) -> float:
    """Calculates IRT log-likelihood averaged over samples."""
    N, C = theta_s.shape
    
    # Probit
    arg = utils.compute_probit_args(R, theta_s, a, b)
    ll_1 = np.sum(utils.safe_log(norm.cdf(arg))) / C
    
    # Prior (Normal)
    ll_prior = - np.sum(theta_s**2) / (2 * C)
    
    return (ll_1 + ll_prior) / N

def _irt_grad_samp(theta_s: np.ndarray, R: np.ndarray, 
                   a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculates IRT gradients for a and b averaged over samples."""
    N, C = theta_s.shape
    
    val = utils.compute_probit_args(R, theta_s, a, b)
    ratio = utils.probit_log_ratio(val)
    
    term_common = ratio * (2 * R[:, np.newaxis, :] - 1)
    
    grad_a = np.sum(term_common * theta_s[:, :, np.newaxis], axis=(0, 1)) / C
    grad_b = np.sum(term_common, axis=(0, 1)) / C
    
    return np.concatenate([grad_a / N, grad_b / N])

# ==============================================================================
# 2. MAP Estimation (Post-SAEM)
# ==============================================================================

def get_indi_map_lart(theta_old: np.ndarray, tau_old: np.ndarray, 
                      R: np.ndarray, log_T: np.ndarray, 
                      params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Computes MAP estimates for LaRT individuals."""
    N = R.shape[0]
    a, b = params['a'], params['b']
    omega, phi = params['omega'], params['phi']
    lam, rho = params['lam'], params['rho']
    
    Sigma_inv = np.linalg.inv(np.array([[1., rho], [rho, 1.]]))

    def obj_fn(x):
        theta, tau = x[:N], x[N:]
        
        # 1. Likelihood (Negative for minimization)
        # Probit
        arg = utils.compute_probit_args(R, theta, a, b)
        ll = np.sum(utils.safe_log(norm.cdf(arg)))
        
        # Time
        resid = log_T - omega[np.newaxis, :] - tau[:, np.newaxis] * phi[np.newaxis, :]
        ll -= 0.5 * np.sum(resid**2 / lam[np.newaxis, :]**2)
        
        # Prior
        xi = np.stack([theta, tau], axis=1)
        quad = np.sum((xi @ Sigma_inv) * xi)
        ll -= 0.5 * quad
        
        # 2. Gradient (Negative)
        # Probit
        ratio = utils.probit_log_ratio(arg)
        grad_theta = np.sum(ratio * (2*R - 1) * a[np.newaxis, :], axis=1)
        
        # Time
        # d/dtau of -0.5 * sum( (logT - omega - tau*phi)^2 / lam^2 )
        # = sum( (logT - omega - tau*phi)/lam^2 * phi )
        grad_tau = np.sum(resid * (phi[np.newaxis, :] / lam[np.newaxis, :]**2), axis=1)
        
        # Prior
        adj = (Sigma_inv @ xi.T).T # (N, 2)
        grad_theta -= adj[:, 0]
        grad_tau -= adj[:, 1]
        
        return -ll, -np.concatenate([grad_theta, grad_tau])

    x0 = np.concatenate([theta_old, tau_old])
    res = minimize(obj_fn, x0, method='L-BFGS-B', jac=True, 
                   options={'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-7})
    return res.x[:N], res.x[N:]

def get_indi_map_irt(theta_old: np.ndarray, R: np.ndarray, 
                     a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes MAP estimates for IRT individuals."""
    def obj_fn(theta):
        # Log Like
        arg = utils.compute_probit_args(R, theta, a, b)
        ll = np.sum(utils.safe_log(norm.cdf(arg))) - np.sum(theta**2)/(2)
        
        # Gradient
        ratio = utils.probit_log_ratio(arg)
        grad = np.sum(ratio * (2*R - 1) * a[np.newaxis, :], axis=1) - theta

        return -ll, -grad

    res = minimize(obj_fn, theta_old, method='L-BFGS-B', jac=True,
                   options={'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-7})
    return res.x

# ==============================================================================
# 3. Main SAEM Algorithms
# ==============================================================================

def LaRT_SAEM_full(R: np.ndarray, T: np.ndarray, 
                   n_samples: int = 1, 
                   weight_scheme_fn: Optional[Callable[[int], float]] = None, 
                   initial_params: Optional[Dict[str, Any]] = None, 
                   eps: float = 1e-4, 
                   max_iter: int = 100, 
                   seed: Optional[int] = None) -> Tuple:
    """
    Full SAEM estimation for LaRT model.
    """
    log_T = np.log(T + utils.CLIP_MIN)
    J = R.shape[1]

    # --- Initialization ---
    if initial_params is None:
        theta_est, tau_est, a, b, omega, phi, lam, rho = utils.spectral_init_lart(R, T)
    else:
        theta_est = initial_params["theta"]
        tau_est = initial_params["tau"]
        a, b = initial_params["a"], initial_params["b"]
        omega, phi = initial_params["omega"], initial_params["phi"]
        lam, rho = initial_params["lam"], initial_params["rho"]

    if weight_scheme_fn is None:
        weight_scheme_fn = lambda x: 1.0 / x

    prev_loss_closure = None
    prev_grad_closure = None
    n_iter = 0
    diff = float('inf')

    # --- SAEM Loop ---
    while diff > eps and n_iter < max_iter:
        n_iter += 1
        current_seed = seed + n_iter if seed is not None else None
        gamma = weight_scheme_fn(n_iter)

        # 1. E-Step: Sampling
        params_curr = {'a': a, 'b': b, 'omega': omega, 'phi': phi, 'lam': lam, 'rho': rho}
        theta_s, tau_s = utils.post_samp_lart(n_samples, R, log_T, params_curr, seed=current_seed)

        # 2. M-Step: Update Rho (Closed Form)
        # Calculate empirical covariance of latent samples
        xi = np.stack([theta_s, tau_s], axis=1) # (N, 2, C)
        Sigma_emp = np.einsum('ipk,iqk->pq', xi, xi) / (R.shape[0] * theta_s.shape[1])
        
        # Normalize to correlation matrix
        d_sqrt = np.sqrt(np.diag(Sigma_emp))
        Sigma_emp = Sigma_emp / np.outer(d_sqrt, d_sqrt)
        
        # SAEM Update
        rho_new = (1 - gamma) * rho + gamma * Sigma_emp[0, 1]

        # 3. M-Step: Update Global Params (Optimization)
        
        # Define closures that capture current samples
        def curr_loss_fn(a_, b_, omega_, phi_, lam_, rho_):
            return _lart_log_like_samp(R, log_T, theta_s, tau_s, a_, b_, omega_, phi_, lam_, rho_)

        def curr_grad_fn(a_, b_, omega_, phi_, lam_):
            return _lart_grad_samp(theta_s, tau_s, R, log_T, a_, b_, omega_, phi_, lam_)

        # Objective Function for Optimizer
        def obj_fn(flat_params):
            _a = flat_params[:J]
            _b = flat_params[J:2*J]
            _o = flat_params[2*J:3*J]
            _p = flat_params[3*J:4*J]
            _log_lam = flat_params[4*J:]
            _lam = np.exp(_log_lam)
            
            # Loss: - [gamma * L_curr + (1-gamma) * L_prev]
            # (We minimize negative log-likelihood)
            l_curr = -gamma * curr_loss_fn(_a, _b, _o, _p, _lam, rho_new)
            l_prev = (1 - gamma) * prev_loss_closure(flat_params) if prev_loss_closure else 0
            loss = l_curr + l_prev
            
            # Gradient
            g_curr_raw = -gamma * curr_grad_fn(_a, _b, _o, _p, _lam)
            # Chain rule for log_lam: dL/d(log_lam) = dL/dlam * lam
            g_curr_raw[4*J:] *= _lam 
            
            g_prev = (1 - gamma) * prev_grad_closure(flat_params) if prev_grad_closure else 0
            grad = g_curr_raw + g_prev
            
            return loss, grad

        # Optimize
        x0 = np.concatenate([a, b, omega, phi, np.log(lam + utils.EPSILON)])
        res = minimize(obj_fn, x0, method='L-BFGS-B', jac=True,
                       options={'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-7})
        
        x_new = res.x
        a_new = x_new[:J]
        b_new = x_new[J:2*J]
        omega_new = x_new[2*J:3*J]
        phi_new = x_new[3*J:4*J]
        lam_new = np.exp(x_new[4*J:])
        rho = rho_new

        # Update Convergence Metric
        diff = np.mean(np.abs(a_new - a)) + np.mean(np.abs(b_new - b))
        
        # Update State
        a, b, omega, phi, lam = a_new, b_new, omega_new, phi_new, lam_new

        # Update Closures for next iteration
        # We wrap the current calculation into a function of params only
        # Note: We must bind current samples using default args or a factory
        def make_next_loss(prev_c, curr_l_fn, g):
            def loss_wrapper(fp):
                _a, _b = fp[:J], fp[J:2*J]
                _o, _p = fp[2*J:3*J], fp[3*J:4*J]
                _l = np.exp(fp[4*J:])
                
                val_curr = -g * curr_l_fn(_a, _b, _o, _p, _l, rho_new)
                val_prev = (1 - g) * prev_c(fp) if prev_c else 0
                return val_curr + val_prev
            return loss_wrapper

        def make_next_grad(prev_g, curr_g_fn, g):
            def grad_wrapper(fp):
                _a, _b = fp[:J], fp[J:2*J]
                _o, _p = fp[2*J:3*J], fp[3*J:4*J]
                _l = np.exp(fp[4*J:])
                
                grad_curr = -g * curr_g_fn(_a, _b, _o, _p, _l)
                grad_curr[4*J:] *= _l # Chain rule
                
                grad_prev = (1 - g) * prev_g(fp) if prev_g else 0
                return grad_curr + grad_prev
            return grad_wrapper
            
        prev_loss_closure = make_next_loss(prev_loss_closure, curr_loss_fn, gamma)
        prev_grad_closure = make_next_grad(prev_grad_closure, curr_grad_fn, gamma)

    # --- Finalization ---
    if np.sum(a) < 0:
        a = -a
        rho = -rho
    if np.sum(phi) < 0:
        phi = -phi
        rho = -rho

    # Final MAP for individuals
    final_params = {'a': a, 'b': b, 'omega': omega, 'phi': phi, 'lam': lam, 'rho': rho}
    theta_est, tau_est = get_indi_map_lart(theta_est, tau_est, R, log_T, final_params)

    return theta_est, tau_est, a, b, omega, phi, lam, rho, n_iter


def IRT_SAEM_full(R: np.ndarray, 
                  n_samples: int = 1, 
                  initial_params: Optional[Dict[str, Any]] = None, 
                  weight_scheme_fn: Optional[Callable[[int], float]] = None, 
                  eps: float = 1e-4, 
                  max_iter: int = 100, 
                  seed: Optional[int] = None) -> Tuple:
    """
    Full SAEM estimation for IRT model.
    """
    N, J = R.shape

    # --- Initialization ---
    if initial_params is not None:
        theta_est = initial_params["theta"]
        a, b = initial_params["a"], initial_params["b"]
    else:
        theta_est, a, b = utils.spectral_init_irt(R)

    if weight_scheme_fn is None:
        weight_scheme_fn = lambda x: 1.0 / x

    prev_loss_closure = None
    prev_grad_closure = None
    n_iter = 0
    diff = float('inf')

    # --- SAEM Loop ---
    while diff > eps and n_iter < max_iter:
        n_iter += 1
        current_seed = seed + n_iter if seed is not None else None
        gamma = weight_scheme_fn(n_iter)

        # 1. E-Step
        theta_s = utils.post_samp_theta_irt(n_samples, R, a, b, seed=current_seed)

        # 2. M-Step
        def curr_loss_fn(a_, b_, s2_):
            return _irt_log_like_samp(theta_s, R, a_, b_, s2_)
        
        def curr_grad_fn(a_, b_):
            return _irt_grad_samp(theta_s, R, a_, b_)

        def obj_fn(flat_params):
            _a, _b = flat_params[:J], flat_params[J:]
            
            l_curr = -gamma * curr_loss_fn(_a, _b)
            l_prev = (1 - gamma) * prev_loss_closure(flat_params) if prev_loss_closure else 0
            
            g_curr = -gamma * curr_grad_fn(_a, _b)
            g_prev = (1 - gamma) * prev_grad_closure(flat_params) if prev_grad_closure else 0
            
            return l_curr + l_prev, g_curr + g_prev

        # Optimize
        x0 = np.concatenate([a, b])
        res = minimize(obj_fn, x0, method='L-BFGS-B', jac=True,
                       options={'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-7})
        
        a_new, b_new = res.x[:J], res.x[J:]
        diff = np.mean(np.abs(a_new - a)) + np.mean(np.abs(b_new - b))
        a, b = a_new, b_new

        # Update closures
        def make_next_loss(prev_c, curr_l_fn, g):
            return lambda fp: -g * curr_l_fn(fp[:J], fp[J:]) + \
                              ((1 - g) * prev_c(fp) if prev_c else 0)

        def make_next_grad(prev_g, curr_g_fn, g):
            return lambda fp: -g * curr_g_fn(fp[:J], fp[J:]) + \
                              ((1 - g) * prev_g(fp) if prev_g else 0)

        prev_loss_closure = make_next_loss(prev_loss_closure, curr_loss_fn, gamma)
        prev_grad_closure = make_next_grad(prev_grad_closure, curr_grad_fn, gamma)

    # --- Finalization ---
    if np.sum(a) < 0:
        a = -a
        theta_est = -theta_est # Correct sign flip for individual estimates

    theta_est = get_indi_map_irt(theta_est, R, a, b)

    return theta_est, a, b, n_iter