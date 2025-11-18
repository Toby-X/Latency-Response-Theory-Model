import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.sparse.linalg import svds
from minimax_tilting_sampler import TruncatedMVN

def log_complete_likelihood(R, log_T, theta, tau, a, b, omega, phi, lam, rho):
    N = R.shape[0]
    J = R.shape[1]

    Sigma = np.array([[1., rho], [rho, 1.]])
    Sigma_inv = np.linalg.inv(Sigma)

    probit_prob = norm.cdf((2*R - 1) * (a[np.newaxis,:] * theta[:,np.newaxis] + b[np.newaxis,:]))
    # log_like_1 = np.sum(R * log_expit(a[np.newaxis,:] * theta[:,np.newaxis] + b[np.newaxis,:]) + \
        #  (1 - R) * log_expit(- (a[np.newaxis,:] * theta[:,np.newaxis] + b[np.newaxis,:])))
    log_like_1 = np.sum(np.log(probit_prob + 1e-12))

    log_like_2 = - N * np.sum(np.log(np.abs(lam)))
    normal_like = (log_T - omega[np.newaxis,:] + tau[:,np.newaxis] * phi[np.newaxis,:]) ** 2 / (lam ** 2)[np.newaxis,:]
    log_like_3 = - np.sum(normal_like) / 2

    log_like4 = - N * np.log(np.linalg.det(Sigma)) / 2

    xi = np.stack([theta, tau], axis=1)
    quadratic_form_sum = np.sum((xi @ Sigma_inv) * xi)
    log_like4 -= quadratic_form_sum / 2

    log_like = log_like_1 + log_like_2 + log_like_3 + log_like4

    return log_like

def LaRT_spectral(R, T):
    N, J = R.shape
    eps = 1e-9
    log_T = np.log(T + eps)

    def ini_params_R(R):
        U_R1, S_R1, Vt_R1 = np.linalg.svd(R, full_matrices=False)
        
        threshold = 1.01 * np.sqrt(N)
        idx = np.where(S_R1 > threshold)[0][-1]
        K_tilde = np.max([1, idx])
        K_tilde += 1

        U_reduced_R1 = U_R1[:, :K_tilde]
        S_reduced_R1 = S_R1[:K_tilde]
        Vt_reduced_R1 = Vt_R1[:K_tilde, :]
        R_est1 = U_reduced_R1 @ np.diag(S_reduced_R1) @ Vt_reduced_R1
        R_est_clip = np.clip(R_est1, 0 + eps, 1 - eps)
        R_inv_est = norm.ppf(R_est_clip)
        # R_inv_est = logit(R_est_clip)

        b_ini = np.mean(R_inv_est, axis=0)
        R_inv_res = R_inv_est - b_ini[np.newaxis, :]
        U_R2, S_R2, Vt_R2 = svds(R_inv_res, k=1)
        theta_ini = U_R2[:, 0] * np.sqrt(N)
        a_ini = Vt_R2[0, :] * S_R2[0] / np.sqrt(N)

        if np.sum(a_ini) < 0:
            a_ini = - a_ini
            theta_ini = - theta_ini

        return a_ini, b_ini, theta_ini

    def ini_params_T(log_T):
        omega_ini = np.mean(log_T, axis=0)
        log_T_res = log_T - omega_ini[np.newaxis, :]
        U_T, S_T, Vt_T = svds(log_T_res, k=1)

        tau_ini = U_T[:, 0] * np.sqrt(N)
        phi_ini = - Vt_T[0, :] * S_T[0] / np.sqrt(N)

        sigma_1 = S_T[0]
        u_1 = U_T[:, 0]
        v_1_t = Vt_T[0, :]
        rank_1_approx = sigma_1 * np.outer(u_1, v_1_t)

        log_T_res2 = log_T_res - rank_1_approx
        log_T_res2 = log_T_res2 ** 2
        lam_ini_2 = np.mean(log_T_res2, axis=0)
        lam_ini = np.sqrt(lam_ini_2 + eps)

        if np.sum(phi_ini) < 0:
            phi_ini = - phi_ini
            tau_ini = - tau_ini

        return tau_ini, phi_ini, omega_ini, lam_ini

    a_est, b_est, theta_est = ini_params_R(R)
    tau_est, phi_est, omega_est, lam_est = ini_params_T(log_T)
    indi_params = np.stack([theta_est, tau_est], axis=1)
    Sigma_est = np.mean(indi_params[:, :, np.newaxis] @ indi_params[:, np.newaxis, :], axis=0)
    rho_est = Sigma_est[0, 1]

    return theta_est, tau_est, a_est, b_est, omega_est, phi_est, lam_est, rho_est

def grad_indi_given_all(theta, tau, R, log_T, a, b, omega, phi, lam, rho):
    """Vectorized calculation of gradients with respect to individual parameters."""
    Sigma = np.array([[1., rho], [rho, 1.]])
    Sigma_inv = np.linalg.inv(Sigma)

    val = (2 * R - 1) * (theta[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :])
    # val = a[np.newaxis, :] * theta[:, np.newaxis] + b[np.newaxis, :]
    # ratio is also an (N, J) matrix
    log_pdf_val = norm.logpdf(val)
    log_cdf_val = norm.logcdf(val)
    ratio = np.exp(np.nan_to_num(log_pdf_val - log_cdf_val, neginf=0.0))

    grad_theta = np.sum(ratio * (2 * R - 1) * a[np.newaxis, :], axis=1)
    # grad_theta_mat = a[np.newaxis, :] * (R * expit(-val) - (1 - R) * expit(val))
    # grad_theta = np.sum(grad_theta_mat, axis=1)

    tau_term = log_T - omega + tau[:, np.newaxis] * phi
    grad_tau = - np.sum(tau_term * (phi/lam**2)[np.newaxis, :], axis=1)

    xi = np.stack([theta, tau], axis=1)
    adjustment = Sigma_inv @ xi.T
    grad_theta -= adjustment[0, :]
    grad_tau -= adjustment[1, :]

    return grad_theta, grad_tau

def grad_all_given_indi(theta, tau, R, log_T, a, b, omega, phi, lam):
    """Vectorized calculation of gradients with respect to global parameters."""
    J = len(a)
    
    val = (2 * R - 1) * (theta[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :])
    # val = a[np.newaxis, :] * theta[:, np.newaxis] + b[np.newaxis, :]
    # grad_mat = (2*R - 1) * expit(val) * (1 - expit(val))

    log_cdf_val = norm.logcdf(val)
    ratio = np.exp(np.nan_to_num(norm.logpdf(val) - log_cdf_val, neginf=0.0))
    term_for_ops = log_T - omega[np.newaxis, :] + tau[:, np.newaxis] * phi[np.newaxis, :]
    
    # --- Vectorized gradients for a, b, omega, phi ---
    grad_a = np.sum(ratio * (2 * R - 1) * theta[:, np.newaxis], axis=0)
    grad_b = np.sum(ratio * (2 * R - 1), axis=0)
    # grad_ab_mat = R * expit(-val) - (1 - R) * expit(val)
    # grad_a = np.sum(grad_ab_mat * theta[:, np.newaxis], axis=0)
    # grad_b = np.sum(grad_ab_mat, axis=0)
    lam_sq_col = lam[:, np.newaxis]**2
    grad_omega = np.sum(term_for_ops / lam_sq_col, axis=0)
    grad_phi = -np.sum(term_for_ops * tau[:, np.newaxis] / lam_sq_col, axis=0)
    
    sum_term_for_lam = np.sum(term_for_ops ** 2, axis=1)
    grad_lam = -J / lam + sum_term_for_lam / (lam**3)
    ## since we are actually optimizing log lam, we need to modify the gradient
    grad_lam = lam * grad_lam
    
    return grad_a, grad_b, grad_omega, grad_phi, grad_lam

def V1_sampler(sigma_theta2, D1, D2, S, seed=None):
    J = D1.shape[0]
    mu = np.zeros(J)
    S_inv = np.linalg.inv(S)
    Sigma = S_inv @ (sigma_theta2 * (D1 @ D1.T) + np.eye(J)) @ S_inv
    truncation_lb = - S_inv @ D2
    truncation_ub = np.inf * np.ones(J)
    truncation_lb = truncation_lb.flatten()
    sampler = TruncatedMVN(mu, Sigma, truncation_lb, truncation_ub, seed=seed)
    # samples = sampler.sample(n_samples)
    return sampler

def V1_sampler_full(mu_theta, sigma_theta2, D1, D2, S, seed=None):
    J = D1.shape[0]
    mu = np.zeros(J)
    S_inv = np.linalg.inv(S)
    Sigma = S_inv @ (sigma_theta2 * (D1 @ D1.T) + np.eye(J)) @ S_inv
    truncation_lb = - S_inv @ (D2 + D1 * mu_theta)
    truncation_ub = np.inf * np.ones(J)
    truncation_lb = truncation_lb.flatten()
    sampler = TruncatedMVN(mu, Sigma, truncation_lb, truncation_ub, seed=seed)

    return sampler

def SUN_sampler(n_samples, sigma_theta2, R_i, a, b, seed=None):
    if seed is not None:
        np.random.seed(seed)
    J = a.shape[0]
    diag_vals = 2 * R_i - 1
    D1 = diag_vals * a
    D2 = diag_vals * b
    s_diag = np.sqrt((D1.flatten()**2) * sigma_theta2 + 1)
    S = np.diag(s_diag)

    D1 = D1.reshape(-1,1)
    D2 = D2.reshape(-1,1)

    sampler_V1 = V1_sampler(sigma_theta2, D1, D2, S, seed=seed)
    samples_V1 = sampler_V1.sample(int(n_samples)) ## this will give shape like (J, n_samples)
    tmp_inv = np.linalg.inv(sigma_theta2 * D1 @ D1.T + np.eye(J))
    sigma_0_2 = 1- sigma_theta2 * D1.T @ tmp_inv @ D1
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    samples_V0 = rng.normal(0, np.sqrt(sigma_0_2.flatten()), n_samples) ## this is (n_samples,)

    sigma_theta = np.sqrt(sigma_theta2)
    V1_transformed = sigma_theta * D1.T @ tmp_inv @ S @ samples_V1

    samples = sigma_theta * (samples_V0 + V1_transformed.flatten())
    return samples

def SUN_sampler_full(n_samples, mu_theta, sigma_theta2, R_i, a, b, seed=None):
    if seed is not None:
        np.random.seed(seed)
    J = a.shape[0]
    diag_vals = 2 * R_i - 1
    D1 = diag_vals * a
    D2 = diag_vals * b
    s_diag = np.sqrt((D1.flatten()**2) * sigma_theta2 + 1)
    S = np.diag(s_diag)

    D1 = D1.reshape(-1,1)
    D2 = D2.reshape(-1,1)

    sampler_V1 = V1_sampler_full(mu_theta, sigma_theta2, D1, D2, S, seed=seed)
    samples_V1 = sampler_V1.sample(int(n_samples)) ## this will give shape like (J, n_samples)
    tmp_inv = np.linalg.inv(sigma_theta2 * D1 @ D1.T + np.eye(J))
    sigma_0_2 = 1- sigma_theta2 * D1.T @ tmp_inv @ D1
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    samples_V0 = rng.normal(0, np.sqrt(sigma_0_2.flatten()), n_samples) ## this is (n_samples,)

    sigma_theta = np.sqrt(sigma_theta2)
    V1_transformed = sigma_theta * D1.T @ tmp_inv @ S @ samples_V1

    samples = sigma_theta * (samples_V0 + V1_transformed.flatten()) + mu_theta
    return samples

def post_samp_theta(n_samples, R, a, b, seed=None):
    sigma_theta2 = 1.0
    N = R.shape[0]
    theta_samples = np.zeros((N, n_samples))
    for i in range(N):
        theta_samples[i, :] = SUN_sampler(n_samples, sigma_theta2, R[i,:], a, b, seed=seed)

    return theta_samples

def post_samp_tau_theta(n_samples, R, log_T, a, b, omega, phi, lam, rho, seed = None):
    sigma_theta2 = 1.0
    sigma_tau2 = 1.0
    sigma_rho_tau = rho
    N = R.shape[0]
    J = R.shape[1]

    if seed is not None:
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
    else:
        rng = np.random.default_rng()
    theta_samples = np.zeros((N, n_samples))
    tau_samples = np.zeros((N, n_samples))
    sigma_tau2_reorg = sigma_tau2 - sigma_rho_tau**2
    sigma_tau2_p = (1 / sigma_tau2_reorg + np.sum((phi/lam)**2)) ** (-1)
    B = -np.sum((log_T - omega[np.newaxis,:]) * (phi/lam**2)[np.newaxis,:], axis=1)
    for i in range(N):
        sigma_theta2_p = (1/(1-rho**2) - sigma_tau2_p * rho ** 2 / (1 - rho ** 2)) ** (-1)
        mu_theta_i = sigma_theta2_p * sigma_tau2_p * B[i] * rho / (1 - rho ** 2)
        theta_samples[i, :] = SUN_sampler_full(n_samples, mu_theta_i, sigma_theta2_p, R[i,:], a, b, seed=seed)
        mu_tau_reorg = theta_samples[i, :] * sigma_rho_tau / sigma_theta2
        # mu_tau = (-np.sum((log_T[i,:] - omega[np.newaxis,:]) * (phi/lam**2)[np.newaxis,:], axis=1) + \
        #            mu_tau_reorg/sigma_tau2_reorg) * sigma_tau2_p
        mu_tau = (B[i] + mu_tau_reorg/sigma_tau2_reorg) * sigma_tau2_p
        tau_samples[i, :] = rng.normal(mu_tau, np.sqrt(sigma_tau2_p), n_samples)

    return theta_samples, tau_samples

def log_complete_like_samp(R, log_T, theta_samples, tau_samples, a, b, omega, phi, lam, rho):
    """
    Calculates the log complete-data likelihood using vectorized NumPy operations.
    """
    # Get dimensions
    N, J = R.shape
    C = theta_samples.shape[1]

    # --- Pre-calculations (Independent of samples) ---
    Sigma = np.array([[1., rho], [rho, 1.]])
    Sigma_inv = np.linalg.inv(Sigma)
    log_like_2 = -N * np.sum(np.log(np.abs(lam)))
    log_like_4_const = -N * np.log(np.linalg.det(Sigma)) / 2

    arg_probit = (2 * R[:, :, np.newaxis] - 1) * \
                 (a[np.newaxis, :, np.newaxis] * theta_samples[:, np.newaxis, :] + b[np.newaxis, :, np.newaxis])
    
    probit_prob = norm.cdf(arg_probit)
    log_like_1 = np.sum(np.log(probit_prob + 1e-12)) / C

    numerator = (log_T[:, :, np.newaxis] - omega[np.newaxis, :, np.newaxis] + \
                 tau_samples[:, np.newaxis, :] * phi[np.newaxis, :, np.newaxis]) ** 2
    denominator = lam[np.newaxis, :, np.newaxis] ** 2
    
    log_like_3 = -np.sum(numerator / denominator) / 2 / C

    xi_samples = np.stack([theta_samples, tau_samples], axis=1)

    total_quadratic_form = np.einsum('ijc,jk,ikc->', xi_samples, Sigma_inv, xi_samples)
    log_like_4_samples = -total_quadratic_form / 2 / C

    log_like = log_like_1 + log_like_2 + log_like_3 + log_like_4_const + log_like_4_samples

    return log_like

def grad_all_given_indi_samp(theta_samp, tau_samp, R, log_T, a, b, omega, phi, lam):
    """
    Vectorized calculation of the gradients of the log-likelihood with respect
    to the global parameters (a, b, omega, phi, lam), averaged over all samples.
    """
    # Get dimensions
    N, C = theta_samp.shape
    J = len(a)

    # --- Term 1: Probit-related calculations ---
    val = (2 * R[:, :, np.newaxis] - 1) * \
          (theta_samp[:, np.newaxis, :] * a[np.newaxis, :, np.newaxis] + \
           b[np.newaxis, :, np.newaxis])
    log_cdf_val = norm.logcdf(val)
    ratio = np.exp(np.nan_to_num(norm.logpdf(val) - log_cdf_val, neginf=0.0))

    # --- Term 2: Normal-related calculations ---
    term_for_ops = log_T[:, :, np.newaxis] - omega[np.newaxis, :, np.newaxis] + \
                   tau_samp[:, np.newaxis, :] * phi[np.newaxis, :, np.newaxis]

    # --- Calculate Gradients (Sum and Average in one step) ---
    grad_a = np.sum(ratio * (2 * R[:, :, np.newaxis] - 1) * theta_samp[:, np.newaxis, :], axis=(0, 2)) / C
    grad_b = np.sum(ratio * (2 * R[:, :, np.newaxis] - 1), axis=(0, 2)) / C
    
    # For omega and phi, we also sum over N and C and average
    lam_sq_col = lam[np.newaxis, :, np.newaxis]**2
    grad_omega = np.sum(term_for_ops / lam_sq_col, axis=(0, 2)) / C
    grad_phi = -np.sum(term_for_ops * tau_samp[:, np.newaxis, :] / lam_sq_col, axis=(0, 2)) / C
    
    # For lam, the logic is slightly different
    sum_term_for_lam = np.sum(term_for_ops ** 2, axis=(0, 2)) / C
    grad_lam = -N / lam + sum_term_for_lam / (lam**3)
    
    return np.concatenate([grad_a, grad_b, grad_omega, grad_phi, grad_lam])

def update_all_given_indi_samp(theta_samples, tau_samples, R, log_T, a_old, b_old, omega_old, phi_old, lam_old, rho_old,
                          weight, loss_old, grad_old):
    ## loss_old and grad_old needs to function of all the population parameters
    N = R.shape[0]
    J = R.shape[1]
    C = theta_samples.shape[1]

    Sigma_old = np.array([[1., rho_old], [rho_old, 1.]])
    all_indi_params = np.stack([theta_samples, tau_samples], axis=1)
    Sigma_new = np.einsum('ipk,iqk->pq', all_indi_params, all_indi_params) / (N * C)
    norm_sigma_new = np.diag(Sigma_new)**(1/2)
    norm_sigma_mat = norm_sigma_new[:, np.newaxis] @ norm_sigma_new[np.newaxis, :]
    Sigma_new = Sigma_new / norm_sigma_mat
    Sigma_new = (1-weight) * Sigma_old + weight * Sigma_new
    # rho_new = np.mean(theta_samples * tau_samples)
    # rho_new = (1-weight) * rho_old + weight * rho_new
    rho_new = Sigma_new[0, 1]

    def get_loss_fn(R, log_T, theta_samples, tau_samples):
        if loss_old is None:
            def loss(a, b, omega, phi, lam, rho):
                return -weight * log_complete_like_samp(R, log_T, theta_samples, tau_samples, a, b, omega, phi, lam, rho)
        else:
            def loss(a, b, omega, phi, lam, rho):
                return -weight * log_complete_like_samp(R, log_T, theta_samples, tau_samples, a, b, omega, phi, lam, rho) + (1-weight) * loss_old(a, b, omega, phi, lam, rho)
        return loss

    def get_grad_fn(R, log_T, theta_samples, tau_samples):
        if grad_old is None:
            def grad(a, b, omega, phi, lam):
                return -weight * grad_all_given_indi_samp(theta_samples, tau_samples, R, log_T, a, b, omega, phi, lam)
        else:
            def grad(a, b, omega, phi, lam):
                return -weight * grad_all_given_indi_samp(theta_samples, tau_samples, R, log_T, a, b, omega, phi, lam) + (1-weight) * grad_old(a, b, omega, phi, lam)
        return grad

    loss_fn = get_loss_fn(R, log_T, theta_samples, tau_samples)
    grad_fn = get_grad_fn(R, log_T, theta_samples, tau_samples)

    def obj_fn(all_params):
        a = all_params[:J]
        b = all_params[J:(2*J)]
        omega = all_params[(2*J):(3*J)]
        phi = all_params[(3*J):(4*J)]
        log_lam = all_params[(4*J):]
        lam = np.exp(log_lam)
        loss = loss_fn(a, b, omega, phi, lam, rho_new)
        grad = grad_fn(a, b, omega, phi, lam)
        grad[4*J:] = lam * grad[4*J:]  # Adjust gradient for log(lam)
        return (loss, grad)
    
    all_params_old = np.concatenate([a_old, b_old, omega_old, phi_old, np.log(lam_old+1e-12)])
    opt_res = minimize(obj_fn, all_params_old, method='L-BFGS-B', jac=True,
                       options={'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-7})

    all_params_new = opt_res.x
    a_new = all_params_new[:J]
    b_new = all_params_new[J:(2*J)]
    omega_new = all_params_new[(2*J):(3*J)]
    phi_new = all_params_new[(3*J):(4*J)]
    log_lam_new = all_params_new[(4*J):]
    lam_new = np.exp(log_lam_new)
    return a_new, b_new, omega_new, phi_new, lam_new, rho_new, loss_fn, grad_fn

def update_indi_fixed_all(theta_old, tau_old, R, log_T, a, b, omega, phi, lam, rho):
    N = R.shape[0]

    def obj_fn(indi_params):
        theta = indi_params[:N]
        tau = indi_params[N:]
        loss = -log_complete_likelihood(R, log_T, theta, tau, a, b, omega, phi, lam, rho)
        grad_theta, grad_tau = grad_indi_given_all(theta, tau, R, log_T, a, b, omega, phi, lam, rho)
        grad = np.concatenate([-grad_theta, -grad_tau])
        return (loss, grad)
    
    indi_params_old = np.concatenate([theta_old, tau_old])
    opt_res = minimize(obj_fn, indi_params_old, method='L-BFGS-B', jac=True,
                       options={'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-7})

    indi_params_new = opt_res.x
    theta_new = indi_params_new[:N]
    tau_new = indi_params_new[N:]
    return theta_new, tau_new

def LaRT_SAEM(R, T, n_samples=1, weight_scheme_fn = None, initial_params = None, eps = 1e-4, max_iter=100, seed=None):
    log_T = np.log(T)

    if weight_scheme_fn is None:
        weight_scheme_fn = lambda x: 1/x

    if initial_params is not None:
        a_init = initial_params["a"]
        b_init = initial_params["b"]
        omega_init = initial_params["omega"]
        phi_init = initial_params["phi"]
        lam_init = initial_params["lam"]
        rho_init = initial_params["rho"]
    else:
        theta_init, tau_init, a_init, b_init, omega_init, phi_init, lam_init, rho_init = LaRT_spectral(R, T)

    n_iter = 1
    # current_n_samples = min(20, n_samples)
    theta_samples, tau_samples = post_samp_tau_theta(n_samples, R, log_T, a_init, b_init, omega_init, phi_init, lam_init, rho_init, seed=seed)
    a, b, omega, phi, lam, rho, loss_old, grad_old = update_all_given_indi_samp(theta_samples, tau_samples, R, log_T, a_init, b_init, omega_init, phi_init, lam_init, rho_init,
                                                                           weight=weight_scheme_fn(n_iter), loss_old=None, grad_old=None)
    err = np.mean(np.abs(a - a_init)) + np.mean(np.abs(b - b_init))

    while (err > eps) and (n_iter < max_iter):
        n_iter += 1
        if seed is not None:
            seed += 1
        
        theta_samples, tau_samples = post_samp_tau_theta(n_samples, R, log_T, a, b, omega, phi, lam, rho, seed=seed)
        a_new, b_new, omega_new, phi_new, lam_new, rho_new, loss_old, grad_old = update_all_given_indi_samp(theta_samples, tau_samples, R, log_T, a, b, omega, phi, lam, rho,
                                                                                                         weight=weight_scheme_fn(n_iter), loss_old=loss_old, grad_old=grad_old)
        err = np.mean(np.abs(a_new - a)) + np.mean(np.abs(b_new - b))

        a, b, omega, phi, lam, rho = a_new, b_new, omega_new, phi_new, lam_new, rho_new

    if np.sum(a) < 0:
        a = - a
        rho = - rho
    if np.sum(phi) < 0:
        phi = - phi
        rho = - rho
    
    return a, b, omega, phi, lam, rho, n_iter

def LaRT_SAEM_full(R, T, n_samples=1, weight_scheme_fn = None, initial_params = None, eps = 1e-4, max_iter=100, seed=None):
    if initial_params is None:
        theta_init, tau_init, a_init, b_init, omega_init, phi_init, lam_init, rho_init = LaRT_spectral(R, T)
        initial_params = {
            "theta": theta_init,
            "tau": tau_init,
            "a": a_init,
            "b": b_init,
            "omega": omega_init,
            "phi": phi_init,
            "lam": lam_init,
            "rho": rho_init
        }
    else:
        theta_init = initial_params["theta"]
        tau_init = initial_params["tau"]

    a_est, b_est, omega_est, phi_est, lam_est, rho_est, n_iter = LaRT_SAEM(R, T, n_samples, weight_scheme_fn, initial_params, eps, max_iter, seed=seed)
    theta_est, tau_est = update_indi_fixed_all(theta_init, tau_init, R, np.log(T), a_est, b_est, omega_est, phi_est, lam_est, rho_est)

    return theta_est, tau_est, a_est, b_est, omega_est, phi_est, lam_est, rho_est, n_iter

def IRT_spectral(R, T):
    N, J = R.shape
    eps = 1e-9

    def ini_params_R(R):
        U_R1, S_R1, Vt_R1 = np.linalg.svd(R, full_matrices=False)
        
        threshold = 1.01 * np.sqrt(N)
        idx = np.where(S_R1 > threshold)[0][-1]
        K_tilde = np.max([1, idx])
        K_tilde += 1

        U_reduced_R1 = U_R1[:, :K_tilde]
        S_reduced_R1 = S_R1[:K_tilde]
        Vt_reduced_R1 = Vt_R1[:K_tilde, :]
        R_est1 = U_reduced_R1 @ np.diag(S_reduced_R1) @ Vt_reduced_R1
        R_est_clip = np.clip(R_est1, 0 + eps, 1 - eps)
        R_inv_est = norm.ppf(R_est_clip)
        # R_inv_est = logit(R_est_clip)

        b_ini = np.mean(R_inv_est, axis=0)
        R_inv_res = R_inv_est - b_ini[np.newaxis, :]
        U_R2, S_R2, Vt_R2 = svds(R_inv_res, k=1)
        theta_ini = U_R2[:, 0] * np.sqrt(N)
        a_ini = Vt_R2[0, :] * S_R2[0] / np.sqrt(N)

        if np.sum(a_ini) < 0:
            a_ini = - a_ini
            theta_ini = - theta_ini

        return a_ini, b_ini, theta_ini

    a_est, b_est, theta_est = ini_params_R(R)
    sigma2_est = np.mean(theta_est**2)

    return theta_est, a_est, b_est, sigma2_est

def IRT_SAEM_full(R, n_samples = 1, initial_params = None, weight_scheme_fn = None, 
                   eps = 1e-4, max_iter=100, seed=None):
    def log_c_like(R, theta, a, b, sigma2):
        N, J = R.shape
        # --- Pre-calculations (Independent of samples) ---
        arg_probit = (2 * R - 1) * \
                        (a[np.newaxis, :] * theta[:, np.newaxis] + b[np.newaxis, :])
        probit_prob = norm.cdf(arg_probit)
        log_like_1 = np.sum(np.log(probit_prob + 1e-12))
        log_like_4_const = -N * np.log(sigma2) / 2
        log_like_4_samples = np.sum(theta**2) / 2 / sigma2
        log_like = log_like_1 + log_like_4_const + log_like_4_samples
        return log_like / N

    def log_c_like_samp(R, theta_samples, a, b, sigma2):
        N, J = R.shape
        C = theta_samples.shape[1]

        # --- Pre-calculations (Independent of samples) ---
        log_like_4_const = -N * np.log(sigma2) / 2

        arg_probit = (2 * R[:, :, np.newaxis] - 1) * \
                        (a[np.newaxis, :, np.newaxis] * theta_samples[:, np.newaxis, :] + b[np.newaxis, :, np.newaxis])

        probit_prob = norm.cdf(arg_probit)
        log_like_1 = np.sum(np.log(probit_prob + 1e-12)) / C

        log_like_4_samples = np.sum(theta_samples**2) / 2 / sigma2 / C
        log_like = log_like_1+ log_like_4_const + log_like_4_samples
        return log_like / N

    def grad_theta_given_other(theta, R, a, b, sigma2):
        """Vectorized calculation of gradients with respect to individual parameters."""
        N, J = R.shape

        val = (2 * R - 1) * (theta[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :])
        # ratio is also an (N, J) matrix
        log_pdf_val = norm.logpdf(val)
        log_cdf_val = norm.logcdf(val)
        ratio = np.exp(np.nan_to_num(log_pdf_val - log_cdf_val, neginf=0.0))

        grad_theta = np.sum(ratio * (2 * R - 1) * a[np.newaxis, :], axis=1)
        # val = a[np.newaxis, :] * theta[:, np.newaxis] + b[np.newaxis, :]
        # grad_theta_mat = a[np.newaxis, :] * (R * expit(-val) - (1 - R) * expit(val))
        # grad_theta = np.sum(grad_theta_mat, axis=1)
        grad_theta -= theta/sigma2
        return grad_theta / N
    
    def grad_all_given_indi_samp(theta_samp, R, a, b):
        """
        Vectorized calculation of the gradients of the log-likelihood with respect
        to the global parameters (a, b, omega, phi, lam), averaged over all samples.
        """
        # Get dimensions
        N, C = theta_samp.shape
        J = len(a)

        # --- Term 1: Probit-related calculations ---
        val = (2 * R[:, :, np.newaxis] - 1) * \
            (theta_samp[:, np.newaxis, :] * a[np.newaxis, :, np.newaxis] + \
            b[np.newaxis, :, np.newaxis])
        log_cdf_val = norm.logcdf(val)
        ratio = np.exp(np.nan_to_num(norm.logpdf(val) - log_cdf_val, neginf=0.0))

        # --- Calculate Gradients (Sum and Average in one step) ---
        grad_a = np.sum(ratio * (2 * R[:, :, np.newaxis] - 1) * theta_samp[:, np.newaxis, :], axis=(0, 2)) / C
        grad_b = np.sum(ratio * (2 * R[:, :, np.newaxis] - 1), axis=(0, 2)) / C
    
        return np.concatenate([grad_a / N, grad_b / N])

    def update_indi_fixed_all(theta_old, R, a, b, sigma2):
        def obj_fn(theta):
            loss = -log_c_like(R, theta, a, b, sigma2)
            grad_theta = grad_theta_given_other(theta, R, a, b, sigma2)
            return (loss, -grad_theta)

        opt_res = minimize(obj_fn, theta_old, method='L-BFGS-B', jac=True,
                           options={'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-7})
        theta_new = opt_res.x
        return theta_new
    
    def update_all_given_indi_samp(theta_samples, R, a_old, b_old, sigma2_old,
                          weight, loss_old, grad_old):
        ## loss_old and grad_old needs to function of all the population parameters
        N = R.shape[0]
        J = R.shape[1]
        C = theta_samples.shape[1]

        # sigma2_new = np.sum(theta_samples ** 2) / (N * C)
        # sigma2_new = (1-weight) * sigma2_old + weight * sigma2_new
        sigma2_new = 1

        def get_loss_fn(R, theta_samples):
            if loss_old is None:
                def loss(a, b, sigma2):
                    return -weight * log_c_like_samp(R, theta_samples, a, b, sigma2)
            else:
                def loss(a, b, sigma2):
                    return -weight * log_c_like_samp(R, theta_samples, a, b, sigma2) + (1-weight) * loss_old(a, b, sigma2)
            return loss

        def get_grad_fn(R, theta_samples):
            if grad_old is None:
                def grad(a, b):
                    return -weight * grad_all_given_indi_samp(theta_samples, R, a, b)
            else:
                def grad(a, b):
                    return -weight * grad_all_given_indi_samp(theta_samples, R, a, b) + (1-weight) * grad_old(a, b)
            return grad

        loss_fn = get_loss_fn(R, theta_samples)
        grad_fn = get_grad_fn(R, theta_samples)

        def obj_fn(all_params):
            a = all_params[:J]
            b = all_params[J:]
            loss = loss_fn(a, b, sigma2_new)
            grad = grad_fn(a, b)
            return (loss, grad)
        
        all_params_old = np.concatenate([a_old, b_old])
        opt_res = minimize(obj_fn, all_params_old, method='L-BFGS-B', jac=True,
                        options={'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-7})

        all_params_new = opt_res.x
        a_new = all_params_new[:J]
        b_new = all_params_new[J:]
        return a_new, b_new, sigma2_new, loss_fn, grad_fn

    if initial_params is not None:
        theta_init = initial_params["theta"]
        a_init = initial_params["a"]
        b_init = initial_params["b"]
        sigma2_init = initial_params["sigma2"]
    else:
        theta_init, a_init, b_init, sigma2_init = IRT_spectral(R)

    if weight_scheme_fn is None:
        weight_scheme_fn = lambda iter: 1 / (iter)

    n_iter = 1
    # current_n_samples = min(20, n_samples)
    theta_samples = post_samp_theta(n_samples, R, a_init, b_init, seed=seed)
    a, b, sigma2, loss_fn, grad_fn = update_all_given_indi_samp(theta_samples, R, a_init, b_init, sigma2_init,
                                      weight_scheme_fn(n_iter), loss_old=None, grad_old=None)
    # log_like = log_c_like_samp(R, theta_samples, a, b, sigma2)
    err = np.mean(np.abs(a - a_init)) + np.mean(np.abs(b - b_init)) 
    # if err > 1:
    #     if seed is not None:
    #         seed += 1
    #     theta_samples = post_samp_theta(current_n_samples, R, a_init, b_init, seed=seed)
    #     a, b, sigma2, loss_fn, grad_fn = update_all_given_indi_samp(theta_samples, R, a_init, b_init, sigma2_init,
    #                                       weight_scheme_fn(n_iter), loss_old=None, grad_old=None)
    #     err = np.mean(np.abs(a - a_init)) + np.mean(np.abs(b - b_init))
    # + np.abs(sigma2 - sigma2_init)
    # a_est, b_est, sigma2_est = a, b, sigma2

    while (err > eps) and (n_iter < max_iter):
        n_iter += 1
        if seed is not None:
            seed += 1

        theta_samples = post_samp_theta(n_samples, R, a, b, seed=seed)
        a_new, b_new, sigma2_new, loss_fn, grad_fn = update_all_given_indi_samp(theta_samples, R, a, b, sigma2,
                                      weight_scheme_fn(n_iter), loss_old=loss_fn, grad_old=grad_fn)
        err = np.mean(np.abs(a_new - a)) + np.mean(np.abs(b_new - b)) 

        a, b, sigma2 = a_new, b_new, sigma2_new

    if np.sum(a) < 0:
        a = - a
    
    # theta_est = update_indi_fixed_all(theta_init, R, a_est, b_est, sigma2_est)
    theta = update_indi_fixed_all(theta_init, R, a, b, sigma2)

    return theta, a, b, sigma2, n_iter

def fisher_info_theta_lart(a, b, theta, Sigma):
    val = theta[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :]
    log_phi_2 = 2 * norm.logpdf(val)
    log_Phi_p = norm.logcdf(val)
    log_Phi_m = norm.logcdf(-val)
    mat_common = np.exp(np.nan_to_num(log_phi_2 - log_Phi_p - log_Phi_m, neginf=0.0)) * (a ** 2)[np.newaxis, :]
    fisher_info_1 = np.sum(mat_common, axis=1)

    Sigma_inv = np.linalg.inv(Sigma)
    fisher_info_2 = Sigma_inv[0, 0]
    return fisher_info_1 + fisher_info_2

def fisher_info_theta_irt(a, b, theta, sigma2):
    val = theta[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :]
    log_phi_2 = 2 * norm.logpdf(val)
    log_Phi_p = norm.logcdf(val)
    log_Phi_m = norm.logcdf(-val)
    mat_common = np.exp(np.nan_to_num(log_phi_2 - log_Phi_p - log_Phi_m, neginf=0.0)) * (a ** 2)[np.newaxis, :]
    
    return np.sum(mat_common, axis=1) + 1/sigma2
