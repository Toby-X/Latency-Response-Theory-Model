import sys
import os
import numpy as np
import pandas as pd
import multiprocess as mp
from tqdm import tqdm
import itertools
from scipy.stats import norm
from scipy.optimize import minimize

binary_df_math500 = pd.read_csv('../data/accuracy_3.csv', index_col=0)
cot_df_math500 = pd.read_csv('../data/cot_length_3.csv', index_col=0)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

estimated_parameters = np.load("../data_processed/rest3_pred_params.npz")
a_joint = estimated_parameters['a_lart_train']
b_joint = estimated_parameters['b_lart_train']
omega_joint = estimated_parameters['omega_lart_train']
phi_joint = estimated_parameters['phi_lart_train']
lam_joint = estimated_parameters['lam_lart_train']
rho_joint = estimated_parameters['rho_lart']
a_irt = estimated_parameters['a_irt_train']
b_irt = estimated_parameters['b_irt_train']

from ..LaRT import update_indi_fixed_all


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

def update_indi_fixed_all_irt(theta_old, R, a, b, sigma2=1.0):
    def obj_fn(theta):
        loss = -log_c_like(R, theta, a, b, sigma2)
        grad_theta = grad_theta_given_other(theta, R, a, b, sigma2)
        return (loss, -grad_theta)

    opt_res = minimize(obj_fn, theta_old, method='L-BFGS-B', jac=True,
                        options={'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-7})
    theta_new = opt_res.x
    return theta_new

## data preprocessing
# rows_to_delete = [
#     "meta_llama_Llama_3.2_1B_one_shot",
#     "meta_llama_Llama_3.2_1B_zero_shot",
#     "meta_llama_Meta_Llama_3_8B_one_shot",
#     "meta_llama_Meta_Llama_3_8B_zero_shot",
#     "microsoft_phi_4_one_shot",
#     "microsoft_phi_4_zero_shot",
#     'TinyLlama_TinyLlama_1.1B_Chat_v1.0_zero_shot',
#     'TinyLlama_TinyLlama_1.1B_Chat_v1.0_one_shot',
#     'google_gemma_3_1b_pt_one_shot',
#     'google_gemma_3_1b_pt_zero_shot',
#     'google_gemma_7b_it_one_shot', 
#     'google_gemma_7b_it_zero_shot',
#     'google_vaultgemma_1b_one_shot', 
#     'google_vaultgemma_1b_zero_shot',
#     'meta_llama_Llama_3.2_3B_one_shot', 
#     'meta_llama_Llama_3.2_3B_zero_shot',
#     'openai_community_gpt2_one_shot', 
#     'openai_community_gpt2_zero_shot'
# ]

# binary_df_math500 = binary_df_math500.drop(rows_to_delete, errors='ignore')
# cot_df_math500 = cot_df_math500.drop(rows_to_delete, errors='ignore')

# binary_array = binary_df_math500.to_numpy()
# cot_array = cot_df_math500.to_numpy()
# cot_array += 1

binary_array = binary_df_math500.to_numpy()
cot_array = cot_df_math500.to_numpy()

N, J = binary_array.shape

# Function Needed to evaluate efficiency by step-wise highest fisher information
def instance_log_fisher_info(a, b, theta):
    '''Compute Item-wise log Fisher Information for Normal-ogive model for each test taker (theta 1-dim)'''
    args = a * theta + b
    log_num = 2 * norm.logpdf(args) + 2 * np.log(a)
    log_denom = norm.logcdf(args) + norm.logcdf(-args)
    log_fisher_info = log_num - log_denom

    return log_fisher_info

def log_fisher_info(a, b, theta):
    '''Compute Item-wise log Fisher Information for Normal-ogive model for each test taker (theta 1-dim)'''
    args = a[np.newaxis,:] * theta[:,np.newaxis] + b[np.newaxis,:]

    log_a = np.log(np.maximum(a, 1e-12))
    log_num = 2 * norm.logpdf(args) + 2 * log_a[np.newaxis,:]
    log_denom = norm.logcdf(args) + norm.logcdf(-args)
    log_fisher_info = log_num - log_denom

    return log_fisher_info

def initialize_theta_tau(R, log_T, a, b, omega, phi, lam, rho, num_items=10):
    N, J = R.shape
    theta_init = np.zeros(N)
    tau_init = np.zeros(N)
    
    theta_init, tau_init = update_indi_fixed_all(
        theta_init, tau_init, R[:, :num_items], log_T[:, :num_items], 
        a[:num_items], b[:num_items], omega[:num_items], 
        phi[:num_items], lam[:num_items], rho
    )

    return theta_init, tau_init

def step_wise_evaluation_joint(R, T, a, b, omega, phi, lam, rho, num_items=10, n_steps=None):
    N, J = R.shape
    if n_steps is None:
        n_steps = J
    log_T = np.log(T)

    selected_items = np.zeros((N, n_steps), dtype=bool)
    selected_items[:, :num_items] = True

    theta, tau = initialize_theta_tau(R, log_T, a, b, omega, phi, lam, rho, num_items=num_items)
    rank_each_step = np.zeros((N, n_steps-num_items), dtype=int)
    theta_each_step = np.zeros((N, n_steps-num_items), dtype=float)
    
    for step in range(n_steps-num_items):
        all_log_info = log_fisher_info(a, b, theta)
        all_log_info[selected_items] = -np.inf
        new_item_indices = np.argmax(all_log_info, axis=1)
        selected_items[np.arange(N), new_item_indices] = True
        for i in range(N):
            current_mask = selected_items[i, :]
            theta[i], tau[i] = update_indi_fixed_all(
                np.atleast_1d(theta[i]), np.atleast_1d(tau[i]), 
                R[i, current_mask].reshape(1, -1), log_T[i, current_mask].reshape(1, -1), 
                a[current_mask], b[current_mask], 
                omega[current_mask], phi[current_mask], 
                lam[current_mask], rho
            )
            # log_info = instance_log_fisher_info(a[~selected_items[i, :]], b[~selected_items[i, :]], theta[i])
            # item_index = np.argmax(log_info)
            # actual_item_index = np.arange(J)[~selected_items[i, :]][item_index]
            # selected_items[i, actual_item_index] = True
            # ## remember to check whether this function works in 1-dim !!!
            # theta[i], tau[i] = update_indi_fixed_all(
            #     np.atleast_1d(theta[i]), np.atleast_1d(tau[i]), 
            #     R[i, selected_items[i, :]].reshape(1, -1), log_T[i, selected_items[i, :]].reshape(1, -1), 
            #     a[selected_items[i, :]], b[selected_items[i, :]], 
            #     omega[selected_items[i, :]], phi[selected_items[i, :]], 
            #     lam[selected_items[i, :]], rho
            # )
        rank_each_step[:, step] = np.argsort(theta)
        theta_each_step[:, step] = theta

    return rank_each_step, theta_each_step

def initialize_theta(R, a, b, num_items=10):
    N, J = R.shape
    theta_init = np.zeros(N)
    
    theta_init = update_indi_fixed_all_irt(
        theta_init, R[:, :num_items],
        a[:num_items], b[:num_items], 1.0
    )

    return theta_init

def step_wise_evaluation_irt(R, a, b, num_items=10, n_steps=None):
    N, J = R.shape
    if n_steps is None:
        n_steps = J

    selected_items = np.zeros((N, n_steps), dtype=bool)
    selected_items[:, :num_items] = True

    theta = initialize_theta(R, a, b, num_items=num_items)
    rank_each_step = np.zeros((N, n_steps-num_items), dtype=int)
    theta_each_step = np.zeros((N, n_steps-num_items), dtype=float)
    
    for step in range(n_steps-num_items):
        all_log_info = log_fisher_info(a, b, theta)
        all_log_info[selected_items] = -np.inf
        new_item_indices = np.argmax(all_log_info, axis=1)
        selected_items[np.arange(N), new_item_indices] = True
        for i in range(N):
            current_mask = selected_items[i, :]
            theta[i]= update_indi_fixed_all_irt(
                np.atleast_1d(theta[i]), R[i, current_mask].reshape(1, -1),
                a[current_mask], b[current_mask], 1.0
            )
            # log_info = instance_log_fisher_info(a[~selected_items[i, :]], b[~selected_items[i, :]], theta[i])
            # item_index = np.argmax(log_info)
            # actual_item_index = np.arange(J)[~selected_items[i, :]][item_index]
            # selected_items[i, actual_item_index] = True
            # ## remember to check whether this function works in 1-dim !!!
            # theta[i]= update_indi_fixed_all_irt(
            #     np.atleast_1d(theta[i]), R[i, selected_items[i, :]].reshape(1, -1),
            #     a[selected_items[i, :]], b[selected_items[i, :]], 1.0
            # )
        rank_each_step[:, step] = np.argsort(theta)
        theta_each_step[:, step] = theta

    return rank_each_step, theta_each_step


# --- Create Subarrays with Random Columns (Unchanged) ---
np.random.seed(42)
chosen_models = np.random.choice(range(N), 100, replace=False)
chosen_models_bool = np.zeros(N, dtype=bool)
chosen_models_bool[chosen_models] = True

# --- 1. Define a More Generic Worker Function ---
# --- 2. Main Execution Block ---
if __name__ == "__main__":
    # Create a flat list of all 10 tasks to run
    with mp.Pool(processes=2) as pool:
        print("Starting cMIRT and MIRT model fitting in parallel...")

        # 1. Submit the first task (cMIRT) to the pool
        cmirt_result_async = pool.apply_async(
            step_wise_evaluation_joint,
            args=(binary_array[~chosen_models_bool], cot_array[~chosen_models_bool], 
                  a_joint, b_joint, 
                  omega_joint, phi_joint, 
                  lam_joint, rho_joint),
            kwds={'num_items': 10}
        )

        # 2. Submit the second task (MIRT) to the pool
        mirt_result_async = pool.apply_async(
            step_wise_evaluation_irt,
            args=(binary_array[~chosen_models_bool], 
                  a_irt, b_irt),
            kwds={'num_items': 10}
        )

        # 3. Wait for the results from both tasks
        print("Waiting for results...")
        rank_cirt, theta_cirt = cmirt_result_async.get()
        rank_irt, theta_irt = mirt_result_async.get()

    print("Model fitting complete. Saving results...")

    # 4. Save the estimated parameters into a single parquet file
    # The arrays are wrapped in lists to be stored correctly in a DataFrame
    np.savez_compressed(
    '../data_processed/efficiency_rest3.npz',  # Note the .npz extension
    rank_cirt=rank_cirt,
    rank_irt=rank_irt,
    theta_cirt=theta_cirt,
    theta_irt=theta_irt
    )

    print("Results saved to 'efficiency_rest3.npz'")
    # results_df = pd.DataFrame({
    #     'rank_cirt': [rank_cirt],
    #     'rank_irt': [rank_irt]
    # })

    # results_df.to_parquet('efficiency_math500.parquet', index=False)
    # print("Results saved to 'efficiency_math500.parquet'")