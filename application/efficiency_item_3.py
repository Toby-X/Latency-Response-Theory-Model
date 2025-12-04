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

estimated_parameters = np.load("three_pred_params.npz")
a_joint = estimated_parameters['a_lart_train']
b_joint = estimated_parameters['b_lart_train']
omega_joint = estimated_parameters['omega_lart_train']
phi_joint = estimated_parameters['phi_lart_train']
lam_joint = estimated_parameters['lam_lart_train']
rho_joint = estimated_parameters['rho_lart']
a_irt = estimated_parameters['a_irt_train']
b_irt = estimated_parameters['b_irt_train']

from ..LaRT import get_indi_map_irt, get_indi_map_lart

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
    
    theta_init, tau_init = get_indi_map_lart(
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
            theta[i], tau[i] = get_indi_map_lart(
                np.atleast_1d(theta[i]), np.atleast_1d(tau[i]), 
                R[i, current_mask].reshape(1, -1), log_T[i, current_mask].reshape(1, -1), 
                a[current_mask], b[current_mask], 
                omega[current_mask], phi[current_mask], 
                lam[current_mask], rho
            )
        rank_each_step[:, step] = np.argsort(theta)
        theta_each_step[:, step] = theta

    return rank_each_step, theta_each_step

def initialize_theta(R, a, b, num_items=10):
    N, J = R.shape
    theta_init = np.zeros(N)
    
    theta_init = get_indi_map_irt(
        theta_init, R[:, :num_items],
        a[:num_items], b[:num_items]
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
            theta[i]= get_indi_map_irt(
                np.atleast_1d(theta[i]), R[i, current_mask].reshape(1, -1),
                a[current_mask], b[current_mask]
            )
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
        print("Starting LaRT and IRT model fitting in parallel...")

        # 1. Submit the first task (LaRT) to the pool
        LaRT_result_async = pool.apply_async(
            step_wise_evaluation_joint,
            args=(binary_array[~chosen_models_bool], cot_array[~chosen_models_bool], 
                  a_joint, b_joint, 
                  omega_joint, phi_joint, 
                  lam_joint, rho_joint),
            kwds={'num_items': 10}
        )

        # 2. Submit the second task (IRT) to the pool
        IRT_result_async = pool.apply_async(
            step_wise_evaluation_irt,
            args=(binary_array[~chosen_models_bool], 
                  a_irt, b_irt),
            kwds={'num_items': 10}
        )

        # 3. Wait for the results from both tasks
        print("Waiting for results...")
        rank_LaRT, theta_LaRT = LaRT_result_async.get()
        rank_IRT, theta_IRT = IRT_result_async.get()

    print("Model fitting complete. Saving results...")

    # 4. Save the estimated parameters into a single parquet file
    # The arrays are wrapped in lists to be stored correctly in a DataFrame
    np.savez_compressed(
    'efficiency_rest3.npz',  # Note the .npz extension
    rank_LaRT=rank_LaRT,
    rank_IRT=rank_IRT,
    theta_LaRT=theta_LaRT,
    theta_IRT=theta_IRT
    )

    print("Results saved to 'efficiency_item_parest3.npz'")