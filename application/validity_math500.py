import sys
import os
import numpy as np
import pandas as pd
import multiprocess as mp
from tqdm import tqdm
import itertools

binary_df_math500 = pd.read_csv('../data/accuracy_math500.csv', index_col=0)
cot_df_math500 = pd.read_csv('../data/cot_length_math500.csv', index_col=0)

from ..LaRT import LaRT_SAEM_full, IRT_SAEM_full

rows_to_delete = [
    "meta_llama_Llama_3.2_1B_one_shot",
    "meta_llama_Llama_3.2_1B_zero_shot",
    "meta_llama_Meta_Llama_3_8B_one_shot",
    "meta_llama_Meta_Llama_3_8B_zero_shot",
    "microsoft_phi_4_one_shot",
    "microsoft_phi_4_zero_shot",
    'TinyLlama_TinyLlama_1.1B_Chat_v1.0_zero_shot',
    'TinyLlama_TinyLlama_1.1B_Chat_v1.0_one_shot',
    'google_gemma_3_1b_pt_one_shot',
    'google_gemma_3_1b_pt_zero_shot',
    'google_gemma_7b_it_one_shot', 
    'google_gemma_7b_it_zero_shot',
    'google_vaultgemma_1b_one_shot', 
    'google_vaultgemma_1b_zero_shot',
    'meta_llama_Llama_3.2_3B_one_shot', 
    'meta_llama_Llama_3.2_3B_zero_shot',
    'openai_community_gpt2_one_shot', 
    'openai_community_gpt2_zero_shot'
]

binary_df_math500 = binary_df_math500.drop(rows_to_delete, errors='ignore')
cot_df_math500 = cot_df_math500.drop(rows_to_delete, errors='ignore')

binary_array = binary_df_math500.to_numpy()
cot_array = cot_df_math500.to_numpy()
cot_array += 1

N, J = binary_array.shape
num_subarrays = 5

# --- Create Subarrays with Random Columns (Unchanged) ---
all_column_indices = np.arange(J)
np.random.seed(42)
np.random.shuffle(all_column_indices)
random_index_chunks = np.array_split(all_column_indices, num_subarrays)
binary_subarrays = [binary_array[:, indices] for indices in random_index_chunks]
cot_subarrays = [cot_array[:, indices] for indices in random_index_chunks]


# --- 1. Define a More Generic Worker Function ---
# This worker now takes a tuple: (model_type, subarray_index)
def run_model(task_info):
    """Runs a single specified model on a single specified subarray."""
    model_type, i = task_info
    print(f"Starting task: {model_type} on subarray {i+1}/{num_subarrays}...")

    if model_type == 'LaRT':
        theta, tau, a, b, omega, phi, lam, rho, n_iter = LaRT_SAEM_full(
            binary_subarrays[i], cot_subarrays[i], n_samples=1, seed=42
        )
        return {
            'model_type': 'LaRT', 'subarray': i, 'theta_joint': theta, 'tau_joint': tau, 
            'a_joint': a, 'b_joint': b, 'omega_joint': omega, 'phi_joint': phi, 
            'lam_joint': lam, 'rho_joint': rho, 'n_iter_joint': n_iter
        }
    elif model_type == 'IRT':
        theta, a, b, n_iter = IRT_SAEM_full(
            binary_subarrays[i], n_samples=1, seed=42
        )
        return {
            'model_type': 'IRT', 'subarray': i, 'theta_irt': theta, 'a_irt': a, 
            'b_irt': b, 'n_iter_irt': n_iter
        }

# --- 2. Main Execution Block ---
if __name__ == "__main__":
    # Create a flat list of all 10 tasks to run
    model_types = ['LaRT', 'IRT']
    tasks = list(itertools.product(model_types, range(num_subarrays)))
    
    # We can use more processes now, up to the number of tasks or CPU cores
    # For example, min(len(tasks), mp.cpu_count())
    n_cores = int(os.getenv('SLURM_CPUS_PER_TASK', 8)) # Let's use 8 or the number of tasks if fewer

    with mp.Pool(processes=n_cores) as pool:
        results = list(tqdm(pool.imap(run_model, tasks), total=len(tasks)))

    # --- 3. Collect and Re-organize Results ---
    joint_results = []
    irt_results = []
    for res in results:
        if res['model_type'] == 'LaRT':
            joint_results.append(res)
        elif res['model_type'] == 'IRT':
            irt_results.append(res)

    # Sort results by subarray index to maintain order
    joint_results.sort(key=lambda x: x['subarray'])
    irt_results.sort(key=lambda x: x['subarray'])

    joint_results_df = pd.DataFrame(joint_results)
    irt_results_df = pd.DataFrame(irt_results)
    
    print("\nSaving joint model results to parquet...")
    joint_results_df.to_parquet('estimated_parameters_joint_validity_math500.parquet', index=False)
    
    print("Saving IRT model results to parquet...")
    irt_results_df.to_parquet('estimated_parameters_irt_validity_math500.parquet', index=False)

    print("Done!")