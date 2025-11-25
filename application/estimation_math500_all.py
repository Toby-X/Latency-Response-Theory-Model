import sys
import os
import numpy as np
import pandas as pd
import multiprocess as mp

# --- Data Loading and Preprocessing (Unchanged) ---
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
cot_array += 1  # To avoid log(0)

N, J = binary_array.shape

# --- Main Execution Block for Parallel Processing ---
if __name__ == "__main__":
    # Create a pool with 2 processes to run both models at the same time
    with mp.Pool(processes=2) as pool:
        print("Starting LaRT and IRT model fitting in parallel...")

        # 1. Submit the first task (LaRT) to the pool
        LaRT_result_async = pool.apply_async(
            LaRT_SAEM_full,
            args=(binary_array, cot_array),
            kwds={'n_samples': 1, 'seed': 42}
        )

        # 2. Submit the second task (IRT) to the pool
        IRT_result_async = pool.apply_async(
            IRT_SAEM_full,
            args=(binary_array,),
            kwds={'n_samples': 1, 'seed': 42}
        )

        # 3. Wait for the results from both tasks
        print("Waiting for results...")
        theta_joint, tau_joint, a_joint, b_joint, omega_joint, phi_joint, lam_joint, rho_joint, n_iter_joint = LaRT_result_async.get()
        theta_irt, a_irt, b_irt, n_iter_irt = IRT_result_async.get()

    print("Model fitting complete. Saving results...")

    # 4. Save the estimated parameters into a single parquet file
    # The arrays are wrapped in lists to be stored correctly in a DataFrame
    results_df = pd.DataFrame({
        'theta_joint': [theta_joint],
        'tau_joint': [tau_joint],
        'a_joint': [a_joint],
        'b_joint': [b_joint],
        'omega_joint': [omega_joint],
        'phi_joint': [phi_joint],
        'lam_joint': [lam_joint],
        'rho_joint': [rho_joint],
        'theta_irt': [theta_irt],
        'a_irt': [a_irt],
        'b_irt': [b_irt],
        'n_iter_joint': n_iter_joint,
        'n_iter_irt': n_iter_irt
    })

    results_df.to_parquet('estimated_parameters_math500_all.parquet', index=False)
    print("Results saved to 'estimated_parameters_math500_all.parquet'")