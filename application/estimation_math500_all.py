import sys
import os
import numpy as np
import pandas as pd
import multiprocess as mp

# --- Data Loading and Preprocessing (Unchanged) ---
binary_df_math500 = pd.read_csv('correctness_matrix_math500.csv', index_col=0)
cot_df_math500 = pd.read_csv('cot_length_matrix_math500.csv', index_col=0)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cMIRT_EM_c import cMIRT_SAEM_full, MIRT_SAEM_full

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

# first_col = binary_df.columns[0]
# zero_rows = binary_df.iloc[:, 1:].eq(0).all(axis=1)
# non_zero_indices = ~zero_rows

# binary_df = binary_df[non_zero_indices].reset_index(drop=True)
# cot_df = cot_df[non_zero_indices].reset_index(drop=True)

# binary_array = binary_df.iloc[:, 1:].values
# cot_array = cot_df.iloc[:, 1:].values
# cot_array = cot_array.astype(np.float64)

# rows_with_zeros = np.any(cot_array == 0, axis=1)
# cot_array = cot_array[~rows_with_zeros].copy()
# binary_array = binary_array[~rows_with_zeros].copy()

N, J = binary_array.shape

# --- Main Execution Block for Parallel Processing ---
if __name__ == "__main__":
    # Create a pool with 2 processes to run both models at the same time
    with mp.Pool(processes=2) as pool:
        print("Starting cMIRT and MIRT model fitting in parallel...")

        # 1. Submit the first task (cMIRT) to the pool
        cmirt_result_async = pool.apply_async(
            cMIRT_SAEM_full,
            args=(binary_array, cot_array),
            kwds={'n_samples': 1, 'seed': 42}
        )

        # 2. Submit the second task (MIRT) to the pool
        mirt_result_async = pool.apply_async(
            MIRT_SAEM_full,
            args=(binary_array,),
            kwds={'n_samples': 1, 'seed': 42}
        )

        # 3. Wait for the results from both tasks
        print("Waiting for results...")
        theta_joint, tau_joint, a_joint, b_joint, omega_joint, phi_joint, lam_joint, rho_joint, n_iter_joint = cmirt_result_async.get()
        theta_irt, a_irt, b_irt, sigma2_irt, n_iter_irt = mirt_result_async.get()

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
        'sigma2_irt': [sigma2_irt],
        'n_iter_joint': n_iter_joint,
        'n_iter_irt': n_iter_irt
    })

    results_df.to_parquet('estimated_parameters_math500_all.parquet', index=False)
    print("Results saved to 'estimated_parameters_math500_all.parquet'")