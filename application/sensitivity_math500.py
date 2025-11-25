import sys
import os
import numpy as np
import pandas as pd
import multiprocess as mp
from functools import partial

# --- Data Loading and Preprocessing (Unchanged) ---
binary_df_math500 = pd.read_csv('correctness_matrix_math500.csv', index_col=0)
cot_df_math500 = pd.read_csv('cot_length_matrix_math500.csv', index_col=0)
question_list = np.load('question_list.npy')

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

binary_array = binary_array[:, question_list]
cot_array = cot_array[:, question_list]

np.random.seed(42)
shuffle_indices = np.random.permutation(binary_array.shape[0])

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
def run_model_task(task_tuple, binary_data, cot_data):
    """
    Runs a single model (cMIRT or MIRT) for a single N value.
    This function is designed to be parallelized by pool.map().
    """
    N, model_type = task_tuple
    print(f"Starting task: N={N}, model={model_type}")
    
    try:
        if model_type == 'cMIRT':
            # Run the cMIRT model
            (theta_joint, tau_joint, a_joint, b_joint, 
             omega_joint, phi_joint, lam_joint, rho_joint, 
             n_iter_joint) = cMIRT_SAEM_full(
                binary_data[:N, :], cot_data[:N, :],
                n_samples=1, seed=42
            )
            
            # Return results as a dictionary
            # Wrap arrays in a list for pyarrow compatibility
            return {
                'N': N, 'model_type': 'cMIRT', 'status': 'success',
                'theta_joint': [theta_joint], 'tau_joint': [tau_joint], 
                'a_joint': [a_joint], 'b_joint': [b_joint], 
                'omega_joint': [omega_joint], 'phi_joint': [phi_joint], 
                'lam_joint': [lam_joint], 'rho_joint': [rho_joint], 
                'n_iter_joint': n_iter_joint
            }

        elif model_type == 'MIRT':
            # Run the MIRT model
            (theta_irt, a_irt, b_irt, 
             sigma2_irt, n_iter_irt) = MIRT_SAEM_full(
                binary_data[:N, :],
                n_samples=1, seed=42
            )
            
            # Return results as a dictionary
            return {
                'N': N, 'model_type': 'MIRT', 'status': 'success',
                'theta_irt': [theta_irt], 'a_irt': [a_irt], 
                'b_irt': [b_irt], 'sigma2_irt': [sigma2_irt], 
                'n_iter_irt': n_iter_irt
            }
    except Exception as e:
        print(f"ERROR on N={N}, model={model_type}: {e}")
        return {'N': N, 'model_type': model_type, 'status': 'error', 'error': str(e)}


if __name__ == "__main__":
    # (Load your data here)
    # binary_array = pd.read_csv(...).to_numpy()
    # cot_array = pd.read_csv(...).to_numpy()
    
    N_values = [25, 50, 75, 100, 125]
    model_types = ['cMIRT', 'MIRT']
    n_cores = int(os.getenv('SLURM_CPUS_PER_TASK', 8))

    # 1. Create the full list of tasks
    # This will be [(25, 'cMIRT'), (25, 'MIRT'), (50, 'cMIRT'), ...]
    tasks_list = [(N, model) for N in N_values for model in model_types]

    # 2. Create a "partial" worker function
    # This "bakes in" the large data arrays so we don't need to pass them 
    # in every task tuple. They will be inherited by the child processes.
    worker_fn = partial(run_model_task, binary_data=binary_array, cot_data=cot_array)
    
    # 3. Run all tasks in parallel
    print(f"Starting parallel pool with {n_cores} cores for {len(tasks_list)} tasks.")
    
    all_results = []
    with mp.Pool(processes=n_cores) as pool:
        # pool.map applies the worker_fn to each item in tasks_list
        all_results = pool.map(worker_fn, tasks_list)

    print("All tasks complete. Processing results...")

    # 4. Process the list of dictionaries into a single DataFrame
    # This merges the 'cMIRT' and 'MIRT' results into a single row for each N
    merged_results = {}
    for res in all_results:
        if res['status'] == 'success':
            N = res['N']
            if N not in merged_results:
                merged_results[N] = {'N': N} # Initialize a dict for this N
            
            # Update the dict, merging keys from cMIRT and MIRT
            merged_results[N].update(res) 
            
    # Convert the merged dictionaries into a DataFrame
    results_df = pd.DataFrame(list(merged_results.values()))

    # Clean up helper columns
    results_df = results_df.drop(columns=['model_type', 'status'], errors='ignore')

    # 5. Save the final file
    results_df.to_parquet('sensitivity_math500.parquet', index=False)
    print("Results saved to 'sensitivity_math500.parquet'")