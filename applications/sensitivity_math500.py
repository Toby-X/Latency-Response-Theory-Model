import os
import numpy as np
import pandas as pd
import multiprocess as mp
from functools import partial
from pathlib import Path
from scipy.stats import norm

from lart import irt_saem_full, lart_saem_full

# --- Data Loading and Preprocessing (Unchanged) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data' / 'benchmarks'
OUTPUT_DIR = REPO_ROOT / 'results' / 'applications'
binary_df_math500 = pd.read_csv(DATA_DIR / 'correctness_matrix_math500.csv', index_col=0)
cot_df_math500 = pd.read_csv(DATA_DIR / 'cot_length_matrix_math500.csv', index_col=0)

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

np.random.seed(42)
shuffle_indices = np.random.permutation(binary_array.shape[0])
binary_array = binary_array[shuffle_indices]
cot_array = cot_array[shuffle_indices]

N, J = binary_array.shape


def item_fisher_information(a, b, theta):
    """Normal-ogive Fisher information for every LLM-item pair."""
    value = a[np.newaxis, :] * theta[:, np.newaxis] + b[np.newaxis, :]
    log_information = (
        2 * norm.logpdf(value)
        + 2 * np.log(np.maximum(a, 1e-12))[np.newaxis, :]
        - norm.logcdf(value)
        - norm.logcdf(-value)
    )
    return np.exp(log_information)


def select_questions(response, latency, num_questions=100, max_iter=100, seed=42):
    """Generate the MATH500 item set used in Section 7.2.4 from full-data fits."""
    lart = lart_saem_full(
        response, latency, n_samples=1, seed=seed, max_iter=max_iter
    )
    irt = irt_saem_full(response, n_samples=1, seed=seed, max_iter=max_iter)
    lart_order = np.argsort(
        -np.sum(item_fisher_information(lart[2], lart[3], lart[0]), axis=0)
    )
    irt_order = np.argsort(
        -np.sum(item_fisher_information(irt[1], irt[2], irt[0]), axis=0)
    )
    combined = np.unique(
        np.concatenate([lart_order[:num_questions], irt_order[:num_questions]])
    )
    if combined.size < num_questions:
        raise RuntimeError("the combined information ranking contains too few unique items")
    return combined[:num_questions]

# --- Main Execution Block for Parallel Processing ---
def run_model_task(task_tuple, binary_data, cot_data, max_iter=100, seed=42):
    """
    Runs a single model (LaRT or IRT) for a single N value.
    This function is designed to be parallelized by pool.map().
    """
    N, model_type = task_tuple
    print(f"Starting task: N={N}, model={model_type}")

    try:
        if model_type == 'LaRT':
            # Run the LaRT model
            (theta_joint, tau_joint, a_joint, b_joint,
             omega_joint, phi_joint, lam_joint, rho_joint,
             n_iter_joint) = lart_saem_full(
                binary_data[:N, :], cot_data[:N, :],
                n_samples=1, seed=seed, max_iter=max_iter
            )

            # Return results as a dictionary
            # Wrap arrays in a list for pyarrow compatibility
            return {
                'N': N, 'model_type': 'LaRT', 'status': 'success',
                'theta_joint': [theta_joint], 'tau_joint': [tau_joint],
                'a_joint': [a_joint], 'b_joint': [b_joint],
                'omega_joint': [omega_joint], 'phi_joint': [phi_joint],
                'lam_joint': [lam_joint], 'rho_joint': [rho_joint],
                'n_iter_joint': n_iter_joint
            }

        elif model_type == 'IRT':
            # Run the IRT model
            (theta_irt, a_irt, b_irt,
             sigma2_irt, n_iter_irt) = irt_saem_full(
                binary_data[:N, :],
                n_samples=1, seed=seed, max_iter=max_iter
            )

            # Return results as a dictionary
            return {
                'N': N, 'model_type': 'IRT', 'status': 'success',
                'theta_irt': [theta_irt], 'a_irt': [a_irt],
                'b_irt': [b_irt], 'sigma2_irt': [sigma2_irt],
                'n_iter_irt': n_iter_irt
            }
    except Exception as e:
        print(f"ERROR on N={N}, model={model_type}: {e}")
        return {'N': N, 'model_type': model_type, 'status': 'error', 'error': str(e)}


if __name__ == "__main__":
    question_list = select_questions(binary_array, cot_array)
    selected_binary = binary_array[:, question_list]
    selected_cot = cot_array[:, question_list]

    # The full N=140 fit is the ground truth; Table 2 reports the four subsets.
    N_values = [50, 75, 100, 125]
    model_types = ['LaRT', 'IRT']
    n_cores = int(os.getenv('SLURM_CPUS_PER_TASK', 8))

    # 1. Create the full list of tasks
    # This will be [(50, 'LaRT'), (50, 'IRT'), (75, 'LaRT'), ...]
    tasks_list = [(N, model) for N in N_values for model in model_types]

    # 2. Create a "partial" worker function
    # This "bakes in" the large data arrays so we don't need to pass them
    # in every task tuple. They will be inherited by the child processes.
    worker_fn = partial(
        run_model_task, binary_data=selected_binary, cot_data=selected_cot
    )

    # 3. Run all tasks in parallel
    print(f"Starting parallel pool with {n_cores} cores for {len(tasks_list)} tasks.")

    all_results = []
    with mp.Pool(processes=n_cores) as pool:
        # pool.map applies the worker_fn to each item in tasks_list
        all_results = pool.map(worker_fn, tasks_list)

    print("All tasks complete. Processing results...")

    # 4. Process the list of dictionaries into a single DataFrame
    # This merges the 'LaRT' and 'IRT' results into a single row for each N
    merged_results = {}
    for res in all_results:
        if res['status'] == 'success':
            N = res['N']
            if N not in merged_results:
                merged_results[N] = {'N': N} # Initialize a dict for this N

            # Update the dict, merging keys from LaRT and IRT
            merged_results[N].update(res)

    # Convert the merged dictionaries into a DataFrame
    results_df = pd.DataFrame(list(merged_results.values()))

    # Clean up helper columns
    results_df = results_df.drop(columns=['model_type', 'status'], errors='ignore')

    # 5. Save the final file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / 'question_list.npy', question_list)
    output_path = OUTPUT_DIR / 'sensitivity_math500.parquet'
    results_df.to_parquet(output_path, index=False)
    print(f"Results saved to '{output_path}'")
