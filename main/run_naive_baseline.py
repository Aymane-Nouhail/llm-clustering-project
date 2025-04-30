import numpy as np
import os
import pandas as pd
# Import f1_score from sklearn.metrics
from sklearn.metrics import f1_score
# Import linear_sum_assignment (Hungarian algorithm) from scipy
from scipy.optimize import linear_sum_assignment as hungarian # Use the same name as in cluster_acc

# Import modules from src
from src.config import OPENAI_API_KEY, DATA_CACHE_PATH, DEFAULT_N_CLUSTERS
from src.data import load_dataset
from src.llm_service import LLMService # Needed to get the embedding model for load_dataset
from src.baselines import run_naive_kmeans

# Import evaluation utility (assuming it's from few_shot_clustering)
# We only need cluster_acc for the accuracy calculation; we'll replicate mapping logic
try:
    from few_shot_clustering.eval_utils import cluster_acc
except ImportError:
    print("Warning: few_shot_clustering.eval_utils not found. Accuracy calculation will not be possible using that utility.")
    cluster_acc = None

# Define the path for the metrics CSV file
METRICS_CSV_PATH = "clustering_metrics_results.csv"


def calculate_clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_clusters: int):
    """
    Calculates clustering metrics (Accuracy, Macro F1, Micro F1) by first
    finding the optimal mapping between predicted and true labels using the
    Hungarian algorithm, replicating the logic from cluster_acc.

    Args:
        y_true: True labels (NumPy array).
        y_pred: Predicted assignments (NumPy array).
        n_clusters: The expected number of clusters.

    Returns:
        A tuple containing (Accuracy, Macro F1, Micro F1, mapped_assignments),
        or (None, None, None, None) if calculation fails.
    """
    if len(y_pred) != len(y_true):
        print("Error: Predicted assignments and true labels must have the same length for metric calculation.")
        return None, None, None, None

    # Filter out any placeholder assignments (like -1 for failed documents) if they exist
    valid_indices = y_pred != -1
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    if len(y_pred_valid) == 0:
         print("Warning: No valid predicted assignments found for metric calculation.")
         # Return 0 for metrics if no valid predictions, or None? Let's use None if no data
         return None, None, None, None


    # Replicate contingency matrix building and Hungarian algorithm from cluster_acc
    # Ensure dimensions are based on unique labels/assignments in the *valid* subset
    true_labels_unique_valid = np.unique(y_true_valid)
    pred_assignments_unique_valid = np.unique(y_pred_valid)

    n_true_classes_valid = len(true_labels_unique_valid)
    n_pred_clusters_valid = len(pred_assignments_unique_valid)

    # Handle edge case where after filtering, one set is empty (shouldn't happen if len(y_pred_valid) > 0 but safety)
    if n_true_classes_valid == 0 or n_pred_clusters_valid == 0:
         return None, None, None, None

    # Create mapping from original label/assignment ID to matrix index (0 to N-1)
    true_label_to_matrix_idx = {label: i for i, label in enumerate(true_labels_unique_valid)}
    pred_assign_to_matrix_idx = {assign: i for i, assign in enumerate(pred_assignments_unique_valid)}

    w = np.zeros((n_pred_clusters_valid, n_true_classes_valid), dtype=np.int64)

    # Populate the contingency matrix
    for i in range(len(y_pred_valid)):
        pred_assign = y_pred_valid[i]
        true_label = y_true_valid[i]
        w[pred_assign_to_matrix_idx[pred_assign], true_label_to_matrix_idx[true_label]] += 1

    # Use Hungarian algorithm to find optimal mapping based on the valid subset contingency matrix
    # row_ind corresponds to predicted clusters (rows of w)
    # col_ind corresponds to true labels (columns of w)
    row_ind, col_ind = hungarian(w.max() - w)

    # Create mapping from matrix index back to original label/assignment ID
    matrix_idx_to_pred_assign = {i: assign for assign, i in pred_assign_to_matrix_idx.items()}
    matrix_idx_to_true_label = {i: label for label, i in true_label_to_matrix_idx.items()}

    # Create the mapped assignments array for the *entire* original dataset length
    mapped_assignments = np.full(y_pred.shape, -2, dtype=np.int64) # Use -2 for placeholder (distinct from -1 failed)

    # Apply the optimal mapping to the valid predicted assignments
    # The optimal mapping (row_ind, col_ind) refers to indices in the contingency matrix w
    # We need to map back to the original assignment/label IDs
    pred_assign_map = {matrix_idx_to_pred_assign[r]: matrix_idx_to_true_label[c] for r, c in zip(row_ind, col_ind)}

    # Apply the mapping to the original predicted assignments array (y_pred)
    for i in range(len(y_pred)):
        original_pred_assign = y_pred[i]
        if original_pred_assign != -1: # If it's not a failed document
             # Map the predicted assignment ID using the optimal mapping
             # If a predicted cluster ID doesn't appear in the optimal mapping (e.g. small clusters),
             # assign a default or leave as placeholder. Let's leave as placeholder (-2).
             mapped_assignments[i] = pred_assign_map.get(original_pred_assign, -2)


    # Calculate Accuracy using the standard utility (should match if mapping is correct)
    # Note: cluster_acc calculates accuracy on the full size, so we should too.
    # However, the mapping logic is derived from the *valid* subset.
    # The accuracy calculation using the *full* original y_true and the new mapped_assignments
    # will implicitly handle the -1 (failed) and -2 (unmapped) cases if they don't match y_true.
    accuracy = None
    if cluster_acc is not None:
         try:
             accuracy = cluster_acc(y_true, y_pred) # Calculate accuracy using the original utility on original y_pred
         except Exception as e:
             print(f"Warning: Error calculating accuracy using cluster_acc: {e}")


    # Calculate Macro and Micro F1 using the MAPPED assignments and the original true labels
    # Filter out placeholder values from mapped_assignments and corresponding true labels for F1 calculation?
    # No, F1 metrics typically expect the full set for macro/micro averages over classes.
    # We need to ensure the mapped_assignments has the same length and categories match y_true.
    # The mapping was derived from the *valid* subset of y_pred and y_true_valid.
    # F1 score needs predictions and true labels with the same set of possible values.
    # If mapped_assignments contains -2, but y_true doesn't, f1_score might error.

    # Let's calculate F1 on the *valid* subset after mapping, where labels correspond.
    # This gives metrics on the successfully clustered portion.
    mapped_assignments_valid = mapped_assignments[valid_indices] # mapped assignments for valid docs
    y_true_valid_subset = y_true[valid_indices] # Corresponding true labels

    macro_f1 = None
    micro_f1 = None

    if len(mapped_assignments_valid) > 0 and len(np.unique(y_true_valid_subset)) > 0:
        try:
            # Ensure the unique values in mapped_assignments_valid are actual true label IDs
            # (i.e., not -2 placeholders that survived indexing) and are present in y_true_valid_subset
            # If the mapping logic is correct, mapped_assignments_valid should only contain values from true_labels_unique_valid

            # Calculate F1 scores on the valid, mapped subset
            # Need to handle case where a true label might be entirely missing in y_true_valid_subset (e.g., if all docs of that label failed)
            macro_f1 = f1_score(y_true_valid_subset, mapped_assignments_valid, average='macro', zero_division=0)
            micro_f1 = f1_score(y_true_valid_subset, mapped_assignments_valid, average='micro', zero_division=0) # Should equal accuracy on the subset


        except Exception as e:
            print(f"Error calculating F1 scores on valid subset: {e}")

    return accuracy, macro_f1, micro_f1, mapped_assignments # Return original accuracy and subset F1s


def run_naive_baseline_experiment():
    print("\n--- Running Naive KMeans Baseline Experiment ---")

    # --- Configuration and Setup ---
    api_key = OPENAI_API_KEY
    if not api_key:
        print("OpenAI API Key not found.")
        return

    llm_service = LLMService(api_key)
    if not llm_service.is_available():
         print("LLM Service could not be initialized.")
         return

    embedding_model_instance = llm_service.get_embedding_model()
    if embedding_model_instance is None:
         print("Embedding model not available.")
         return

    # --- Load Data ---
    print("\nLoading data and embeddings...")
    features, labels, documents = load_dataset(cache_path=DATA_CACHE_PATH, embedding_model=embedding_model_instance)

    if features.size == 0 or labels.size == 0 or not documents:
        print("Data loading failed.")
        return

    labels_np = np.array(labels)
    n_clusters = len(np.unique(labels_np))
    print(f"Target number of clusters: {n_clusters}")

    # --- Run Naive Baseline ---
    print("\nRunning Naive KMeans...")
    naive_assignments = run_naive_kmeans(features, n_clusters)

    # --- Evaluate and Report ---
    naive_accuracy = None
    naive_macro_f1 = None
    naive_micro_f1 = None
    method_status = "Failed" # Default status

    if naive_assignments is not None:
        assignments_np = np.array(naive_assignments)

        # Calculate metrics using the helper function that handles mapping
        # Note: calculate_clustering_metrics assumes naive_assignments has the same length as labels
        accuracy, macro_f1, micro_f1, mapped_assignments = calculate_clustering_metrics(labels_np, assignments_np, n_clusters)

        naive_accuracy = accuracy
        naive_macro_f1 = macro_f1
        naive_micro_f1 = micro_f1

        if naive_accuracy is not None:
             print("\n--- Naive KMeans Baseline Results ---")
             print(f"Accuracy: {naive_accuracy}")
             # F1 scores are calculated on the valid subset by the helper
             if naive_macro_f1 is not None:
                 print(f"Macro F1 (on valid subset): {naive_macro_f1}")
             if naive_micro_f1 is not None:
                 print(f"Micro F1 (on valid subset): {naive_micro_f1}")

             method_status = "Success" if all(m is not None for m in [naive_accuracy, naive_macro_f1, naive_micro_f1]) else "Completed (Partial Eval)"


    else:
        print("\nNaive KMeans baseline failed.")


    # --- Save Metrics to CSV ---
    metrics_data = {
        'Method': 'Naive KMeans',
        'Status': method_status,
        'Accuracy': naive_accuracy,
        'Macro_F1': naive_macro_f1,
        'Micro_F1': naive_micro_f1
    }

    try:
        df_metrics = pd.DataFrame([metrics_data])
        if not os.path.exists(METRICS_CSV_PATH):
            df_metrics.to_csv(METRICS_CSV_PATH, index=False, mode='w', header=True)
            print(f"\nCreated {METRICS_CSV_PATH} and saved metrics.")
        else:
            df_metrics.to_csv(METRICS_CSV_PATH, index=False, mode='a', header=False)
            print(f"\nAppended metrics to {METRICS_CSV_PATH}.")

    except Exception as e:
        print(f"\nError saving metrics to CSV: {e}")


if __name__ == "__main__":
    run_naive_baseline_experiment()