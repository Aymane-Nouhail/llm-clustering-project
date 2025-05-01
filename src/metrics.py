import numpy as np
from few_shot_clustering.eval_utils import cluster_acc
from scipy.optimize import linear_sum_assignment as hungarian # Use the same name as in cluster_acc
from sklearn.metrics import f1_score, precision_score, recall_score, normalized_mutual_info_score, adjusted_rand_score


def calculate_clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_clusters: int):
    """
    Calculates clustering metrics (Accuracy, Precision, Recall, F1 scores, NMI, ARI) by first
    finding the optimal mapping between predicted and true labels using the
    Hungarian algorithm, replicating the logic from cluster_acc.
    Args:
        y_true: True labels (NumPy array).
        y_pred: Predicted assignments (NumPy array).
        n_clusters: The expected number of clusters.
    Returns:
        A tuple containing (Accuracy, Precision, Recall, Macro F1, Micro F1, NMI, ARI, mapped_assignments),
        or (None, None, None, None, None, None, None, None) if calculation fails.
    """
    if len(y_pred) != len(y_true):
        print("Error: Predicted assignments and true labels must have the same length for metric calculation.")
        return None, None, None, None, None, None, None, None
    # Filter out any placeholder assignments (like -1 for failed documents) if they exist
    valid_indices = y_pred != -1
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    if len(y_pred_valid) == 0:
         print("Warning: No valid predicted assignments found for metric calculation.")
         # Return 0 for metrics if no valid predictions, or None? Let's use None if no data
         return None, None, None, None, None, None, None, None
    # Replicate contingency matrix building and Hungarian algorithm from cluster_acc
    # Ensure dimensions are based on unique labels/assignments in the *valid* subset
    true_labels_unique_valid = np.unique(y_true_valid)
    pred_assignments_unique_valid = np.unique(y_pred_valid)
    n_true_classes_valid = len(true_labels_unique_valid)
    n_pred_clusters_valid = len(pred_assignments_unique_valid)
    # Handle edge case where after filtering, one set is empty (shouldn't happen if len(y_pred_valid) > 0 but safety)
    if n_true_classes_valid == 0 or n_pred_clusters_valid == 0:
         return None, None, None, None, None, None, None, None
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
    # Calculate metrics using the MAPPED assignments and the original true labels
    # Filter out placeholder values from mapped_assignments and corresponding true labels for metric calculation
    # F1 score needs predictions and true labels with the same set of possible values.
    # Let's calculate metrics on the *valid* subset after mapping, where labels correspond.
    # This gives metrics on the successfully clustered portion.
    mapped_assignments_valid = mapped_assignments[valid_indices] # mapped assignments for valid docs
    y_true_valid_subset = y_true[valid_indices] # Corresponding true labels
    precision = None
    recall = None
    macro_f1 = None
    micro_f1 = None
    nmi = None
    ari = None
    
    # Calculate NMI and ARI directly on original predictions (before mapping)
    # These metrics are invariant to permutation of labels, so they don't need the mapping
    try:
        # Filter out invalid predictions for NMI and ARI calculation
        nmi = normalized_mutual_info_score(y_true_valid, y_pred_valid)
        ari = adjusted_rand_score(y_true_valid, y_pred_valid)
    except Exception as e:
        print(f"Error calculating NMI or ARI: {e}")
    
    if len(mapped_assignments_valid) > 0 and len(np.unique(y_true_valid_subset)) > 0:
        try:
            # Calculate precision and recall scores on the valid, mapped subset
            precision = precision_score(y_true_valid_subset, mapped_assignments_valid, average='macro', zero_division=0)
            recall = recall_score(y_true_valid_subset, mapped_assignments_valid, average='macro', zero_division=0)
            
            # Calculate F1 scores on the valid, mapped subset
            macro_f1 = f1_score(y_true_valid_subset, mapped_assignments_valid, average='macro', zero_division=0)
            micro_f1 = f1_score(y_true_valid_subset, mapped_assignments_valid, average='micro', zero_division=0) # Should equal accuracy on the subset
        except Exception as e:
            print(f"Error calculating metrics on valid subset: {e}")
    
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Macro_F1": macro_f1,
            "Micro_F1": micro_f1,
            "NMI": nmi,
            "ARI": ari
        }