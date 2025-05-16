import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    normalized_mutual_info_score, adjusted_rand_score
)
from itertools import combinations
from few_shot_clustering.eval_utils import cluster_acc

def calculate_clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_clusters: int) -> dict:
    """
    Calculates clustering metrics with identical functionality and return format as original.
    Returns dictionary with:
    {
        "Accuracy", "Precision", "Recall", "Macro_F1", "Micro_F1",
        "NMI", "ARI", "Pairwise_Precision", "Pairwise_Recall", "Pairwise_F1"
    }
    """
    # Input validation
    if len(y_pred) != len(y_true):
        print("Error: Predicted assignments and true labels must have the same length.")
        return _create_empty_metrics_dict()

    # Filter out invalid predictions (-1)
    valid_indices = y_pred != -1
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    if len(y_pred_valid) == 0:
        print("Warning: No valid predicted assignments found.")
        return _create_empty_metrics_dict()

    # Get unique labels in valid subset
    true_labels_unique = np.unique(y_true_valid)
    pred_assignments_unique = np.unique(y_pred_valid)
    
    if len(true_labels_unique) == 0 or len(pred_assignments_unique) == 0:
        return _create_empty_metrics_dict()

    # Build contingency matrix
    true_to_idx = {label: i for i, label in enumerate(true_labels_unique)}
    pred_to_idx = {assign: i for i, assign in enumerate(pred_assignments_unique)}
    
    w = np.zeros((len(pred_assignments_unique), len(true_labels_unique)), dtype=np.int64)
    for true, pred in zip(y_true_valid, y_pred_valid):
        w[pred_to_idx[pred], true_to_idx[true]] += 1

    # Hungarian algorithm for optimal mapping
    row_ind, col_ind = hungarian(w.max() - w)
    idx_to_pred = {i: assign for assign, i in pred_to_idx.items()}
    idx_to_true = {i: label for label, i in true_to_idx.items()}
    
    # Create mapped assignments
    pred_assign_map = {idx_to_pred[r]: idx_to_true[c] for r, c in zip(row_ind, col_ind)}
    mapped_assignments = np.full(y_pred.shape, -2, dtype=np.int64)
    
    for i in range(len(y_pred)):
        if y_pred[i] != -1:
            mapped_assignments[i] = pred_assign_map.get(y_pred[i], -2)

    # Calculate accuracy using cluster_acc if available
    accuracy = None
    if cluster_acc is not None:
        try:
            accuracy = cluster_acc(y_true, y_pred)
        except Exception as e:
            print(f"Warning: Error calculating accuracy: {e}")

    # Calculate metrics on valid mapped subset
    mapped_valid = mapped_assignments[valid_indices]
    y_true_valid_subset = y_true[valid_indices]
    
    # Initialize metrics
    metrics = _create_empty_metrics_dict()
    
    # Calculate NMI and ARI on original valid predictions
    try:
        metrics["NMI"] = normalized_mutual_info_score(y_true_valid, y_pred_valid)
        metrics["ARI"] = adjusted_rand_score(y_true_valid, y_pred_valid)
    except Exception as e:
        print(f"Error calculating NMI/ARI: {e}")

    # Calculate pairwise metrics
    try:
        (metrics["Pairwise_Precision"], 
         metrics["Pairwise_Recall"], 
         metrics["Pairwise_F1"]) = _calculate_pairwise_metrics(y_true_valid, y_pred_valid)
    except Exception as e:
        print(f"Error calculating pairwise metrics: {e}")

    # Calculate other metrics on mapped subset
    if len(mapped_valid) > 0 and len(np.unique(y_true_valid_subset)) > 0:
        try:
            metrics.update({
                "Accuracy": accuracy,
                "Precision": precision_score(y_true_valid_subset, mapped_valid, average='macro', zero_division=0),
                "Recall": recall_score(y_true_valid_subset, mapped_valid, average='macro', zero_division=0),
                "Macro_F1": f1_score(y_true_valid_subset, mapped_valid, average='macro', zero_division=0),
                "Micro_F1": f1_score(y_true_valid_subset, mapped_valid, average='micro', zero_division=0)
            })
        except Exception as e:
            print(f"Error calculating metrics: {e}")

    return metrics

def _create_empty_metrics_dict() -> dict:
    """Returns dictionary with all metrics set to None"""
    return {
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "Macro_F1": None,
        "Micro_F1": None,
        "NMI": None,
        "ARI": None,
        "Pairwise_Precision": None,
        "Pairwise_Recall": None,
        "Pairwise_F1": None
    }

def _calculate_pairwise_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Calculates pairwise precision, recall and F1"""
    n = len(y_true)
    true_pairs = set()
    pred_pairs = set()

    for i, j in combinations(range(n), 2):
        if y_true[i] == y_true[j]:
            true_pairs.add((i, j))
        if y_pred[i] == y_pred[j]:
            pred_pairs.add((i, j))

    tp = len(true_pairs & pred_pairs)
    pred_pos = len(pred_pairs)
    true_pos = len(true_pairs)

    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / true_pos if true_pos > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1