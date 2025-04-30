import numpy as np
import random
import os # Import os for path manipulation
import pandas as pd # Import pandas for CSV handling
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score # Import f1_score for evaluation
from scipy.optimize import linear_sum_assignment as hungarian # Import Hungarian algorithm
from langchain_core.prompts import ChatPromptTemplate
from src.llm_service import LLMService # Import LLM service
from typing import List, Tuple, Any # Import necessary types for type hinting
import time
# Import PCKMeans from the correct package and path
try:
    from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
    print("PCKMeans from 'active-semi-clustering' imported successfully.")
except ImportError:
    print("PCKMeans from 'active-semi-clustering' not found.")
    print("The pairwise constraints method requires this library to run.")
    PCKMeans = None # Define as None if not available

# Define the path for the metrics CSV file (use the same path as other run scripts)
METRICS_CSV_PATH = "clustering_metrics_results.csv"


# Replicate the metrics calculation helper function
def calculate_clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculates clustering metrics (Accuracy, Macro F1, Micro F1) by first
    finding the optimal mapping between predicted and true labels using the
    Hungarian algorithm, replicating the logic from cluster_acc.

    Args:
        y_true: True labels (NumPy array).
        y_pred: Predicted assignments (NumPy array).

    Returns:
        A tuple containing (Accuracy, Macro F1, Micro F1),
        or (None, None, None) if calculation fails.
    """
    if len(y_pred) != len(y_true):
        print("Error: Predicted assignments and true labels must have the same length for metric calculation.")
        return None, None, None

    # Filter out any placeholder assignments (like -1 or -2 for failed documents) if they exist
    valid_indices = y_pred >= 0
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    if len(y_pred_valid) == 0:
         print("Warning: No valid predicted assignments found for metric calculation.")
         return None, None, None

    # Replicate contingency matrix building and Hungarian algorithm from cluster_acc
    true_labels_unique_valid = np.unique(y_true_valid)
    pred_assignments_unique_valid = np.unique(y_pred_valid)

    n_true_classes_valid = len(true_labels_unique_valid)
    n_pred_clusters_valid = len(pred_assignments_unique_valid)

    if n_true_classes_valid == 0 or n_pred_clusters_valid == 0:
         return None, None, None


    true_label_to_matrix_idx = {label: i for i, label in enumerate(true_labels_unique_valid)}
    pred_assign_to_matrix_idx = {assign: i for i, assign in enumerate(pred_assignments_unique_valid)}

    w = np.zeros((n_pred_clusters_valid, n_true_classes_valid), dtype=np.int64)

    for i in range(len(y_pred_valid)):
        pred_assign = y_pred_valid[i]
        true_label = y_true_valid[i]
        w[pred_assign_to_matrix_idx[pred_assign], true_label_to_matrix_idx[true_label]] += 1

    try:
         row_ind, col_ind = hungarian(w.max() - w)
    except Exception as e:
         print(f"Error running Hungarian algorithm: {e}")
         return None, None, None

    # Create mapping from matrix index back to original label/assignment ID
    matrix_idx_to_pred_assign = {i: assign for assign, i in pred_assign_to_matrix_idx.items()}
    matrix_idx_to_true_label = {i: label for label, i in true_label_to_matrix_idx.items()}

    # Create the mapped assignments array for the *valid* subset
    mapped_assignments_valid = np.full(len(y_pred_valid), -2, dtype=np.int64) # Use -2 for placeholder

    # Apply the optimal mapping to the valid predicted assignments
    pred_assign_map = {matrix_idx_to_pred_assign[r]: matrix_idx_to_true_label[c] for r, c in zip(row_ind, col_ind)}

    for i in range(len(y_pred_valid)):
        original_pred_assign = y_pred_valid[i]
        mapped_assignments_valid[i] = pred_assign_map.get(original_pred_assign, -2)


    # Calculate Accuracy on the full dataset vs true labels
    # We need the full mapped_assignments array for this.
    full_mapped_assignments = np.full(y_pred.shape, -2, dtype=np.int64)
    full_mapped_assignments[valid_indices] = mapped_assignments_valid # Place mapped valid assignments

    correct_count = np.sum(full_mapped_assignments == y_true)
    accuracy = correct_count * 1.0 / y_true.size


    # Calculate Macro and Micro F1 using the MAPPED assignments VALID subset
    macro_f1 = None
    micro_f1 = None

    if len(mapped_assignments_valid) > 0 and len(np.unique(y_true_valid)) > 0:
        try:
            macro_f1 = f1_score(y_true_valid, mapped_assignments_valid, average='macro', zero_division=0)
            micro_f1 = f1_score(y_true_valid, mapped_assignments_valid, average='micro', zero_division=0)

        except Exception as e:
            print(f"Error calculating F1 scores on valid subset: {e}")

    return accuracy, macro_f1, micro_f1


# The main function for pairwise constraints clustering
# Added labels_np as an argument, and a path for query output CSV
def cluster_via_pairwise_constraints(
    documents: List[str],
    features: np.ndarray,
    labels_np: np.ndarray, # Added true labels as argument
    n_clusters: int,
    llm_service: LLMService | None, # Allow llm_service to be None
    pairwise_prompt_template: str,
    num_pairs_to_query: int,
    constraint_selection_strategy: str = 'random',
    pairwise_queries_output_csv_path: str = "pairwise_queries_output.csv" # Add path for saving queries
) -> np.ndarray | None: # Returns assignments for the full dataset
    """
    Implements clustering via pseudo-oracle pairwise constraints (Section 2.2).
    Queries LLM for constraints, runs PCKMeans, calculates and saves metrics to CSV,
    and saves LLM queries/responses to a separate CSV.

    Args:
        documents: List of original document texts.
        features: Original document embeddings (NumPy array).
        labels_np: True labels (NumPy array) for evaluation.
        n_clusters: The number of clusters.
        llm_service: An initialized LLMService instance (can be None).
        pairwise_prompt_template: A string template for pairwise comparison.
                                  Example: "Are these two texts similar? Text 1: {text1}\nText 2: {text2}\nRespond YES or NO."
        num_pairs_to_query: The number of pairs to query the LLM for constraints.
        constraint_selection_strategy: Strategy for selecting pairs ('random', 'similarity').
        pairwise_queries_output_csv_path: Path to save the generated queries/responses CSV.

    Returns:
        A NumPy array of cluster assignments for the full original dataset length,
        or None if PCKMeans is not available or clustering fails.
    """
    print("\n--- Running Clustering via Pseudo-Oracle Pairwise Constraints ---")
    method_status = "Failed" # Default status
    assignments = None # Initialize assignments

    # --- Initial Checks ---
    # Check for PCKMeans availability immediately
    if PCKMeans is None:
        print("Cannot run Method 2: PCKMeans implementation is not available.")
        method_status = "Skipped (PCKMeans Missing)"
    # Check if LLMService instance is provided and available
    elif llm_service is None or not llm_service.is_available():
        print("LLMService is not available. Cannot run pairwise constraints.")
        method_status = "Failed (LLM Service Missing)"
    # Check if essential data is provided (documents, features, labels)
    elif len(documents) == 0 or features.size == 0 or len(labels_np) == 0 or len(documents) != features.shape[0] or len(documents) != len(labels_np):
        print("Input data (documents, features, or labels) is missing or inconsistent.")
        method_status = "Failed (Invalid Input Data)"
    else:
        # --- Proceed with Method Logic if all checks pass ---
        n_samples = len(documents)
        must_link_constraints = []
        cannot_link_constraints = []

        # List to store query data for CSV
        pairwise_query_data = []

        # Define Langchain prompt template
        prompt_template = ChatPromptTemplate.from_template(pairwise_prompt_template)

        # 1. Select pairs based on strategy
        selected_pairs = []
        all_possible_pairs = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]

        # Ensure num_pairs_to_query does not exceed the number of available pairs
        num_pairs_to_query = min(num_pairs_to_query, len(all_possible_pairs))

        if num_pairs_to_query > 0:
            if constraint_selection_strategy == 'random':
                 selected_pairs = random.sample(all_possible_pairs, num_pairs_to_query)
            elif constraint_selection_strategy == 'similarity':
                 # Simple similarity-based selection: get some most similar and some most dissimilar pairs
                 similarity_matrix = cosine_similarity(features)
                 similarity_pairs = []
                 for i in range(n_samples):
                     for j in range(i + 1, n_samples):
                         similarity_pairs.append(((i, j), similarity_matrix[i, j]))

                 similarity_pairs.sort(key=lambda item: item[1], reverse=True) # Sort by similarity descending

                 num_similar = num_pairs_to_query // 2
                 num_dissimilar = num_pairs_to_query - num_similar

                 selected_pairs.extend([pair for pair, sim in similarity_pairs[:num_similar]])
                 selected_pairs.extend([pair for pair, sim in similarity_pairs[-num_dissimilar:]])

                 selected_pairs = list(set(selected_pairs))
                 # If after taking top/bottom similarity, we still don't have enough unique pairs
                 while len(selected_pairs) < num_pairs_to_query and len(selected_pairs) < len(all_possible_pairs):
                      remaining_pairs = list(set(all_possible_pairs) - set(selected_pairs))
                      selected_pairs.extend(random.sample(remaining_pairs, min(num_pairs_to_query - len(selected_pairs), len(remaining_pairs))))

                 selected_pairs = selected_pairs[:num_pairs_to_query] # Ensure we don't exceed the requested number

            else:
                print(f"Unknown constraint selection strategy: {constraint_selection_strategy}. Using 'random'.")
                selected_pairs = random.sample(all_possible_pairs, num_pairs_to_query)
        else:
            print("Num pairs to query is 0 or less. No constraints will be generated.")


        print(f"Selected {len(selected_pairs)} pairs using '{constraint_selection_strategy}' strategy.")

        # 2. Query LLM for constraints for selected pairs
        if selected_pairs: # Only query if there are pairs to query
            print(f"Querying LLM for {len(selected_pairs)} pairs...")
            start_time = time.time() # Start timing LLM calls
            for i, (idx1, idx2) in enumerate(selected_pairs):
                # Print a simple progress counter
                if (i + 1) % 50 == 0 or (i + 1) == len(selected_pairs):
                     print(f"  Queried {i + 1}/{len(selected_pairs)} pairs...")

                doc1 = documents[idx1]
                doc2 = documents[idx2]

                # Format the prompt for the current pair
                prompt_text = prompt_template.format(text1=doc1, text2=doc2)
                llm_response_text = "" # Initialize response text
                llm_response_text_cleaned = "" # Initialize cleaned response
                constraint_type = "Skipped (LLM Error)" # Default constraint type if something goes wrong

                # Check if llm_service is None before calling get_chat_completion
                if llm_service is not None:
                    try:
                         llm_response_text = llm_service.get_chat_completion(prompt_text) # get_chat_completion returns "ERROR" on failure
                         llm_response_text_cleaned = llm_response_text.strip().upper()

                         if llm_response_text_cleaned == "YES":
                             must_link_constraints.append((idx1, idx2))
                             constraint_type = "Must-Link"
                         elif llm_response_text_cleaned == "NO":
                             cannot_link_constraints.append((idx1, idx2))
                             constraint_type = "Cannot-Link"
                         else:
                             print(f"  Warning: Ambiguous LLM response for pair ({idx1}, {idx2}): '{llm_response_text}'. Skipping constraint.")
                             constraint_type = "Ambiguous Response"

                    except Exception as e:
                         print(f"  Error querying LLM for pair ({idx1}, {idx2}): {e}. Skipping constraint.")
                         constraint_type = "Query Exception"
                else:
                     print(f"  Warning: LLMService not available, cannot query pair ({idx1}, {idx2}).")
                     constraint_type = "Skipped (LLM Unavailable)"


                # Collect data for the query CSV
                pairwise_query_data.append({
                    "pair_indices": f"({idx1}, {idx2})",
                    "document1_text_preview": doc1[:100] + "...", # Preview of doc1
                    "document2_text_preview": doc2[:100] + "...", # Preview of doc2
                    "full_prompt": prompt_text,
                    "raw_llm_response": llm_response_text,
                    "cleaned_llm_response": llm_response_text_cleaned,
                    "constraint_type": constraint_type,
                    "doc1_full_text": doc1, # Include full text for reference
                    "doc2_full_text": doc2, # Include full text for reference
                })


            end_time = time.time() # End timing LLM calls
            print(f"LLM querying finished. Time taken: {end_time - start_time:.2f} seconds.")

        # --- Save Pairwise Queries to CSV ---
        if pairwise_query_data:
            try:
                df_queries = pd.DataFrame(pairwise_query_data)
                # Ensure directory exists
                output_dir = os.path.dirname(pairwise_queries_output_csv_path)
                if output_dir and not os.path.exists(output_dir):
                     os.makedirs(output_dir)
                df_queries.to_csv(pairwise_queries_output_csv_path, index=False)
                print(f"\nPairwise queries and responses saved to {pairwise_queries_output_csv_path}")
            except Exception as e:
                print(f"\nError saving pairwise queries to CSV: {e}")
        else:
            print("\nNo pairwise query data collected to save.")


        print(f"Generated {len(must_link_constraints)} must-link constraints and {len(cannot_link_constraints)} cannot-link constraints.")

        # 3. Perform Constrained Clustering (PCKMeans)
        print("\nRunning Constrained Clustering (PCKMeans)...")
        # PCKMeans requires at least 1 must-link or 1 cannot-link constraint to run
        if not must_link_constraints and not cannot_link_constraints:
             print("No constraints generated. Cannot run PCKMeans.")
             method_status = "Completed (No Constraints)"
             assignments = None # Return None if no constraints
        else:
            try:
                # FIX: Removed the random_state=0 argument
                pckmeans = PCKMeans(n_clusters=n_clusters)
                # PCKMeans expects lists of tuples (index1, index2)
                pckmeans.fit(features, ml=must_link_constraints, cl=cannot_link_constraints)
                assignments = pckmeans.labels_ # Get labels after fitting

                print("PCKMeans completed.")
                method_status = "Success"

            except Exception as e:
                print(f"Error during Constrained Clustering (PCKMeans): {e}")
                method_status = "Failed (Clustering Error)"


    # --- Evaluate and Save Metrics to CSV ---
    pairwise_accuracy = None
    pairwise_macro_f1 = None
    pairwise_micro_f1 = None

    # Only attempt metric calculation if assignments were obtained and match label length
    if assignments is not None and len(assignments) == len(labels_np):
         try:
             # Call the calculate_clustering_metrics helper
             # Calculate metrics on the full dataset as PCKMeans provides assignments for all
             accuracy, macro_f1, micro_f1 = calculate_clustering_metrics(labels_np, assignments)

             pairwise_accuracy = accuracy
             pairwise_macro_f1 = macro_f1
             pairwise_micro_f1 = micro_f1

             # Update status if evaluation was successful after clustering success
             if method_status == "Success" and all(m is not None for m in [pairwise_accuracy, pairwise_macro_f1, pairwise_micro_f1]):
                  pass # Keep status as Success
             elif method_status == "Success": # Clustering succeeded, but eval failed
                  method_status = "Completed (Eval Failed)"


         except Exception as e:
              print(f"Error during metrics calculation: {e}")
              if method_status == "Success": # Clustering succeeded, but eval failed
                   method_status = "Completed (Eval Failed)"
              else: # Clustering failed earlier
                   pass # Keep the earlier failed status


    # --- Save Metrics to CSV ---
    metrics_data = {
        'Method': 'Pairwise Constraints',
        'Status': method_status,
        'Accuracy': pairwise_accuracy,
        'Macro_F1': pairwise_macro_f1,
        'Micro_F1': pairwise_micro_f1
    }

    try:
        df_metrics = pd.DataFrame([metrics_data])

        # Append logic remains the same - check if file exists to write header
        if not os.path.exists(METRICS_CSV_PATH):
            df_metrics.to_csv(METRICS_CSV_PATH, index=False, mode='w', header=True)
            print(f"\nCreated {METRICS_CSV_PATH} and saved metrics.")
        else:
            df_metrics.to_csv(METRICS_CSV_PATH, index=False, mode='a', header=False)
            print(f"\nAppended metrics to {METRICS_CSV_PATH}.")

    except Exception as e:
        print(f"\nError saving metrics to CSV: {e}")


    return assignments # Return the assignments (full dataset length)