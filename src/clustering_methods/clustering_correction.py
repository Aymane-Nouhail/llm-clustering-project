import numpy as np
import os
import pandas as pd
import time # Import time for potential timing
from sklearn.metrics import f1_score, precision_score, recall_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn.cluster import KMeans # Needed to find centroids based on initial assignments
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity # Needed for distances
from langchain_core.prompts import ChatPromptTemplate
from src.llm_service import LLMService
from src.metrics import calculate_clustering_metrics
from typing import List, Dict, Any, Tuple
import concurrent.futures # Import for parallel execution
from tqdm import tqdm # Import tqdm for the loading bar
from few_shot_clustering.eval_utils import cluster_acc


# Define the path for the metrics CSV file (use the same path as other run scripts)
METRICS_CSV_PATH = "clustering_metrics_results.csv"


# Helper to find cluster centroids and representatives (closest document to centroid)
def find_cluster_info(features: np.ndarray, assignments: np.ndarray, documents: List[str], n_clusters: int) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Finds cluster centroids and the index of the document closest to each centroid
    for clusters that have at least one assigned document.
    """
    centroids = np.zeros((n_clusters, features.shape[1]))
    representative_doc_indices: Dict[int, int] = {}

    for cluster_id in range(n_clusters):
        # Consider only documents assigned to this cluster (and are valid assignments >= 0)
        cluster_points_indices = np.where(assignments == cluster_id)[0]
        if len(cluster_points_indices) > 0:
            cluster_points = features[cluster_points_indices]
            centroid = np.mean(cluster_points, axis=0)
            centroids[cluster_id] = centroid

            # Find document closest to the centroid within this cluster's points
            distances_to_centroid = euclidean_distances([centroid], cluster_points)
            closest_point_index_in_subset = np.argmin(distances_to_centroid)
            original_doc_index = cluster_points_indices[closest_point_index_in_subset]
            representative_doc_indices[cluster_id] = original_doc_index
        # If a cluster is empty in initial assignments, its centroid remains zero, and it won't have a representative

    return centroids, representative_doc_indices


# Function to identify low-confidence points based on the paper's margin criteria
def identify_low_confidence_points(features: np.ndarray, assignments: np.ndarray, centroids: np.ndarray, n_clusters: int, k_low_confidence: int) -> List[int]:
    """
    Identifies the top k points with the least margin between the nearest and
    second-nearest clusters (including the assigned cluster).

    Args:
        features: Document embeddings.
        assignments: Initial cluster assignments.
        centroids: Cluster centroids.
        n_clusters: Total number of clusters.
        k_low_confidence: Number of low-confidence points to select.

    Returns:
        A list of original document indices identified as low-confidence.
    """
    n_samples = features.shape[0]
    margins = np.full(n_samples, -np.inf) # Initialize margins; larger margin means higher confidence

    # Calculate distances to all centroids for all points
    distances_to_all_centroids = euclidean_distances(features, centroids)

    for i in range(n_samples):
        assigned_cluster = assignments[i]
        # Skip documents that weren't assigned or had invalid initial assignments
        if assigned_cluster < 0 or assigned_cluster >= n_clusters:
             continue

        # Get distances from this point to all centroids
        dists = distances_to_all_centroids[i, :]

        # Sort distances to find nearest and second nearest *among all clusters*
        # Ensure there are at least two distinct distances to avoid index errors
        unique_dists = np.unique(dists)
        if len(unique_dists) < 2:
             # Cannot calculate margin if less than 2 distinct distances
             continue


        sorted_dist_indices = np.argsort(dists)
        # The nearest is sorted_dist_indices[0], the second nearest is sorted_dist_indices[1]
        nearest_centroid_dist = dists[sorted_dist_indices[0]]
        second_nearest_centroid_dist = dists[sorted_dist_indices[1]]

        # The paper defines margin as "between the nearest and second-nearest clusters"
        # This implies the difference in distance: second_nearest_dist - nearest_dist.
        # Points close to a decision boundary will have a small margin.
        margin = second_nearest_centroid_dist - nearest_centroid_dist
        margins[i] = margin

    # Get indices of points with finite margins (successfully processed)
    valid_margin_indices = np.isfinite(margins)
    if np.sum(valid_margin_indices) == 0:
         print("Warning: No points with finite margins found for low-confidence identification.")
         return []

    # Sort indices based on margin (ascending, smallest margin first)
    sorted_indices_in_subset = np.argsort(margins[valid_margin_indices])
    # Get the original indices of the points with the smallest margins
    low_confidence_original_indices = np.where(valid_margin_indices)[0][sorted_indices_in_subset]

    # Return the top k low-confidence points
    return low_confidence_original_indices[:k_low_confidence].tolist()

# Helper function to process a single low-confidence point with LLM queries
# Returns a tuple containing the assignment result and a list of query data dictionaries
def process_low_confidence_point(
    doc_index: int,
    document_text: str,
    current_assignment: int,
    features: np.ndarray,
    centroids: np.ndarray,
    representative_doc_indices: Dict[int, int],
    n_clusters: int,
    llm_service: LLMService,
    correction_prompt_template_langchain: ChatPromptTemplate,
    num_candidate_clusters: int,
    documents: List[str] # Pass full documents list to get representative text
) -> Tuple[int, int, int, List[Dict[str, Any]]]: # Returns (original_doc_index, old_assignment, new_assignment, query_data_list)
    """
    Processes a single low-confidence document using the two-step LLM query process.
    Returns the original document index, its old assignment, the new assignment,
    and a list of dictionaries containing data for each LLM query made for this point.
    Returns old_assignment for new_assignment if no correction occurs or fails.
    The query data is simplified to include only prompt, answer, and action.
    """
    # print(f"  Processing low-confidence point (Document {doc_index})...") # Optional verbose print

    query_data_list: List[Dict[str, Any]] = [] # List to store query data for this point
    final_assignment = current_assignment # Initialize final assignment

    # Skip if the initial assignment was invalid (-1) - should be handled before calling this helper
    # if current_assignment < 0 or current_assignment >= n_clusters:
    #      return (doc_index, current_assignment, current_assignment, query_data_list) # Return original if invalid

    # Get representative text for the currently assigned cluster
    assigned_cluster_rep_index = representative_doc_indices.get(current_assignment)
    if assigned_cluster_rep_index is None:
         print(f"  Skipping document {doc_index}: No representative found for assigned cluster {current_assignment}.")
         # Log this skip in query data? Maybe add a status for the point itself rather than per query
         return (doc_index, current_assignment, current_assignment, query_data_list) # Return original if no representative


    assigned_cluster_rep_document = documents[assigned_cluster_rep_index]

    # --- Step 1: First LLM Query - Is current assignment correct? ---
    prompt_q1_text = correction_prompt_template_langchain.format(
        document_text=document_text,
        rep_doc_text=assigned_cluster_rep_document,
        query_type="is_linked_to_assigned"
    )

    llm_response_q1 = llm_service.get_chat_completion(prompt_q1_text)
    llm_response_q1_cleaned = llm_response_q1.strip().upper()

    # Collect data for Query 1 (Simplified)
    query_data_list.append({
        "document_index": doc_index, # Keep index for reference
        "full_prompt": prompt_q1_text, # Keep the full prompt
        "llm_answer": llm_response_q1_cleaned, # LLM's cleaned answer
        "resulting_action": "Evaluated Current Assignment", # Placeholder, updated later
    })


    # If LLM predicts current assignment is *not* correct
    if llm_response_q1_cleaned != "YES":
        # Update the action for Query 1
        query_data_list[-1]["resulting_action"] = "LLM Said NO (Checking Candidates)"

        # --- Step 2: Second LLM Query - Which candidate cluster? ---
        # Find the next 'num_candidate_clusters' nearest clusters *excluding* the assigned one
        distances_to_all_centroids = euclidean_distances([features[doc_index]], centroids)[0]
        sorted_centroid_indices = np.argsort(distances_to_all_centroids)

        candidate_cluster_ids = []
        for c_id in sorted_centroid_indices:
             if c_id != current_assignment and len(candidate_cluster_ids) < num_candidate_clusters:
                  # Ensure the candidate cluster actually has a representative
                  if c_id not in representative_doc_indices:
                       continue # Skip this candidate if no representative

                  candidate_cluster_ids.append(c_id)

             if len(candidate_cluster_ids) == num_candidate_clusters:
                  break # Found enough candidates


        # Ensure we actually found candidate clusters with representatives
        if not candidate_cluster_ids:
             # print(f"  Warning: Could not find {num_candidate_clusters} candidate clusters with representatives for document {doc_index}.") # Optional warning
             query_data_list[-1]["resulting_action"] = "LLM Said NO (No Candidates Found)" # Update action for Q1 failure
             return (doc_index, current_assignment, current_assignment, query_data_list) # Return original if no candidates found


        # --- LLM Query per Candidate ---
        reassigned_this_point = False
        for candidate_c_id in candidate_cluster_ids:
            # Get representative document text for the candidate cluster
            candidate_cluster_rep_index = representative_doc_indices.get(candidate_c_id)
            # We already checked if representative exists when building candidate_cluster_ids, but double check
            if candidate_cluster_rep_index is None:
                 continue # Skip this candidate

            candidate_cluster_rep_document = documents[candidate_cluster_rep_index]

            prompt_q2_candidate_text = correction_prompt_template_langchain.format(
                document_text=document_text,
                rep_doc_text=candidate_cluster_rep_document,
                query_type=f"is_linked_to_candidate_{candidate_c_id}"
            )

            llm_response_q2_candidate = llm_service.get_chat_completion(prompt_q2_candidate_text)
            llm_response_q2_candidate_cleaned = llm_response_q2_candidate.strip().upper()

            # Collect data for this candidate query (Step 2 - Simplified)
            query_data_list.append({
                "document_index": doc_index, # Keep index for reference
                "full_prompt": prompt_q2_candidate_text, # Keep the full prompt
                "llm_answer": llm_response_q2_candidate_cleaned, # LLM's cleaned answer
                "resulting_action": "Evaluated Candidate", # Placeholder, updated later
            })


            # If LLM responds positively for this candidate
            if llm_response_q2_candidate_cleaned == "YES":
                # Reassign the point to this candidate cluster
                final_assignment = candidate_c_id # Update final assignment
                reassigned_this_point = True
                # print(f"  Point {doc_index} reassigned from cluster {current_assignment} to {candidate_c_id}.") # Optional print
                query_data_list[-1]["resulting_action"] = f"LLM Said YES (Reassigned to {candidate_c_id})" # Update action for this query
                break # Stop checking other candidates once reassigned


        # If the point was not reassigned after checking all candidates
        if not reassigned_this_point:
            # print(f"  Point {doc_index} remains in cluster {current_assignment} (no positive link to candidates).") # Optional print
            # Update the action for any candidate queries that didn't result in reassignment
            # FIX: Check if resulting_action is still the placeholder before updating
            for qd in query_data_list:
                 # Only update if it's a candidate query log AND it wasn't already marked as a successful reassignment
                 if qd.get("resulting_action") == "Evaluated Candidate":
                      qd["resulting_action"] = "LLM Said NO (Remains in Original)"


    else:
        # If LLM predicted current assignment *is* correct (Q1 response was YES)
        # print(f"  Point {doc_index} confirmed in cluster {current_assignment} by LLM.") # Optional print
        query_data_list[-1]["resulting_action"] = "LLM Said YES (Remains in Original)" # Update action for Q1 success


    return (doc_index, current_assignment, final_assignment, query_data_list) # Return original index, old, new assignment, and query data list


# The main function for clustering correction (Multithreaded)
def cluster_via_correction(
    dataset_name: str,
    documents: List[str],
    features: np.ndarray,
    initial_assignments: np.ndarray, # Assignments from initial clustering (e.g., Naive KMeans)
    labels_np: np.ndarray, # True labels for evaluation
    n_clusters: int,
    llm_service: LLMService | None, # Allow llm_service to be None
    correction_prompt_template: str, # Single template for both query types
    k_low_confidence: int, # Number of low-confidence points to consider
    num_candidate_clusters: int, # Number of nearest clusters to offer as candidates (paper uses 4)
    correction_queries_output_csv_path: str = "correction_queries_output.csv" # Add path for saving queries
) -> np.ndarray | None:
    """
    Implements clustering correction using a two-step LLM querying process
    for low-confidence points, based on the paper's description.
    Uses multithreading for LLM calls.
    Calculates and saves metrics to CSV, and saves LLM queries/responses to a separate CSV
    with simplified data.
    """
    print("\n--- Running Clustering Correction with LLM ---")
    method_status = "Failed" # Default status
    # Create a mutable copy of initial assignments to modify
    corrected_assignments = np.copy(initial_assignments)

    # --- Initial Checks ---
    if llm_service is None or not llm_service.is_available():
        print("LLMService is not available. Cannot run clustering correction.")
        method_status = "Skipped (LLM Service Missing)"
    elif len(documents) == 0 or features.size == 0 or len(labels_np) == 0 or len(documents) != features.shape[0] or len(documents) != len(labels_np) or len(documents) != len(initial_assignments):
         print("Input data (documents, features, labels, or initial assignments) is missing or inconsistent.")
         method_status = "Failed (Invalid Input Data)"
    elif k_low_confidence <= 0:
         print("k_low_confidence must be positive for Clustering Correction.")
         method_status = "Skipped (Invalid k_low_confidence)"
    elif num_candidate_clusters <= 0:
         print("num_candidate_clusters must be positive for Clustering Correction.")
         method_status = "Skipped (Invalid num_candidate_clusters)"
    # Check if initial assignments seem plausible (e.g., not all -1)
    elif np.all(initial_assignments < 0):
         print("Initial assignments contain no valid assignments. Cannot run correction.")
         method_status = "Skipped (Invalid Initial Assignments)"

    else:
        # --- Proceed with Method Logic if all checks pass ---
        n_samples = len(documents)
        initial_assignments_np = np.array(initial_assignments) # Ensure numpy array

        # Calculate centroids and find representative documents based on initial assignments
        print("Finding cluster centroids and representatives based on initial assignments...")
        centroids, representative_doc_indices = find_cluster_info(features, initial_assignments_np, documents, n_clusters)

        # Ensure we have representatives for relevant clusters
        if len(representative_doc_indices) < n_clusters:
             print(f"Warning: Found representatives for only {len(representative_doc_indices)} out of {n_clusters} clusters.")
             # This might happen if some clusters are empty in the initial assignments.
             # We can proceed but need to handle cases where a representative isn't found.


        print(f"Identifying top {k_low_confidence} low-confidence points...")
        # Identify low-confidence points based on the paper's margin criteria
        low_confidence_indices = identify_low_confidence_points(features, initial_assignments_np, centroids, n_clusters, k_low_confidence)

        print(f"Identified {len(low_confidence_indices)} low-confidence points.")

        if not low_confidence_indices:
            print("No low-confidence points identified. Skipping LLM correction process.")
            method_status = "Completed (No Low Confidence Points)"
        else:
            print(f"Attempting to correct {len(low_confidence_indices)} low-confidence points using LLM queries in parallel...")
            llm_calls_count = 0 # Counter for LLM calls (approximate with parallel)
            points_reassigned_count = 0 # Counter for points that were reassigned

            correction_prompt_template_langchain = ChatPromptTemplate.from_template(correction_prompt_template)

            # Use ThreadPoolExecutor for parallel execution of LLM queries for low-confidence points
            MAX_WORKERS = 20 # Adjust based on your machine and OpenAI rate limits (lower than keyphrase due to sequential queries per point)
            print(f"Using {MAX_WORKERS} workers for parallel correction queries.")

            start_time = time.time() # Start timing correction process

            # List to store results from parallel processing: (original_doc_index, old_assignment, new_assignment, query_data_list)
            correction_results: List[Tuple[int, int, int, List[Dict[str, Any]]]] = []
            all_query_data: List[Dict[str, Any]] = [] # List to collect all query data from all points

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit tasks for each low-confidence document index
                future_to_doc_index = {
                    executor.submit(
                        process_low_confidence_point,
                        doc_index,
                        documents[doc_index],
                        corrected_assignments[doc_index], # Pass the current assignment
                        features,
                        centroids,
                        representative_doc_indices,
                        n_clusters,
                        llm_service,
                        correction_prompt_template_langchain,
                        num_candidate_clusters,
                        documents # Pass documents list to helper
                    ): doc_index for doc_index in low_confidence_indices
                    # Filter out documents with invalid initial assignments before submitting
                    if corrected_assignments[doc_index] >= 0 and corrected_assignments[doc_index] < n_clusters
                }

                # Wrap the as_completed iterator with tqdm for a progress bar
                # The total is the number of tasks submitted (valid low-confidence points)
                for future in tqdm(concurrent.futures.as_completed(future_to_doc_index), total=len(future_to_doc_index), desc="Correcting Assignments"):
                    doc_index = future_to_doc_index[future]
                    try:
                        # Retrieve the result tuple (original_doc_index, old_assignment, new_assignment, query_data_list)
                        result_tuple = future.result()
                        correction_results.append(result_tuple)
                        # Extend the main query data list with data from this point
                        all_query_data.extend(result_tuple[3])


                    except Exception as exc:
                        # Handle exceptions from the helper function
                        print(f"\n--- Exception processing document {doc_index} during correction: {exc} ---")
                        # The point's assignment will remain unchanged in corrected_assignments


            end_time = time.time() # End timing correction process
            print(f"Finished parallel correction processing. Time taken: {end_time - start_time:.2f} seconds.")

            # --- Apply Corrections from Results ---
            # Sort correction_results by original_doc_index to apply updates in order (optional but good practice)
            correction_results.sort(key=lambda x: x[0])

            for original_doc_index, old_assignment, new_assignment, _ in correction_results: # Ignore query_data_list here
                 if new_assignment != old_assignment:
                      corrected_assignments[original_doc_index] = new_assignment
                      # points_reassigned_count += 1 # Increment counter if reassignment occurred

            # Recalculate points_reassigned_count based on the actual changes made
            points_reassigned_count = np.sum(corrected_assignments != initial_assignments_np)


            print(f"\nFinished correction pass. {points_reassigned_count} points were reassigned out of {len(low_confidence_indices)} low-confidence points considered.")
            print(f"Total LLM queries made: {len(all_query_data)}") # Total queries is the length of the collected list

            method_status = "Success" # Assume success if the pass completed

        # --- Save Correction Queries to CSV ---
        if all_query_data:
            try:
                df_queries = pd.DataFrame(all_query_data)
                # Ensure directory exists
                output_dir = os.path.dirname(correction_queries_output_csv_path)
                if output_dir and not os.path.exists(output_dir):
                     os.makedirs(output_dir)
                df_queries.to_csv(correction_queries_output_csv_path, index=False)
                print(f"\nCorrection queries and responses saved to {correction_queries_output_csv_path}")
            except Exception as e:
                print(f"\nError saving correction queries to CSV: {e}")
        else:
            print("\nNo correction query data collected to save.")


    # --- Evaluate and Save Metrics to CSV ---
    metrics = calculate_clustering_metrics(labels_np, corrected_assignments, n_clusters)             
    # --- Save Metrics to CSV ---
    metrics_data = {
        'Dataset': dataset_name,
        'Method': 'LLM Correction', # Use 'LLM Correction' as method name
        'Status': method_status,
        **metrics
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


    return corrected_assignments # Return the corrected assignments
