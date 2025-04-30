import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity # cosine_similarity for LLM simulation if needed
from sklearn.cluster import KMeans # For initial clustering if needed by this method
from langchain_core.prompts import ChatPromptTemplate
from src.llm_service import LLMService # Import LLM service
from typing import List, Tuple


def correct_clustering_with_llm(
    documents: List[str],
    features: np.ndarray,
    initial_assignments: np.ndarray,
    n_clusters: int, # Pass n_clusters explicitly for clarity
    llm_service: LLMService, # Accept LLMService instance
    correction_prompt_template: str,
    k_low_confidence: int = 500,
    num_candidate_clusters: int = 4
) -> np.ndarray | None:
    """
    Implements correcting an existing clustering using an LLM (Section 2.3).

    Args:
        documents: List of original document texts.
        features: Original document embeddings (NumPy array).
        initial_assignments: Initial cluster assignments (NumPy array).
        n_clusters: The total number of clusters expected (including potential empty ones from initial run).
        llm_service: An initialized LLMService instance.
        correction_prompt_template: A string template for correction decisions.
                                    Should include placeholders for the point's text
                                    and the representative's text.
                                    Example: "Should the following text snippet be in the same cluster as this representative text? Respond with YES or NO.\nSnippet: {point_text}\nRepresentative: {representative_text}"
        k_low_confidence: The number of lowest-confidence points to consider for correction.
        num_candidate_clusters: The number of nearest clusters to consider as alternatives.

    Returns:
        A NumPy array of corrected cluster assignments, or None if processing fails.
    """
    print("\n--- Running Clustering Correction with LLM ---")
    if not llm_service.is_available():
        print("LLMService is not available. Cannot run clustering correction.")
        return None

    corrected_assignments = np.copy(initial_assignments)
    n_samples = len(documents)
    unique_clusters = np.unique(initial_assignments)


    # Exclude noise cluster if present and identify valid cluster IDs
    valid_cluster_ids = [c for c in unique_clusters if c != -1]
    if not valid_cluster_ids:
        print("Only noise points or no clusters found initially. Skipping correction.")
        return corrected_assignments
    n_valid_clusters = len(valid_cluster_ids)

    if n_valid_clusters <= 1:
        print("Only one valid cluster found initially. Skipping correction.")
        return corrected_assignments


    # Define Langchain prompt template
    prompt_template = ChatPromptTemplate.from_template(correction_prompt_template)

    # 1. Find cluster centroids and representatives
    # Use the full range of n_clusters for centroids array size, even if some are empty
    # This helps if original assignments were 0, 1, 3 (cluster 2 was empty)
    centroids = np.zeros((n_clusters, features.shape[1]))
    representatives_indices = {}
    # Map cluster ID to its index in the centroids/valid_cluster_ids arrays
    # We need a mapping from the original cluster ID (0 to n_clusters-1)
    # to an index if we only store centroids for non-empty clusters.
    # Let's simplify and use the cluster ID directly as index in centroids if possible,
    # filling only for non-empty clusters.

    print("Finding cluster centroids and representatives...")
    for cluster_id in range(n_clusters): # Iterate through all potential cluster IDs
        if cluster_id not in valid_cluster_ids:
             # print(f"  Cluster {cluster_id} is empty in initial assignments.")
             continue # Skip empty clusters

        points_in_cluster_indices = np.where(initial_assignments == cluster_id)[0]
        if len(points_in_cluster_indices) > 0:
            cluster_features = features[points_in_cluster_indices]
            centroid = np.mean(cluster_features, axis=0)
            centroids[cluster_id] = centroid # Store centroid at index = cluster_id

            # Find representative: point nearest to the centroid
            # Calculate distances from centroid to all points in the cluster
            distances_to_centroid = euclidean_distances(centroid.reshape(1, -1), cluster_features)[0]
            nearest_point_in_cluster_index = points_in_cluster_indices[np.argmin(distances_to_centroid)]
            representatives_indices[cluster_id] = nearest_point_in_cluster_index
        else:
             # This case should be caught by the valid_cluster_ids check, but as safety
             print(f"Warning: Cluster {cluster_id} is in valid_cluster_ids but appears empty.")


    # Ensure all valid clusters in initial assignments have representatives
    valid_cluster_ids_with_reps = list(representatives_indices.keys())
    if len(valid_cluster_ids_with_reps) != n_valid_clusters:
         print("Warning: Some valid clusters do not have representatives. These will be excluded from correction logic.")


    # 2. Identify low-confidence points
    if not valid_cluster_ids_with_reps:
         print("No clusters with representatives found. Skipping low-confidence point identification.")
         low_confidence_point_indices = []
    else:
        print("Identifying low-confidence points...")
        # Calculate distance of each point to its assigned centroid and the second nearest centroid
        # Need distances to all centroids WITH representatives for each point
        centroid_matrix_for_dist = np.array([centroids[cid] for cid in valid_cluster_ids_with_reps]) # Use centroids at their cluster ID index
        distances_to_all_valid_centroids = euclidean_distances(features, centroid_matrix_for_dist)

        confidence_margins = []
        point_indices = [] # Store original indices of points being evaluated

        for i in range(n_samples):
            current_assignment = initial_assignments[i]
            # Only consider points assigned to valid clusters that have representatives
            if current_assignment == -1 or current_assignment not in representatives_indices:
                 continue

            point_indices.append(i)

            # Get distances to valid centroids with representatives
            # Find the index of the current assignment within the list of valid cluster IDs with representatives
            try:
                assigned_cluster_rep_idx = valid_cluster_ids_with_reps.index(current_assignment)
            except ValueError:
                 # Should not happen based on the outer check, but as safety
                 continue


            dists = distances_to_all_valid_centroids[i]

            # Find distance to assigned cluster's centroid
            dist_to_assigned_centroid = dists[assigned_cluster_rep_idx]

            # Find distance to the second nearest centroid
            # Exclude the distance to the assigned centroid
            dists_excluding_assigned = np.delete(dists, assigned_cluster_rep_idx)
            if len(dists_excluding_assigned) > 0:
                 dist_to_second_nearest_centroid = np.min(dists_excluding_assigned)
                 margin = dist_to_second_nearest_centroid - dist_to_assigned_centroid
                 confidence_margins.append(margin)
            else:
                 # Only one relevant cluster exists for this point, handle as high confidence
                 confidence_margins.append(np.inf) # Infinite margin means very high confidence


        if not confidence_margins:
             print("Could not calculate confidence margins for any points. Skipping correction.")
             low_confidence_point_indices = []
        else:
            # Get indices of points with the lowest confidence margins
            sorted_indices = np.argsort(confidence_margins)[:min(k_low_confidence, len(confidence_margins))]
            low_confidence_point_indices = [point_indices[i] for i in sorted_indices]

        print(f"Identified {len(low_confidence_point_indices)} low-confidence points.")

    # 3. Attempt to correct low-confidence points using LLM
    print("Attempting to correct assignments for low-confidence points using LLM...")
    corrected_count = 0
    for i, point_idx in enumerate(low_confidence_point_indices):
        original_assignment = corrected_assignments[point_idx]
        point_text = documents[point_idx]

        # Re-check if original_assignment is still valid and has a representative (safety check)
        if original_assignment == -1 or original_assignment not in representatives_indices:
            # print(f"  Skipping point {point_idx}: Original assignment {original_assignment} is invalid or has no representative.")
            continue


        current_representative_idx = representatives_indices[original_assignment]
        current_representative_text = documents[current_representative_idx]

        # Ask LLM if the point should be linked to its current cluster's representative
        prompt_current = ChatPromptTemplate.from_template(correction_prompt_template).format(point_text=point_text, representative_text=current_representative_text)
        llm_response_current = llm_service.get_chat_completion(prompt_current).strip().upper()
        # print(f"  Checking point {i+1}/{len(low_confidence_point_indices)} against current cluster {original_assignment}. Response: {llm_response_current}")


        if llm_response_current == "NO":
            # If LLM says NO to current cluster, consider alternatives
            # print(f"  LLM says NO to current cluster {original_assignment}. Checking candidate clusters.")

            # Find IDs of other valid clusters with representatives
            other_valid_cluster_ids = [cid for cid in representatives_indices.keys() if cid != original_assignment]
            if not other_valid_cluster_ids:
                 # print("  No other valid clusters to consider.")
                 continue

            # Get distances from the point to all other valid centroids with representatives
            other_centroids_matrix = np.array([centroids[cid] for cid in other_valid_cluster_ids])
            if other_centroids_matrix.shape[0] == 0:
                 # print("  No other valid clusters to consider (matrix empty).")
                 continue

            dists_to_other_centroids = euclidean_distances(features[point_idx].reshape(1, -1), other_centroids_matrix)[0]


            # Get IDs of candidate clusters sorted by proximity
            sorted_other_cluster_indices = np.argsort(dists_to_other_centroids)
            candidate_cluster_ids = [other_valid_cluster_ids[k] for k in sorted_other_cluster_indices[:num_candidate_clusters]]

            new_assignment = original_assignment # Default to keeping original
            reassigned = False

            # print(f"  Candidate clusters for point {point_idx}: {candidate_cluster_ids}")

            for candidate_cluster_id in candidate_cluster_ids:
                # Check if candidate cluster still has a representative (safety check)
                if candidate_cluster_id not in representatives_indices:
                    continue

                candidate_representative_idx = representatives_indices[candidate_cluster_id]
                candidate_representative_text = documents[candidate_representative_idx]

                # Ask LLM if the point should be linked to the candidate cluster's representative
                prompt_candidate = ChatPromptTemplate.from_template(correction_prompt_template).format(point_text=point_text, representative_text=candidate_representative_text)
                llm_response_candidate = llm_service.get_chat_completion(prompt_candidate).strip().upper()
                # print(f"    Checking against candidate cluster {candidate_cluster_id}. Response: {llm_response_candidate}")


                if llm_response_candidate == "YES":
                    # LLM says YES to this candidate cluster
                    new_assignment = candidate_cluster_id
                    corrected_assignments[point_idx] = new_assignment
                    corrected_count += 1
                    print(f"  Point {point_idx} reassigned from cluster {original_assignment} to {new_assignment}.")
                    reassigned = True
                    break # Move to the next low-confidence point

            # If loop finishes and the point was not reassigned
            # if not reassigned:
                 # print(f"  Point {point_idx} remains in original cluster {original_assignment} after checking candidates.")


        # else: # LLM says YES to current cluster, keep assignment
        #     pass # No change needed


    print(f"\nFinished correction pass. {corrected_count} points were reassigned.")
    return corrected_assignments