import numpy as np
import random
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate
from src.llm_service import LLMService
from src.metrics import calculate_clustering_metrics
from typing import List, Dict, Tuple, Any, Set
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans

# Define the path for the metrics CSV file
METRICS_CSV_PATH = "clustering_metrics_results.csv"

def process_pair(pair_data):
    """
    Process a single document pair to generate constraint.
    This function is designed to be run in parallel.
    
    Args:
        pair_data: Tuple containing (idx1, idx2, doc1, doc2, prompt_template, llm_service)
        
    Returns:
        Dict with all query data and constraint information
    """
    idx1, idx2, doc1, doc2, prompt_template, llm_service = pair_data
    
    # Format the prompt for the current pair
    prompt_text = prompt_template.format(text1=doc1, text2=doc2)
    llm_response_text = ""
    llm_response_text_cleaned = ""
    constraint_type = "Skipped"
    constraint = None
    
    try:
        llm_response_text = llm_service.get_chat_completion(prompt_text)
        llm_response_text_cleaned = llm_response_text.strip().upper()
        
        if llm_response_text_cleaned == "YES":
            constraint = ("MUST", (idx1, idx2))
            constraint_type = "Must-Link"
        elif llm_response_text_cleaned == "NO":
            constraint = ("CANNOT", (idx1, idx2))
            constraint_type = "Cannot-Link"
        else:
            constraint_type = "Ambiguous Response"
    except Exception as e:
        constraint_type = f"Error: {str(e)[:50]}..."
    
    # Return all query data
    return {
        "pair_indices": f"({idx1}, {idx2})",
        "doc1_full_text": doc1,
        "doc2_full_text": doc2,
        "full_prompt": prompt_text,
        "raw_llm_response": llm_response_text,
        "constraint_type": constraint_type,
        "constraint": constraint
    }

def cluster_via_pairwise_constraints(
    dataset_name: str,
    documents: List[str],
    features: np.ndarray,
    labels_np: np.ndarray,
    n_clusters: int,
    llm_service: LLMService,
    pairwise_prompt_template: str,
    num_pairs_to_query: int,
    constraint_selection_strategy: str = 'random',
    pairwise_queries_output_csv_path: str = "pairwise_queries_output.csv",
    max_workers: int = 50  # Number of parallel workers for multithreading
) -> np.ndarray | None:
    """
    Implements clustering via pseudo-oracle pairwise constraints with multithreading.
    Queries LLM for constraints in parallel, runs PCKMeans, calculates and saves metrics.
    
    Args:
        dataset_name: Name of the dataset being processed.
        documents: List of original document texts.
        features: Original document embeddings (NumPy array).
        labels_np: True labels (NumPy array) for evaluation.
        n_clusters: The number of clusters.
        llm_service: An initialized LLMService instance.
        pairwise_prompt_template: A string template for pairwise comparison.
        num_pairs_to_query: The number of pairs to query the LLM for constraints.
        constraint_selection_strategy: Strategy for selecting pairs ('random', 'similarity').
        pairwise_queries_output_csv_path: Path to save the generated queries/responses CSV.
        max_workers: Maximum number of parallel threads for LLM querying.
        
    Returns:
        A NumPy array of cluster assignments or None if clustering fails.
    """
    print("\n--- Running Clustering via Pseudo-Oracle Pairwise Constraints ---")
    method_status = "Running"
    assignments = None
    
    # --- Main clustering logic ---
    n_samples = len(documents)
    must_link_constraints = []
    cannot_link_constraints = []
    
    # Dictionary to track constraints between pairs for consistency checking
    constraint_dict = {}
    
    # List to store query data for CSV
    pairwise_query_data = []
    
    # Define Langchain prompt template
    prompt_template = ChatPromptTemplate.from_template(pairwise_prompt_template)
    
    # 1. Select pairs based on strategy
    selected_pairs = []
    all_possible_pairs = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]
    
    # Ensure num_pairs_to_query doesn't exceed available pairs
    num_pairs_to_query = min(num_pairs_to_query, len(all_possible_pairs))
    
    if num_pairs_to_query > 0:
        if constraint_selection_strategy == 'random':
            selected_pairs = random.sample(all_possible_pairs, num_pairs_to_query)
        elif constraint_selection_strategy == 'similarity':
            # Get both most similar and most dissimilar pairs
            similarity_matrix = cosine_similarity(features)
            similarity_pairs = []
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    similarity_pairs.append(((i, j), similarity_matrix[i, j]))
            
            similarity_pairs.sort(key=lambda item: item[1], reverse=True)  # Sort by similarity descending
            
            num_similar = num_pairs_to_query // 2
            num_dissimilar = num_pairs_to_query - num_similar
            
            selected_pairs.extend([pair for pair, sim in similarity_pairs[:num_similar]])
            selected_pairs.extend([pair for pair, sim in similarity_pairs[-num_dissimilar:]])
            
            selected_pairs = list(set(selected_pairs))
            # If we still don't have enough unique pairs
            while len(selected_pairs) < num_pairs_to_query and len(selected_pairs) < len(all_possible_pairs):
                remaining_pairs = list(set(all_possible_pairs) - set(selected_pairs))
                selected_pairs.extend(random.sample(remaining_pairs, min(num_pairs_to_query - len(selected_pairs), len(remaining_pairs))))
            
            selected_pairs = selected_pairs[:num_pairs_to_query]  # Ensure we don't exceed the requested number
        else:
            print(f"Unknown constraint selection strategy: {constraint_selection_strategy}. Using 'random'.")
            selected_pairs = random.sample(all_possible_pairs, num_pairs_to_query)
    else:
        print("Num pairs to query is 0 or less. No constraints will be generated.")
    
    print(f"Selected {len(selected_pairs)} pairs using '{constraint_selection_strategy}' strategy.")
    
    # 2. Query LLM for constraints for selected pairs using multithreading
    if selected_pairs:
        print(f"Querying LLM for {len(selected_pairs)} pairs using {max_workers} parallel workers...")
        start_time = time.time()
        
        # Prepare data for parallel processing
        pair_data_list = [(idx1, idx2, documents[idx1], documents[idx2], prompt_template, llm_service) 
                          for idx1, idx2 in selected_pairs]
        
        # Run queries in parallel with a progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_pair, pair_data): i for i, pair_data in enumerate(pair_data_list)}
            
            # Process results as they complete with a progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Querying LLM (parallel)", unit="pair"):
                try:
                    result = future.result()
                    pairwise_query_data.append(result)
                    
                    # Process constraint if valid
                    if result["constraint"]:
                        constraint_type, (idx1, idx2) = result["constraint"]
                        key = (min(idx1, idx2), max(idx1, idx2))
                        
                        # Check for consistency before adding constraint
                        if key not in constraint_dict or constraint_dict[key] == constraint_type:
                            if constraint_type == "MUST":
                                must_link_constraints.append((idx1, idx2))
                            else:  # CANNOT
                                cannot_link_constraints.append((idx1, idx2))
                            constraint_dict[key] = constraint_type
                except Exception as e:
                    print(f"  Error processing result: {e}")
        
        end_time = time.time()
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
    
    print(f"Generated {len(must_link_constraints)} must-link constraints and {len(cannot_link_constraints)} cannot-link constraints.")
    
    # 3. Validate transitive constraints for consistency
    print("\nValidating constraint consistency...")
    
    # Build connected components for must-link constraints
    must_link_groups = []
    point_to_group = {}
    
    for i, j in must_link_constraints:
        if i in point_to_group and j in point_to_group:
            # Both points already in groups
            if point_to_group[i] != point_to_group[j]:
                # Merge groups
                group_i = point_to_group[i]
                group_j = point_to_group[j]
                new_group = must_link_groups[group_i].union(must_link_groups[group_j])
                must_link_groups[group_i] = new_group
                
                # Update all points in group_j to point to group_i
                for point in must_link_groups[group_j]:
                    point_to_group[point] = group_i
                
                # Empty group_j (but keep the index)
                must_link_groups[group_j] = set()
        elif i in point_to_group:
            # Add j to i's group
            group_idx = point_to_group[i]
            must_link_groups[group_idx].add(j)
            point_to_group[j] = group_idx
        elif j in point_to_group:
            # Add i to j's group
            group_idx = point_to_group[j]
            must_link_groups[group_idx].add(i)
            point_to_group[i] = group_idx
        else:
            # Create new group
            must_link_groups.append({i, j})
            group_idx = len(must_link_groups) - 1
            point_to_group[i] = group_idx
            point_to_group[j] = group_idx
    
    # Filter out empty groups
    must_link_groups = [g for g in must_link_groups if g]
    
    # Check for cannot-link consistency
    consistent_cannot_links = []
    
    for i, j in cannot_link_constraints:
        # If both points are in the same must-link group, this is inconsistent
        if i in point_to_group and j in point_to_group and point_to_group[i] == point_to_group[j]:
            print(f"  Warning: Inconsistent constraint between {i} and {j} - both in same must-link group")
            continue
        
        consistent_cannot_links.append((i, j))
    
    print(f"Validated constraints: {len(must_link_constraints)} must-link and {len(consistent_cannot_links)} consistent cannot-link")
    
    # 4. Perform Constrained Clustering (PCKMeans)
    print("\nRunning Constrained Clustering (PCKMeans)...")
    
    if not must_link_constraints and not consistent_cannot_links:
        print("No valid constraints generated. Cannot run PCKMeans.")
        method_status = "Completed (No Valid Constraints)"
    else:
        try:
            pckmeans = PCKMeans(n_clusters=n_clusters)
            pckmeans.fit(features, ml=must_link_constraints, cl=consistent_cannot_links)
            assignments = pckmeans.labels_
            print("PCKMeans completed successfully.")
            method_status = "Success"
        except Exception as e:
            print(f"Error during Constrained Clustering (PCKMeans): {e}")
            method_status = "Failed (Clustering Error)"
    
    # --- Evaluate and Save Metrics to CSV ---
    metrics = {}
    
    # Only attempt metric calculation if assignments were obtained
    if assignments is not None and len(assignments) == len(labels_np):
        try:
            metrics = calculate_clustering_metrics(labels_np, assignments, n_clusters)
        except Exception as e:
            print(f"Error during metrics calculation: {e}")
    
    # --- Save Metrics to CSV ---
    metrics_data = {
        "Dataset": dataset_name,
        'Method': 'Pairwise Constraints',
        'Status': method_status,
        **metrics
    }
    
    try:
        df_metrics = pd.DataFrame([metrics_data])
        
        # Check if file exists to write header
        if not os.path.exists(METRICS_CSV_PATH):
            df_metrics.to_csv(METRICS_CSV_PATH, index=False, mode='w', header=True)
            print(f"\nCreated {METRICS_CSV_PATH} and saved metrics.")
        else:
            df_metrics.to_csv(METRICS_CSV_PATH, index=False, mode='a', header=False)
            print(f"\nAppended metrics to {METRICS_CSV_PATH}.")
    except Exception as e:
        print(f"\nError saving metrics to CSV: {e}")
    
    return assignments