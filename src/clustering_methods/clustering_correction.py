import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Tuple
from src.llm_service import LLMService
import concurrent.futures
from typing import Any
from tqdm import tqdm
import os
from src.metrics import calculate_clustering_metrics

METRICS_CSV_PATH = "clustering_metrics_results.csv"

def find_cluster_info(features: np.ndarray, assignments: np.ndarray, 
                     documents: List[str], n_clusters: int) -> Tuple[np.ndarray, Dict[int, int]]:
    """Calculate cluster centroids and find representative documents.
    
    Args:
        features: Document embeddings
        assignments: Cluster assignments for each document
        documents: List of document texts
        n_clusters: Total number of clusters
        
    Returns:
        Tuple of:
        - Centroids matrix (n_clusters x feature_dim)
        - Dictionary mapping cluster IDs to index of representative document
    """
    centroids = np.zeros((n_clusters, features.shape[1]))
    representatives = {}

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(assignments == cluster_id)[0]
        if cluster_indices.size > 0:
            centroids[cluster_id] = np.mean(features[cluster_indices], axis=0)
            distances = euclidean_distances([centroids[cluster_id]], features[cluster_indices])
            representatives[cluster_id] = cluster_indices[np.argmin(distances)]

    return centroids, representatives

def identify_low_confidence_points(features: np.ndarray, assignments: np.ndarray,
                                 centroids: np.ndarray, n_clusters: int,
                                 k: int) -> List[int]:
    """Identify documents with least confidence in their cluster assignments.
    
    Args:
        features: Document embeddings
        assignments: Current cluster assignments
        centroids: Cluster centroids
        n_clusters: Total number of clusters
        k: Number of low-confidence points to return
        
    Returns:
        List of document indices with lowest confidence
    """
    distances = euclidean_distances(features, centroids)
    margins = np.array([np.sort(row)[1] - np.sort(row)[0] if len(np.unique(row)) > 1 
                       else -np.inf for row in distances])
    valid = np.isfinite(margins)
    return np.where(valid)[0][np.argsort(margins[valid])[:k]].tolist()

def process_low_confidence_point(
    doc_index: int,
    document: str,
    current_assignment: int,
    features: np.ndarray,
    centroids: np.ndarray,
    representatives: Dict[int, int],
    n_clusters: int,
    llm_service: LLMService,
    prompt_template: ChatPromptTemplate,
    num_candidates: int,
    documents: List[str]
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    """Process a single low-confidence document through LLM correction pipeline.
    
    Args:
        doc_index: Index of document to process
        document: Text content of document
        current_assignment: Current cluster assignment
        features: All document embeddings
        centroids: Cluster centroids
        representatives: Cluster representative indices
        n_clusters: Total clusters
        llm_service: LLM service instance
        prompt_template: Template for correction prompts
        num_candidates: Number of alternative clusters to consider
        documents: Full list of documents
        
    Returns:
        Tuple containing:
        - Original document index
        - Old cluster assignment
        - New cluster assignment
        - List of query metadata dictionaries
    """
    queries = []
    new_assignment = current_assignment
    
    # Get current cluster representative
    current_rep_idx = representatives.get(current_assignment)
    if current_rep_idx is None:
        return (doc_index, current_assignment, current_assignment, queries)
    
    # Step 1: Verify current assignment
    prompt = prompt_template.format(
        document_text=document,
        rep_doc_text=documents[current_rep_idx],
        query_type="is_linked_to_assigned"
    )
    response = llm_service.get_chat_completion(prompt).strip().upper()
    
    queries.append({
        "document_index": doc_index,
        "full_prompt": prompt,
        "llm_answer": response,
        "resulting_action": "Evaluated Current Assignment"
    })

    if response != "YES":
        queries[-1]["resulting_action"] = "LLM Said NO (Checking Candidates)"
        
        # Step 2: Check alternative clusters
        distances = euclidean_distances([features[doc_index]], centroids)[0]
        candidates = [c for c in np.argsort(distances) 
                     if c != current_assignment and c in representatives][:num_candidates]
        
        for candidate in candidates:
            prompt = prompt_template.format(
                document_text=document,
                rep_doc_text=documents[representatives[candidate]],
                query_type=f"is_linked_to_candidate_{candidate}"
            )
            response = llm_service.get_chat_completion(prompt).strip().upper()
            
            queries.append({
                "document_index": doc_index,
                "full_prompt": prompt,
                "llm_answer": response,
                "resulting_action": "Evaluated Candidate"
            })

            if response == "YES":
                new_assignment = candidate
                queries[-1]["resulting_action"] = f"LLM Said YES (Reassigned to {candidate})"
                break
            else:
                queries[-1]["resulting_action"] = "LLM Said NO (Remains in Original)"
    
    else:
        queries[-1]["resulting_action"] = "LLM Said YES (Remains in Original)"
    
    return (doc_index, current_assignment, new_assignment, queries)

def cluster_via_correction(
    dataset_name: str,
    documents: List[str],
    features: np.ndarray,
    initial_assignments: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    llm_service: LLMService,
    correction_prompt: str,
    k_low_confidence: int,
    num_candidates: int,
    queries_output_path: str = "correction_queries_output.csv"
) -> np.ndarray:
    """Perform cluster assignment correction using LLM queries.
    
    Args:
        dataset_name: Name of dataset being processed
        documents: List of document texts
        features: Document embeddings
        initial_assignments: Starting cluster assignments
        labels: Ground truth labels
        n_clusters: Number of clusters
        llm_service: LLM service instance
        correction_prompt: Template for correction prompts
        k_low_confidence: Number of low-confidence points to correct
        num_candidates: Number of alternative clusters to consider
        queries_output_path: Path to save query logs
        
    Returns:
        Corrected cluster assignments
    """
    if (not llm_service or not llm_service.is_available() or 
        len(documents) != features.shape[0] or 
        len(documents) != len(labels) or
        k_low_confidence <= 0 or
        num_candidates <= 0):
        return initial_assignments

    corrected = np.copy(initial_assignments)
    centroids, reps = find_cluster_info(features, corrected, documents, n_clusters)
    low_conf_indices = identify_low_confidence_points(features, corrected, centroids, n_clusters, k_low_confidence)

    if not low_conf_indices:
        return corrected

    template = ChatPromptTemplate.from_template(correction_prompt)
    all_queries = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(
                process_low_confidence_point,
                i, documents[i], corrected[i], features, centroids, reps,
                n_clusters, llm_service, template, num_candidates, documents
            ): i for i in low_conf_indices if 0 <= corrected[i] < n_clusters
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Correcting"):
            doc_idx, old, new, queries = future.result()
            if new != old:
                corrected[doc_idx] = new
            all_queries.extend(queries)

    # Save queries and metrics
    if all_queries:
        pd.DataFrame(all_queries).to_csv(queries_output_path, index=False)
    
    metrics = calculate_clustering_metrics(labels, corrected, n_clusters)
    metrics_data = {'Dataset': dataset_name, 'Method': 'LLM Correction', **metrics}
    
    pd.DataFrame([metrics_data]).to_csv(
        METRICS_CSV_PATH,
        mode='a',
        header=not os.path.exists(METRICS_CSV_PATH),
        index=False
    )

    return corrected