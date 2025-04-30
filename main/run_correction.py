import numpy as np
import os
import pandas as pd # Import pandas (needed because the called function uses it)
# Need f1_score and linear_sum_assignment imports as the called function uses them
# It's good practice to import them here as well, although not strictly necessary
# if they are only used inside calculate_clustering_metrics within the other file.
# from sklearn.metrics import f1_score
# from scipy.optimize import linear_sum_assignment

# Import modules from src
from src.config import (
    OPENAI_API_KEY, DATA_CACHE_PATH, DEFAULT_N_CLUSTERS,
    CORRECTION_PROMPT_TEMPLATE, CORRECTION_K_LOW_CONFIDENCE,
    CORRECTION_NUM_CANDIDATE_CLUSTERS
)
from src.data import load_dataset
from src.llm_service import LLMService
# Import the RENAMED function from src
from src.clustering_methods.clustering_correction import cluster_via_correction
from src.baselines import run_naive_kmeans


# Import evaluation utility (assuming it's from few_shot_clustering)
try:
    from few_shot_clustering.eval_utils import cluster_acc
except ImportError:
    print("Warning: few_shot_clustering.eval_utils not found. Evaluation will not be possible.")
    cluster_acc = None


def run_clustering_correction_experiment():
    print("\n--- Running Clustering Correction Experiment ---")

    # --- Configuration and Setup ---
    api_key = OPENAI_API_KEY # Get API key from config (loads from env)
    if not api_key:
        print("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable or in a .env file.")
        # The called function saves status to CSV, but needs dummy data
        dummy_docs = []
        dummy_features = np.array([])
        dummy_labels = np.array([])
        # Call the correction function with dummy data and None LLM service to log the skipped status
        cluster_via_correction(
            dummy_docs, dummy_features, dummy_labels, 0, # Pass dummy data (docs, features, labels, n_clusters)
            None, "", 0, 0 # Pass None for llm_service, dummy prompt, k, num_candidates
        )
        return

    # Initialize LLM Service
    llm_service = LLMService(api_key)
    if not llm_service.is_available():
        print("LLM Service could not be initialized or is not available. Exiting.")
        # The called function saves status to CSV, but needs dummy data
        dummy_docs = []
        dummy_features = np.array([])
        dummy_labels = np.array([])
        # Call the correction function with dummy data and the LLM service instance to log the failed status
        cluster_via_correction(
            dummy_docs, dummy_features, dummy_labels, 0, # Pass dummy data
            llm_service, "", 0, 0 # Pass the llm_service instance, dummy config
        )
        return

    # Get the embedding model instance from the service to pass to data loading
    embedding_model_instance = llm_service.get_embedding_model()
    if embedding_model_instance is None:
         print("Embedding model not available from LLM Service.")
         # The called function saves status to CSV, but needs dummy data
         dummy_docs = []
         dummy_features = np.array([])
         dummy_labels = np.array([])
         # Call the correction function with dummy data and the LLM service instance to log the failed status
         cluster_via_correction(
             dummy_docs, dummy_features, dummy_labels, 0, # Pass dummy data
             llm_service, "", 0, 0 # Pass the llm_service instance, dummy config
         )
         return


    # --- Load Data ---
    print("\nLoading data and embeddings...")
    # Pass the embedding model to load_dataset for consistent embeddings
    features, labels, documents = load_dataset(cache_path=DATA_CACHE_PATH, embedding_model=embedding_model_instance)

    if features.size == 0 or len(labels) == 0 or not documents:
        print("Data loading failed or produced no data. Cannot proceed.")
        # The called function saves status to CSV, but needs dummy data
        dummy_docs = []
        dummy_features = np.array([])
        dummy_labels = np.array([])
        # Call the correction function with dummy data to log the failed status
        cluster_via_correction(
            dummy_docs, dummy_features, dummy_labels, 0, # Pass dummy data (empty)
            llm_service, "", 0, 0 # Pass the llm_service instance, dummy config
        )
        return

    # Ensure labels are numpy array for scikit-learn metrics
    labels_np = np.array(labels)

    # Determine the number of clusters from the true labels
    n_clusters = len(np.unique(labels_np))
    print(f"Target number of clusters (from true labels): {n_clusters}")

    # --- Run Initial Clustering (Naive KMeans) for Correction ---
    print("\nRunning Naive KMeans for initial assignments...")
    # Note: Using features loaded with OpenAI embeddings
    naive_assignments = run_naive_kmeans(features, n_clusters)

    if naive_assignments is None:
        print("\nNaive KMeans failed. Cannot proceed with Clustering Correction.")
        # The called function saves status to CSV, but needs dummy data
        dummy_docs = []
        dummy_features = np.array([])
        dummy_labels = np.array([])
        # Call the correction function with dummy data to log the failed status
        cluster_via_correction(
            dummy_docs, dummy_features, dummy_labels, n_clusters, # Pass dummy data, but correct n_clusters
            llm_service, "", 0, 0 # Pass the llm_service instance, dummy config
        )
        return

    # --- Run LLM Method 3: Clustering Correction ---
    print(f"\nRunning Method 3: Clustering Correction...")
    # Pass the true labels (labels_np) and the initial_assignments explicitly to the function
    corrected_assignments = cluster_via_correction(
        documents, features, naive_assignments, labels_np, n_clusters, llm_service, # Pass all required data
        CORRECTION_PROMPT_TEMPLATE,
        k_low_confidence=CORRECTION_K_LOW_CONFIDENCE,
        num_candidate_clusters=CORRECTION_NUM_CANDIDATE_CLUSTERS
    )

    # --- Report Final Status (Metrics saved internally by cluster_via_correction) ---
    if corrected_assignments is not None:
        print("\nClustering Correction method completed. Metrics saved to CSV.")
    else:
        print("\nClustering Correction method failed or skipped. Status saved to CSV.")


if __name__ == "__main__":
    run_clustering_correction_experiment()
