import numpy as np
import os

# Import modules from src
from src.config import (
    OPENAI_API_KEY, DATA_CACHE_PATH, DEFAULT_N_CLUSTERS,
    CORRECTION_PROMPT_TEMPLATE, CORRECTION_K_LOW_CONFIDENCE,
    CORRECTION_NUM_CANDIDATE_CLUSTERS
)
from src.data import load_dataset
from src.llm_service import LLMService
from src.baselines import run_naive_kmeans # Need baseline for initial assignments
from src.clustering_methods.clustering_correction import correct_clustering_with_llm

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
        return

    llm_service = LLMService(api_key)
    if not llm_service.is_available():
        print("LLM Service could not be initialized. Exiting.")
        return

    # Get the embedding model instance from the service to pass to data loading
    embedding_model_instance = llm_service.get_embedding_model()
    if embedding_model_instance is None:
         print("Embedding model not available from LLM Service. Exiting.")
         return

    # --- Load Data ---
    # Pass the embedding model to load_dataset for consistent embeddings
    features, labels, documents = load_dataset(cache_path=DATA_CACHE_PATH, embedding_model=embedding_model_instance)

    if features.size == 0 or labels.size == 0 or not documents:
        print("Data loading failed or produced no data. Cannot proceed.")
        return

    n_clusters = len(np.unique(labels))
    print(f"Target number of clusters (from true labels): {n_clusters}")

    # --- Run Initial Clustering (Naive KMeans) for Correction ---
    print("\nRunning Naive KMeans for initial assignments...")
    # Note: Using features loaded with OpenAI embeddings
    naive_assignments = run_naive_kmeans(features, n_clusters)

    if naive_assignments is None:
        print("\nNaive KMeans failed. Cannot proceed with Clustering Correction.")
        return

    # --- Run LLM Method 3: Clustering Correction ---
    print(f"\nRunning Method 3: Clustering Correction...")
    corrected_assignments = cluster_via_clustering_correction(
        documents, features, naive_assignments, n_clusters, llm_service,
        CORRECTION_PROMPT_TEMPLATE,
        k_low_confidence=CORRECTION_K_LOW_CONFIDENCE,
        num_candidate_clusters=CORRECTION_NUM_CANDIDATE_CLUSTERS
    )

    # --- Evaluate and Report ---
    if corrected_assignments is not None:
        if cluster_acc is not None:
            correction_accuracy = cluster_acc(np.array(corrected_assignments), np.array(labels))
            print(f"\nMethod 3 (Clustering Correction) Accuracy: {correction_accuracy}")
        else:
            print("\nMethod 3 (Clustering Correction) completed, but evaluation utility is missing.")
    else:
        print("\nMethod 3 (Clustering Correction) failed.")


if __name__ == "__main__":
    run_clustering_correction_experiment()