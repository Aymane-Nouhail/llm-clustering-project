import numpy as np
import os

# Import modules from src
from src.config import (
    OPENAI_API_KEY, DATA_CACHE_PATH, DEFAULT_N_CLUSTERS,
    PC_PROMPT_TEMPLATE, PC_NUM_PAIRS_TO_QUERY, PC_CONSTRAINT_SELECTION_STRATEGY
)
from src.data import load_dataset
from src.llm_service import LLMService
from src.clustering_methods.pairwise_constraints import cluster_via_pairwise_constraints

# Import evaluation utility (assuming it's from few_shot_clustering)
try:
    from few_shot_clustering.eval_utils import cluster_acc
except ImportError:
    print("Warning: few_shot_clustering.eval_utils not found. Evaluation will not be possible.")
    cluster_acc = None

# Check if PCKMeans is available (required by this method)
try:
    from active_semi_supervised_clustering.pairwise_constraints import PCKMeans
except ImportError:
    print("PCKMeans from 'active-semi-supervised-clustering' not found.")
    print("The pairwise constraints method cannot run without this library.")
    PCKMeans = None # Define as None if not available


def run_pairwise_constraints_experiment():
    print("\n--- Running Pairwise Constraints Experiment ---")

    # Check if the required library is available first
    if PCKMeans is None:
         print("Skipping Pairwise Constraints Method: PCKMeans library not found.")
         return


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

    # --- Run LLM Method 2: Pairwise Constraints ---
    print(f"\nRunning Method 2: Pairwise Constraints...")
    pairwise_assignments = cluster_via_pairwise_constraints(
        documents, features, n_clusters, llm_service, PC_PROMPT_TEMPLATE,
        num_pairs_to_query=PC_NUM_PAIRS_TO_QUERY,
        constraint_selection_strategy=PC_CONSTRAINT_SELECTION_STRATEGY
    )

    # --- Evaluate and Report ---
    if pairwise_assignments is not None:
        if cluster_acc is not None:
            pairwise_accuracy = cluster_acc(np.array(pairwise_assignments), np.array(labels))
            print(f"\nMethod 2 (Pairwise Constraints) Accuracy: {pairwise_accuracy}")
        else:
             print("\nMethod 2 (Pairwise Constraints) completed, but evaluation utility is missing.")
    else:
        print("\nMethod 2 (Pairwise Constraints) failed.")


if __name__ == "__main__":
    run_pairwise_constraints_experiment()