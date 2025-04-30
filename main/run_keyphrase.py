import numpy as np
import os

# Import modules from src
from src.config import (
    OPENAI_API_KEY, DATA_CACHE_PATH, DEFAULT_N_CLUSTERS,
    KP_PROMPT_TEMPLATE
)
from src.data import load_dataset
from src.llm_service import LLMService
from src.clustering_methods.keyphrase_expansion import cluster_via_keyphrase_expansion

# Import evaluation utility (assuming it's from few_shot_clustering)
try:
    from few_shot_clustering.eval_utils import cluster_acc
except ImportError:
    print("Warning: few_shot_clustering.eval_utils not found. Evaluation will not be possible.")
    cluster_acc = None


def run_keyphrase_expansion_experiment():
    print("\n--- Running Keyphrase Expansion Experiment ---")

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


    # --- Run LLM Method 1: Keyphrase Expansion ---
    print(f"\nRunning Method 1: Keyphrase Expansion...")
    keyphrase_assignments = cluster_via_keyphrase_expansion(
        documents, features, n_clusters, llm_service, KP_PROMPT_TEMPLATE
    )

    # --- Evaluate and Report ---
    if keyphrase_assignments is not None:
        if cluster_acc is not None:
            keyphrase_accuracy = cluster_acc(np.array(keyphrase_assignments), np.array(labels))
            print(f"\nMethod 1 (Keyphrase Expansion) Accuracy: {keyphrase_accuracy}")
        else:
            print("\nMethod 1 (Keyphrase Expansion) completed, but evaluation utility is missing.")
    else:
        print("\nMethod 1 (Keyphrase Expansion) failed.")


if __name__ == "__main__":
    run_keyphrase_expansion_experiment()