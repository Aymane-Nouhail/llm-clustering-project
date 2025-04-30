import numpy as np
import os

# Import modules from src
from src.config import OPENAI_API_KEY, DATA_CACHE_PATH, DEFAULT_N_CLUSTERS
from src.data import load_dataset
from src.llm_service import LLMService # Needed to get the embedding model for load_dataset
from src.baselines import run_naive_kmeans

# Import evaluation utility (assuming it's from few_shot_clustering)
try:
    from few_shot_clustering.eval_utils import cluster_acc
except ImportError:
    print("Warning: few_shot_clustering.eval_utils not found. Evaluation will not be possible.")
    cluster_acc = None


def run_naive_baseline_experiment():
    print("\n--- Running Naive KMeans Baseline Experiment ---")

    # --- Configuration and Setup ---
    api_key = OPENAI_API_KEY # Get API key from config (loads from env)
    if not api_key:
        print("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable or in a .env file.")
        return

    llm_service = LLMService(api_key)
    # Although Naive KMeans itself doesn't call the LLM directly,
    # the data loading might require the embedding model if not cached.
    if not llm_service.is_available():
         print("LLM Service could not be initialized. Embedding model might be needed for data loading. Exiting.")
         return

    # Get the embedding model instance from the service to pass to data loading
    embedding_model_instance = llm_service.get_embedding_model()
    if embedding_model_instance is None:
         print("Embedding model not available from LLM Service. Needed for data loading if cache is not found. Exiting.")
         return


    # --- Load Data ---
    # This will load data and embeddings. If DATA_CACHE_PATH exists and is valid,
    # it will load embeddings from there. Otherwise, it will use embedding_model_instance
    # to generate them (involving embedding API calls, but NOT generation API calls).
    print("\nLoading data and embeddings...")
    features, labels, documents = load_dataset(cache_path=DATA_CACHE_PATH, embedding_model=embedding_model_instance)

    if features.size == 0 or labels.size == 0 or not documents:
        print("Data loading failed or produced no data. Cannot proceed.")
        return

    # Determine the number of clusters from the true labels
    n_clusters = len(np.unique(labels))
    print(f"Target number of clusters (from true labels): {n_clusters}")


    # --- Run Naive Baseline ---
    print("\nRunning Naive KMeans on loaded embeddings...")
    naive_assignments = run_naive_kmeans(features, n_clusters)

    # --- Evaluate and Report ---
    if naive_assignments is not None:
        if cluster_acc is not None:
            naive_accuracy = cluster_acc(np.array(naive_assignments), np.array(labels))
            print(f"\nNaive KMeans Baseline Accuracy: {naive_accuracy}")
        else:
            print("\nNaive KMeans completed, but evaluation utility is missing.")
    else:
        print("\nNaive KMeans baseline failed.")


if __name__ == "__main__":
    run_naive_baseline_experiment()