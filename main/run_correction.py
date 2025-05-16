import numpy as np
from src.config import (
    OPENAI_API_KEY, DATA_CACHE_PATH,
    CORRECTION_PROMPT_TEMPLATE, CORRECTION_K_LOW_CONFIDENCE,
    CORRECTION_NUM_CANDIDATE_CLUSTERS
)
from src.data import load_dataset
from src.llm_service import LLMService
from src.clustering_methods.clustering_correction import cluster_via_correction
from src.baselines import run_naive_kmeans

def run_clustering_correction_experiment(dataset_name: str) -> None:
    """Run the complete clustering correction experiment pipeline.
    
    Args:
        dataset_name: Name of dataset to process (e.g. 'tweet', 'bank77')
        
    Pipeline steps:
    1. Initialize LLM service with API key
    2. Load dataset with embeddings
    3. Run naive KMeans for initial clustering
    4. Perform LLM-based clustering correction
    5. Save results and metrics
    
    Returns:
        None (results are saved to CSV files)
    """
    print("\n--- Running Clustering Correction Experiment ---")
    
    # Initialize LLM service
    llm_service = LLMService(OPENAI_API_KEY)
    if not llm_service.is_available():
        print("LLM Service unavailable. Logging failure status.")
        _log_failed_run(dataset_name, llm_service)
        return

    # Load data with embeddings
    features, labels, docs = load_dataset(
        dataset_name=dataset_name,
        cache_path=DATA_CACHE_PATH,
        embedding_model=llm_service.get_embedding_model()
    )

    # Determine cluster count from labels
    n_clusters = len(np.unique(labels))
    print(f"Target clusters: {n_clusters}")

    # Get initial KMeans assignments
    initial_assignments = run_naive_kmeans(features, n_clusters)
    if initial_assignments is None:
        print("Initial clustering failed. Logging failure status.")
        _log_failed_run(dataset_name, llm_service, n_clusters)
        return

    # Run clustering correction
    output_csv = f"{dataset_name}_correction_queries_output.csv"
    corrected = cluster_via_correction(
        dataset_name, docs, features, initial_assignments, np.array(labels),
        n_clusters, llm_service, CORRECTION_PROMPT_TEMPLATE,
        CORRECTION_K_LOW_CONFIDENCE, CORRECTION_NUM_CANDIDATE_CLUSTERS,
        output_csv
    )

    print("\nCompleted." if corrected is not None else "\nFailed.")

def _log_failed_run(dataset_name: str, llm_service: LLMService, n_clusters: int = 0) -> None:
    """Helper to log failed runs with dummy data."""
    cluster_via_correction(
        dataset_name, [], np.array([]), np.array([]), n_clusters,
        llm_service, "", 0, 0
    )

if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "tweet"
    run_clustering_correction_experiment(dataset)