import numpy as np
import pandas as pd
import os
from typing import List
from src.config import OPENAI_API_KEY, DATA_CACHE_PATH, KP_PROMPT_TEMPLATE
from src.data import load_dataset
from src.llm_service import LLMService
from src.clustering_methods.keyphrase_expansion import cluster_via_keyphrase_expansion
from src.metrics import calculate_clustering_metrics

METRICS_CSV_PATH = "clustering_metrics_results.csv"

def run_keyphrase_expansion_experiment(dataset_name: str) -> None:
    """Run the complete keyphrase expansion clustering experiment pipeline.
    
    Args:
        dataset_name: Name of dataset to process (e.g. 'tweet', 'bank77')
        
    Pipeline steps:
    1. Initialize LLM service with API key
    2. Load dataset with embeddings 
    3. Run keyphrase expansion clustering
    4. Evaluate and save results for all clustering variants
    5. Save metrics to CSV
    
    Returns:
        None (results are saved to CSV files)
    """
    print("\n--- Running Keyphrase Expansion Experiment ---")
    
    # Initialize LLM service
    llm_service = LLMService(OPENAI_API_KEY)
    if not llm_service.is_available():
        print("LLM Service unavailable. Exiting.")
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

    # Run keyphrase expansion clustering
    output_csv = f"{dataset_name}_keyphrase_expansions_output.csv"
    results = cluster_via_keyphrase_expansion(
        docs, features, n_clusters, llm_service, 
        KP_PROMPT_TEMPLATE, output_csv
    )

    # Evaluate and save metrics for each method
    metrics_data = []
    for method, assignments in results.items():
        status = "Success" if assignments is not None else "Failed"
        metrics = calculate_clustering_metrics(labels, assignments, n_clusters) if assignments is not None else {}
        
        metrics_data.append({
            'Dataset': dataset_name,
            'Method': f'Keyphrase Expansion - {method}',
            'Status': status,
            **metrics
        })

    # Save all metrics
    _save_metrics(metrics_data)

def _save_metrics(metrics_data: List[dict]) -> None:
    """Helper function to save metrics to CSV."""
    try:
        df = pd.DataFrame(metrics_data)
        df.to_csv(
            METRICS_CSV_PATH,
            mode='a',
            header=not os.path.exists(METRICS_CSV_PATH),
            index=False
        )
        print(f"Metrics saved to {METRICS_CSV_PATH}")
    except Exception as e:
        print(f"Error saving metrics: {e}")

if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "tweet"
    run_keyphrase_expansion_experiment(dataset)