import numpy as np
import os
import pandas as pd
from src.config import OPENAI_API_KEY, DATA_CACHE_PATH
from src.data import load_dataset
from src.llm_service import LLMService
from src.baselines import run_naive_kmeans
from src.metrics import calculate_clustering_metrics

METRICS_PATH = "clustering_metrics_results.csv"

def run_naive_baseline_experiment(dataset_name="tweet"):
    print(f"\n--- Running Naive KMeans on {dataset_name} dataset ---")
    
    # Validate environment and setup
    if not OPENAI_API_KEY or not LLMService(OPENAI_API_KEY).is_available():
        print("LLM Service unavailable." if OPENAI_API_KEY else "OpenAI API Key not found.")
        return
    
    # Get embeddings and load data
    embeddings, true_labels, documents = load_dataset(
        dataset_name, DATA_CACHE_PATH, LLMService(OPENAI_API_KEY).get_embedding_model()
    )

    # Run clustering
    cluster_count = len(np.unique(true_labels))
    print(f"Identified {cluster_count} target clusters\nClustering with KMeans...")
    if (assignments := run_naive_kmeans(embeddings, cluster_count)) is None:
        print("Clustering failed.")
        return
    
    # Process results
    metrics = calculate_clustering_metrics(true_labels, assignments, cluster_count)
    if metrics["Accuracy"] is not None:
        print(f"\n--- Results for {dataset_name} ---")
        for metric, value in metrics.items():
            if value is not None:
                suffix = " (valid subset)" if metric in ["Precision", "Recall", "Macro_F1", "Micro_F1"] else ""
                print(f"{metric.replace('_', ' ')}{suffix}: {value}")
    
    # Save metrics
    results = {'Dataset': dataset_name, 'Method': 'Naive KMeans', 
               'Status': 'Success' if all(metrics.values()) else 'Partial', **metrics}
    try:
        exists = os.path.exists(METRICS_PATH)
        pd.DataFrame([results]).to_csv(METRICS_PATH, index=False, mode='a' if exists else 'w', header=not exists)
        print(f"\n{'Updated' if exists else 'Created'} metrics file")
    except Exception as e:
        print(f"\nFailed to save metrics: {e}")

if __name__ == "__main__":
    import sys
    run_naive_baseline_experiment(sys.argv[1] if len(sys.argv) > 1 else "tweet")