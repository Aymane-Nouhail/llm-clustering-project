import numpy as np
import os
import pandas as pd
# Import modules from src
from src.config import OPENAI_API_KEY, DATA_CACHE_PATH, DEFAULT_N_CLUSTERS
from src.data import load_dataset
from src.llm_service import LLMService # Needed to get the embedding model for load_dataset
from src.baselines import run_naive_kmeans
from src.metrics import calculate_clustering_metrics

# Define the path for the metrics CSV file
METRICS_CSV_PATH = "clustering_metrics_results.csv"


def run_naive_baseline_experiment(dataset_name="tweet"):
    print(f"\n--- Running Naive KMeans Baseline Experiment on {dataset_name} dataset ---")
    # --- Configuration and Setup ---
    api_key = OPENAI_API_KEY
    if not api_key:
        print("OpenAI API Key not found.")
        return
    llm_service = LLMService(api_key)
    if not llm_service.is_available():
         print("LLM Service could not be initialized.")
         return
    embedding_model_instance = llm_service.get_embedding_model()
    if embedding_model_instance is None:
         print("Embedding model not available.")
         return
    # --- Load Data ---
    print(f"\nLoading data and embeddings for dataset: {dataset_name}...")
    features, labels, documents = load_dataset(dataset_name, cache_path=DATA_CACHE_PATH, embedding_model=embedding_model_instance)
    if features.size == 0 or labels.size == 0 or not documents:
        print("Data loading failed.")
        return
    labels_np = np.array(labels)
    n_clusters = len(np.unique(labels_np))
    print(f"Target number of clusters: {n_clusters}")
    # --- Run Naive Baseline ---
    print("\nRunning Naive KMeans...")
    naive_assignments = run_naive_kmeans(features, n_clusters)
    # --- Evaluate and Report ---
    method_status = "Failed" # Default status
    if naive_assignments is not None:
        assignments_np = np.array(naive_assignments)
        # Calculate metrics using the helper function that handles mapping
        # Note: calculate_clustering_metrics assumes naive_assignments has the same length as labels
        metrics = calculate_clustering_metrics(labels_np, assignments_np, n_clusters)
    # Print metrics
        if metrics["Accuracy"] is not None:
            print(f"\n--- Naive KMeans Baseline Results for {dataset_name} dataset ---")
            print(f"Accuracy: {metrics['Accuracy']}")
            
            for metric_name in ["Precision", "Recall", "Macro_F1", "Micro_F1", "NMI", "ARI"]:
                if metrics[metric_name] is not None:
                    # Add special suffix for F1 and precision/recall metrics
                    suffix = " (on valid subset)" if metric_name in ["Precision", "Recall", "Macro_F1", "Micro_F1"] else ""
                    # Normalize metric name for display
                    display_name = metric_name.replace("_", " ")
                    print(f"{display_name}{suffix}: {metrics[metric_name]}")
            
            # Determine status based on metrics completeness
            method_status = "Success" if all(v is not None for v in metrics.values()) else "Completed (Partial Eval)"
    else:
        print("\nNaive KMeans baseline failed.")
    # --- Save Metrics to CSV ---
    metrics_data = {
        'Dataset': dataset_name,
        'Method': 'Naive KMeans',
        'Status': method_status,
        **metrics
    }
    try:
        df_metrics = pd.DataFrame([metrics_data])
        if not os.path.exists(METRICS_CSV_PATH):
            df_metrics.to_csv(METRICS_CSV_PATH, index=False, mode='w', header=True)
            print(f"\nCreated {METRICS_CSV_PATH} and saved metrics.")
        else:
            df_metrics.to_csv(METRICS_CSV_PATH, index=False, mode='a', header=False)
            print(f"\nAppended metrics to {METRICS_CSV_PATH}.")
    except Exception as e:
        print(f"\nError saving metrics to CSV: {e}")
if __name__ == "__main__":
    # You can specify the dataset name when calling the script
    import sys
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        run_naive_baseline_experiment(dataset_name)
    else:
        # Default to "tweet" dataset if none specified
        run_naive_baseline_experiment("tweet")