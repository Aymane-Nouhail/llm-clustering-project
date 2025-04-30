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
    PC_PROMPT_TEMPLATE, PC_NUM_PAIRS_TO_QUERY, PC_CONSTRAINT_SELECTION_STRATEGY
)
from src.data import load_dataset
from src.llm_service import LLMService
# Import the updated function from src
from src.clustering_methods.pairwise_constraints import cluster_via_pairwise_constraints

# Import evaluation utility (assuming it's from few_shot_clustering)
try:
    from few_shot_clustering.eval_utils import cluster_acc
except ImportError:
    print("Warning: few_shot_clustering.eval_utils not found. Evaluation will not be possible.")
    cluster_acc = None

# Check if PCKMeans is available (required by this method)
try:
    from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
    print("PCKMeans from 'active-semi-clustering' imported successfully.")
except ImportError:
    print("PCKMeans from 'active-semi-clustering' not found.")
    print("The pairwise constraints method cannot run without this library.")
    PCKMeans = None # Define as None if not available


def run_pairwise_constraints_experiment():
    print("\n--- Running Pairwise Constraints Experiment ---")

    # Check if the required library is available first
    if PCKMeans is None:
         print("Skipping Pairwise Constraints Method: PCKMeans library not found.")
         # This method saves its status to CSV internally even if skipped
         # Call the function with None assignments to trigger the skipped status save
         # Need dummy data structure to avoid errors in the called function if it tries to access docs/features/labels
         dummy_docs = []
         dummy_features = np.array([])
         dummy_labels = np.array([])
         # Define dummy query output path for the skipped case
         dummy_query_csv_path = "dummy_pairwise_queries_output.csv"
         cluster_via_pairwise_constraints(
             dummy_docs, dummy_features, dummy_labels, None, # Pass dummy labels, None for llm_service
             "", 0, "", # Dummy prompt, num_pairs, strategy
             pairwise_queries_output_csv_path=dummy_query_csv_path # Pass dummy query path
         )
         # Optionally clean up dummy file if it was created
         if os.path.exists(dummy_query_csv_path):
             os.remove(dummy_query_csv_path)
         return


    # --- Configuration and Setup ---
    api_key = OPENAI_API_KEY # Get API key from config (loads from env)
    if not api_key:
        print("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable or in a .env file.")
        # Save failed status to CSV internally
        # Need dummy data structure to avoid errors in the called function if it tries to access docs/features/labels
        dummy_docs = []
        dummy_features = np.array([])
        dummy_labels = np.array([])
        # Define dummy query output path for the failed case
        dummy_query_csv_path = "dummy_pairwise_queries_output.csv"
        cluster_via_pairwise_constraints(
            dummy_docs, dummy_features, dummy_labels, None, # Pass dummy labels, None for llm_service
            "", 0, "", # Dummy prompt, num_pairs, strategy
            pairwise_queries_output_csv_path=dummy_query_csv_path # Pass dummy query path
        )
         # Optionally clean up dummy file if it was created
        if os.path.exists(dummy_query_csv_path):
            os.remove(dummy_query_csv_path)
        return


    llm_service = LLMService(api_key) # Creates LLMService instance or it raises

    # This check *uses* the instance:
    if not llm_service.is_available():
        print("LLM Service could not be initialized or is not available. Exiting.")
        # Save failed status to CSV internally
        # Need dummy data structure to avoid errors in the called function if it tries to access docs/features/labels
        dummy_docs = []
        dummy_features = np.array([])
        dummy_labels = np.array([])
        # Define dummy query output path for the failed case
        dummy_query_csv_path = "dummy_pairwise_queries_output.csv"
        cluster_via_pairwise_constraints(
            dummy_docs, dummy_features, dummy_labels, llm_service, # Pass dummy labels, llm_service instance
            "", 0, "", # Dummy prompt, num_pairs, strategy
            pairwise_queries_output_csv_path=dummy_query_csv_path # Pass dummy query path
        )
         # Optionally clean up dummy file if it was created
        if os.path.exists(dummy_query_csv_path):
            os.remove(dummy_query_csv_path)
        return


    # Get the embedding model instance from the service to pass to data loading
    embedding_model_instance = llm_service.get_embedding_model()
    if embedding_model_instance is None:
         print("Embedding model not available from LLM Service.")
         # Save failed status to CSV internally
         # Need dummy data structure to avoid errors in the called function if it tries to access docs/features/labels
         dummy_docs = []
         dummy_features = np.array([])
         dummy_labels = np.array([])
         # Define dummy query output path for the failed case
         dummy_query_csv_path = "dummy_pairwise_queries_output.csv"
         cluster_via_pairwise_constraints(
             dummy_docs, dummy_features, dummy_labels, llm_service, # Pass dummy labels, llm_service instance
             "", 0, "", # Dummy prompt, num_pairs, strategy
             pairwise_queries_output_csv_path=dummy_query_csv_path # Pass dummy query path
         )
         # Optionally clean up dummy file if it was created
         if os.path.exists(dummy_query_csv_path):
            os.remove(dummy_query_csv_path)
         return


    # --- Load Data ---
    print("\nLoading data and embeddings...")
    # Pass the embedding model to load_dataset for consistent embeddings
    features, labels, documents = load_dataset(cache_path=DATA_CACHE_PATH, embedding_model=embedding_model_instance)

    if features.size == 0 or len(labels) == 0 or not documents:
        print("Data loading failed or produced no data. Cannot proceed.")
        # Save failed status to CSV internally
        # Need dummy data structure to trigger save logic in the called function
        dummy_docs = []
        dummy_features = np.array([])
        dummy_labels = np.array([])
        # Define dummy query output path for the failed case
        dummy_query_csv_path = "dummy_pairwise_queries_output.csv"
        cluster_via_pairwise_constraints(
            dummy_docs, dummy_features, dummy_labels, llm_service, # Pass dummy labels (empty), llm_service instance
            "", 0, "", # Dummy prompt, num_pairs, strategy
            pairwise_queries_output_csv_path=dummy_query_csv_path # Pass dummy query path
        )
         # Optionally clean up dummy file if it was created
        if os.path.exists(dummy_query_csv_path):
            os.remove(dummy_query_csv_path)
        return

    # Ensure labels are numpy array for scikit-learn metrics
    labels_np = np.array(labels)

    # Determine the number of clusters from the true labels
    n_clusters = len(np.unique(labels_np))
    print(f"Target number of clusters (from true labels): {n_clusters}")

    # --- Run LLM Method 2: Pairwise Constraints ---
    print(f"\nRunning Method 2: Pairwise Constraints...")
    # Define the desired CSV output path for pairwise queries generated by this method
    PAIRWISE_QUERIES_CSV = "pairwise_queries_output.csv"
    # Pass the true labels (labels_np) and the query output path explicitly to the function
    pairwise_assignments = cluster_via_pairwise_constraints(
        documents, features, labels_np, n_clusters, llm_service, # Pass labels_np, llm_service instance
        PC_PROMPT_TEMPLATE,
        num_pairs_to_query=PC_NUM_PAIRS_TO_QUERY,
        constraint_selection_strategy=PC_CONSTRAINT_SELECTION_STRATEGY,
        pairwise_queries_output_csv_path=PAIRWISE_QUERIES_CSV # Pass the path here
    )

    # --- Report Final Status (Metrics saved internally) ---
    if pairwise_assignments is not None:
        print("\nPairwise Constraints method completed. Metrics and queries saved to CSV.")
    else:
        print("\nPairwise Constraints method failed or skipped. Status saved to CSV.")


if __name__ == "__main__":
    run_pairwise_constraints_experiment()