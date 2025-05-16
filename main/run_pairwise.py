import numpy as np
from src.config import (
    OPENAI_API_KEY, DATA_CACHE_PATH,
    PC_PROMPT_TEMPLATE, PC_NUM_PAIRS_TO_QUERY, PC_CONSTRAINT_SELECTION_STRATEGY
)
from src.data import load_dataset
from src.llm_service import LLMService
from src.clustering_methods.pairwise_constraints import cluster_via_pairwise_constraints

def run_pairwise_constraints_experiment(dataset_name="tweet"):
    print("\n--- Running Pairwise Constraints Experiment ---")
    
    # Setup and data loading
    if not OPENAI_API_KEY:
        print("OpenAI API Key not found.")
        return
        
    llm_service = LLMService(OPENAI_API_KEY)
    features, labels, documents = load_dataset(
        dataset_name, 
        DATA_CACHE_PATH, 
        llm_service.get_embedding_model(),
        max_samples_per_class=500
    )

    # Run clustering
    n_clusters = len(np.unique(labels))
    print(f"\nTarget clusters: {n_clusters}\nRunning Pairwise Constraints...")
    
    pairwise_assignments = cluster_via_pairwise_constraints(
        dataset_name, documents, features, np.array(labels), n_clusters, llm_service,
        PC_PROMPT_TEMPLATE,
        num_pairs_to_query=PC_NUM_PAIRS_TO_QUERY,
        constraint_selection_strategy=PC_CONSTRAINT_SELECTION_STRATEGY,
        pairwise_queries_output_csv_path=f"{dataset_name}_pairwise_queries_output.csv"
    )

    print("\nPairwise Constraints method " + 
          ("completed. Metrics and queries saved to CSV." if pairwise_assignments is not None 
           else "failed or skipped. Status saved to CSV."))

if __name__ == "__main__":
    import sys
    run_pairwise_constraints_experiment(sys.argv[1] if len(sys.argv) > 1 else "tweet")