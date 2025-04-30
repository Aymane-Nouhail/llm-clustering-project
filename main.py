import numpy as np
import os
# Optional: Load environment variables from .env file
# from dotenv import load_dotenv

# Import modules from src
from src.config import (
    OPENAI_API_KEY, DATA_CACHE_PATH, DEFAULT_N_CLUSTERS,
    KP_PROMPT_TEMPLATE, PC_PROMPT_TEMPLATE, CORRECTION_PROMPT_TEMPLATE,
    PC_NUM_PAIRS_TO_QUERY, PC_CONSTRAINT_SELECTION_STRATEGY,
    CORRECTION_K_LOW_CONFIDENCE, CORRECTION_NUM_CANDIDATE_CLUSTERS
)
from src.data import load_dataset # Import the modified load_dataset
from src.llm_service import LLMService
from src.baselines import run_naive_kmeans
from src.clustering_methods.keyphrase_expansion import cluster_via_keyphrase_expansion
from src.clustering_methods.pairwise_constraints import cluster_via_pairwise_constraints
from src.clustering_methods.clustering_correction import correct_clustering_with_llm

# Import evaluation utility (assuming it's from few_shot_clustering)
from few_shot_clustering.eval_utils import cluster_acc


def main():
    # # Optional: Load environment variables from .env file
    # load_dotenv()

    # --- Configuration and Setup ---
    api_key = OPENAI_API_KEY # Get API key from config (loads from env)
    if not api_key:
        print("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable or in a .env file.")
        return

    # Initialize LLM Service first to get the embedding model
    llm_service = LLMService(api_key)
    if not llm_service.is_available():
        print("LLM Service could not be initialized. Exiting.")
        return

    # Get the embedding model instance from the service
    embedding_model_instance = llm_service.get_embedding_model()
    if embedding_model_instance is None:
         print("Embedding model not available from LLM Service. Exiting.")
         return

    # --- Load Data ---
    # Pass the embedding model to load_dataset
    features, labels, documents = load_dataset(cache_path=DATA_CACHE_PATH, embedding_model=embedding_model_instance)

    if features.size == 0 or labels.size == 0 or not documents:
        print("Data loading failed or produced no data. Cannot proceed.")
        return

    n_clusters = len(np.unique(labels))
    print(f"Using n_clusters = {n_clusters} (from true labels) for clustering methods.")


    # --- Run Clustering Methods ---

    # 1. Naive Baseline
    # Note: The naive baseline will now also run on OpenAI embeddings
    # loaded by load_dataset.
    naive_assignments = run_naive_kmeans(features, n_clusters)
    if naive_assignments is not None:
        naive_accuracy = cluster_acc(np.array(naive_assignments), np.array(labels))
        print(f"\nNaive KMeans (Cosine Similarity) Accuracy: {naive_accuracy}")
    else:
        naive_accuracy = None
        print("\nNaive KMeans baseline failed.")


    # 2. LLM Method 1: Keyphrase Expansion
    keyphrase_assignments = cluster_via_keyphrase_expansion(
        documents, features, n_clusters, llm_service, KP_PROMPT_TEMPLATE
    )
    if keyphrase_assignments is not None:
        keyphrase_accuracy = cluster_acc(np.array(keyphrase_assignments), np.array(labels))
        print(f"\nMethod 1 (Keyphrase Expansion) Accuracy: {keyphrase_accuracy}")
    else:
        keyphrase_accuracy = None
        print("\nMethod 1 (Keyphrase Expansion) failed.")

    # 3. LLM Method 2: Pairwise Constraints
    # Requires active-semi-supervised-clustering library
    # This method also uses the original features (OpenAI embeddings)
    pairwise_assignments = cluster_via_pairwise_constraints(
        documents, features, n_clusters, llm_service, PC_PROMPT_TEMPLATE,
        num_pairs_to_query=PC_NUM_PAIRS_TO_QUERY,
        constraint_selection_strategy=PC_CONSTRAINT_SELECTION_STRATEGY
    )
    if pairwise_assignments is not None:
        pairwise_accuracy = cluster_acc(np.array(pairwise_assignments), np.array(labels))
        print(f"\nMethod 2 (Pairwise Constraints) Accuracy: {pairwise_accuracy}")
    else:
        pairwise_accuracy = None
        print("\nMethod 2 (Pairwise Constraints) failed (PCKMeans likely not available or failed).")


    # 4. LLM Method 3: Clustering Correction
    # Requires initial assignments - can use Naive KMeans assignments for this
    if naive_assignments is not None:
        print("\nUsing Naive KMeans assignments as initial assignments for Method 3.")
        initial_assignments_for_correction = naive_assignments
        # This method also uses the original features (OpenAI embeddings)
        correction_assignments = correct_clustering_with_llm(
            documents, features, initial_assignments_for_correction, n_clusters, llm_service,
            CORRECTION_PROMPT_TEMPLATE,
            k_low_confidence=CORRECTION_K_LOW_CONFIDENCE,
            num_candidate_clusters=CORRECTION_NUM_CANDIDATE_CLUSTERS
        )
        if correction_assignments is not None:
            correction_accuracy = cluster_acc(np.array(correction_assignments), np.array(labels))
            print(f"\nMethod 3 (Clustering Correction) Accuracy: {correction_accuracy}")
        else:
            correction_accuracy = None
            print("\nMethod 3 (Clustering Correction) failed.")
    else:
        correction_accuracy = None
        print("\nSkipping Method 3 (Clustering Correction) as initial assignments are not available.")

    # --- Final Summary ---
    print("\n--- Overall Results ---")
    print(f"Naive KMeans (Cosine Similarity) Accuracy: {naive_accuracy if naive_accuracy is not None else 'Failed'}")
    print(f"Method 1 (Keyphrase Expansion) Accuracy: {keyphrase_accuracy if keyphrase_accuracy is not None else 'Failed'}")
    print(f"Method 2 (Pairwise Constraints) Accuracy: {pairwise_accuracy if pairwise_accuracy is not None else 'Failed'}")
    print(f"Method 3 (Clustering Correction) Accuracy: {correction_accuracy if correction_accuracy is not None else 'Failed'}")


if __name__ == "__main__":
    main()