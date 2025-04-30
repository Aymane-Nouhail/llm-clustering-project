import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate
from src.llm_service import LLMService # Import LLM service
from typing import List, Tuple

# Try importing PCKMeans
try:
    from active_semi_supervised_clustering.pairwise_constraints import PCKMeans
except ImportError:
    print("PCKMeans from 'active-semi-supervised-clustering' not found.")
    print("The pairwise constraints method requires this library to run.")
    PCKMeans = None # Define as None if not available


def cluster_via_pairwise_constraints(
    documents: List[str],
    features: np.ndarray,
    n_clusters: int,
    llm_service: LLMService, # Accept LLMService instance
    pairwise_prompt_template: str,
    num_pairs_to_query: int,
    constraint_selection_strategy: str = 'random'
) -> np.ndarray | None:
    """
    Implements clustering via pseudo-oracle pairwise constraints (Section 2.2).

    Args:
        documents: List of original document texts.
        features: Original document embeddings (NumPy array).
        n_clusters: The number of clusters.
        llm_service: An initialized LLMService instance.
        pairwise_prompt_template: A string template for pairwise comparison.
                                  Should include placeholders for two document texts.
                                  Example: "Are these two texts similar? Text 1: {text1}\nText 2: {text2}\nRespond YES or NO."
        num_pairs_to_query: The number of pairs to query the LLM for constraints.
        constraint_selection_strategy: Strategy for selecting pairs ('random', 'similarity').
                                      'random': Select pairs randomly.
                                      'similarity': Select pairs based on initial feature similarity
                                                    (e.g., a mix of very similar and very dissimilar).

    Returns:
        A NumPy array of cluster assignments, or None if PCKMeans is not available or clustering fails.
    """
    print("\n--- Running Clustering via Pseudo-Oracle Pairwise Constraints ---")
    if PCKMeans is None:
        print("Cannot run Method 2: PCKMeans implementation is not available.")
        return None
    if not llm_service.is_available():
        print("LLMService is not available. Cannot run pairwise constraints.")
        return None


    n_samples = len(documents)
    must_link_constraints = []
    cannot_link_constraints = []

    # Define Langchain prompt template
    prompt_template = ChatPromptTemplate.from_template(pairwise_prompt_template)

    # 1. Select pairs based on strategy
    selected_pairs = []
    all_possible_pairs = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]

    if constraint_selection_strategy == 'random':
        if len(all_possible_pairs) > num_pairs_to_query:
            selected_pairs = random.sample(all_possible_pairs, num_pairs_to_query)
        else:
            selected_pairs = all_possible_pairs
            num_pairs_to_query = len(selected_pairs) # Adjust if fewer pairs than requested
    elif constraint_selection_strategy == 'similarity':
         # Simple similarity-based selection: get some most similar and some most dissimilar pairs
         similarity_matrix = cosine_similarity(features)
         # Flatten and sort indices to get pairs from most similar to least similar
         # Exclude self-similarity (diagonal) and duplicate pairs
         similarity_pairs = []
         for i in range(n_samples):
             for j in range(i + 1, n_samples):
                 similarity_pairs.append(((i, j), similarity_matrix[i, j]))

         similarity_pairs.sort(key=lambda item: item[1], reverse=True) # Sort by similarity descending

         # Take some most similar (potential must-links) and some least similar (potential cannot-links)
         num_similar = num_pairs_to_query // 2
         num_dissimilar = num_pairs_to_query - num_similar

         selected_pairs.extend([pair for pair, sim in similarity_pairs[:num_similar]])
         selected_pairs.extend([pair for pair, sim in similarity_pairs[-num_dissimilar:]])

         # Remove potential duplicates if any overlap in selection
         selected_pairs = list(set(selected_pairs))
         # Resample if not enough unique pairs found (unlikely with enough data)
         while len(selected_pairs) < num_pairs_to_query and len(selected_pairs) < len(all_possible_pairs):
             remaining_pairs = list(set(all_possible_pairs) - set(selected_pairs))
             selected_pairs.extend(random.sample(remaining_pairs, min(num_pairs_to_query - len(selected_pairs), len(remaining_pairs))))

         selected_pairs = selected_pairs[:num_pairs_to_query] # Ensure we don't exceed the requested number

    else:
        print(f"Unknown constraint selection strategy: {constraint_selection_strategy}. Using 'random'.")
        if len(all_possible_pairs) > num_pairs_to_query:
            selected_pairs = random.sample(all_possible_pairs, num_pairs_to_query)
        else:
            selected_pairs = all_possible_pairs
            num_pairs_to_query = len(selected_pairs)


    print(f"Selected {len(selected_pairs)} pairs using '{constraint_selection_strategy}' strategy.")

    # 2. Query LLM for constraints for selected pairs
    print(f"Querying LLM for {len(selected_pairs)} pairs...")
    for i, (idx1, idx2) in enumerate(selected_pairs):
        doc1 = documents[idx1]
        doc2 = documents[idx2]

        prompt = ChatPromptTemplate.from_template(pairwise_prompt_template).format(text1=doc1, text2=doc2)
        llm_response_text = llm_service.get_chat_completion(prompt).strip().upper()
        # print(f"  Querying pair {i+1}/{len(selected_pairs)}. LLM response: {llm_response_text}") # Optional: print response


        if llm_response_text == "YES":
            must_link_constraints.append((idx1, idx2))
        elif llm_response_text == "NO":
            cannot_link_constraints.append((idx1, idx2))
        else:
            # Handle ambiguous or error responses - e.g., skip or log
            print(f"  Warning: Ambiguous LLM response for pair ({idx1}, {idx2}): {llm_response_text}")


    print(f"Generated {len(must_link_constraints)} must-link constraints and {len(cannot_link_constraints)} cannot-link constraints.")

    # 3. Perform Constrained Clustering (PCKMeans)
    print("\nRunning Constrained Clustering (PCKMeans)...")
    try:
        # Use the imported PCKMeans class
        # The fit method expects constraints in a specific format, check library docs.
        # active-semi-supervised-clustering expects lists of tuples (index1, index2)
        pckmeans = PCKMeans(n_clusters=n_clusters, random_state=0)
        # PCKMeans typically works on the original feature space
        # Note: The author's wrapper has a pckmeans_w parameter - you might need to add this
        # or find the equivalent parameter in the PCKMeans library if tuning constraint weight.
        # Example fit call:
        pckmeans.fit(features, ml=must_link_constraints, cl=cannot_link_constraints)
        constrained_cluster_assignments = pckmeans.labels_ # Get labels after fitting

        print("PCKMeans completed.")
        return constrained_cluster_assignments

    except Exception as e:
        print(f"Error during Constrained Clustering (PCKMeans): {e}")
        return None # Indicate failure