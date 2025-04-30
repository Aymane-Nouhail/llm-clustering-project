import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple

def run_naive_kmeans(features: np.ndarray, n_clusters: int, random_state: int = 0) -> np.ndarray:
    """
    Runs Naive KMeans clustering on the provided features using cosine similarity.

    Args:
        features (np.ndarray): The input features/embeddings.
        n_clusters (int): The number of clusters.
        random_state (int): Random state for KMeans reproducibility.

    Returns:
        np.ndarray: Cluster assignments.
    """
    print("\n--- Running Naive KMeans Baseline (Cosine Similarity) ---")
    # Normalize features for cosine similarity based clustering
    features_normalized = features / np.linalg.norm(features, axis=1)[:, np.newaxis]

    # Initialize and run KMeans
    # Use n_init='auto' to suppress warning
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        cluster_assignments = kmeans.fit_predict(features_normalized)
        print("Naive KMeans completed.")
        return cluster_assignments
    except Exception as e:
        print(f"Error during Naive KMeans clustering: {e}")
        # Return an array of -1 or handle error appropriately
        return np.array([-1] * features.shape[0]) # Indicate failure