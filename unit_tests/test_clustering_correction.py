import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.metrics.pairwise import euclidean_distances # Need actual euclidean_distances
from sklearn.cluster import KMeans # Need KMeans for the initial step in a real run, but mocked here if needed
from src.clustering_methods.clustering_correction import correct_clustering_with_llm
from src.llm_service import LLMService # Import the actual LLMService for mocking

# Define mock data
MOCK_DOCUMENTS = [
    "doc 0 cluster 0", "doc 1 cluster 0", "doc 2 cluster 0 low confidence",
    "doc 3 cluster 1", "doc 4 cluster 1", "doc 5 cluster 1 low confidence",
    "doc 6 cluster 2", "doc 7 cluster 2"
]
MOCK_FEATURES = np.array([
    [0.1, 0.1], [0.2, 0.2], [0.5, 0.5], # Cluster 0 (doc 2 is closer to center but still low confidence margin)
    [1.1, 1.1], [1.2, 1.2], [1.5, 1.5], # Cluster 1 (doc 5 is closer to center but still low confidence margin)
    [2.1, 2.1], [2.2, 2.2]              # Cluster 2
])
MOCK_INITIAL_ASSIGNMENTS = np.array([0, 0, 0, 1, 1, 1, 2, 2])
MOCK_N_CLUSTERS = 3 # Total expected clusters
MOCK_CORRECTION_PROMPT_TEMPLATE = "Should {point_text} be with {representative_text}? YES/NO."
MOCK_K_LOW_CONFIDENCE = 2 # Look at the 2 lowest confidence points
MOCK_NUM_CANDIDATE_CLUSTERS = 2 # Check 2 nearest alternative clusters

class TestClusteringCorrection(unittest.TestCase):

    @patch('src.clustering_methods.clustering_correction.LLMService')
    @patch('src.clustering_methods.clustering_correction.euclidean_distances') # Patch distances to control confidence
    def test_correct_clustering_with_llm(self, MockEuclideanDistances, MockLLMService):
        """Test correct_clustering_with_llm method."""
        print("\nTesting correct_clustering_with_llm...")

        # Configure the mock LLMService
        mock_llm_service_instance = MagicMock()
        mock_llm_service_instance.is_available.return_value = True
        MockLLMService.return_value = mock_llm_service_instance

        # Configure the mock Euclidean Distances
        # We need to control the distances to simulate low confidence points
        # and control which clusters are candidates.
        # Mock distances_to_all_valid_centroids (shape: n_samples, n_valid_clusters_with_reps)
        # valid_cluster_ids_with_reps will be [0, 1, 2] (assuming all have reps)
        mock_distances_to_all_valid_centroids = np.array([
            [0.1, 1.5, 3.0], # doc 0: high margin
            [0.05, 1.6, 3.1], # doc 1: high margin
            [0.3, 0.4, 2.5],  # doc 2: low margin (closest to 0, second to 1, third to 2)
            [1.8, 0.1, 1.7],  # doc 3: high margin
            [1.9, 0.05, 1.8], # doc 4: high margin
            [2.6, 0.3, 0.4],  # doc 5: low margin (closest to 1, second to 2, third to 0)
            [3.5, 2.0, 0.1],  # doc 6: high margin
            [3.6, 2.1, 0.05]  # doc 7: high margin
        ])

        # Mock euclidean_distances side effect: return the pre-defined matrix for the main call,
        # and use the real function for centroid-to-point/point-to-centroid calls.
        original_euclidean_distances = euclidean_distances # Keep a reference to the real function

        def mock_euclidean_distances_side_effect(X, Y):
            # Check if this is the main call from all points to all valid centroids
            if X.shape[0] == len(MOCK_FEATURES) and Y.shape[0] == MOCK_N_CLUSTERS: # Y shape is n_valid_clusters_with_reps
                 # For this test data and setup, valid_cluster_ids_with_reps will be [0, 1, 2], shape is 3
                 if Y.shape[0] == 3:
                      return mock_distances_to_all_valid_centroids
                 else:
                      # If the code changes how centroids are handled, this mock might need adjustment.
                      print(f"Warning: Unexpected shape for centroids in distance call: {Y.shape}")
                      return original_euclidean_distances(X, Y) # Fallback

            elif X.shape[0] == 1 or Y.shape[0] == 1:
                 # This is likely a call from a single point to multiple points/centroids or vice versa
                 # Use the real function for these
                 return original_euclidean_distances(X, Y)
            else:
                 # Fallback for unexpected calls
                 return original_euclidean_distances(X, Y)

        MockEuclideanDistances.side_effect = mock_euclidean_distances_side_effect


        # Configure the mock LLM get_chat_completion for correction decisions
        # Fix: Use a list of expected responses in the exact order of calls
        # Expected call sequence for k_low_confidence=2 (doc 2, doc 5), num_candidate_clusters=2:
        # Point 2 (original 0):
        # 1. Check vs rep 0 (current cluster) -> Should be NO
        # 2. Check vs rep 1 (candidate 1, nearest) -> Should be YES (reassigns to 1)
        # Point 5 (original 1):
        # 3. Check vs rep 1 (current cluster) -> Should be NO
        # 4. Check vs rep 2 (candidate 1, nearest) -> Should be YES (reassigns to 2)
        # Total expected calls: 4

        mock_llm_response_sequence = [
            "NO",  # doc 2 vs rep 0
            "YES", # doc 2 vs rep 1
            "NO",  # doc 5 vs rep 1
            "YES", # doc 5 vs rep 2
        ]

        mock_llm_service_instance.get_chat_completion.side_effect = mock_llm_response_sequence


        # Run the method
        corrected_assignments = correct_clustering_with_llm(
            MOCK_DOCUMENTS, MOCK_FEATURES, MOCK_INITIAL_ASSIGNMENTS, MOCK_N_CLUSTERS,
            mock_llm_service_instance, MOCK_CORRECTION_PROMPT_TEMPLATE,
            k_low_confidence=MOCK_K_LOW_CONFIDENCE,
            num_candidate_clusters=MOCK_NUM_CANDIDATE_CLUSTERS
        )

        # Assertions
        self.assertIsNotNone(corrected_assignments, "Clustering correction returned None.")
        self.assertIsInstance(corrected_assignments, np.ndarray, "Corrected assignments are not a numpy array.")
        self.assertEqual(len(corrected_assignments), len(MOCK_DOCUMENTS), "Number of corrected assignments does not match number of documents.")

        # Verify LLMService methods were called
        self.assertTrue(mock_llm_service_instance.is_available.called)
        # Check LLM call count
        # Fix: Assert against the expected number of calls (4)
        self.assertEqual(mock_llm_service_instance.get_chat_completion.call_count, len(mock_llm_response_sequence), "Incorrect number of LLM calls for correction.")


        # Verify assignments were corrected as expected
        expected_corrected_assignments = np.array([0, 0, 1, 1, 1, 2, 2, 2]) # doc 2 to cluster 1, doc 5 to cluster 2
        self.assertTrue(np.array_equal(corrected_assignments, expected_corrected_assignments), "Assignments were not corrected as expected.")

        # Verify the number of points reassigned
        initial_reassignments_count = np.sum(corrected_assignments != MOCK_INITIAL_ASSIGNMENTS)
        self.assertEqual(initial_reassignments_count, 2, "Incorrect number of points reassigned.")


        print("correct_clustering_with_llm test passed.")


# This allows running the tests directly from the terminal
if __name__ == '__main__':
    unittest.main(verbosity=2)