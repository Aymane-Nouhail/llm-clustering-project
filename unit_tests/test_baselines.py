import unittest
import numpy as np
from unittest.mock import patch, MagicMock # Import patch and MagicMock
from sklearn.cluster import KMeans # Need KMeans for the original code logic
from src.baselines import run_naive_kmeans

class TestBaselines(unittest.TestCase):

    @patch('src.baselines.KMeans') # Patch KMeans within the baselines module
    def test_run_naive_kmeans(self, MockKMeans):
        """Test run_naive_kmeans by mocking KMeans behavior."""
        print("\nTesting run_naive_kmeans...")
        # Create simple dummy data
        features = np.array([
            [0.1, 0.1], [0.2, 0.1], [0.1, 0.2], # Cluster 0
            [5.1, 5.1], [5.2, 5.1], [5.1, 5.2], # Cluster 1
            [10.1, 10.1], [10.2, 10.1], [10.1, 10.2]  # Cluster 2
        ])
        n_clusters = 3
        random_state = 42 # Use a fixed random state for reproducibility

        # Configure the mock KMeans instance
        mock_kmeans_instance = MagicMock()
        # Mock fit_predict to return a predictable, correct assignment
        mock_kmeans_instance.fit_predict.return_value = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        MockKMeans.return_value = mock_kmeans_instance # Ensure the patch returns our mock instance

        assignments = run_naive_kmeans(features, n_clusters, random_state=random_state)

        self.assertIsNotNone(assignments, "run_naive_kmeans returned None.")
        self.assertIsInstance(assignments, np.ndarray, "Assignments are not a numpy array.")
        self.assertEqual(len(assignments), len(features), "Number of assignments does not match number of features.")
        # Assert that KMeans was called with the correct parameters
        MockKMeans.assert_called_once_with(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        # Assert that fit_predict was called with the normalized features
        mock_kmeans_instance.fit_predict.assert_called_once()
        called_with_features = mock_kmeans_instance.fit_predict.call_args[0][0]
        self.assertEqual(called_with_features.shape, features.shape, "KMeans fit_predict called with incorrect feature shape.")
        # Optional: Check normalization by comparing a few points
        original_norm = np.linalg.norm(features[0])
        called_norm = np.linalg.norm(called_with_features[0])
        # Check if the called features are normalized (norm should be close to 1)
        self.assertAlmostEqual(called_norm, 1.0, places=6, msg="Features passed to KMeans were not normalized.")


        # Assert against the predictable mock return value
        self.assertTrue(np.array_equal(assignments, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])), "Assignments do not match expected mock output.")
        self.assertEqual(len(np.unique(assignments)), n_clusters, "Number of unique clusters in assignments does not match n_clusters.")


        print("run_naive_kmeans test passed.")

# This allows running the tests directly from the terminal
if __name__ == '__main__':
    unittest.main(verbosity=2)