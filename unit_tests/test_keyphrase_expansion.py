import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.cluster import KMeans # Need KMeans for the final step
from src.clustering_methods.keyphrase_expansion import cluster_via_keyphrase_expansion
from src.llm_service import LLMService # Import the actual LLMService for mocking

# Define mock data
MOCK_DOCUMENTS = ["doc one about apples", "doc two about bananas", "doc three about cherries"]
MOCK_FEATURES = np.array([
    [0.1, 0.2],
    [1.1, 1.2],
    [2.1, 2.2]
])
MOCK_N_CLUSTERS = 3
MOCK_KP_PROMPT_TEMPLATE = "Generate keyphrases for: {document_text}"
MOCK_EMBEDDING_DIM = MOCK_FEATURES.shape[1] # Match original feature dimension

class TestKeyphraseExpansion(unittest.TestCase):

    @patch('src.clustering_methods.keyphrase_expansion.KMeans')
    @patch('src.clustering_methods.keyphrase_expansion.LLMService')
    def test_cluster_via_keyphrase_expansion(self, MockLLMService, MockKMeans):
        """Test cluster_via_keyphrase_expansion method."""
        print("\nTesting cluster_via_keyphrase_expansion...")

        # Configure the mock LLMService
        mock_llm_service_instance = MagicMock()
        mock_llm_service_instance.is_available.return_value = True

        # Mock get_chat_completion to return simulated JSON keyphrases
        def mock_get_chat_completion(prompt):
            if "apples" in prompt:
                return '["red fruit", "sweet", "orchard"]'
            elif "bananas" in prompt:
                return '["yellow fruit", "tropical", "peel"]'
            elif "cherries" in prompt:
                return '["small fruit", "red", "pit"]'
            return '[]' # Default empty list

        mock_llm_service_instance.get_chat_completion.side_effect = mock_get_chat_completion

        # Mock get_embedding to return simulated embedding vectors
        def mock_get_embedding(text):
            # Return a random vector of the expected dimension
            return list(np.random.rand(MOCK_EMBEDDING_DIM))

        mock_llm_service_instance.get_embedding.side_effect = mock_get_embedding

        MockLLMService.return_value = mock_llm_service_instance # Ensure the patch returns our mock instance

        # Configure the mock KMeans
        mock_kmeans_instance = MagicMock()
        # Mock the fit_predict method to return dummy assignments
        mock_kmeans_instance.fit_predict.return_value = np.array([0, 1, 2])
        MockKMeans.return_value = mock_kmeans_instance # Ensure the patch returns our mock instance

        # Run the method
        assignments = cluster_via_keyphrase_expansion(
            MOCK_DOCUMENTS, MOCK_FEATURES, MOCK_N_CLUSTERS,
            mock_llm_service_instance, MOCK_KP_PROMPT_TEMPLATE
        )

        # Assertions
        self.assertIsNotNone(assignments, "Clustering returned None.")
        self.assertIsInstance(assignments, np.ndarray, "Assignments are not a numpy array.")
        self.assertEqual(len(assignments), len(MOCK_DOCUMENTS), "Number of assignments does not match number of documents.")
        self.assertEqual(len(np.unique(assignments)), MOCK_N_CLUSTERS, "Number of unique clusters does not match n_clusters.")

        # Verify LLMService methods were called
        self.assertTrue(mock_llm_service_instance.is_available.called)
        self.assertEqual(mock_llm_service_instance.get_chat_completion.call_count, len(MOCK_DOCUMENTS), "LLM not called for each document.")
        # Embedding should be called for each document's joined text
        self.assertEqual(mock_llm_service_instance.get_embedding.call_count, len(MOCK_DOCUMENTS), "Embedding not called for each joined text.")

        # Verify KMeans was called with the correct data shape
        MockKMeans.assert_called_once_with(n_clusters=MOCK_N_CLUSTERS, random_state=0, n_init='auto')
        mock_kmeans_instance.fit_predict.assert_called_once()
        called_with_features = mock_kmeans_instance.fit_predict.call_args[0][0]
        self.assertEqual(called_with_features.shape[0], len(MOCK_DOCUMENTS), "KMeans called with incorrect number of samples.")
        # Expanded features should have original_dim + expansion_dim (which is MOCK_EMBEDDING_DIM)
        self.assertEqual(called_with_features.shape[1], MOCK_EMBEDDING_DIM * 2, "KMeans called with incorrect feature dimension.")

        print("cluster_via_keyphrase_expansion test passed.")

# This allows running the tests directly from the terminal
if __name__ == '__main__':
    unittest.main(verbosity=2)