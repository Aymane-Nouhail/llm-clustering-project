import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from src.clustering_methods.pairwise_constraints import cluster_via_pairwise_constraints
from src.llm_service import LLMService # Import the actual LLMService for mocking

# Define mock data
MOCK_DOCUMENTS = ["doc A", "doc B", "doc C", "doc D"]
MOCK_FEATURES = np.array([
    [0.1, 0.1], # A
    [0.15, 0.15], # B (similar to A)
    [1.1, 1.1], # C
    [1.2, 1.2]  # D (similar to C)
])
MOCK_N_CLUSTERS = 2
MOCK_PC_PROMPT_TEMPLATE = "Are {text1} and {text2} similar? YES/NO."
MOCK_NUM_PAIRS_TO_QUERY = 3 # Query 3 pairs

# Try importing PCKMeans for patching and skipping
try:
    from active_semi_supervised_clustering.pairwise_constraints import PCKMeans as ActualPCKMeans
except ImportError:
    print("PCKMeans from 'active-semi-supervised-clustering' not found.")
    print("The pairwise constraints tests that require PCKMeans will be skipped.")
    ActualPCKMeans = None


class TestPairwiseConstraints(unittest.TestCase):

    @patch('src.clustering_methods.pairwise_constraints.PCKMeans')
    @patch('src.clustering_methods.pairwise_constraints.LLMService')
    def test_cluster_via_pairwise_constraints_random(self, MockLLMService, MockPCKMeans):
        """Test cluster_via_pairwise_constraints with random strategy."""
        print("\nTesting cluster_via_pairwise_constraints (random strategy)...")
        # Ensure PCKMeans is available for this test (this check is separate from the source code)
        if ActualPCKMeans is None:
            raise unittest.SkipTest("PCKMeans not available, skipping test.")


        # Configure the mock LLMService
        mock_llm_service_instance = MagicMock()
        mock_llm_service_instance.is_available.return_value = True

        # Mock get_chat_completion to return alternating YES/NO responses for simplicity
        mock_llm_service_instance.get_chat_completion.side_effect = ["YES", "NO", "YES", "NO", "YES", "NO"] * 10 # Enough responses

        MockLLMService.return_value = mock_llm_service_instance

        # Configure the mock PCKMeans
        mock_pckmeans_instance = MagicMock()
        # Mock the labels_ attribute to return dummy assignments
        mock_pckmeans_instance.labels_ = np.array([0, 0, 1, 1]) # Example assignments
        MockPCKMeans.return_value = mock_pckmeans_instance

        # Run the method
        assignments = cluster_via_pairwise_constraints(
            MOCK_DOCUMENTS, MOCK_FEATURES, MOCK_N_CLUSTERS,
            mock_llm_service_instance, MOCK_PC_PROMPT_TEMPLATE,
            num_pairs_to_query=MOCK_NUM_PAIRS_TO_QUERY,
            constraint_selection_strategy='random'
        )

        # Assertions
        self.assertIsNotNone(assignments, "Clustering returned None.")
        self.assertIsInstance(assignments, np.ndarray, "Assignments are not a numpy array.")
        self.assertEqual(len(assignments), len(MOCK_DOCUMENTS), "Number of assignments does not match number of documents.")

        # Verify LLMService methods were called
        self.assertTrue(mock_llm_service_instance.is_available.called)
        self.assertEqual(mock_llm_service_instance.get_chat_completion.call_count, MOCK_NUM_PAIRS_TO_QUERY, "LLM not called for the specified number of pairs.")

        # Verify PCKMeans was called
        MockPCKMeans.assert_called_once_with(n_clusters=MOCK_N_CLUSTERS, random_state=0)
        mock_pckmeans_instance.fit.assert_called_once()
        fit_args, fit_kwargs = mock_pckmeans_instance.fit.call_args
        self.assertTrue(np.array_equal(fit_args[0], MOCK_FEATURES), "PCKMeans fit called with incorrect features.")
        self.assertIn('ml', fit_kwargs, "PCKMeans fit not called with must-link constraints.")
        self.assertIn('cl', fit_kwargs, "PCKMeans fit not called with cannot-link constraints.")
        self.assertIsInstance(fit_kwargs['ml'], list, "Must-link constraints not a list.")
        self.assertIsInstance(fit_kwargs['cl'], list, "Cannot-link constraints not a list.")
        self.assertTrue(len(fit_kwargs['ml']) + len(fit_kwargs['cl']) > 0, "No constraints generated.")
        self.assertEqual(len(fit_kwargs['ml']) + len(fit_kwargs['cl']), MOCK_NUM_PAIRS_TO_QUERY, "Incorrect total number of constraints generated.")


        print("cluster_via_pairwise_constraints (random) test passed.")

    @patch('src.clustering_methods.pairwise_constraints.PCKMeans')
    @patch('src.clustering_methods.pairwise_constraints.LLMService')
    def test_cluster_via_pairwise_constraints_similarity(self, MockLLMService, MockPCKMeans):
        """Test cluster_via_pairwise_constraints with similarity strategy."""
        print("\nTesting cluster_via_pairwise_constraints (similarity strategy)...")
        # Ensure PCKMeans is available for this test
        if ActualPCKMeans is None:
            raise unittest.SkipTest("PCKMeans not available, skipping test.")

        # Configure the mock LLMService
        mock_llm_service_instance = MagicMock()
        mock_llm_service_instance.is_available.return_value = True

        # Mock get_chat_completion based on the dummy data's similarity
        def mock_get_chat_completion(prompt):
            prompt_text = str(prompt)
            if "doc A" in prompt_text and "doc B" in prompt_text: return "YES"
            if "doc A" in prompt_text and "doc C" in prompt_text: return "NO"
            if "doc A" in prompt_text and "doc D" in prompt_text: return "NO"
            if "doc B" in prompt_text and "doc C" in prompt_text: return "NO"
            if "doc B" in prompt_text and "doc D" in prompt_text: return "NO"
            if "doc C" in prompt_text and "doc D" in prompt_text: return "YES"
            return "ERROR"

        mock_llm_service_instance.get_chat_completion.side_effect = mock_get_chat_completion

        MockLLMService.return_value = mock_llm_service_instance

        # Configure the mock PCKMeans
        mock_pckmeans_instance = MagicMock()
        mock_pckmeans_instance.labels_ = np.array([0, 0, 1, 1])
        MockPCKMeans.return_value = mock_pckmeans_instance

        # Run the method
        assignments = cluster_via_pairwise_constraints(
            MOCK_DOCUMENTS, MOCK_FEATURES, MOCK_N_CLUSTERS,
            mock_llm_service_instance, MOCK_PC_PROMPT_TEMPLATE,
            num_pairs_to_query=MOCK_NUM_PAIRS_TO_QUERY,
            constraint_selection_strategy='similarity'
        )

        # Assertions
        self.assertIsNotNone(assignments, "Clustering returned None.")
        self.assertIsInstance(assignments, np.ndarray, "Assignments are not a numpy array.")
        self.assertEqual(len(assignments), len(MOCK_DOCUMENTS), "Number of assignments does not match number of documents.")

        self.assertTrue(mock_llm_service_instance.is_available.called)
        self.assertEqual(mock_llm_service_instance.get_chat_completion.call_count, MOCK_NUM_PAIRS_TO_QUERY, "LLM not called for the specified number of pairs.")

        MockPCKMeans.assert_called_once_with(n_clusters=MOCK_N_CLUSTERS, random_state=0)
        mock_pckmeans_instance.fit.assert_called_once()
        fit_args, fit_kwargs = mock_pckmeans_instance.fit.call_args
        self.assertTrue(np.array_equal(fit_args[0], MOCK_FEATURES), "PCKMeans fit called with incorrect features.")
        self.assertIn('ml', fit_kwargs, "PCKMeans fit not called with must-link constraints.")
        self.assertIn('cl', fit_kwargs, "PCKMeans fit not called with cannot-link constraints.")
        self.assertIsInstance(fit_kwargs['ml'], list, "Must-link constraints not a list.")
        self.assertIsInstance(fit_kwargs['cl'], list, "Cannot-link constraints not a list.")
        self.assertTrue(len(fit_kwargs['ml']) + len(fit_kwargs['cl']) > 0, "No constraints generated.")
        self.assertEqual(len(fit_kwargs['ml']) + len(fit_kwargs['cl']), MOCK_NUM_PAIRS_TO_QUERY, "Incorrect total number of constraints generated.")

        print("cluster_via_pairwise_constraints (similarity) test passed.")


    @patch('src.clustering_methods.pairwise_constraints.PCKMeans', new=None) # Explicitly patch PCKMeans to be None
    @patch('src.clustering_methods.pairwise_constraints.LLMService')
    def test_cluster_via_pairwise_constraints_no_pckmeans(self, MockLLMService):
        """Test cluster_via_pairwise_constraints when PCKMeans is not available."""
        print("\nTesting cluster_via_pairwise_constraints (no PCKMeans)...")

        mock_llm_service_instance = MagicMock()
        mock_llm_service_instance.is_available.return_value = True # Still mock as it might be called in other scenarios
        # No need to mock get_chat_completion, it shouldn't be called

        MockLLMService.return_value = mock_llm_service_instance

        # Run the method
        assignments = cluster_via_pairwise_constraints(
            MOCK_DOCUMENTS, MOCK_FEATURES, MOCK_N_CLUSTERS,
            mock_llm_service_instance, MOCK_PC_PROMPT_TEMPLATE,
            num_pairs_to_query=MOCK_NUM_PAIRS_TO_QUERY,
            constraint_selection_strategy='random'
        )

        # Assertions
        self.assertIsNone(assignments, "Clustering should return None when PCKMeans is not available.")
        # Fix: Remove the assertion that LLMService.is_available is called, as it's skipped in the source code logic.
        # self.assertTrue(mock_llm_service_instance.is_available.called) # REMOVE THIS LINE
        # Check that the LLM was NOT called for constraints
        self.assertFalse(mock_llm_service_instance.get_chat_completion.called, "LLM should not be called if PCKMeans is not available.")

        print("cluster_via_pairwise_constraints (no PCKMeans) test passed.")