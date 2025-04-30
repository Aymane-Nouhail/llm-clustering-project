import unittest
import numpy as np
import os
import pickle
from unittest.mock import patch, MagicMock, mock_open # Import mock_open
from src.data import load_dataset

# Define a mock dataset structure similar to clinc_oos "small" test split
MOCK_DATASET = {
    "test": {
        "text": ["text1", "text2", "text3", "text4", "text5", "text6", "text7", "text8", "text9", "text10", "text11", "text12", "text13", "text14", "text15", "text16", "text17", "text18", "text19", "text20", "text_intent_42"],
        "intent": [0, 1, 0, 2, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 42] # Include intent 42
    }
}

# Define a mock embedding dimension (though not used in this specific test)
MOCK_EMBEDDING_DIM = 10

class TestDataLoading(unittest.TestCase):

    @patch('src.data.load_dataset_hf')
    @patch('builtins.open', new_callable=mock_open) # Patch builtins.open
    @patch('src.data.pickle.load') # Patch pickle.load
    def test_load_dataset_from_cache(self, mock_pickle_load, mock_open_file, mock_load_dataset_hf):
        """Test loading data from cache when cache file exists."""
        print("\nTesting load_dataset from cache...")
        # Mock the cache loading process
        mock_features = np.random.rand(10, MOCK_EMBEDDING_DIM) # Mock features
        mock_labels_original = [0, 1, 0, 2, 1, 2, 0, 1, 2, 3] # Mock original labels from HF load
        mock_documents = [f"cached_text_{i}" for i in range(10)] # Mock documents
        mock_pickle_load.return_value = mock_features

        # Mock load_dataset_hf to return the necessary documents and original intents
        mock_load_dataset_hf.return_value = {
             "test": {
                 "text": mock_documents + ["text_intent_42"], # Include dummy intent 42 text
                 "intent": list(mock_labels_original) + [42] # Include dummy intent 42 label
             }
        }

        temp_cache_path = "/tmp/test_clinc_cache_exists.pkl"
        # Configure mock_open_file to succeed when called for reading and return a mock file handle
        mock_read_file_handle = MagicMock()
        mock_open_file.return_value = mock_read_file_handle


        features, labels, documents = load_dataset(cache_path=temp_cache_path, embedding_model=MagicMock()) # Pass a dummy model

        mock_load_dataset_hf.assert_called_once_with("clinc_oos", "small") # load_dataset_hf is called to get texts/labels even with cache
        mock_open_file.assert_called_once_with(temp_cache_path, 'rb') # Check open was called for reading
        # Fix: Check load was called with the result of mock_open_file().__enter__()
        # When mock_open is used as a context manager, the object yielded
        # is mock_open_file().__enter__(). pickle.load receives this object.
        # We need to check if pickle.load was called with this specific mock object.
        # The error message shows Actual: load(<MagicMock name='open().__enter__()' id='...'>)
        # The Expected argument should match this. The object yielded by mock_open_file()
        # (the mock file handle) is the object on which __enter__ is called.
        # So, the correct assertion argument is mock_open_file().__enter__()
        # Let's be explicit:
        mock_file_handle_inside_with = mock_open_file().__enter__()
        mock_pickle_load.assert_called_once_with(mock_file_handle_inside_with)


        # Check that embed_documents was NOT called on the embedding model
        self.assertFalse(MagicMock().embed_documents.called, "Embedding model was called when cache should be used.")


        self.assertEqual(len(documents), len(mock_documents), "Incorrect number of documents loaded from cache.")
        # Re-map the mocked labels based on the mocked documents/intents from load_dataset_hf mock
        expected_remapped_labels = np.array([0, 1, 0, 2, 1, 2, 0, 1, 2, 3]) # Based on MOCK_DATASET structure excluding 42
        self.assertTrue(np.array_equal(labels, expected_remapped_labels), "Incorrect labels loaded/remapped from cache.")
        self.assertTrue(np.array_equal(features, mock_features), "Incorrect features loaded from cache.")


# This allows running the tests directly from the terminal
if __name__ == '__main__':
    unittest.main(verbosity=2)