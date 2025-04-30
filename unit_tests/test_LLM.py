import unittest
import os
import numpy as np
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL_NAME
from src.llm_service import LLMService

# Define the expected embedding dimension for the configured model
# Update this if you change EMBEDDING_MODEL_NAME in src/config.py
# Common dimensions: text-embedding-ada-002 (1536)
EXPECTED_EMBEDDING_DIM = 1536 # Default for text-embedding-ada-002

class TestLLMService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up LLMService instance for all tests."""
        print("\nSetting up TestLLMService...")
        # Ensure API key is available
        cls.api_key = OPENAI_API_KEY
        if not cls.api_key:
            raise unittest.SkipTest("OPENAI_API_KEY environment variable not set.")

        try:
            cls.llm_service = LLMService(cls.api_key)
            if not cls.llm_service.is_available():
                 raise RuntimeError("LLMService failed to initialize.")
            print("LLMService setup successful.")
        except Exception as e:
            print(f"Failed to initialize LLMService during setup: {e}")
            raise unittest.SkipTest(f"Failed to initialize LLMService: {e}")


    def test_01_service_availability(self):
        """Test if the LLMService is initialized and available."""
        print("\nTesting LLMService availability...")
        self.assertIsNotNone(self.llm_service, "LLMService instance is None.")
        self.assertTrue(self.llm_service.is_available(), "LLMService reported as not available.")
        print("LLMService is available.")

    def test_02_get_embedding_model(self):
        """Test if the embedding model is accessible."""
        print("\nTesting get_embedding_model...")
        embedding_model = self.llm_service.get_embedding_model()
        self.assertIsNotNone(embedding_model, "Embedding model is None.")
        # You could add more specific checks here if needed, e.g., check its type
        print("Embedding model is accessible.")


    def test_03_get_generation_model(self):
        """Test if the generation model is accessible."""
        print("\nTesting get_generation_model...")
        generation_model = self.llm_service.get_generation_model()
        self.assertIsNotNone(generation_model, "Generation model is None.")
         # You could add more specific checks here if needed, e.g., check its type
        print("Generation model is accessible.")


    def test_04_get_embedding(self):
        """Test if get_embedding returns an embedding of the correct dimension."""
        print("\nTesting get_embedding...")
        test_text = "This is a simple test sentence."
        embedding = self.llm_service.get_embedding(test_text)

        self.assertIsNotNone(embedding, "Embedding returned is None.")
        self.assertIsInstance(embedding, list, "Embedding is not a list.")
        self.assertEqual(len(embedding), EXPECTED_EMBEDDING_DIM,
                         f"Embedding dimension mismatch. Expected {EXPECTED_EMBEDDING_DIM}, got {len(embedding)}.")

        # Optional: Check if the embedding contains valid numbers
        self.assertTrue(all(isinstance(x, (int, float)) for x in embedding), "Embedding contains non-numeric values.")
        # Optional: Check if it's not a zero vector (unlikely for real text unless model fails)
        # self.assertFalse(np.allclose(embedding, np.zeros(len(embedding))), "Embedding is a zero vector.")
        print(f"Embedding generated successfully with dimension {len(embedding)}.")


    def test_05_get_chat_completion(self):
        """Test if get_chat_completion returns a non-empty string response."""
        print("\nTesting get_chat_completion...")
        test_prompt = "Hello, what is your name?"
        response = self.llm_service.get_chat_completion(test_prompt)

        self.assertIsNotNone(response, "Chat completion response is None.")
        self.assertIsInstance(response, str, "Chat completion response is not a string.")
        self.assertTrue(len(response.strip()) > 0, "Chat completion response is empty or whitespace.")
        print(f"Chat completion received a non-empty response (starts with: '{response[:50]}...').")


# This allows running the tests directly from the terminal
if __name__ == '__main__':
    unittest.main(verbosity=2) # verbosity=2 shows more details about test execution