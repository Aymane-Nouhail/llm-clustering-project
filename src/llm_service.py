import time
import os
import json # Still needed for potential error logging or inspection
from typing import List, Any, Type
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from src.config import EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, MOCKING_MODE
from langchain_core.embeddings import Embeddings # Import Embeddings base class
# REMOVED: from langchain_core.runnables import with_structured_output # Import for structured output (no longer needed for import)
from pydantic import BaseModel, Field # Import for defining structured output schema


# Define a Pydantic model for the keyphrase output structure
class KeyphraseList(BaseModel):
    """List of keyphrases describing the document intent."""
    keyphrases: List[str] = Field(description="A list of keyphrases.")


class LLMService:
    """Handles interaction with OpenAI Embedding and Generation models."""

    def __init__(self, api_key: str):
        # Ensure API key is set in the environment for Langchain
        os.environ["OPENAI_API_KEY"] = api_key
        self.embedding_model: Embeddings | None = None
        self.generation_model: ChatOpenAI | None = None
        self._embedding_dim = 0 # Store embedding dimension

        print("Initializing LLM models...")
        try:
            self.embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
             # A quick test call to get dimension and check availability
            try:
                 test_embedding = self.embedding_model.embed_query("test")
                 self._embedding_dim = len(test_embedding)
                 print(f"Embedding model '{EMBEDDING_MODEL_NAME}' initialized. Dimension: {self._embedding_dim}")
            except Exception as e:
                 print(f"Error testing embedding model '{EMBEDDING_MODEL_NAME}': {e}")
                 self.embedding_model = None # Mark as failed


            # Use a low temperature for more consistent/deterministic responses
            # Bind with structured output capability here or in the call method
            self.generation_model = ChatOpenAI(model=GENERATION_MODEL_NAME, temperature=0)
            # A quick test call for generation model
            try:
                 self.generation_model.invoke("test")
                 print(f"Generation model '{GENERATION_MODEL_NAME}' initialized.")
            except Exception as e:
                 print(f"Error testing generation model '{GENERATION_MODEL_NAME}': {e}")
                 self.generation_model = None # Mark as failed


        except Exception as e:
            # Catch errors during initial model instantiation if they weren't caught by test calls
            print(f"Error initializing LLM models: {e}")
            print("Please ensure your OPENAI_API_KEY is set and check model availability.")
            self.embedding_model = None
            self.generation_model = None


        # A short delay to avoid hitting API rate limits quickly
        self._api_delay_seconds = 0 # Adjust as needed

    def is_available(self) -> bool:
        """Checks if both embedding and generation models were initialized successfully."""
        return self.embedding_model is not None and self.generation_model is not None

    def get_embedding_model(self) -> Embeddings:
         """Returns the initialized embedding model instance."""
         if not self.is_available() or self.embedding_model is None:
              raise RuntimeError("Embedding model is not available.")
         return self.embedding_model


    def get_generation_model(self) -> ChatOpenAI:
        """Returns the initialized generation model instance."""
        if not self.is_available() or self.generation_model is None:
            raise RuntimeError("Generation model is not available.")
        return self.generation_model


    def get_embedding_dimension(self) -> int:
        """Returns the dimension of the embedding vectors."""
        if self._embedding_dim == 0 and self.is_available() and self.embedding_model:
            # Try a test call if dimension wasn't set during init
            try:
                test_embedding = self.get_embedding("test")
                if test_embedding:
                    self._embedding_dim = len(test_embedding)
            except Exception:
                pass # Ignore error, _embedding_dim remains 0

        # If model failed or dimension couldn't be determined, return a known default for common models
        if self._embedding_dim == 0:
             # Assuming text-embedding-ada-002 which has 1536 dimensions
             print(f"Warning: Embedding dimension unknown, assuming 1536 for {EMBEDDING_MODEL_NAME}.")
             return 1536 # Default dimension for text-embedding-ada-002

        return self._embedding_dim


    def get_embedding(self, text: str) -> List[float]:
        """Gets the embedding vector for a single text."""
        if not self.is_available() or self.embedding_model is None:
            # Return zero vector if model not available, try to use known dimension
            dim = self.get_embedding_dimension()
            if dim > 0:
                # print(f"LLMService not available, returning zero vector of dimension {dim}.")
                return [0.0] * dim
            else:
                print("LLMService not available and embedding dimension unknown, returning empty list.")
                return []

        time.sleep(self._api_delay_seconds) # Add a small delay
        try:
            # Use embed_query for single string embeddings
            return self.embedding_model.embed_query(text)
        except Exception as e:
            print(f"Error during embedding call for text '{text[:50]}...': {e}")
            # Return a vector of zeros of the known dimension in case of embedding failure
            dim = self.get_embedding_dimension()
            if dim > 0:
                 print(f"  Returning zero vector of dimension {dim} due to embedding error.")
                 return [0.0] * dim
            else:
                 print("  Embedding error and dimension unknown, returning empty list.")
                 return []


    # Modify get_chat_completion to handle optional structured output
    def get_chat_completion(self, prompt: Any, output_structure: Type[BaseModel] | None = None) -> str | BaseModel | None:
        """
        Gets a chat completion response from the generation model.
        Optionally returns a structured output if output_structure is provided.
        """
        if MOCKING_MODE:
            return "YES"
        if not self.is_available() or self.generation_model is None:
            print("LLMService generation model not available.")
            return "ERROR" if output_structure is None else None # Indicate failure based on expected return type

        time.sleep(self._api_delay_seconds) # Add a small delay

        try:
            chain = self.generation_model
            if output_structure:
                # Bind the model with the structured output schema
                # with_structured_output is a method on the model instance itself
                chain = chain.with_structured_output(output_structure)

            response = chain.invoke(prompt)

            # If output_structure was used, response is already a BaseModel instance
            # Otherwise, it's a Langchain message object, return its content
            if output_structure:
                return response # Return the Pydantic model instance
            else:
                # For non-structured output, return the string content
                return response.content

        except Exception as e:
            print(f"Error during generation call (structured: {output_structure is not None}): {e}")
            # Log the prompt that caused the error (be cautious with sensitive data)
            # print(f"  Prompt: {prompt}")
            # Depending on the error, you might want to retry or return a specific error indicator
            return "ERROR" if output_structure is None else None # Indicate failure