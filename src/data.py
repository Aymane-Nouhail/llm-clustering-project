import numpy as np
# Assuming load_dataset_hf is available from datasets library, need to install it
from datasets import load_dataset as load_dataset_hf
import pickle
import os
from typing import Tuple, List
from langchain_core.embeddings import Embeddings # Use the base class for type hinting
# Note: The actual embedding model passed will be OpenAIEmbeddings

def load_dataset(cache_path: str = None, embedding_model: Embeddings = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads the CLINC dataset texts and labels, and generates/loads features.
    Uses the provided embedding_model to generate features if not cached.

    Args:
        cache_path: Optional path to cache generated features.
        embedding_model: An initialized Langchain-compatible embedding model
                         to generate features if not cached. Required if cache_path is None
                         or cache doesn't exist.

    Returns:
        A tuple containing:
        - features (np.ndarray): The document features/embeddings.
        - labels (np.ndarray): The true cluster labels.
        - documents (List[str]): The original document texts.
         Returns empty features array and prints error if embedding_model is missing
         when needed or if embedding fails, but returns populated labels and documents.
    """
    print(f"Loading CLINC dataset from Hugging Face...")
    try:
        dataset = load_dataset_hf("clinc_oos", "small")
        test_split = dataset["test"]
        texts = test_split["text"]
        intents = test_split["intent"]
        # Filter out intent 42 as in the original load_clinc
        filtered_pairs = [(t, i) for (t, i) in zip(texts, intents) if i != 42]
        filtered_texts, filtered_intents = zip(*filtered_pairs)
        remapped_intents_list = []
        intent_mapping = {}
        current_map_id = 0
        for intent in filtered_intents:
            if intent not in intent_mapping:
                intent_mapping[intent] = current_map_id
                current_map_id += 1
            remapped_intents_list.append(intent_mapping[intent])
        remapped_intents = np.array(remapped_intents_list)
        documents = list(filtered_texts)

        print(f"Loaded {len(documents)} documents.")
        print(f"Found {len(np.unique(remapped_intents))} unique intents (clusters) after filtering.")

    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        print("Please ensure 'datasets' library is installed.")
        # Return empty if dataset loading fails entirely
        return np.array([]), np.array([]), []


    embeddings = None
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                 embeddings = pickle.load(f)
            print("Embeddings loaded from cache.")
        except Exception as e:
            print(f"Error loading cache file {cache_path}: {e}")
            embeddings = None # Set to None so it will regenerate


    if embeddings is None:
        print("Generating embeddings...")
        if embedding_model is None:
            print("Error: Embedding model not provided and cache not available.")
            # Fix: Return populated labels and documents even if embedding model is missing
            return np.array([]), remapped_intents, documents # <-- Make sure this is correct
        try:
            # Use the provided embedding_model to generate embeddings
            embeddings_list = embedding_model.embed_documents(documents)
            embeddings = np.array(embeddings_list)
            print(f"Embeddings generated. Shape: {embeddings.shape}")

            if cache_path is not None:
                print(f"Saving embeddings to cache: {cache_path}")
                try:
                    # Ensure cache directory exists
                    cache_dir = os.path.dirname(cache_path)
                    if cache_dir and not os.path.exists(cache_dir):
                        os.makedirs(cache_dir)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(embeddings, f)
                    print("Embeddings saved to cache.")
                except Exception as e:
                    print(f"Error saving cache file {cache_path}: {e}")

        except Exception as e:
            print(f"Error generating embeddings with the provided model: {e}")
            # Fix: Return populated labels and documents even if embedding generation fails
            return np.array([]), remapped_intents, documents # <-- Make sure this is correct

    return embeddings, remapped_intents, documents