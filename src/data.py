import numpy as np
import pickle
import os
import pandas as pd # Needed for reading CSV datasets like Tweet
from typing import Tuple, List, Any # Import Any for type hinting raw data/text
from langchain_core.embeddings import Embeddings # Use the base class for type hinting
# Assuming load_dataset_hf is available from datasets library, need to install it
from datasets import load_dataset as load_dataset_hf
from sklearn import datasets as sklearn_datasets # Needed for Iris
import random # Needed for dummy embedding model in example usage
import pickle
from typing import List, Tuple
from collections import defaultdict


# --- Helper Functions for Dataset Loading ---

def _load_iris_data() -> Tuple[List[Any], List[int], List[str]]:
    """Loads the Iris dataset (numerical features, labels, empty documents)."""
    print("Loading Iris dataset...")
    try:
        samples_raw, labels_raw = sklearn_datasets.load_iris(return_X_y=True)
        # Iris features are already numerical, return as list of lists for consistency with text data before conversion to np
        features_list = samples_raw.tolist()
        labels_list = list(labels_raw)
        documents = ["" for _ in samples_raw] # Iris has no text documents
        print(f"Loaded {len(features_list)} samples from Iris.")
        return features_list, labels_list, documents
    except Exception as e:
        print(f"Error loading Iris dataset: {e}")
        print("Please ensure scikit-learn data loading is working.")
        return [], [], [] # Return empty on failure

def _load_clinc_data() -> Tuple[List[str], List[int], List[str]]:
    """Loads the CLINC dataset (texts, labels, documents)."""
    print("Loading CLINC dataset from Hugging Face...")
    try:
        dataset = load_dataset_hf("clinc_oos", "small")
        test_split = dataset["test"]
        texts = test_split["text"]
        intents = test_split["intent"]
        # Filter out intent 42
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
        labels_list = remapped_intents_list
        documents = list(filtered_texts)

        print(f"Loaded {len(documents)} documents from CLINC.")
        print(f"Found {len(set(labels_list))} unique intents (clusters) after filtering.")
        return documents, labels_list, documents # For text datasets, raw data is the documents list

    except Exception as e:
        print(f"Error loading CLINC dataset from Hugging Face: {e}")
        print("Please ensure 'datasets' library is installed.")
        return [], [], [] # Return empty on failure

def _load_tweet_data(dataset_split: str = 'test') -> Tuple[List[str], List[int], List[str]]:
    """Loads the cardiffnlp/tweet_eval (emotion) dataset (texts, labels, documents)."""
    # Ensure dataset_split is a string, default to 'test' if None or not a string
    if not isinstance(dataset_split, str) or not dataset_split:
         print(f"Warning: Invalid dataset_split '{dataset_split}' provided for tweet dataset. Defaulting to 'test'.")
         dataset_split = 'test'

    print(f"Loading cardiffnlp/tweet_eval (emotion) dataset from Hugging Face, split: {dataset_split}...")
    try:
        # Load the specific dataset and configuration
        dataset = load_dataset_hf("cardiffnlp/tweet_eval", "sentiment")
        # Select the specified split
        if dataset_split not in dataset:
             print(f"Error: Split '{dataset_split}' not found in cardiffnlp/tweet_eval (emotion). Available splits: {list(dataset.keys())}")
             # If the requested split is not found, return empty data
             return [], [], []
        else:
            data_split = dataset[dataset_split]


        texts = data_split["text"]
        labels = data_split["label"] # Labels are already integers from HF datasets

        documents = list(texts)
        labels_list = list(labels)

        print(f"Loaded {len(documents)} documents from cardiffnlp/tweet_eval (emotion).")
        print(f"Found {len(set(labels_list))} unique labels.")
        return documents, labels_list, documents # For text datasets, raw data is the documents list

    except Exception as e:
        print(f"Error loading cardiffnlp/tweet_eval (emotion) dataset from Hugging Face: {e}")
        print("Please ensure 'datasets' library is installed and you have internet access.")
        return [], [], [] # Return empty on failure

def _load_bank77_data() -> Tuple[List[str], List[int], List[str]]:
    """Loads the BANKING77 dataset (texts, labels, documents)."""
    print("Loading BANKING77 dataset from Hugging Face...")
    try:
        dataset = load_dataset_hf("banking77")
        test_split = dataset["test"]
        texts = test_split["text"]
        labels = test_split["label"] # Labels are already integers from HF datasets
        documents = list(texts)
        labels_list = list(labels)
        print(f"Loaded {len(documents)} documents from BANKING77.")
        print(f"Found {len(set(labels_list))} unique intents (clusters).")
        return documents, labels_list, documents # For text datasets, raw data is the documents list

    except Exception as e:
        print(f"Error loading BANKING77 dataset from Hugging Face: {e}")
        print("Please ensure 'datasets' library is installed.")
        return [], [], [] # Return empty on failure

def _load_agnews_data(dataset_split: str = 'test') -> Tuple[List[str], List[int], List[str]]:
    """Loads the AG News dataset (text, labels, documents).
    
    Args:
        dataset_split: The dataset split to load ('train', 'test'). Defaults to 'test'.
        
    Returns:
        A tuple containing (documents, labels_list, documents).
    """
    # Ensure dataset_split is a string, default to 'test' if None or not a string
    if not isinstance(dataset_split, str) or not dataset_split:
        print(f"Warning: Invalid dataset_split '{dataset_split}' provided for AG News dataset. Defaulting to 'test'.")
        dataset_split = 'test'

    print(f"Loading AG News dataset from Hugging Face, split: {dataset_split}...")
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset_hf("ag_news")
        
        # Select the specified split
        if dataset_split not in dataset:
            print(f"Error: Split '{dataset_split}' not found in AG News. Available splits: {list(dataset.keys())}")
            # If the requested split is not found, return empty data
            return [], [], []
        else:
            data_split = dataset[dataset_split]

        # Extract texts and labels
        # AG News has 'text' field for content and 'label' for class (0-3)
        texts = data_split["text"]
        labels = data_split["label"]  # Labels are already integers (0-3)

        documents = list(texts)
        labels_list = list(labels)

        print(f"Loaded {len(documents)} documents from AG News.")
        print(f"Found {len(set(labels_list))} unique categories (World=0, Sports=1, Business=2, Sci/Tech=3).")
        return documents, labels_list, documents  # For text datasets, raw_data is the documents list

    except Exception as e:
        print(f"Error loading AG News dataset from Hugging Face: {e}")
        print("Please ensure 'datasets' library is installed and you have internet access.")
        return [], [], []  # Return empty on failure

def _load_hwu64_data(dataset_split: str = 'test') -> Tuple[List[str], List[int], List[str]]:
    """Loads the HWU64 dataset (text, labels, documents).
    
    Args:
        dataset_split: The dataset split to load ('train', 'test', 'val'). Defaults to 'test'.
        
    Returns:
        A tuple containing (documents, labels_list, documents).
    """
    # Ensure dataset_split is a string, default to 'test' if None or not a string
    if not isinstance(dataset_split, str) or not dataset_split:
        print(f"Warning: Invalid dataset_split '{dataset_split}' provided for HWU64 dataset. Defaulting to 'test'.")
        dataset_split = 'test'

    print(f"Loading HWU64 'small' dataset from Hugging Face, split: {dataset_split}...")
    try:
        # Load the dataset from Hugging Face with the 'small' configuration
        dataset = load_dataset_hf("DeepPavlov/hwu64", "default")
        
        # Select the specified split
        if dataset_split not in dataset:
            print(f"Error: Split '{dataset_split}' not found in HWU64 small. Available splits: {list(dataset.keys())}")
            # If the requested split is not found, return empty data
            return [], [], []
        else:
            data_split = dataset[dataset_split]

        # Extract texts and labels
        texts = data_split["utterance"]
        labels = data_split["label"]
        
        # Convert string labels to integers
        labels_list = []
        intent_mapping = {}
        current_map_id = 0
        
        for intent in labels:
            if intent not in intent_mapping:
                intent_mapping[intent] = current_map_id
                current_map_id += 1
            labels_list.append(intent_mapping[intent])
        
        documents = list(texts)

        print(f"Loaded {len(documents)} documents from HWU64 small.")
        print(f"Found {len(intent_mapping)} unique intents (clusters).")
        return documents, labels_list, documents  # For text datasets, raw_data is the documents list

    except Exception as e:
        print(f"Error loading HWU64 small dataset from Hugging Face: {e}")
        print("Please ensure 'datasets' library is installed and you have internet access.")
        return [], [], []  # Return empty on failure


def _load_opiec59k(path: str = "datasets/OPIEC59K/OPIEC59K_valid") -> Tuple[List[str], List[int], List[str]]:
    print("Loading OpieC59k dataset...")
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    documents = []
    labels = []
    label_mapping = {}
    current_label_id = 0
    
    for entry in data:
        tokens = entry.get("sentence_linked", {}).get("tokens", [])
        sentence = " ".join([t.get("word", "") for t in tokens]).strip()
        if not sentence:
            continue
            
        # Look for tokens with wiki_links
        wiki_link = None
        for token in tokens:
            link = token.get("w_link", {}).get("wiki_link", "")
            if link:
                wiki_link = link
                break
                
        if not wiki_link:
            continue  # skip if no wiki link
            
        # Map wiki_link to label ID
        if wiki_link not in label_mapping:
            label_mapping[wiki_link] = current_label_id
            current_label_id += 1
            
        label = label_mapping[wiki_link]
        documents.append(sentence)
        labels.append(label)
    
    print(f"Loaded {len(documents)} documents from OpieC59k.")
    print(f"Found {len(set(labels))} unique wiki links (clusters): {list(label_mapping.keys())[:10] + ['...'] if len(label_mapping) > 10 else list(label_mapping.keys())}")
    
    return documents, labels, documents


from typing import List, Tuple
from collections import Counter

def _load_reverb45k(path: str = "datasets/reverb45k_change/reverb45k_valid") -> Tuple[List[str], List[int], List[str]]:
    import json

    print("Loading ReVerb45k dataset...")
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    documents = []
    labels = []
    label_mapping = {}
    current_label_id = 0

    # Step 1: Collect all the labels
    for entry in data:
        triple = entry.get("triple", [])
        sentence = " ".join(triple).strip()
        if not sentence:
            continue

        wiki_object = entry.get("entity_linking", {}).get("object", "")
        if not wiki_object:
            continue

        if wiki_object not in label_mapping:
            label_mapping[wiki_object] = current_label_id
            current_label_id += 1

        label = label_mapping[wiki_object]
        documents.append(sentence)
        labels.append(label)

    # Step 2: Filter out single occurrence labels (clusters with only one item)
    label_counts = Counter(labels)
    filtered_docs = [doc for doc, lbl in zip(documents, labels) if label_counts[lbl] > 1]
    filtered_labels = [lbl for lbl in labels if label_counts[lbl] > 1]

    print(f"Loaded {len(filtered_docs)} triples from ReVerb45k after filtering out singleton clusters.")
    print(f"Found {len(set(filtered_labels))} unique object entities (clusters): "
          f"{list(set(filtered_labels))[:10] + ['...'] if len(set(filtered_labels)) > 10 else list(set(filtered_labels))}")

    return filtered_docs, filtered_labels, filtered_docs

# --- Main Dataset Loading Function ---

def load_dataset(dataset_name: str, cache_path: str = None, embedding_model: Embeddings = None, max_samples_per_class: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads a specified dataset (iris, clinc, tweet, bank77), its labels, and documents.
    Generates/loads features using the provided embedding_model if not cached for text datasets.

    Args:
        dataset_name: Name of the dataset to load ('iris', 'clinc', 'tweet', 'bank77').
        data_path: Optional path to the data file (needed for 'tweet').
        cache_path: Optional path to a *directory* to cache generated features.
        embedding_model: An initialized Langchain-compatible embedding model
                         to generate features if not cached. Required if cache_path is None
                         or cache doesn't exist for text datasets.
        max_samples_per_class: Optional limit on the number of samples per class.
                              If provided, will ensure balanced representation across classes.


    Returns:
        A tuple containing:
        - features (np.ndarray): The document features/embeddings or raw data points for Iris.
        - labels (np.ndarray): The true cluster labels.
        - documents (List[str]): The original document texts (empty for datasets without text).
         Returns empty features array and prints error if embedding fails or is missing
         when needed, but returns populated labels and documents if they could be loaded.
    """
    print(f"Loading dataset: {dataset_name}...")

    raw_data: List[Any] = [] # Can be numerical points (Iris) or text (others)
    labels_list: List[int] = []
    documents: List[str] = []
    features = np.array([]) # Initialize features as empty numpy array

    # --- Call appropriate helper to load raw data ---
    if dataset_name == "iris":
        raw_data, labels_list, documents = _load_iris_data()
        if raw_data: # If loading was successful
            features = np.array(raw_data) # For Iris, raw data is the features

    elif dataset_name == "clinc":
        raw_data, labels_list, documents = _load_clinc_data()
        # For text datasets, raw_data is the documents list, features will be embeddings

    elif dataset_name == "tweet":
        raw_data, labels_list, documents = _load_tweet_data()
        # For text datasets, raw_data is the documents list, features will be embeddings

    elif dataset_name == "bank77":
        raw_data, labels_list, documents = _load_bank77_data()
        # For text datasets, raw_data is the documents list, features will be embeddings
    elif dataset_name == "agnews":
        raw_data, labels_list, documents = _load_agnews_data()
    elif dataset_name == "hwu64":
        raw_data, labels_list, documents = _load_hwu64_data()
    elif dataset_name == "opiec59k":
        raw_data, labels_list, documents = _load_opiec59k()
    elif dataset_name == "reverb45k":
        raw_data, labels_list, documents = _load_reverb45k()

    else:
        print(f"Unknown dataset name: {dataset_name}")
        # Return empty for unknown dataset
        return np.array([]), np.array([]), []
    # --- Limit dataset size if max_samples_per_class is specified ---
    if max_samples_per_class is not None and labels_list:
        print(f"Limiting dataset to max {max_samples_per_class} samples per class...")
        
        # Convert to numpy array for easier manipulation
        labels_np_temp = np.array(labels_list)
        unique_labels = np.unique(labels_np_temp)
        
        # Track indices to keep
        indices_to_keep = []
        
        # For each class, randomly select up to max_samples_per_class
        for label in unique_labels:
            class_indices = np.where(labels_np_temp == label)[0]
            
            # If we have more samples than the limit, randomly select
            if len(class_indices) > max_samples_per_class:
                # Randomly select indices without replacement
                selected_indices = np.random.choice(
                    class_indices, 
                    size=max_samples_per_class, 
                    replace=False
                )
                indices_to_keep.extend(selected_indices)
            else:
                # Keep all samples for this class
                indices_to_keep.extend(class_indices)
        
        # Sort indices to maintain original order
        indices_to_keep.sort()
        
        # Update raw_data, labels_list, and documents
        if dataset_name == "iris":  # For numerical datasets
            raw_data = [raw_data[i] for i in indices_to_keep]
            features = np.array(raw_data)  # Update features for iris
        
        # Update labels and documents for all datasets
        labels_list = [labels_list[i] for i in indices_to_keep]
        documents = [documents[i] for i in indices_to_keep]
        
        print(f"Dataset limited to {len(labels_list)} samples total")
        print(f"Class distribution after limiting:")
        for label in np.unique(np.array(labels_list)):
            count = np.sum(np.array(labels_list) == label)
            print(f"  Class {label}: {count} samples")
    
    # --- Handle Feature Generation/Loading for Text-Based Datasets ---
    # This applies if raw_data/documents were loaded, but features are still empty (i.e., not Iris)
    if features.size == 0 and documents:
        print("Attempting to load or generate features from documents...")
        # Construct a specific cache file path within the cache_path directory
        dataset_cache_file_path = None
        if cache_path:
             # Ensure cache_path is treated as a directory
             cache_directory = cache_path # Assume cache_path is the directory

             # Get a name for the embedding model to include in the filename
             model_name = "unknown_model"
             if embedding_model:
                 # Use the class name of the embedding model
                 model_name = type(embedding_model).__name__
                 # Sanitize the name for use in a filename (replace invalid characters)
                 model_name = model_name.replace(" ", "_").replace("-", "_").replace("/", "_").lower()


             # Simple cache naming: dataset_name_split_modelname_embeddings.pkl
             # Assuming 'test' split for all text datasets loaded this way for simplicity
             cache_filename = f"{dataset_name}_test_{model_name}_embeddings.pkl"
             dataset_cache_file_path = os.path.join(cache_directory, cache_filename)
             print(f"Using cache file path: {dataset_cache_file_path}")


        if dataset_cache_file_path is not None and os.path.exists(dataset_cache_file_path):
             print(f"Loading embeddings from cache: {dataset_cache_file_path}")
             try:
                 with open(dataset_cache_file_path, 'rb') as f:
                      features = pickle.load(f)
                 print("Embeddings loaded from cache.")
             except Exception as e:
                 print(f"Error loading cache file {dataset_cache_file_path}: {e}")
                 features = np.array([]) # Set to empty so it will regenerate


        if features.size == 0: # If still no features (cache failed or doesn't exist)
            print("Generating embeddings using the provided embedding_model...")
            if embedding_model is None:
                print("Error: Embedding model not provided and cache not available.")
                # Return populated labels and documents even if embedding model is missing
                return np.array([]), np.array(labels_list), documents
            try:
                # Use the provided embedding_model to generate embeddings
                embeddings_list = embedding_model.embed_documents(documents)
                features = np.array(embeddings_list)
                print(f"Embeddings generated. Shape: {features.shape}")

                if dataset_cache_file_path is not None and features.size > 0:
                    print(f"Saving embeddings to cache: {dataset_cache_file_path}")
                    try:
                        # Ensure the *directory* for the cache file exists
                        cache_directory_for_file = os.path.dirname(dataset_cache_file_path)
                        if cache_directory_for_file and not os.path.exists(cache_directory_for_file):
                            os.makedirs(cache_directory_for_file, exist_ok=True) # Use exist_ok=True to avoid error if dir already exists
                        with open(dataset_cache_file_path, 'wb') as f:
                            pickle.dump(features, f)
                        print("Embeddings saved to cache.")
                    except Exception as e:
                        print(f"Error saving cache file {dataset_cache_file_path}: {e}")

            except Exception as e:
                print(f"Error generating embeddings with the provided model: {e}")
                # Return populated labels and documents even if embedding generation fails
                return np.array([]), np.array(labels_list), documents


    # --- Final Checks and Return ---
    # Ensure consistency in number of samples across features, labels, and documents if features were loaded/generated
    if features.shape[0] > 0 and (features.shape[0] != len(labels_list) or features.shape[0] != len(documents)):
        print(f"Warning: Inconsistent number of samples loaded for dataset {dataset_name}:")
        print(f"  Features: {features.shape[0]}, Labels: {len(labels_list)}, Documents: {len(documents)}")
        # Decide how to handle inconsistency - for now, proceed but warn.
        # A more robust solution might truncate or raise an error.
        # For now, we'll return what we have, but this might cause issues later.


    # Ensure labels are a numpy array before returning
    labels_np = np.array(labels_list)

    print(f"Successfully loaded dataset {dataset_name}.")
    print(f"Features shape: {features.shape}")
    print(f"Labels count: {len(labels_np)}")
    print(f"Documents count: {len(documents)}")
    if labels_np.size > 0:
         print(f"Number of unique gold clusters: {len(np.unique(labels_np))}")


    return features, labels_np, documents

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    # Define dummy paths for testing
    dummy_data_path = "./data"
    if not os.path.exists(dummy_data_path):
        os.makedirs(dummy_data_path)

    # Define a cache *directory* for testing
    dummy_cache_directory = "./cache_embeddings" # Changed to emphasize it's a directory
    if not os.path.exists(dummy_cache_directory):
        os.makedirs(dummy_cache_directory)

    # Create a dummy tweet.tsv file for testing the 'tweet' dataset
    dummy_tweet_path = os.path.join(dummy_data_path, "tweet.tsv")
    if not os.path.exists(dummy_tweet_path):
        print(f"Creating dummy tweet data file at {dummy_tweet_path}")
        with open(dummy_tweet_path, "w") as f:
            f.write("text\tlabel\n")
            f.write("This is tweet 1.\t0\n")
            f.write("This is tweet 2.\t0\n")
            f.write("Another tweet.\t1\n")
            f.write("Different topic.\t2\n")


    # Mock a dummy embedding model for testing the embedding generation path
    class DummyEmbeddingModel(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            print("Using DummyEmbeddingModel to generate embeddings...")
            # Return dummy embeddings (e.g., random vectors) with a fixed dimension
            return [[random.random() for _ in range(10)] for _ in texts]
        def embed_query(self, text: str) -> List[float]:
             # Implement if needed for query embedding
             pass

    dummy_embedding_model = DummyEmbeddingModel()

    # Mock another dummy embedding model with a different name
    class AnotherDummyModel(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            print("Using AnotherDummyModel to generate embeddings...")
            return [[random.random() for _ in range(20)] for _ in texts] # Different dimension
        def embed_query(self, text: str) -> List[float]:
             pass

    another_dummy_model = AnotherDummyModel()


    print("--- Testing Iris Data ---")
    # Pass the dummy_data_path, although not strictly needed for iris
    features_iris, labels_iris, docs_iris = load_dataset("iris", dummy_data_path)
    print(f"Iris Data - Features shape: {features_iris.shape}, Labels count: {len(labels_iris)}, Documents count: {len(docs_iris)}")
    print(f"Iris Data - Sample labels: {labels_iris[:10]}")


    print("\n--- Testing CLINC Data with DummyEmbeddingModel and Cache ---")
    # Pass the dummy_cache_directory as the cache_path
    features_clinc, labels_clinc, docs_clinc = load_dataset("clinc", dummy_data_path, cache_path=dummy_cache_directory, embedding_model=dummy_embedding_model)
    print(f"CLINC Data - Features shape: {features_clinc.shape}, Labels count: {len(labels_clinc)}, Documents count: {len(docs_clinc)}")
    print(f"CLINC Data - Sample labels: {labels_clinc[:10]}")
    print(f"CLINC Data - Sample document: {docs_clinc[0][:200]}...") # Print preview


    print("\n--- Testing CLINC Data with AnotherDummyModel and Cache ---")
    # This should create a *different* cache file for CLINC in the same directory
    features_clinc_another, labels_clinc_another, docs_clinc_another = load_dataset("clinc", dummy_data_path, cache_path=dummy_cache_directory, embedding_model=another_dummy_model)
    print(f"CLINC Data (Another Model) - Features shape: {features_clinc_another.shape}, Labels count: {len(labels_clinc_another)}, Documents count: {len(docs_clinc_another)}")
    print(f"CLINC Data (Another Model) - Sample labels: {labels_clinc_another[:10]}")


    print("\n--- Testing Tweet Data with Dummy Embeddings and Cache ---")
    # Pass the dummy_tweet_path as data_path and dummy_cache_directory as cache_path
    features_tweet, labels_tweet, docs_tweet = load_dataset("tweet", dummy_tweet_path, cache_path=dummy_cache_directory, embedding_model=dummy_embedding_model)
    print(f"Tweet Data - Features shape: {features_tweet.shape}, Labels count: {len(labels_tweet)}, Documents count: {len(docs_tweet)}")
    print(f"Tweet Data - Sample labels: {labels_tweet[:10]}")
    print(f"Tweet Data - Sample document: {docs_tweet[0][:200]}...") # Print preview


    print("\n--- Testing BANKING77 Data with Dummy Embeddings and Cache ---")
    # Pass the dummy_data_path and dummy_cache_directory as cache_path
    features_bank, labels_bank, docs_bank = load_dataset("bank77", dummy_data_path, cache_path=dummy_cache_directory, embedding_model=dummy_embedding_model)
    print(f"BANKING77 Data - Features shape: {features_bank.shape}, Labels count: {len(labels_bank)}, Documents count: {len(docs_bank)}")
    print(f"BANKING77 Data - Sample labels: {labels_bank[:10]}")
    print(f"BANKING77 Data - Sample document: {docs_bank[0][:200]}...") # Print preview

