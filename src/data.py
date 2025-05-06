import os
import pickle
import random
from typing import Tuple, List, Any

import numpy as np
import pandas as pd
from datasets import load_dataset as load_dataset_hf
from langchain_core.embeddings import Embeddings
from sklearn import datasets as sklearn_datasets

# --- Dataset Loaders ---

def _load_iris() -> Tuple[List[Any], List[int], List[str]]:
    print("Loading Iris dataset...")
    samples_raw, labels_raw = sklearn_datasets.load_iris(return_X_y=True)
    return samples_raw.tolist(), list(labels_raw), [""] * len(samples_raw)

def _load_clinc() -> Tuple[List[str], List[int], List[str]]:
    print("Loading CLINC dataset...")
    ds = load_dataset_hf("clinc_oos", "small")["test"]
    texts, intents = ds["text"], ds["intent"]
    filtered = [(t, i) for t, i in zip(texts, intents) if i != 42]
    docs, labels = zip(*filtered)
    label_map = {v: i for i, v in enumerate(sorted(set(labels)))}
    remapped = [label_map[l] for l in labels]
    return list(docs), remapped, list(docs)

def _load_opiec59k() -> Tuple[List[str], List[int], List[str]]:
    print("Loading OPIEC59K...")
    ds = load_dataset_hf("openpef/opiec59k")["test"]
    docs = ds["sentence"]
    rels = ds["relation"]
    label_map = {v: i for i, v in enumerate(sorted(set(rels)))}
    labels = [label_map[r] for r in rels]
    return docs, labels, docs

def _load_reverb45k() -> Tuple[List[str], List[int], List[str]]:
    print("Loading ReVerb45K...")
    ds = load_dataset_hf("openpef/reverb45k")["test"]
    docs = ds["sentence"]
    rels = ds["relation"]
    label_map = {v: i for i, v in enumerate(sorted(set(rels)))}
    labels = [label_map[r] for r in rels]
    return docs, labels, docs

LOADERS = {
    "iris": _load_iris,
    "clinc": _load_clinc,
    "opiec59k": _load_opiec59k,
    "reverb45k": _load_reverb45k,
}

# --- Main Loader ---

def load_dataset(dataset_name: str,
                 cache_path: str = None,
                 embedding_model: Embeddings = None,
                 max_samples_per_class: int = None
                 ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    print(f"\n--- Loading dataset: {dataset_name} ---")

    if dataset_name not in LOADERS:
        print(f"Dataset '{dataset_name}' not supported.")
        return np.array([]), np.array([]), []

    raw_data, labels, documents = LOADERS[dataset_name]()

    if max_samples_per_class is not None:
        print(f"Limiting to {max_samples_per_class} samples per class...")
        indices = []
        labels_np = np.array(labels)
        for lbl in np.unique(labels_np):
            idx = np.where(labels_np == lbl)[0]
            chosen = np.random.choice(idx, min(len(idx), max_samples_per_class), replace=False)
            indices.extend(chosen)
        indices.sort()
        raw_data = [raw_data[i] for i in indices]
        documents = [documents[i] for i in indices]
        labels = [labels[i] for i in indices]

    if dataset_name == "iris":
        return np.array(raw_data), np.array(labels), documents

    # Handle embeddings
    features = np.array([])
    if documents:
        model_name = type(embedding_model).__name__.lower() if embedding_model else "no_model"
        if cache_path:
            os.makedirs(cache_path, exist_ok=True)
            cache_file = os.path.join(cache_path, f"{dataset_name}_{model_name}_embeddings.pkl")
            if os.path.exists(cache_file):
                print(f"Loading embeddings from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    features = pickle.load(f)
            elif embedding_model:
                print("Generating embeddings...")
                features = np.array(embedding_model.embed_documents(documents))
                with open(cache_file, 'wb') as f:
                    pickle.dump(features, f)
            else:
                print("No embedding model provided and no cache found.")
        elif embedding_model:
            features = np.array(embedding_model.embed_documents(documents))
        else:
            print("Embedding model missing and no cache directory specified.")

    return features, np.array(labels), documents


# --- Dummy Model for Testing ---

class DummyEmbeddingModel(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print("Generating dummy embeddings...")
        return [[random.random() for _ in range(10)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [random.random() for _ in range(10)]


# --- Example Usage ---

if __name__ == '__main__':
    dummy_model = DummyEmbeddingModel()
    cache_dir = "./cache"

    for name in ["iris", "clinc", "opiec59k", "reverb45k"]:
        features, labels, docs = load_dataset(name, cache_path=cache_dir, embedding_model=dummy_model, max_samples_per_class=5)
        print(f"{name.upper()} => Features: {features.shape}, Labels: {len(labels)}, Documents: {len(docs)}\n")
