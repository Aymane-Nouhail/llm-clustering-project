import numpy as np
import os
import pickle
import json
from typing import List, Tuple, Any, Optional, Callable, Dict
from langchain_core.embeddings import Embeddings
from datasets import load_dataset as load_dataset_hf
from sklearn import datasets as sklearn_datasets
from collections import defaultdict, Counter

def _balance_classes(docs: List[str], labels: List[int], raw_data: List[Any], 
                    max_samples: int) -> Tuple[List[str], List[int], List[Any]]:
    """Balance classes by limiting samples per class."""
    if not max_samples or not labels:
        return docs, labels, raw_data
    
    labels_np = np.array(labels)
    indices = []
    for lbl in np.unique(labels_np):
        class_idx = np.where(labels_np == lbl)[0]
        indices.extend(np.random.choice(class_idx, min(len(class_idx), max_samples), replace=False))
    indices.sort()
    
    balanced_docs = [docs[i] for i in indices]
    balanced_labels = [labels[i] for i in indices]
    balanced_raw = [raw_data[i] for i in indices] if raw_data else balanced_docs
    
    print(f"Balanced to {len(balanced_labels)} samples ({max_samples}/class)")
    return balanced_docs, balanced_labels, balanced_raw

def _get_embeddings(docs: List[str], cache_path: Optional[str], 
                   embedding_model: Embeddings, dataset_name: str) -> np.ndarray:
    """Get embeddings either from cache or by generating them."""
    if not docs or not embedding_model:
        return np.array([])
    
    cache_file = None
    if cache_path:
        model_name = type(embedding_model).__name__.lower().replace(" ", "_")
        cache_file = f"{cache_path}/{dataset_name}_{model_name}_embeddings.pkl"
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Cache load failed: {e}")
    
    try:
        embeddings = np.array(embedding_model.embed_documents(docs))
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
        return embeddings
    except Exception as e:
        print(f"Embedding failed: {e}")
        return np.array([])

def _load_iris() -> Tuple[List[List[float]], List[int], List[str]]:
    """Load Iris dataset."""
    try:
        X, y = sklearn_datasets.load_iris(return_X_y=True)
        return X.tolist(), y.tolist(), [""]*len(X)
    except Exception as e:
        print(f"Error loading Iris: {e}")
        return [], [], []

def _load_clinc() -> Tuple[List[str], List[int], List[str]]:
    """Load CLINC dataset"""
    try:
        dataset = load_dataset_hf("clinc_oos", "small")["test"]
        texts = dataset["text"]
        intents = dataset["intent"]
        
        # Filter out intent 42 and remap remaining intents
        filtered_data = [(text, intent) for text, intent in zip(texts, intents) if intent != 42]
        filtered_texts, filtered_intents = zip(*filtered_data)
        
        # Remap intents to contiguous integers
        intent_mapping = {intent: idx for idx, intent in enumerate(sorted(set(filtered_intents)))}
        remapped_intents = [intent_mapping[intent] for intent in filtered_intents]
        
        print(f"Loaded CLINC: {len(filtered_texts)} samples, {len(intent_mapping)} intents")
        return list(filtered_texts), remapped_intents, list(filtered_texts)
    except Exception as e:
        print(f"Error loading CLINC dataset: {e}")
        return [], [], []
    

def _load_hf_dataset(dataset: str, text_field: str, label_field: str, 
                    filter_func: Optional[Callable] = None) -> Tuple[List[str], List[int], List[str]]:
    """Load HuggingFace dataset with optional filtering."""
    try:
        data = load_dataset_hf(*dataset.split('/'))['test']
        texts, labels = data[text_field], data[label_field]
        
        if filter_func:
            texts, labels = zip(*filter(filter_func, zip(texts, labels)))
        
        if isinstance(labels[0], str):
            label_map = {v:i for i,v in enumerate(set(labels))}
            labels = [label_map[l] for l in labels]
            
        return list(texts), list(labels), list(texts)
    except Exception as e:
        print(f"Error loading {dataset}: {e}")
        return [], [], []

def _load_opiec59k(path: str = "datasets/OPIEC59K/OPIEC59K_valid") -> Tuple[List[str], List[int], List[str]]:
    """Load OPIEC59K dataset from pickle file."""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        documents = []
        labels = []
        label_mapping = {}
        current_id = 0
        
        for entry in data:
            tokens = entry.get("sentence_linked", {}).get("tokens", [])
            sentence = " ".join(t.get("word", "") for t in tokens).strip()
            if not sentence:
                continue
                
            wiki_link = next((t.get("w_link", {}).get("wiki_link", "") for t in tokens if t.get("w_link", {}).get("wiki_link", "")), None)
            if not wiki_link:
                continue
                
            if wiki_link not in label_mapping:
                label_mapping[wiki_link] = current_id
                current_id += 1
                
            documents.append(sentence)
            labels.append(label_mapping[wiki_link])
        
        print(f"Loaded OPIEC59K: {len(documents)} samples, {len(label_mapping)} entities")
        return documents, labels, documents
    except Exception as e:
        print(f"Error loading OPIEC59K: {e}")
        return [], [], []

def _load_reverb45k(path: str = "datasets/reverb45k_change/reverb45k_valid") -> Tuple[List[str], List[int], List[str]]:
    """Load ReVerb45K dataset from JSON lines."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        documents = []
        labels = []
        label_mapping = {}
        current_id = 0
        
        for entry in data:
            triple = entry.get("triple", [])
            sentence = " ".join(triple).strip()
            if not sentence:
                continue
                
            wiki_object = entry.get("entity_linking", {}).get("object", "")
            if not wiki_object:
                continue
                
            if wiki_object not in label_mapping:
                label_mapping[wiki_object] = current_id
                current_id += 1
                
            documents.append(sentence)
            labels.append(label_mapping[wiki_object])
        
        # Filter singleton clusters
        label_counts = Counter(labels)
        filtered_docs = [doc for doc, lbl in zip(documents, labels) if label_counts[lbl] > 1]
        filtered_labels = [lbl for lbl in labels if label_counts[lbl] > 1]
        
        print(f"Loaded ReVerb45K: {len(filtered_docs)} samples, {len(set(filtered_labels))} entities after filtering")
        return filtered_docs, filtered_labels, filtered_docs
    except Exception as e:
        print(f"Error loading ReVerb45K: {e}")
        return [], [], []

def load_dataset(
    dataset_name: str, 
    cache_path: Optional[str] = None,
    embedding_model: Optional[Embeddings] = None,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Main dataset loading function with all supported datasets."""
    if dataset_name == "iris":
        docs, labels, raw_data = _load_iris()
    elif dataset_name == "clinc":
        docs, labels, raw_data = _load_clinc()  # Simplified call
    elif dataset_name == "tweet":
        docs, labels, raw_data = _load_hf_dataset(
            dataset='cardiffnlp/tweet_eval/sentiment',
            text_field='text',
            label_field='label'
        )
    elif dataset_name == "bank77":
        docs, labels, raw_data = _load_hf_dataset(
            dataset='banking77',
            text_field='text',
            label_field='label'
        )
    elif dataset_name == "agnews":
        docs, labels, raw_data = _load_hf_dataset(
            dataset='ag_news',
            text_field='text',
            label_field='label'
        )
    elif dataset_name == "hwu64":
        docs, labels, raw_data = _load_hf_dataset(
            dataset='DeepPavlov/hwu64/default',
            text_field='utterance',
            label_field='label'
        )
    elif dataset_name == "opiec59k":
        docs, labels, raw_data = _load_opiec59k()
    elif dataset_name == "reverb45k":
        docs, labels, raw_data = _load_reverb45k()
    else:
        print(f"Unknown dataset: {dataset_name}")
        return np.array([]), np.array([]), []
    
    # Balance classes if requested
    docs, labels, raw_data = _balance_classes(docs, labels, raw_data, max_samples)
    
    # Get features
    features = (
        np.array(raw_data) if dataset_name == "iris"
        else _get_embeddings(docs, cache_path, embedding_model, dataset_name)
    )
    
    print(f"Loaded {dataset_name}: {len(docs)} samples, {len(set(labels))} classes")
    return features, np.array(labels), docs