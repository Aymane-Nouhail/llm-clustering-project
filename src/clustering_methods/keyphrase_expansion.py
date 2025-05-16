import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Tuple
from src.llm_service import LLMService, KeyphraseList
import concurrent.futures
import pandas as pd
from tqdm import tqdm
import os

def process_document(doc_index: int, document: str, features: np.ndarray, 
                    llm_service: LLMService, prompt_template: ChatPromptTemplate, 
                    embedding_dim: int) -> Tuple[int, str, List[str], np.ndarray, np.ndarray]:
    """Process a single document for keyphrase expansion.
    
    Args:
        doc_index: Index of document in original list
        document: Text content of document
        features: Original document embeddings
        llm_service: LLM service for keyphrase generation
        prompt_template: Template for keyphrase generation prompt
        embedding_dim: Expected dimension of embeddings
        
    Returns:
        Tuple containing:
        - Original document index
        - Document text
        - Generated keyphrases
        - Normalized original embedding (or None if failed)
        - Normalized expanded embedding (or None if failed)
    """
    try:
        prompt = prompt_template.format(document_text=document)
        llm_response = llm_service.get_chat_completion(prompt, output_structure=KeyphraseList)
        keyphrases = llm_response.keyphrases if llm_response and hasattr(llm_response, 'keyphrases') else []
        
        joined_text = ", ".join([document] + keyphrases)
        expansion_embedding = np.array(llm_service.get_embedding(joined_text)) if keyphrases else None
        
        orig_feature = features[doc_index].reshape(1, -1)
        orig_norm = normalize(orig_feature, axis=1, norm='l2').flatten() if np.linalg.norm(orig_feature) > 0 else np.zeros_like(orig_feature).flatten()
        
        exp_norm = None
        if expansion_embedding is not None and len(expansion_embedding) == embedding_dim:
            exp_2d = expansion_embedding.reshape(1, -1)
            exp_norm = normalize(exp_2d, axis=1, norm='l2').flatten() if np.linalg.norm(exp_2d) > 0 else np.zeros_like(exp_2d).flatten()
            
        return (doc_index, document, keyphrases, orig_norm, exp_norm)
    except Exception:
        return (doc_index, document, [], None, None)

def cluster_via_keyphrase_expansion(
    documents: List[str],
    features: np.ndarray,
    n_clusters: int,
    llm_service: LLMService,
    keyphrase_prompt_template: str,
    keyphrase_output_csv_path: str = "keyphrase_expansions_output.csv"
) -> Dict[str, np.ndarray]:
    """Perform clustering using LLM-generated keyphrase expansions.
    
    Args:
        documents: List of document texts
        features: Original document embeddings
        n_clusters: Number of clusters to create
        llm_service: Initialized LLM service
        keyphrase_prompt_template: Template for keyphrase generation
        keyphrase_output_csv_path: Path to save generated keyphrases
        
    Returns:
        Dictionary with clustering results for different methods:
        - 'concatenated': Using concatenated original+expanded features
        - 'average': Using averaged features  
        - 'weighted_X': Using weighted features (X = weight 0.1-1.0)
        Each value is a numpy array of cluster assignments (-1 for failed docs)
    """
    if not llm_service.is_available():
        return {'concatenated': None, 'average': None}

    prompt_template = ChatPromptTemplate.from_template(keyphrase_prompt_template + "\nDocument: {document_text}")
    n_samples = len(documents)
    embedding_dim = features.shape[1]
    results = [None] * n_samples
    max_workers = min(50, n_samples)

    # Process documents in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_document,
                i, documents[i], features, llm_service, prompt_template, embedding_dim
            ): i for i in range(n_samples)
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=n_samples, desc="Processing"):
            results[futures[future]] = future.result()

    # Prepare results for CSV and clustering
    csv_data = []
    original_features = []
    expanded_features = [] 
    successful_indices = []
    
    for i, result in enumerate(results):
        if result:
            doc_index, doc, keyphrases, orig, exp = result
            csv_data.append({
                "document_index": doc_index,
                "document_text": doc,
                "generated_keyphrases": ", ".join(keyphrases)
            })
            if orig is not None and exp is not None:
                original_features.append(orig)
                expanded_features.append(exp)
                successful_indices.append(doc_index)

    # Save keyphrases to CSV
    if csv_data:
        os.makedirs(os.path.dirname(keyphrase_output_csv_path), exist_ok=True)
        pd.DataFrame(csv_data).to_csv(keyphrase_output_csv_path, index=False)

    # Cluster using different feature combinations
    cluster_results = {}
    if not original_features:
        return {'concatenated': None, 'average': None}

    orig_features = np.array(original_features)
    exp_features = np.array(expanded_features)
    full_assignments = np.full(n_samples, -1, dtype=int)

    # Helper function for clustering
    def run_clustering(features, method_name):
        try:
            clusters = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit_predict(features)
            for i, idx in enumerate(successful_indices):
                full_assignments[idx] = clusters[i]
            return full_assignments.copy()
        except Exception:
            return None

    # Run all clustering variants
    cluster_results['concatenated'] = run_clustering(np.hstack([orig_features, exp_features]), 'concatenated')
    cluster_results['average'] = run_clustering((orig_features + exp_features) / 2, 'average')
    
    for weight in [round(w, 1) for w in np.arange(0.1, 1.1, 0.1)]:
        weighted = (1-weight)*orig_features + weight*exp_features
        cluster_results[f'weighted_{weight}'] = run_clustering(weighted, f'weighted_{weight}')

    return cluster_results