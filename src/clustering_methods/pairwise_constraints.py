import numpy as np
import random
import pandas as pd
from typing import Any
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate
from src.llm_service import LLMService
from src.metrics import calculate_clustering_metrics
from typing import List, Dict, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
import os

METRICS_CSV_PATH = "clustering_metrics_results.csv"

def process_pair(pair_data: Tuple[int, int, str, str, ChatPromptTemplate, LLMService]) -> Dict[str, Any]:
    """Process a single document pair to generate constraint via LLM.
    
    Args:
        pair_data: Tuple containing (idx1, idx2, doc1, doc2, prompt_template, llm_service)
        
    Returns:
        Dictionary containing:
        - pair_indices: Tuple of document indices
        - doc1_full_text: First document text
        - doc2_full_text: Second document text  
        - full_prompt: Complete LLM prompt used
        - raw_llm_response: Raw LLM response
        - constraint_type: Type of constraint generated
        - constraint: Actual constraint tuple (type, indices) or None
    """
    idx1, idx2, doc1, doc2, template, llm = pair_data
    
    prompt = template.format(text1=doc1, text2=doc2)
    response = ""
    constraint = None
    
    try:
        response = llm.get_chat_completion(prompt).strip().upper()
        if response == "YES":
            constraint = ("MUST", (idx1, idx2))
            const_type = "Must-Link"
        elif response == "NO":
            constraint = ("CANNOT", (idx1, idx2)) 
            const_type = "Cannot-Link"
        else:
            const_type = "Ambiguous Response"
    except Exception as e:
        const_type = f"Error: {str(e)[:50]}..."
    
    return {
        "pair_indices": f"({idx1}, {idx2})",
        "doc1_full_text": doc1,
        "doc2_full_text": doc2,
        "full_prompt": prompt,
        "raw_llm_response": response,
        "constraint_type": const_type,
        "constraint": constraint
    }

def cluster_via_pairwise_constraints(
    dataset_name: str,
    documents: List[str],
    features: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    llm_service: LLMService,
    prompt_template: str,
    num_pairs: int,
    strategy: str = 'random',
    output_path: str = "pairwise_queries_output.csv",
    max_workers: int = 50
) -> np.ndarray:
    """Perform semi-supervised clustering using LLM-generated pairwise constraints.
    
    Args:
        dataset_name: Name of dataset being processed
        documents: List of document texts
        features: Document embeddings
        labels: Ground truth labels for evaluation
        n_clusters: Number of clusters to create
        llm_service: Initialized LLM service
        prompt_template: Template for pairwise comparison prompts
        num_pairs: Number of document pairs to query
        strategy: Pair selection strategy ('random' or 'similarity')
        output_path: Path to save query results
        max_workers: Max parallel threads for LLM queries
        
    Returns:
        Array of cluster assignments or None if clustering fails
    """
    print("\n--- Running Pairwise Constraint Clustering ---")
    
    # 1. Select document pairs
    n_samples = len(documents)
    all_pairs = [(i, j) for i in range(n_samples) for j in range(i+1, n_samples)]
    num_pairs = min(num_pairs, len(all_pairs))
    
    if strategy == 'similarity':
        sim_matrix = cosine_similarity(features)
        sim_pairs = [((i,j), sim_matrix[i,j]) for i,j in all_pairs]
        sim_pairs.sort(key=lambda x: x[1])
        selected = [p for p,s in sim_pairs[:num_pairs//2]] + [p for p,s in sim_pairs[-num_pairs//2:]]
    else:
        selected = random.sample(all_pairs, num_pairs)
    
    # 2. Generate constraints via parallel LLM queries
    must_links = []
    cannot_links = []
    query_data = []
    template = ChatPromptTemplate.from_template(prompt_template)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_pair, 
                (i, j, documents[i], documents[j], template, llm_service)
            ): (i,j) for i,j in selected
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Querying LLM"):
            result = future.result()
            query_data.append(result)
            
            if result["constraint"]:
                const_type, (i,j) = result["constraint"]
                if const_type == "MUST":
                    must_links.append((i,j))
                else:
                    cannot_links.append((i,j))
    
    # Save query results
    if query_data:
        pd.DataFrame(query_data).to_csv(output_path, index=False)
    
    # 3. Run constrained clustering
    assignments = None
    if must_links or cannot_links:
        try:
            pckmeans = PCKMeans(n_clusters=n_clusters)
            pckmeans.fit(features, ml=must_links, cl=cannot_links)
            assignments = pckmeans.labels_
        except Exception as e:
            print(f"Clustering error: {e}")
    
    # Save metrics
    metrics = calculate_clustering_metrics(labels, assignments, n_clusters) if assignments is not None else {}
    metrics_data = {
        "Dataset": dataset_name,
        "Method": "Pairwise Constraints",
        "Status": "Success" if assignments is not None else "Failed",
        **metrics
    }
    
    pd.DataFrame([metrics_data]).to_csv(
        METRICS_CSV_PATH,
        mode='a',
        header=not os.path.exists(METRICS_CSV_PATH),
        index=False
    )
    
    return assignments