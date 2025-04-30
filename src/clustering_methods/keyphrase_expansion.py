import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from langchain_core.prompts import ChatPromptTemplate
from src.llm_service import LLMService, KeyphraseList
from typing import List, Dict, Any
import concurrent.futures
import time


def process_document_for_expansion(doc_index: int, document: str, features: np.ndarray, llm_service: LLMService, prompt_template: ChatPromptTemplate, original_embedding_dim: int) -> np.ndarray | None:
    # Add prints to trace execution within each thread
    print(f"--- Starting processing for doc {doc_index+1} ---") # Add this print to see thread start
    try:
        # 1. Generate keyphrases using LLM with structured output
        prompt = prompt_template.format(document_text=document)
        print(f"  Doc {doc_index+1}: Calling LLM for generation...") # See if it gets here
        llm_response_structured = llm_service.get_chat_completion(prompt, output_structure=KeyphraseList)
        print(f"  Doc {doc_index+1}: LLM generation call finished.") # See if this finishes

        keyphrases = []
        if llm_response_structured is not None:
            if hasattr(llm_response_structured, 'keyphrases') and isinstance(llm_response_structured.keyphrases, list):
                 keyphrases = llm_response_structured.keyphrases
            # Warning about unexpected structure is now handled within LLMService or earlier


        # 2. Prepare text for expansion encoding
        all_text_for_encoding = [document]
        if keyphrases:
            all_text_for_encoding.extend(keyphrases)
        joined_text_for_encoding = ", ".join(all_text_for_encoding)

        # 3. Encode the joined text to get the expansion embedding
        print(f"  Doc {doc_index+1}: Calling LLM for embedding...") # See if it gets here
        expansion_embedding_list = llm_service.get_embedding(joined_text_for_encoding)
        print(f"  Doc {doc_index+1}: LLM embedding call finished.") # See if this finishes

        if not expansion_embedding_list or len(expansion_embedding_list) != original_embedding_dim:
             print(f"  Doc {doc_index+1}: Embedding failed or dimension mismatch ({len(expansion_embedding_list)} vs {original_embedding_dim}). Using zero vector.")
             expansion_embedding = np.zeros(original_embedding_dim)
        else:
             expansion_embedding = np.array(expansion_embedding_list)

        # 4. Normalize original and expansion embeddings
        original_feature = features[doc_index].reshape(1, -1)
        expansion_embedding_2d = expansion_embedding.reshape(1, -1)

        # Handle zero vectors before normalization
        if np.linalg.norm(original_feature) == 0:
             # print(f"  Doc {doc_index+1}: Warning: Original feature vector is zero.") # Too noisy?
             original_feature_normalized = np.zeros_like(original_feature).flatten()
        else:
             original_feature_normalized = normalize(original_feature, axis=1, norm='l2').flatten()

        if np.linalg.norm(expansion_embedding_2d) == 0:
             # print(f"  Doc {doc_index+1}: Warning: Expansion embedding vector is zero.") # Too noisy?
             expansion_embedding_normalized = np.zeros_like(expansion_embedding_2d).flatten()
        else:
             expansion_embedding_normalized = normalize(expansion_embedding_2d, axis=1, norm='l2').flatten()


        # 5. Concatenate
        expanded_feature = np.concatenate([original_feature_normalized, expansion_embedding_normalized])

        print(f"--- Finished processing for doc {doc_index+1} ---") # Add this print to see thread finish
        return expanded_feature

    except Exception as e:
        # Print a clear message if an exception occurs within a thread
        print(f"--- Exception processing doc {doc_index+1}: {e} ---") # See if any exceptions are caught here
        return None


# ... (keep the rest of the cluster_via_keyphrase_expansion function the same)
def cluster_via_keyphrase_expansion(
    documents: List[str],
    features: np.ndarray,
    n_clusters: int,
    llm_service: LLMService,
    keyphrase_prompt_template: str
) -> np.ndarray | None:
    """
    Implements clustering via LLM keyphrase expansion (Section 2.1),
    using parallel processing for LLM calls.

    Args:
        documents: List of original document texts.
        features: Original document embeddings (NumPy array).
        n_clusters: The number of clusters.
        llm_service: An initialized LLMService instance.
        keyphrase_prompt_template: A string template for the core instruction.

    Returns:
        A NumPy array of cluster assignments, or None if clustering fails.
    """
    print("\n--- Running Clustering via LLM Keyphrase Expansion ---")
    if not llm_service.is_available():
        print("LLMService is not available. Cannot run keyphrase expansion.")
        return None

    n_samples = len(documents)
    original_embedding_dim = features.shape[1]

    prompt_template = ChatPromptTemplate.from_template(keyphrase_prompt_template + "\nDocument: {document_text}")

    expanded_features_dict: Dict[int, np.ndarray] = {}
    MAX_WORKERS = 40 # Adjust based on your machine and OpenAI rate limits

    print(f"Processing {n_samples} documents in parallel with {MAX_WORKERS} workers...")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_doc_index = {
            executor.submit(
                process_document_for_expansion,
                i, doc, features, llm_service, prompt_template, original_embedding_dim
            ): i for i, doc in enumerate(documents)
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_doc_index):
            doc_index = future_to_doc_index[future]
            try:
                # Calling future.result() waits for the specific thread to complete
                expanded_feature = future.result()
                if expanded_feature is not None:
                    expanded_features_dict[doc_index] = expanded_feature
                # Optional: Print simple progress
                # print(f"  Completed {len(expanded_features_dict)}/{n_samples} documents.")

            except Exception as exc:
                # This catches exceptions from the process_document_for_expansion helper
                print(f"  Document {doc_index+1} generated an exception caught in main loop: {exc}")


    end_time = time.time()
    print(f"Finished parallel processing. Time taken: {end_time - start_time:.2f} seconds.")

    # Reassemble features in the original order and filter out failures
    expanded_features_list_ordered = [expanded_features_dict.get(i) for i in range(n_samples)]
    successfully_processed_features = [f for f in expanded_features_list_ordered if f is not None]

    if not successfully_processed_features:
        print("No documents were successfully processed for keyphrase expansion.")
        return None

    expanded_features = np.array(successfully_processed_features)
    print(f"Proceeding with KMeans on {expanded_features.shape[0]} successfully expanded documents. Shape: {expanded_features.shape}")

    # Note: If some documents failed, the number of returned assignments will
    # be less than the original number of documents. The evaluation will
    # need to handle this if you compare against original labels.

    # 6. Perform standard K-Means on expanded features
    print(f"\nRunning KMeans on expanded features...")
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        cluster_assignments = kmeans.fit_predict(expanded_features)
        print("KMeans completed.")
        return cluster_assignments
    except Exception as e:
        print(f"Error during KMeans clustering: {e}")
        return None