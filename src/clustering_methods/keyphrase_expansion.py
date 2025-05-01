import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from langchain_core.prompts import ChatPromptTemplate
# Import necessary types for type hinting
from typing import List, Dict, Any, Type, Tuple
# Import BaseModel for structured output definition
from pydantic import BaseModel
# Import LLMService and the specific KeyphraseList model
from src.llm_service import LLMService, KeyphraseList
# Import for parallel execution
import concurrent.futures
# Import for time measurement
import time
# Import os for path manipulation
import os
# Import pandas for CSV handling
import pandas as pd
# Import tqdm for the loading bar
from tqdm import tqdm


# Define the helper function that processes a single document in parallel
# This function now returns a tuple containing processing results, including keyphrases and the expanded feature
def process_document_for_expansion(doc_index: int, document: str, features: np.ndarray, llm_service: LLMService, prompt_template: ChatPromptTemplate, original_embedding_dim: int) -> Tuple[int, str, List[str], np.ndarray | None]:
    """
    Helper function to process a single document for keyphrase expansion,
    intended to be run in a thread or asynchronously.
    Returns (document_index, document_text, generated_keyphrases, expanded_feature_or_None).
    """
    # print(f"--- Starting processing for doc {doc_index+1} ---") # Uncomment for verbose logging per document
    try:
        # 1. Generate keyphrases using LLM with structured output
        prompt = prompt_template.format(document_text=document)
        # print(f"  Doc {doc_index+1}: Calling LLM for generation...") # Uncomment for verbose logging
        # Call the LLM service requesting structured output using the KeyphraseList model
        llm_response_structured = llm_service.get_chat_completion(prompt, output_structure=KeyphraseList)
        # print(f"  Doc {doc_index+1}: LLM generation call finished.") # Uncomment for verbose logging

        keyphrases = []
        # Check if the structured response was successful and contains the expected keyphrases list
        if llm_response_structured is not None:
            if hasattr(llm_response_structured, 'keyphrases') and isinstance(llm_response_structured.keyphrases, list):
                 keyphrases = llm_response_structured.keyphrases
            # Warning about unexpected structure is now handled within LLMService or earlier

        # 2. Prepare text for expansion encoding (Join original text and keyphrases)
        all_text_for_encoding = [document]
        if keyphrases: # Check if the list of keyphrases is non-empty
            all_text_for_encoding.extend(keyphrases)

        joined_text_for_encoding = ", ".join(all_text_for_encoding)

        # 3. Encode the joined text to get the expansion embedding
        # print(f"  Doc {doc_index+1}: Calling LLM for embedding...") # Uncomment for verbose logging
        expansion_embedding_list = llm_service.get_embedding(joined_text_for_encoding)
        # print(f"  Doc {doc_index+1}: LLM embedding call finished.") # Uncomment for verbose logging

        expanded_feature = None # Initialize expanded_feature as None in case of embedding issues

        # Check if embedding was successful and has the correct dimension
        if not expansion_embedding_list or len(expansion_embedding_list) != original_embedding_dim:
             # print(f"  Doc {doc_index+1}: Embedding failed or dimension mismatch ({len(expansion_embedding_list)} vs {original_embedding_dim}). Expanded feature will be None.") # Optional warning
             pass # expanded_feature remains None
        else:
             expansion_embedding = np.array(expansion_embedding_list)

             # 4. Normalize original and expansion embeddings
             # Reshape original feature for normalization
             original_feature = features[doc_index].reshape(1, -1)
             # Reshape expansion embedding for normalization
             expansion_embedding_2d = expansion_embedding.reshape(1, -1)

             # Handle zero vectors before normalization (can lead to NaNs)
             if np.linalg.norm(original_feature) == 0:
                  # print(f"  Doc {doc_index+1}: Warning: Original feature vector is zero.") # Optional warning
                  original_feature_normalized = np.zeros_like(original_feature).flatten()
             else:
                  original_feature_normalized = normalize(original_feature, axis=1, norm='l2').flatten()

             if np.linalg.norm(expansion_embedding_2d) == 0:
                  # print(f"  Doc {doc_index+1}: Warning: Expansion embedding vector is zero.") # Optional warning
                  expansion_embedding_normalized = np.zeros_like(expansion_embedding_2d).flatten()
             else:
                  expansion_embedding_normalized = normalize(expansion_embedding_2d, axis=1, norm='l2').flatten()

             # 5. Concatenate the normalized features
             expanded_feature = np.concatenate([original_feature_normalized, expansion_embedding_normalized])

        # print(f"--- Finished processing for doc {doc_index+1} ---") # Uncomment for verbose logging
        # Return the results as a tuple
        return (doc_index, document, keyphrases, expanded_feature)

    except Exception as e:
        # Print a clear message if an exception occurs within a thread
        print(f"\n--- Exception processing doc {doc_index+1}: {e} ---") # Added newline for clarity
        # Return results indicating failure for the feature but keeping other info
        return (doc_index, document, [], None) # Indicate failure for feature but keep other info


# The main function for keyphrase expansion clustering
# This function orchestrates the parallel processing and results handling
def cluster_via_keyphrase_expansion(
    documents: List[str],
    features: np.ndarray,
    n_clusters: int,
    llm_service: LLMService,
    keyphrase_prompt_template: str,
    keyphrase_output_csv_path: str = "keyphrase_expansions_output.csv" # Default CSV path parameter
) -> np.ndarray | None:
    """
    Implements clustering via LLM keyphrase expansion (Section 2.1),
    using parallel processing for LLM calls and saving generated keyphrases.

    Args:
        documents: List of original document texts.
        features: Original document embeddings (NumPy array).
        n_clusters: The number of clusters.
        llm_service: An initialized LLMService instance.
        keyphrase_prompt_template: A string template for the core instruction.
        keyphrase_output_csv_path: Path to save the generated keyphrases CSV.

    Returns:
        A NumPy array of cluster assignments for the full original dataset length
        (with -1 for documents that failed processing), or None if the overall
        process fails or no documents are successfully processed.
    """
    print("\n--- Running Clustering via LLM Keyphrase Expansion ---")
    if not llm_service.is_available():
        print("LLMService is not available. Cannot run keyphrase expansion.")
        return None
    documents = documents
    n_samples = len(documents)
    original_embedding_dim = features.shape[1]

    # Define Langchain prompt template for generating keyphrases
    prompt_template = ChatPromptTemplate.from_template(keyphrase_prompt_template + "\nDocument: {document_text}")

    # Use a list to store results from parallel processing, pre-allocated
    # Each element will store the tuple (index, doc, keyphrases, expanded_feature) or None initially
    processed_results: List[Tuple[int, str, List[str], np.ndarray | None] | None] = [None] * n_samples

    # Define maximum workers for the thread pool
    # REDUCED MAX_WORKERS for stability
    MAX_WORKERS = 50 # Reduced from 40
    print(f"Processing {n_samples} documents in parallel with {MAX_WORKERS} workers...")
    start_time = time.time()

    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks to the executor, storing the future and its corresponding document index
        future_to_doc_index = {
            executor.submit(
                process_document_for_expansion,
                i, documents[i], features, llm_service, prompt_template, original_embedding_dim
            ): i for i in range(n_samples)
        }

        print("Submitted tasks to executor. Waiting for completion...") # Added print

        # Wrap the as_completed iterator with tqdm for a progress bar
        # The total parameter is automatically inferred from the length of future_to_doc_index
        for future in tqdm(concurrent.futures.as_completed(future_to_doc_index), total=n_samples, desc="Expanding Keyphrases"):
            doc_index = future_to_doc_index[future]
            try:
                # Retrieve the result tuple from the completed future
                result_tuple = future.result() # This is where exceptions from the thread are raised
                # Store the result tuple in the pre-allocated list at the correct index
                processed_results[doc_index] = result_tuple

            except Exception as exc:
                # This catches exceptions that were NOT handled inside process_document_for_expansion
                # or exceptions specifically raised by future.result() (e.g., CancelledError)
                print(f"\n--- UNHANDLED Exception retrieving result for document {doc_index+1}: {exc} ---")
                # The entry in processed_results[doc_index] might still be None if the helper
                # failed very early before setting the result. It will be handled later during collection.


    end_time = time.time()
    print(f"Finished parallel processing. Time taken: {end_time - start_time:.2f} seconds.")

    # --- Collect Data for CSV and KMeans ---
    data_for_csv = []
    successfully_processed_features = []
    successful_doc_indices = [] # Keep track of indices of successfully processed docs

    # Process results in the original document order by iterating through the pre-allocated list
    for doc_index in range(n_samples):
        result_tuple = processed_results[doc_index]
        if result_tuple is not None:
             current_doc_index, document, keyphrases, expanded_feature = result_tuple

             # Add data for CSV logging
             data_for_csv.append({
                 "document_index": current_doc_index,
                 "document_text": document,
                 "generated_keyphrases": ", ".join(keyphrases) # Join keyphrases for CSV cell
             })

             # If the feature was successfully generated (not None), add it for KMeans
             if expanded_feature is not None:
                 successfully_processed_features.append(expanded_feature)
                 successful_doc_indices.append(current_doc_index)
        else:
             # Handle cases where a document completely failed processing and its slot is None
             # print(f"Warning: Document {doc_index+1} had no processing result tuple.") # Optional warning
             # Add entry for failed documents too, with empty keyphrases for logging
             data_for_csv.append({
                 "document_index": doc_index,
                 "document_text": documents[doc_index], # Use original document text
                 "generated_keyphrases": "" # Empty string for failed
             })


    # --- Save Keyphrase Expansions to CSV ---
    if data_for_csv:
        try:
            df = pd.DataFrame(data_for_csv)
            # Ensure directory exists
            output_dir = os.path.dirname(keyphrase_output_csv_path)
            if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir)
            df.to_csv(keyphrase_output_csv_path, index=False)
            print(f"\nGenerated keyphrases saved to {keyphrase_output_csv_path}")
        except Exception as e:
            print(f"\nError saving keyphrases to CSV: {e}")
    else:
        print("\nNo data collected to save keyphrases.")


    # --- Proceed with KMeans on successfully processed documents ---
    if not successfully_processed_features:
        print("\nNo documents were successfully processed for keyphrase expansion. Cannot run KMeans.")
        return None

    expanded_features = np.array(successfully_processed_features)
    print(f"Proceeding with KMeans on {expanded_features.shape[0]} successfully expanded documents. Shape: {expanded_features.shape}")

    # 6. Perform standard K-Means on expanded features
    print(f"\nRunning KMeans on expanded features...")
    try:
        # Run KMeans on the successfully processed features subset
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        subset_assignments = kmeans.fit_predict(expanded_features)
        print("KMeans completed.")

        # Map the subset assignments back to the full original dataset length
        # Create a full assignments array, initialized with -1 for failed docs
        full_assignments = np.full(n_samples, -1, dtype=int)
        # Place the subset assignments into their correct positions in the full array
        for i, original_idx in enumerate(successful_doc_indices):
             full_assignments[original_idx] = subset_assignments[i]

        # Return the full assignments array, which can be evaluated against the full set of true labels
        return full_assignments

    except Exception as e:
        print(f"Error during KMeans clustering: {e}")
        # Even if KMeans fails, the CSV was saved.
        return None
