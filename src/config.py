import os
# from dotenv import load_dotenv

# # Optional: Load environment variables from .env file
# load_dotenv()

# --- API Configuration ---
# Ensure OPENAI_API_KEY environment variable is set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Specify OpenAI Models to use
EMBEDDING_MODEL_NAME = "text-embedding-ada-002" # Or a newer embedding model
GENERATION_MODEL_NAME = "gpt-4.1-nano" # Or a suitable chat model
MOCKING_MODE = False
# --- Clustering Parameters ---
# Parameters for the different methods (can be moved or made method-specific)
DEFAULT_N_CLUSTERS = 3 # Example, will be overridden by true clusters from data

KP_PROMPT_TEMPLATE = """
You are an expert in semantic understanding and intent classification. 
Your task is to extract a comprehensive set of keyphrases that capture the core meaning, intent, topics, and semantic content of the following document.
Extract 5-7 keyphrases that capture the core meaning and intent of this text.
Include both specific terms and broader concepts.
Consider synonyms and related terminology.
Format as a JSON list of strings.
"""

PC_PROMPT_TEMPLATE = """
Evaluate if the following two text snippets are about the same core topic or express fairly similar semantic intent. Answer ONLY with YES or NO.

Snippet 1: {text1}
Snippet 2: {text2}
"""

CORRECTION_PROMPT_TEMPLATE = """
You are a master of semantic understanding and clustering analysis.

Task: The dataset you are working with is highly diverse, containing documents from a wide range of topics and styles. Your goal is to evaluate whether the following document should belong to the same cluster as the given representative document, based strictly on their meaning.

Document: {document_text}

Representative Document for a Cluster: {rep_doc_text}

Question: Based on their semantic similarity, should the Document be linked to the Representative Document? Respond only with YES or NO.

Response:
"""


PC_NUM_PAIRS_TO_QUERY = 150
PC_CONSTRAINT_SELECTION_STRATEGY = 'similarity' # 'random' or 'similarity'

CORRECTION_K_LOW_CONFIDENCE = 100
CORRECTION_NUM_CANDIDATE_CLUSTERS = 3

# --- Data Loading ---
DATA_CACHE_PATH ="cache_embeddings/" # Set a path like "/tmp/clinc_feature_cache.pkl" if you want to cache features