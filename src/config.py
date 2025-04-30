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
GENERATION_MODEL_NAME = "gpt-4.1-nano-2025-04-14" # Or a suitable chat model

# --- Clustering Parameters ---
# Parameters for the different methods (can be moved or made method-specific)
DEFAULT_N_CLUSTERS = 3 # Example, will be overridden by true clusters from data
KP_PROMPT_TEMPLATE = "Generate a comprehensive set of keyphrases that could describe the intent of the following text, as a JSON-formatted list of strings."
PC_PROMPT_TEMPLATE = "Are these two text snippets related to the same topic or express the same general intent? Respond with YES or NO.\nSnippet 1: {text1}\nSnippet 2: {text2}"
CORRECTION_PROMPT_TEMPLATE = "Should the following text snippet be in the same cluster as this representative text? Respond with YES or NO.\nSnippet: {point_text}\nRepresentative: {representative_text}"

PC_NUM_PAIRS_TO_QUERY = 150
PC_CONSTRAINT_SELECTION_STRATEGY = 'similarity' # 'random' or 'similarity'

CORRECTION_K_LOW_CONFIDENCE = 100
CORRECTION_NUM_CANDIDATE_CLUSTERS = 3

# --- Data Loading ---
DATA_CACHE_PATH = None # Set a path like "/tmp/clinc_feature_cache.pkl" if you want to cache features