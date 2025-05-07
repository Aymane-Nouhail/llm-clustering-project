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
You are an expert in semantic understanding and entity-centric text analysis.

Task: Extract 5 to 7 keyphrases that best capture the main ideas, entities, roles, and core concepts mentioned in the following text. These keyphrases should reflect both specific names (e.g., people, organizations, titles) and broader semantic themes.

- Focus on relevance to meaning and identity.
- Include both exact terms and semantically related variants.
- Use concise noun phrases.
- Avoid full sentences.

Output format: JSON list of strings.
"""

PC_PROMPT_TEMPLATE = """
Determine if the following two text fragments refer to the same real-world entity. This is not about general topic similarity — focus only on whether both phrases point to the same individual, organization, location, or concept, despite possible differences in wording.

Be aware: the mentions may be short, ambiguous, or lack context. Use your knowledge and reasoning to assess identity.

Examples:

Mention 1: Joe Biden  
Mention 2: President Obama  
Answer ONLY with YES or NO.  
NO

Mention 1: Barack Obama  
Mention 2: Barack H. Obama  
Answer ONLY with YES or NO.  
YES

Mention 1: Biden  
Mention 2: Vice President under Obama  
Answer ONLY with YES or NO.  
YES

Mention 1: Barack Obama  
Mention 2: Michelle Obama  
Answer ONLY with YES or NO.  
NO

Mention 1: Google  
Mention 2: Alphabet Inc.  
Answer ONLY with YES or NO.  
NO

Mention 1: Alphabet Inc.  
Mention 2: Parent company of Google  
Answer ONLY with YES or NO.  
YES

Mention 1: NYC  
Mention 2: New York City  
Answer ONLY with YES or NO.  
YES

Mention 1: Washington  
Mention 2: George Washington  
Answer ONLY with YES or NO.  
NO

Mention 1: AI  
Mention 2: Artificial Intelligence  
Answer ONLY with YES or NO.  
YES

Mention 1: OpenAI  
Mention 2: Company behind ChatGPT  
Answer ONLY with YES or NO.  
YES

Mention 1: The UN  
Mention 2: United Nations  
Answer ONLY with YES or NO.  
YES

Mention 1: Twitter  
Mention 2: X  
Answer ONLY with YES or NO.  
YES

Mention 1: Meta  
Mention 2: Facebook  
Answer ONLY with YES or NO.  
NO

Mention 1: Meta  
Mention 2: Parent company of Facebook  
Answer ONLY with YES or NO.  
YES

Now, determine the following:

Mention 1: {text1}  
Mention 2: {text2}  
Answer ONLY with YES or NO.
"""



CORRECTION_PROMPT_TEMPLATE = """
You are an expert in entity canonicalization and knowledge representation.

Task: Given two textual fragments, decide whether they refer to the same real-world entity. The dataset involves many ambiguous, short, and context-light mentions.

Examples:

Document Mention: Joe Biden  
Representative Mention (Cluster Anchor): President Obama  
Question: Do these two mentions refer to the same underlying entity (person, organization, location, etc.)?  
NO

Document Mention: Barack H. Obama  
Representative Mention (Cluster Anchor): Barack Obama  
Question: Do these two mentions refer to the same underlying entity (person, organization, location, etc.)?  
YES

Document Mention: NYC  
Representative Mention (Cluster Anchor): New York City  
Question: Do these two mentions refer to the same underlying entity (person, organization, location, etc.)?  
YES

Document Mention: Google  
Representative Mention (Cluster Anchor): Alphabet Inc.  
Question: Do these two mentions refer to the same underlying entity (person, organization, location, etc.)?  
NO

Document Mention: Alphabet  
Representative Mention (Cluster Anchor): Parent company of Google  
Question: Do these two mentions refer to the same underlying entity (person, organization, location, etc.)?  
YES

Document Mention: Washington  
Representative Mention (Cluster Anchor): George Washington  
Question: Do these two mentions refer to the same underlying entity (person, organization, location, etc.)?  
NO

Document Mention: Twitter  
Representative Mention (Cluster Anchor): X  
Question: Do these two mentions refer to the same underlying entity (person, organization, location, etc.)?  
YES

Document Mention: Meta  
Representative Mention (Cluster Anchor): Facebook  
Question: Do these two mentions refer to the same underlying entity (person, organization, location, etc.)?  
NO

Document Mention: Meta  
Representative Mention (Cluster Anchor): Parent company of Facebook  
Question: Do these two mentions refer to the same underlying entity (person, organization, location, etc.)?  
YES

Document Mention: AI  
Representative Mention (Cluster Anchor): Artificial Intelligence  
Question: Do these two mentions refer to the same underlying entity (person, organization, location, etc.)?  
YES

Now, evaluate the following:

Document Mention: {document_text}  
Representative Mention (Cluster Anchor): {rep_doc_text}  
Question: Do these two mentions refer to the same underlying entity (person, organization, location, etc.)?

Respond ONLY with YES or NO.
"""




PC_NUM_PAIRS_TO_QUERY = 2000
PC_CONSTRAINT_SELECTION_STRATEGY = 'similarity' # 'random' or 'similarity'

CORRECTION_K_LOW_CONFIDENCE = 100
CORRECTION_NUM_CANDIDATE_CLUSTERS = 3

# --- Data Loading ---
DATA_CACHE_PATH ="cache_embeddings/" # Set a path like "/tmp/clinc_feature_cache.pkl" if you want to cache features