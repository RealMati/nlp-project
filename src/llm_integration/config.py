import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # openai, anthropic, ollama
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
LLM_API_BASE = os.getenv("LLM_API_BASE", "") # Optional: for local inference (e.g., http://localhost:11434/v1)

# Vector Store Configuration
VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "vector_store")
# Using BM25 (no embeddings required)
EMBEDDING_MODEL_NAME = "bm25"
