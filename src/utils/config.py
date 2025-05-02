import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache/rag-datasets"
MODELS_DIR = PROJECT_ROOT / "models"
VECTORSTORES_DIR = PROJECT_ROOT / "vectorstores"

# Create necessary directories
for directory in [DATA_DIR, CACHE_DIR, MODELS_DIR, VECTORSTORES_DIR]:
    directory.mkdir(exist_ok=True)

# Model configurations
# Embedding model configuration
# Recommended models that work well on GPU:
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight model (384 dimensions)
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Higher quality model (768 dimensions)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Good performance/size tradeoff

# QA model configuration
QA_MODEL = "gpt-4o-mini"
QA_TEMPERATURE = 0

DEFAULT_K = 3

# API configurations for QA model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set (required for QA model)")
