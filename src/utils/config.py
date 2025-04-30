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
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
VECTORSTORES_DIR = PROJECT_ROOT / "vectorstores"

# Create necessary directories
for directory in [DATA_DIR, CACHE_DIR, MODELS_DIR, VECTORSTORES_DIR]:
    directory.mkdir(exist_ok=True)

# Model configurations
EMBEDDING_MODEL = "text-embedding-3-small"
QA_MODEL = "gpt-4o-mini"
QA_TEMPERATURE = 0

# Retrieval configurations
SIMILARITY_THRESHOLD = 0.7
DEFAULT_K = 3

# API configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")
