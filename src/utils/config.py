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
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" 
HYDE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DEFAULT_K = 3
