import logging
from pathlib import Path
from data.dataset_loader import DatasetLoader
from models.embedding import DocumentEmbedder
from utils.config import (
    CACHE_DIR,
    VECTORSTORES_DIR,
    EMBEDDING_MODEL,
)
from rich.logging import RichHandler
from rich.console import Console

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


def build_vectorstore(dataset_name: str, num_passages: int = 2000, all_passages: bool = False):
    """Build and save a vectorstore from a dataset.

    Args:
        dataset_name: Name of the dataset to use
        num_passages: Number of passages to process
        all_passages: Whether to use all passages
    """
    logger.info(f"Building vectorstore for dataset: {dataset_name}")

    # Load dataset
    loader = DatasetLoader(str(CACHE_DIR))
    if all_passages:
        df_passages = loader.load_passages(dataset_name)
    else:
        df_passages = loader.load_passages(dataset_name).head(num_passages)

    # Create documents and vectorstore
    logger.info(f"Using embedding model: {EMBEDDING_MODEL}")
    embedder = DocumentEmbedder(model_name=EMBEDDING_MODEL)
    documents = embedder.create_documents(df_passages)

    vectorstore_path = VECTORSTORES_DIR / f"{dataset_name}_vectorstore"
    embedder.create_vectorstore(documents, save_path=str(vectorstore_path))

    logger.info(f"Vectorstore built and saved to {vectorstore_path}")


def main():
    """Main function to build a vectorstore."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build a vectorstore from a dataset")
    parser.add_argument("dataset_name", help="Name of the dataset to use")
    parser.add_argument("--num_passages", type=int, default=2000, help="Number of passages to process")
    parser.add_argument("--all_passages", action="store_true", help="Whether to use all passages")
    
    args = parser.parse_args()
    
    build_vectorstore(
        dataset_name=args.dataset_name,
        num_passages=args.num_passages,
        all_passages=args.all_passages
    )


if __name__ == "__main__":
    main() 