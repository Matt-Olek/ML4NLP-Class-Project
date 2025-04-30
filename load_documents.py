import pandas as pd
from embedding import DocumentEmbedder
import os
from dotenv import load_dotenv

load_dotenv()


def load_or_cache_data(url, cache_path):
    """Load data from URL or cache if available."""
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    df = pd.read_parquet(url)
    df.to_parquet(cache_path)
    return df


def main():
    # Create cache directory
    os.makedirs("cache", exist_ok=True)

    # Load data
    print("Loading documents from dataset...")
    df_passages = load_or_cache_data(
        "hf://datasets/rag-datasets/rag-mini-bioasq/data/passages.parquet/part.0.parquet",
        "cache/passages.parquet",
    )

    # Initialize and setup embedder
    print("Initializing document embedder...")
    embedder = DocumentEmbedder()
    documents = embedder.create_documents(df_passages.head(2000))

    print("Creating and saving vectorstore...")
    embedder.create_vectorstore(documents)
    embedder.save_vectorstore("vectorstore")

    print("Vectorstore created and saved successfully!")


if __name__ == "__main__":
    main()
