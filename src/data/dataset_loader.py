import pandas as pd
import os
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    def __init__(self, cache_dir: str = "cache"):
        """Initialize the dataset loader with a cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def load_or_cache_data(self, url: str, cache_name: str) -> pd.DataFrame:
        """Load data from URL or cache if available.

        Args:
            url: URL to load data from
            cache_name: Name of the cache file (without extension)

        Returns:
            Loaded DataFrame
        """
        cache_path = self.cache_dir / f"{cache_name}.parquet"

        if cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Downloading data from {url}")
        df = pd.read_parquet(url)
        df.to_parquet(cache_path)
        logger.info(f"Data cached to {cache_path}")
        return df

    def load_qa_dataset(self, dataset_name: str, split: str = "test") -> pd.DataFrame:
        """Load a QA dataset from HuggingFace.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split to load

        Returns:
            DataFrame containing the QA dataset
        """
        cache_name = f"{dataset_name}_{split}"
        url = f"hf://datasets/{dataset_name}/data/{split}.parquet/part.0.parquet"
        return self.load_or_cache_data(url, cache_name)

    def load_passages(self, dataset_name: str) -> pd.DataFrame:
        """Load passages from a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            DataFrame containing the passages
        """
        cache_name = f"{dataset_name}_passages"
        url = f"hf://datasets/{dataset_name}/data/passages.parquet/part.0.parquet"
        return self.load_or_cache_data(url, cache_name)
