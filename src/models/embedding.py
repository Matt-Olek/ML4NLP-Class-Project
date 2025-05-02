import os
from typing import List, Optional
import torch
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import logging
from pathlib import Path

load_dotenv()
logger = logging.getLogger(__name__)


class DocumentEmbedder:
    def __init__(
        self, 
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = None
    ):
        """Initialize the document embedder with local HuggingFace embeddings.

        Args:
            model_name: Name of the HuggingFace embedding model to use
            device: Device to run the model on (None for auto-detection, "cpu", "cuda", etc.)
        """
        # Automatically detect GPU if available and device not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"device": device, "batch_size": 32}
        )
        logger.info(f"Using embedding model: {model_name} on {device}")
            
        self.vectorstore = None

    def create_documents(self, df_passages: pd.DataFrame) -> List[Document]:
        """Convert DataFrame rows to LangChain Documents.

        Args:
            df_passages: DataFrame containing passages

        Returns:
            List of Document objects
        """
        return [
            Document(page_content=row["passage"], metadata={"id": idx, **row.to_dict()})
            for idx, row in df_passages.iterrows()
        ]

    def create_vectorstore(
        self,
        documents: List[Document],
        batch_size: int = 1000,
        save_path: Optional[str] = None,
    ) -> None:
        """Create a FAISS vector store from documents in batches.

        Args:
            documents: List of documents to embed
            batch_size: Number of documents to process in each batch
            save_path: Optional path to save the vectorstore
        """
        if not documents:
            raise ValueError("No documents provided")

        num_batches = (len(documents) + batch_size - 1) // batch_size
        self.vectorstore = None

        for i in tqdm(range(num_batches), desc="Processing document batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(documents))
            batch_docs = documents[start_idx:end_idx]

            if i == 0:
                self.vectorstore = FAISS.from_documents(batch_docs, self.embeddings)
            else:
                self.vectorstore.add_documents(batch_docs)

        if save_path:
            self.save_vectorstore(save_path)

    def save_vectorstore(self, path: str) -> None:
        """Save the vector store to disk.

        Args:
            path: Path to save the vectorstore
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not created. Call create_vectorstore first.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(save_path))
        logger.info(f"Vectorstore saved to {save_path}")

    def load_vectorstore(self, path: str) -> None:
        """Load a vector store from disk.

        Args:
            path: Path to load the vectorstore from
        """
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        logger.info(f"Vectorstore loaded from {path}")

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for a single text.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        return self.embeddings.embed_query(text)
