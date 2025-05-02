from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import numpy as np
import logging
from .embedding import DocumentEmbedder



class DocumentRetriever:
    def __init__(
        self,
        vectorstore: Optional[FAISS] = None,
        embedder: Optional[DocumentEmbedder] = None,
        similarity_threshold: float = 0.7,
    ):
        """Initialize the document retriever.

        Args:
            vectorstore: FAISS vectorstore for retrieval
            embedder: DocumentEmbedder for query encoding
            similarity_threshold: Minimum similarity score for retrieved documents
        """
        self.vectorstore = vectorstore
        self.embedder = embedder

    def set_vectorstore(self, vectorstore: FAISS) -> None:
        """Set the vector store for retrieval.

        Args:
            vectorstore: FAISS vectorstore to use
        """
        self.vectorstore = vectorstore

    def set_embedder(self, embedder: DocumentEmbedder) -> None:
        """Set the embedder for query encoding.

        Args:
            embedder: DocumentEmbedder to use
        """
        self.embedder = embedder

    def retrieve(
        self, query: str, k: int = 5, filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve the top k most relevant documents for a query.

        Args:
            query: Query string
            k: Number of documents to retrieve
            filter_criteria: Optional criteria to filter documents

        Returns:
            List of dictionaries containing document content, metadata, and similarity score
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not set. Call set_vectorstore first.")

        # Get the most similar documents
        docs = self.vectorstore.similarity_search_with_score(
            query, k=k, filter=filter_criteria
        )

        return docs