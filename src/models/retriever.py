from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import numpy as np
import logging
from .embedding import DocumentEmbedder

logger = logging.getLogger(__name__)


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
        self.similarity_threshold = similarity_threshold

    def set_vectorstore(self, vectorstore: FAISS) -> None:
        """Set the vector store for retrieval.

        Args:
            vectorstore: FAISS vectorstore to use
        """
        self.vectorstore = vectorstore
        logger.info("Vectorstore set successfully")

    def set_embedder(self, embedder: DocumentEmbedder) -> None:
        """Set the embedder for query encoding.

        Args:
            embedder: DocumentEmbedder to use
        """
        self.embedder = embedder
        logger.info("Embedder set successfully")

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
        docs = self.vectorstore.similarity_search(query, k=k, filter=filter_criteria)

        # Format the results
        results = []
        for doc in docs:
            score = self._calculate_similarity_score(query, doc.page_content)
            if score >= self.similarity_threshold:
                results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                    }
                )

        logger.info(f"Retrieved {len(results)} documents for query: {query}")
        return results

    def _calculate_similarity_score(self, query: str, document: str) -> float:
        """Calculate similarity score between query and document.

        Args:
            query: Query string
            document: Document content

        Returns:
            Similarity score between 0 and 1
        """
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder first.")

        query_embedding = self.embedder.get_embeddings(query)
        doc_embedding = self.embedder.get_embeddings(document)

        # Calculate cosine similarity
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        return float(similarity)
