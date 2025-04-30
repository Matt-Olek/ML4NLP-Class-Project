from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from embedding import DocumentEmbedder


class DocumentRetriever:
    def __init__(self, vectorstore: FAISS = None, embedder: DocumentEmbedder = None):
        """Initialize the document retriever."""
        self.vectorstore = vectorstore
        self.embedder = embedder

    def set_vectorstore(self, vectorstore: FAISS) -> None:
        """Set the vector store for retrieval."""
        self.vectorstore = vectorstore

    def set_embedder(self, embedder: DocumentEmbedder) -> None:
        """Set the embedder for query encoding."""
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top k most relevant documents for a query."""
        if self.vectorstore is None:
            raise ValueError("Vector store not set. Call set_vectorstore first.")

        # Get the most similar documents
        docs = self.vectorstore.similarity_search(query, k=k)

        # Format the results
        results = []
        for doc in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": self._calculate_similarity_score(query, doc.page_content),
                }
            )

        return results

    def _calculate_similarity_score(self, query: str, document: str) -> float:
        """Calculate similarity score between query and document."""
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder first.")

        query_embedding = self.embedder.get_embeddings(query)
        doc_embedding = self.embedder.get_embeddings(document)

        # Calculate cosine similarity
        import numpy as np

        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        return float(similarity)
