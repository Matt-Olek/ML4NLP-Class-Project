from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import numpy as np
import logging
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class BM25DocumentRetriever:
    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        tokenizer: Optional[callable] = None,
    ):
        """Initialize the BM25 document retriever.

        Args:
            documents: List of documents to index
            tokenizer: Optional custom tokenizer function
        """
        self.documents = documents or []
        self.tokenizer = tokenizer or word_tokenize
        self.bm25 = None
        
        if documents:
            self._build_index()

    def _build_index(self):
        """Build the BM25 index from documents."""
        tokenized_docs = [self.tokenizer(doc.page_content.lower()) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index.

        Args:
            documents: List of documents to add
        """
        self.documents.extend(documents)
        self._build_index()

    def retrieve(
        self, query: str, k: int = 5, filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve the top k most relevant documents for a query.

        Args:
            query: Query string
            k: Number of documents to retrieve
            filter_criteria: Optional criteria to filter documents (not implemented)

        Returns:
            List of dictionaries containing document content, metadata, and similarity score
        """
        if self.bm25 is None:
            raise ValueError("No documents indexed. Call add_documents first.")

        # Tokenize query
        tokenized_query = self.tokenizer(query.lower())
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k documents
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            doc = self.documents[idx]
            results.append((
                doc,
                scores[idx]
            ))
        
        return results 