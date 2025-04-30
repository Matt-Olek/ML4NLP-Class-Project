import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


class DocumentEmbedder:
    def __init__(self):
        """Initialize the document embedder with OpenAI embeddings."""

        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.vectorstore = None

    def create_documents(self, df_passages: pd.DataFrame) -> List[Document]:
        """Convert DataFrame rows to LangChain Documents."""
        return [
            Document(page_content=row["passage"], metadata={"id": idx})
            for idx, row in df_passages.iterrows()
        ]

    def create_vectorstore(
        self, documents: List[Document], batch_size: int = 1000
    ) -> None:
        """Create a FAISS vector store from documents in batches."""
        if not documents:
            raise ValueError("No documents provided")

        # Calculate number of batches
        num_batches = (len(documents) + batch_size - 1) // batch_size

        # Initialize empty vectorstore
        self.vectorstore = None

        # Process documents in batches
        for i in tqdm(range(num_batches), desc="Processing document batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(documents))
            batch_docs = documents[start_idx:end_idx]

            # Create vectorstore for the first batch
            if i == 0:
                self.vectorstore = FAISS.from_documents(batch_docs, self.embeddings)
            else:
                # Add documents to existing vectorstore for subsequent batches
                self.vectorstore.add_documents(batch_docs)

    def save_vectorstore(self, path: str) -> None:
        """Save the vector store to disk."""
        if self.vectorstore is None:
            raise ValueError("Vector store not created. Call create_vectorstore first.")
        self.vectorstore.save_local(path)

    def load_vectorstore(self, path: str) -> None:
        """Load a vector store from disk."""
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for a single text."""
        return self.embeddings.embed_query(text)
