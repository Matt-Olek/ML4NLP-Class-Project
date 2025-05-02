from typing import List, Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
import os
import logging
from .retriever import DocumentRetriever
load_dotenv()

# Silence httpx logger to prevent OpenAI API call logs
logging.getLogger("httpx").setLevel(logging.WARNING)


class QuestionAnswerer:
    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        model_name: str = "gpt-4",
        temperature: float = 0,
    ):
        """Initialize the question answerer.

        Args:
            retriever: DocumentRetriever for retrieving relevant documents
            model_name: Name of the OpenAI model to use
            temperature: Temperature for model generation
        """
        self.retriever = retriever
        self.llm = None
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an expert at answering questions based on the provided context.
            Use the following pieces of context to answer the question at the end.
            If you cannot find the answer in the context, say "I cannot find the answer in the provided context."
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
        )

        # Create the chain
        # self.chain = (
        #     {
        #         "context": lambda x: x["context"],
        #         "question": lambda x: x["question"],
        #     }
        #     | self.prompt
        #     | self.llm
        #     | StrOutputParser()
        # )

    def _format_context(self, context_list: List[str]) -> str:
        """Format a list of context strings into a single context string.

        Args:
            context_list: List of context strings

        Returns:
            Formatted context string
        """
        return "\n\n".join(
            [f"Document {i+1}:\n{context}" for i, context in enumerate(context_list)]
        )

    def answer_question(
        self, question: str, context_list: Optional[List[str]] = None, k: int = 10, no_answer: bool = False
    ) -> Dict[str, Any]:
        """Answer a question based on either provided context or retrieved documents.

        Args:
            question: The question to answer
            context_list: Optional list of context strings
            k: Number of documents to retrieve if using retriever

        Returns:
            Dictionary containing the question, answer, and context used
        """
        if context_list is None:
            if self.retriever is None:
                raise ValueError(
                    "Either context_list must be provided or retriever must be set"
                )
            # Use retriever to get context
            results = self.retriever.retrieve(question, k=k)
            context_list = [result[0].metadata["passage"] for result in results]
            context_ids = [result[0].metadata["id"] for result in results]
        # Format the context
        formatted_context = self._format_context(context_list)

        # Get the answer from the chain
        if no_answer:
            answer = "No answer generated"
        else:
            answer = self.chain.invoke({"context": formatted_context, "question": question})
        return {
            "question": question,
            "answer": answer,
            "context": formatted_context,
            "context_ids": context_ids,
        }
