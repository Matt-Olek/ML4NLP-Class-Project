from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
import os
from retrieval import DocumentRetriever

load_dotenv()


class QuestionAnswerer:
    def __init__(self, retriever: DocumentRetriever = None):
        """Initialize the question answerer with an optional document retriever."""
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
        )

        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """
        You are an expert at answering questions based on the provided context.
        Use the following pieces of context to answer the question at the end.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        )

        # Create the chain
        self.chain = (
            {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"],
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_context(self, context_list: List[str]) -> str:
        """Format a list of context strings into a single context string."""
        return "\n\n".join(
            [f"Document {i+1}:\n{context}" for i, context in enumerate(context_list)]
        )

    def answer_question(
        self, question: str, context_list: List[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a question based on either provided context or retrieved documents.

        Args:
            question: The question to answer
            context_list: Optional list of context strings. If not provided, will use the retriever.

        Returns:
            Dictionary containing the question, answer, and context used
        """
        if context_list is None:
            if self.retriever is None:
                raise ValueError(
                    "Either context_list must be provided or retriever must be set"
                )
            # Use retriever to get context
            results = self.retriever.retrieve(question, k=3)
            context_list = [result["content"] for result in results]

        # Format the context
        formatted_context = self._format_context(context_list)

        # Get the answer from the chain
        answer = self.chain.invoke({"context": formatted_context, "question": question})

        return {
            "question": question,
            "answer": answer,
            "context": formatted_context,
        }
