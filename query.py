from embedding import DocumentEmbedder
from retrieval import DocumentRetriever
from completion import QuestionAnswerer
from dotenv import load_dotenv

load_dotenv()


def main():
    # Load existing vectorstore
    print("Loading vectorstore...")
    embedder = DocumentEmbedder()
    embedder.load_vectorstore("vectorstore")

    # Setup retriever
    print("Setting up document retriever...")
    retriever = DocumentRetriever(embedder.vectorstore, embedder)

    # Setup question answerer
    print("Setting up question answerer...")
    qa = QuestionAnswerer()

    # Example question
    question = "What is the role of DNA in genetics?"
    print(f"\nQuestion: {question}")

    # First, retrieve relevant documents using the retriever
    print("\nRetrieving relevant documents...")
    results = retriever.retrieve(question, k=3)
    context_list = [result["content"] for result in results]

    # Then, use the QuestionAnswerer with the retrieved context
    print("\nGenerating answer using retrieved context...")
    result = qa.answer_question(question, context_list=context_list)

    print(f"\nAnswer: {result['answer']}")
    print("\nContext used:")
    print(result["context"])


if __name__ == "__main__":
    main()
