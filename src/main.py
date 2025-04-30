import logging
from pathlib import Path
from data.dataset_loader import DatasetLoader
from models.embedding import DocumentEmbedder
from models.retriever import DocumentRetriever
from models.qa_model import QuestionAnswerer
from utils.config import (
    CACHE_DIR,
    VECTORSTORES_DIR,
    EMBEDDING_MODEL,
    QA_MODEL,
    SIMILARITY_THRESHOLD,
)
import pandas as pd

logger = logging.getLogger(__name__)


def build_vectorstore(dataset_name: str, num_passages: int = 2000):
    """Build and save a vectorstore from a dataset.

    Args:
        dataset_name: Name of the dataset to use
        num_passages: Number of passages to process
    """
    logger.info(f"Building vectorstore for dataset: {dataset_name}")

    # Load dataset
    loader = DatasetLoader(str(CACHE_DIR))
    df_passages = loader.load_passages(dataset_name).head(num_passages)

    # Create documents and vectorstore
    embedder = DocumentEmbedder(model_name=EMBEDDING_MODEL)
    documents = embedder.create_documents(df_passages)

    vectorstore_path = VECTORSTORES_DIR / f"{dataset_name}_vectorstore"
    embedder.create_vectorstore(documents, save_path=str(vectorstore_path))

    logger.info(f"Vectorstore built and saved to {vectorstore_path}")


def evaluate_qa(dataset_name: str, num_questions: int = 10):
    """Evaluate QA performance on a dataset.

    Args:
        dataset_name: Name of the dataset to evaluate on
        num_questions: Number of questions to evaluate
    """
    logger.info(f"Evaluating QA on dataset: {dataset_name}")

    # Load dataset
    loader = DatasetLoader(str(CACHE_DIR))
    df_questions = loader.load_qa_dataset(dataset_name).head(num_questions)

    # Load vectorstore
    embedder = DocumentEmbedder(model_name=EMBEDDING_MODEL)
    vectorstore_path = VECTORSTORES_DIR / f"{dataset_name}_vectorstore"
    embedder.load_vectorstore(str(vectorstore_path))

    # Setup retriever and QA model
    retriever = DocumentRetriever(
        vectorstore=embedder.vectorstore,
        embedder=embedder,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    qa_model = QuestionAnswerer(retriever=retriever, model_name=QA_MODEL)
    evaluation_results = []
    # Evaluate on questions
    for idx, row in df_questions.iterrows():
        question = row["question"]
        logger.info(f"\nQuestion {idx + 1}: {question}")

        result = qa_model.answer_question(question)
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Context used: {result['context']}")
        evaluation_results.append(
            {
                "question": question,
                "ground_truth": row["answer"],
                "predicted_answer": result["answer"],
                "context": result["context"],
            }
        )

    result_df = pd.DataFrame(evaluation_results)
    result_df.to_csv(f"{dataset_name}_evaluation_results.csv", index=False)


def main():
    """Main function to demonstrate the workflow."""
    dataset_name = "rag-datasets/rag-mini-bioasq"

    # Build vectorstore
    build_vectorstore(dataset_name)

    # Evaluate QA
    evaluate_qa(dataset_name)


if __name__ == "__main__":
    main()
