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
)
import pandas as pd
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.box import ROUNDED
from rich import print as rprint

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


def build_vectorstore(dataset_name: str, num_passages: int = 2000, all_passages: bool = False):
    """Build and save a vectorstore from a dataset.

    Args:
        dataset_name: Name of the dataset to use
        num_passages: Number of passages to process
        all_passages: Whether to use all passages
    """
    logger.info(f"Building vectorstore for dataset: {dataset_name}")

    # Load dataset
    loader = DatasetLoader(str(CACHE_DIR))
    if all_passages:
        df_passages = loader.load_passages(dataset_name)
    else:
        df_passages = loader.load_passages(dataset_name).head(num_passages)

    # Create documents and vectorstore
    logger.info(f"Using embedding model: {EMBEDDING_MODEL}")
    embedder = DocumentEmbedder(model_name=EMBEDDING_MODEL)
    documents = embedder.create_documents(df_passages)

    vectorstore_path = VECTORSTORES_DIR / f"{dataset_name}_vectorstore"
    embedder.create_vectorstore(documents, save_path=str(vectorstore_path))

    logger.info(f"Vectorstore built and saved to {vectorstore_path}")


def evaluate_qa(dataset_name: str, num_questions: int = 1000, top_k: int = 20):
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
    )

    qa_model = QuestionAnswerer(retriever=retriever, model_name=QA_MODEL)
    evaluation_results = []
    # Evaluate on questions
    for idx, row in df_questions.iterrows():
        question = row["question"]
        real_answer = row["answer"]
        relevant_passages_ids = row["relevant_passage_ids"]
        
        # Get answer
        result = qa_model.answer_question(question, no_answer=True, k=top_k)
        
        # Number of passages retrieved that are in the relevant passages ids
        num_relevant_passages = len([id for id in result['context_ids'] if id in eval(relevant_passages_ids)])

        # Rich formatted question and answer display in a single panel
        console.print()
        table = Table(
            show_header=True,
            box=ROUNDED,
            title=f"Q&A {idx + 1}",
            border_style="magenta",
            show_lines=True,
        )
        table.add_column("Category", style="bold cyan")
        table.add_column("Content")
        table.add_row("Question", question)
        table.add_row("Answer", result['answer'])
        table.add_row("Real Answer", real_answer, style="bold red")
        table.add_row("Context IDs", str(result['context_ids']), style="bold blue")
        table.add_row("Relevant Passage IDs", str(relevant_passages_ids), style="bold blue")
        table.add_row("Number of relevant passages retrieved", str(num_relevant_passages) + "/" + str(len(eval(relevant_passages_ids))), style="bold green")
        console.print(table)
        
        evaluation_results.append(
            {
                "id": idx,
                "question": question,
                "ground_truth": row["answer"],
                "top_k": top_k,
                "predicted_answer": result["answer"],
                "context_ids": result["context_ids"],
                "relevant_passages_ids": relevant_passages_ids,
                "num_relevant_passages": num_relevant_passages,
                "max_num_relevant_passages": len(eval(relevant_passages_ids)),
                "retrieved_ratio": num_relevant_passages / len(eval(relevant_passages_ids)),
            }
        )

    result_df = pd.DataFrame(evaluation_results)
    result_df.to_csv(f"{dataset_name}__evaluation_results.csv", index=False)
    logger.info(f"Evaluation results saved to {dataset_name}_evaluation_results.csv")


def main():
    """Main function to demonstrate the workflow."""
    dataset_name = "rag-mini-bioasq"

    # Build vectorstore
    # build_vectorstore(dataset_name, all_passages=True)

    # Evaluate QA
    evaluate_qa(dataset_name)


if __name__ == "__main__":
    main()
