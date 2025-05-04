import logging
from pathlib import Path
import pandas as pd
from data.dataset_loader import DatasetLoader
from models.embedding import DocumentEmbedder
from models.bm25_retriever import BM25DocumentRetriever
from models.qa_model import QuestionAnswerer
from models.hyde import generate_hyde_answer
from utils.config import (
    CACHE_DIR,
    VECTORSTORES_DIR,
    EMBEDDING_MODEL,
)
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.box import ROUNDED
from tqdm import tqdm
import math

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

def evaluate_qa_bm25(dataset_name: str, num_questions: int = 1000, top_k: int = 50, print_results: bool = False):
    """Evaluate QA performance on a dataset using BM25 for retrieval.

    Args:
        dataset_name: Name of the dataset to evaluate on
        num_questions: Number of questions to evaluate
        top_k: Number of top passages to retrieve
        print_results: Whether to print detailed results
    """
    logger.info(f"Evaluating QA with BM25 on dataset: {dataset_name}")

    # Load dataset
    loader = DatasetLoader(str(CACHE_DIR))
    df_questions = loader.load_qa_dataset(dataset_name).head(num_questions)
    df_passages = loader.load_passages(dataset_name)

    # Create documents
    embedder = DocumentEmbedder(model_name=EMBEDDING_MODEL)
    documents = embedder.create_documents(df_passages)

    # Setup BM25 retriever
    retriever = BM25DocumentRetriever(documents=documents)
    qa_model = QuestionAnswerer(retriever=retriever)

    evaluation_results = []
    # Evaluate on questions
    for idx, row in tqdm(df_questions.iterrows(), total=len(df_questions), desc="Evaluating QA"):
        question = row["question"]
        real_answer = row["answer"]
        relevant_passages_ids = row["relevant_passage_ids"]
        
        # Get answer
        result = qa_model.answer_question(question, no_answer=True, k=top_k)
        
        # Number of passages retrieved that are in the relevant passages ids
        num_relevant_passages = len([id for id in result['context_ids'] if id in eval(relevant_passages_ids)])
        
        # Calculation of Normalized Discounted Cumulative Gain (NDCG)
        ndcg = 0
        for i, id in enumerate(result['context_ids']):
            if id in eval(relevant_passages_ids):
                ndcg += 1 / math.log2(i + 1 + 1)
        idcg = 0
        for i, id in enumerate(eval(relevant_passages_ids)):
            idcg += 1 / math.log2(i + 1 + 1)
        ndcg = ndcg / idcg if idcg > 0 else 0

        if print_results:
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
                "ndcg": ndcg,
                "num_relevant_passages": num_relevant_passages,
                "max_num_relevant_passages": len(eval(relevant_passages_ids)),
                "retrieved_ratio": num_relevant_passages / len(eval(relevant_passages_ids)) if len(eval(relevant_passages_ids)) > 0 else 0,
                "question": question,
                "ground_truth": row["answer"],
                "top_k": top_k,
                "predicted_answer": result["answer"],
                "context_ids": result["context_ids"],
                "relevant_passages_ids": relevant_passages_ids
            }
        )

    result_df = pd.DataFrame(evaluation_results)
    result_df.to_csv(f"evaluations/{dataset_name}_bm25_evaluation_results.csv", index=False)
    logger.info(f"Evaluation results saved to evaluations/{dataset_name}_bm25_evaluation_results.csv")


def main():
    """Main function to evaluate QA on a dataset using BM25."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate QA on a dataset using BM25")
    parser.add_argument("dataset_name", help="Name of the dataset to evaluate on")
    parser.add_argument("--num_questions", type=int, default=100, help="Number of questions to evaluate")
    parser.add_argument("--top_k", type=int, default=1000, help="Number of top passages to retrieve")
    parser.add_argument("--print_results", action="store_true", help="Whether to print results")
    
    args = parser.parse_args()
    
    evaluate_qa_bm25(
        dataset_name=args.dataset_name,
        num_questions=args.num_questions,
        top_k=args.top_k,
        print_results=args.print_results
    )


if __name__ == "__main__":
    main() 