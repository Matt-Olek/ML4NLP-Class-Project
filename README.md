# ML4NLP QA System

This project implements a question-answering system using vector embeddings and retrieval-augmented generation (RAG). It allows you to build vectorstores from datasets and evaluate question-answering performance.

## Project Structure

```
├── src/
│   ├── data/
│   │   └── dataset_loader.py      # Dataset loading and caching
│   ├── models/
│   │   ├── embedding.py           # Document embedding
│   │   ├── bm25_retriever.py      # BM25 retriever
│   │   ├── hyde.py                # HyDE model
│   │   ├── retriever.py           # Document retriever (encoder)
│   │   └── qa_model.py            # Question answering model (not used in experiments)
│   ├── utils/
│   │   └── config.py              # Configuration management
│   └── build_vectorstore.py       # Building vectorstore
│   └── evaluate_bm25.py           # Evaluating BM25
│   └── evaluate.py                # Evaluating encoder-based methods
```

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd ML4NLP-Class-Project
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Building a vectorstore from a dataset (for encoding and hyde evaluations):

```bash
python .\src\build_vectorstore.py rag-datasets/rag-mini-bioasq --all_passages
```

2. Evaluating methods:

- Run BM25 evaluation:

```bash
python .\src\evaluate_bm25.py rag-datasets/rag-mini-bioasq --num_questions 1000 --top_k 1000
```

- Run encoder evaluation:

```bash
python .\src\evaluate.py rag-datasets/rag-mini-bioasq --num_questions 1000 --top_k 1000
```

- Run hyde evaluation:

```bash
python .\src\evaluate.py rag-datasets/rag-mini-bioasq --num_questions 1000 --top_k 1000 --hyde
```
