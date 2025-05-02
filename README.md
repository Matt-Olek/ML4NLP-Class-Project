# ML4NLP QA System

This project implements a question-answering system using vector embeddings and retrieval-augmented generation (RAG). It allows you to build vectorstores from datasets and evaluate question-answering performance.

## Project Structure

```
├── src/
│   ├── data/
│   │   └── dataset_loader.py      # Dataset loading and caching
│   ├── models/
│   │   ├── embedding.py           # Document embedding
│   │   ├── retriever.py           # Document retrieval
│   │   └── qa_model.py            # Question answering
│   ├── utils/
│   │   └── config.py              # Configuration management
│   └── main.py                    # Main workflow
├── data/                          # Dataset storage
├── cache/                         # Cached data
├── models/                        # Model storage
├── vectorstores/                  # Vectorstore storage
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
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

3. Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

The project provides two main workflows:

1. Building a vectorstore from a dataset:

```python
from src.main import build_vectorstore

build_vectorstore("rag-datasets/rag-mini-bioasq", num_passages=2000)
```

2. Evaluating QA performance:

```python
from src.main import evaluate_qa

evaluate_qa("rag-datasets/rag-mini-bioasq", num_questions=10)
```

You can also run the complete workflow using the main script:

```bash
python src/main.py
```

## Configuration

The project's configuration can be modified in `src/utils/config.py`. Key settings include:

- Model configurations (embedding model, QA model)
- Retrieval settings (similarity threshold, number of documents)
- Directory paths
- API configurations
