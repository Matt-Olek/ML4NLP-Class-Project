# Hypothetical Document Embeddings (HyDE) - Analysis and Evaluation for Multi-Retrieval

This repository contains the code and resources for the project analyzing the performance of Hypothetical Document Embeddings (HyDE) in multi-retrieval scenarios, as detailed in the accompanying paper: "[Hypotheticals Document Embedding (HyDE) - Analysis and evaluation](ProjectAnalysis.pdf)".

## Overview

Hypothetical Document Embeddings (HyDE) is a powerful zero-shot dense retrieval technique that leverages Large Language Models (LLMs) to generate hypothetical documents, which are then encoded using unsupervised contrastive encoders (like Contriever or BGE) to perform retrieval without relevance labels.

While HyDE excels in tasks focused on finding the single best answer (e.g., web search, simple QA), its effectiveness in **multi-retrieval settings**—where identifying a diverse set of relevant documents is crucial—has been less explored. This project specifically evaluates HyDE's performance on such tasks, using the RAG-Mini-BioASQ dataset as a benchmark.

We compare HyDE against classical baselines:

1. **BM25:** A traditional sparse retrieval method.
2. **Encoder Only:** Using the unsupervised contrastive encoder directly on the query (without the HyDE LLM generation step).

## Key Findings

Our experiments on the RAG-Mini-BioASQ dataset (focused on biomedical question answering requiring multiple relevant passages) revealed:

* The **Encoder Only** baseline (using `BAAI/bge-small-en-v1.5` directly on queries) outperformed both BM25 and HyDE (using TinyLlama-1.1B-Chat + BGE) in terms of mean and median nDCG@1000.
* HyDE's single hypothetical document generation may not sufficiently capture the semantic diversity needed to retrieve multiple distinct relevant documents effectively in this specific multi-retrieval context.
* HyDE introduces significant computational overhead due to the LLM generation step compared to the direct encoder or BM25.

**Main Results (nDCG@1000 on RAG-Mini-BioASQ subset):**

| Method       | Mean nDCG | Median nDCG |
| :----------- | :-------- | :---------- |
| BM25         | 0.561     | 0.595       |
| Encoder Only | **0.617** | **0.647**   |
| HyDE         | 0.570     | 0.606       |

*(See Figures 1-3 in the paper for nDCG distributions)*
<!-- ![nDCG Distribution BM25](path/to/bm25_plot.png) -->
<!-- ![nDCG Distribution Encoder Only](path/to/encoder_plot.png) -->
<!-- ![nDCG Distribution HyDE](path/to/hyde_plot.png) -->

## Repository Structure

```
.
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

## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Matt-Olek/ML4NLP-Class-Project.git
    cd ML4NLP-Class-Project
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Note: This project was developed using Python 3.12 and requires libraries like `transformers`, `datasets`, `faiss-cpu`, `torch`, `sentence-transformers`. Ensure you have compatible CUDA drivers if using GPU acceleration for model inference (though FAISS index runs on CPU as configured).*

4. **Download Models:** The scripts will automatically download required models (like `BAAI/bge-small-en-v1.5`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) via Hugging Face `transformers`. Ensure you have an internet connection the first time you run them, or download them manually if needed.

## Usage

1. **Build a vectorstore from the dataset (for encoder and HyDE evaluations):**

    ```bash
    python src/build_vectorstore.py rag-datasets/rag-mini-bioasq --all_passages
    ```

2. **Run BM25 Baseline:**

    ```bash
    python src/evaluate_bm25.py rag-datasets/rag-mini-bioasq --num_questions 1000 --top_k 1000
    ```

3. **Run Encoder Only Baseline:**

    ```bash
    python src/evaluate.py rag-datasets/rag-mini-bioasq --num_questions 1000 --top_k 1000
    ```

4. **Run HyDE Evaluation:**
    * *Note: This step can be time-consuming due to LLM generation.*

    ```bash
    python src/evaluate.py rag-datasets/rag-mini-bioasq --num_questions 1000 --top_k 1000 --hyde
    ```

## Dataset

* **Dataset:** RAG-Mini-BioASQ
* **Source:** Hugging Face Datasets library (`rag-datasets/rag-mini-bioasq`)
* **Characteristics:** Biomedical QA focus, pre-split passages, includes `relevant_passage_ids` for multi-retrieval evaluation.
* **Subset Used:** Evaluation was performed on the first 500 questions from the test set for computational feasibility of HyDE, while baselines could have been run on the full set (4,719 questions).

## Future Work & Extensions

As discussed in the paper, potential extensions include:

* Multi-hypothesis HyDE generation.
* Filtering/diversifying generated documents.
* Hybrid approaches combining Encoder Only retrieval with generative refinement or reranking.
