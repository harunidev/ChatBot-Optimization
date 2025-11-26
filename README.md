# RAG Architecture 1: Semantic Accuracy Optimized

This project implements a production-quality RAG pipeline focused on maximizing semantic accuracy and factual consistency using DPR (Dense Passage Retrieval), FAISS (HNSW Index), and ColBERT-style reranking.

## Goal
To provide a robust, modular codebase for RAG research and deployment, optimized for Google Colab Pro+ (A100).

## Environment & Requirements
- Python 3.12
- PyTorch 2.3.1
- Transformers 4.57.1
- faiss-gpu
- A100 GPU recommended for full scale, but runs on CPU/smaller GPUs with smaller batch sizes.

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
Downloads Tiny Shakespeare dataset and generates passages/questions.
```bash
python scripts/arch1_prepare_data.py --output-dir data --output-passages-txt indexes/passages.txt
```

### 2. Generate Embeddings
Encodes passages using DPR.
```bash
python scripts/arch1_embeddings.py --passages-txt indexes/passages.txt --output-embeddings indexes/passage_emb.npy --batch-size 16
```

### 3. Build Index
Builds FAISS HNSW index.
```bash
python scripts/arch1_faiss.py --mode build --embedding-path indexes/passage_emb.npy --index-path indexes/nq_hnsw.index
```

### 4. Evaluation
Runs the full pipeline (Retrieve -> Rerank -> Eval) and generates a metrics report.
```bash
python scripts/arch1_eval.py --eval-file data/gold_eval.jsonl --passages-txt indexes/passages.txt --output-report outputs/metrics_report.txt
```

## Metrics & Testing (T1-T5)
The system evaluates the following:
- **T1 Semantic Accuracy**: Exact Match (EM), F1.
- **T2 Retrieval Stability**: Recall@k (1, 5, 20, 100).
- **T3 Latency**: End-to-end query time.
- **T4 Memory**: GPU/CPU usage.
- **T5 Output Coherence**: ROUGE-L, BLEU.

Target metrics are defined in `scripts/arch1_eval.py` and the report will flag status (Meets/Below Expected).

## Notes for Colab Pro+
- Ensure you select a GPU runtime.
- If OOM occurs, reduce `--batch-size` in `arch1_embeddings.py` or `arch1_rerank.py`.
