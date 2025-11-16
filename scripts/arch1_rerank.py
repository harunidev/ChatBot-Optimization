import numpy as np, os
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import CrossEncoder

# MacBook için device seçimi: MPS (Apple Silicon) > CUDA > CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

passages=[l.strip() for l in open("indexes/passages.txt",encoding="utf-8")]
Q_MODEL="facebook/dpr-question_encoder-single-nq-base"
q_tok=AutoTokenizer.from_pretrained(Q_MODEL)
q_enc=AutoModel.from_pretrained(Q_MODEL).eval().to(device)
ce=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

FAISS_INDEX_PATH = "indexes/nq_hnsw.index"
_faiss_index = None

def dpr_qvec(q):
    t=q_tok([q],padding=True,truncation=True,return_tensors="pt",max_length=64).to(device)
    with torch.no_grad(): v=q_enc(**t).pooler_output
    v_np = v.detach().cpu().numpy().astype("float32")
    # Normalize for cosine similarity (IP with normalized vectors = cosine)
    norm = np.linalg.norm(v_np, axis=1, keepdims=True)
    norm[norm == 0] = 1  # Avoid division by zero
    return (v_np / norm).astype("float32")

def _load_faiss_index():
    global _faiss_index
    if _faiss_index is None:
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"{FAISS_INDEX_PATH} bulunamadı. Lütfen önce index oluşturun.")
        import faiss
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    return _faiss_index

def search_topk(qv, k=50):
    ix=_load_faiss_index()
    D,I=ix.search(qv,k)
    return [int(i) for i in I[0] if i >= 0]

def search_and_rerank(query, k_retr=50, k_final=10):
    """Güçlendirilmiş reranker ve arama - k_retr artırıldı, IP + normalize eklendi."""
    try:
        qv=dpr_qvec(query)  # Zaten normalize edilmiş
        # k_retr artırıldı - daha geniş arama kapsamı
        candidate_indices=search_topk(qv, k_retr)
    except FileNotFoundError:
        return []

    if not candidate_indices:
        return []

    cands=[passages[i] for i in candidate_indices]
    # Reranking ile en iyileri seç (k_final kadar)
    if len(cands) > k_final:
        scores=ce.predict([(query,c) for c in cands])
        order=np.argsort(-np.array(scores))[:k_final]
        return [cands[i] for i in order]
    else:
        # Eğer yeterli aday yoksa, reranking yapmadan döndür
        scores=ce.predict([(query,c) for c in cands])
        order=np.argsort(-np.array(scores))
        return [cands[i] for i in order]

if __name__ == "__main__":
    sample_q="What is the capital of France?"
    print(search_and_rerank(sample_q, 50, 5))
