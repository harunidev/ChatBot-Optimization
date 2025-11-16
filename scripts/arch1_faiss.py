import faiss, numpy as np
import os

os.makedirs("indexes", exist_ok=True)

xb = np.load("indexes/passage_emb.npy").astype("float32")
# IP (Inner Product) + normalize için: vektörleri normalize et
# Normalize edilmiş vektörlerle L2 distance = cosine similarity (IP)
norms = np.linalg.norm(xb, axis=1, keepdims=True)
norms[norms == 0] = 1  # Avoid division by zero
xb_normalized = (xb / norms).astype("float32")

# Hedef değerlere ulaşmak için daha iyi parametreler
index = faiss.IndexHNSWFlat(xb_normalized.shape[1], 32)  # M=32
index.hnsw.efConstruction = 200  # İyi construction
index.hnsw.efSearch = 128  # Daha yüksek search (daha iyi recall için)
index.add(xb_normalized)
faiss.write_index(index, "indexes/nq_hnsw.index")
print("OK: indexes/nq_hnsw.index")
