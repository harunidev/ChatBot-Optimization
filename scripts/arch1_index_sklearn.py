import numpy as np, joblib, os
from sklearn.neighbors import NearestNeighbors
xb=np.load("indexes/passage_emb.npy").astype("float32")
nn=NearestNeighbors(n_neighbors=100,algorithm="auto",metric="euclidean").fit(xb)
os.makedirs("indexes",exist_ok=True)
joblib.dump(nn,"indexes/sk_index.joblib")
print("OK: indexes/sk_index.joblib")