import argparse
import os
import numpy as np
import faiss

CONFIG = {
    "INDEX_PATH": "indexes/nq_hnsw.index",
    "EMBEDDING_PATH": "indexes/passage_emb.npy",
    "M": 64,
    "EF_CONSTRUCTION": 200,
    "EF_SEARCH": 128
}

def build_index(embedding_path: str, index_path: str):
    """Builds an HNSW index from embeddings."""
    print(f"Loading embeddings from {embedding_path}...")
    embeddings = np.load(embedding_path)
    d = embeddings.shape[1]
    print(f"Embedding dimension: {d}, Count: {embeddings.shape[0]}")

    print("Building HNSW index...")
    # HNSWFlat: HNSW graph with full vectors stored
    index = faiss.IndexHNSWFlat(d, CONFIG["M"])
    index.hnsw.efConstruction = CONFIG["EF_CONSTRUCTION"]
    
    # Train not needed for HNSWFlat, but adding vectors
    print("Adding vectors to index...")
    index.add(embeddings)
    
    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)

def search_index(index_path: str, queries: np.ndarray, k: int = 100):
    """Searches the index."""
    print(f"Loading index from {index_path}...")
    index = faiss.read_index(index_path)
    index.hnsw.efSearch = CONFIG["EF_SEARCH"]
    
    print(f"Searching for {queries.shape[0]} queries (k={k})...")
    distances, indices = index.search(queries, k)
    return distances, indices

def main():
    parser = argparse.ArgumentParser(description="FAISS HNSW Indexing")
    parser.add_argument("--mode", type=str, choices=["build", "dry-run"], required=True, help="Mode: build or dry-run")
    parser.add_argument("--embedding-path", type=str, default=CONFIG["EMBEDDING_PATH"], help="Path to embeddings")
    parser.add_argument("--index-path", type=str, default=CONFIG["INDEX_PATH"], help="Path to save/load index")
    
    args = parser.parse_args()
    
    if args.mode == "build":
        os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
        build_index(args.embedding_path, args.index_path)
        
    elif args.mode == "dry-run":
        # Generate a random vector for testing
        print("Running dry-run search with random vector...")
        if not os.path.exists(args.index_path):
            print("Index not found! Build it first.")
            return
            
        # Load index to get dimension
        index = faiss.read_index(args.index_path)
        d = index.d
        
        dummy_query = np.random.rand(1, d).astype('float32')
        distances, indices = search_index(args.index_path, dummy_query, k=5)
        print("Top 5 results (indices):", indices)
        print("Top 5 results (distances):", distances)

if __name__ == "__main__":
    main()
