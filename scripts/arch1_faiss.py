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


# ============================================================================
# INCREMENTAL INDEX OPERATIONS
# ============================================================================

def add_documents(
    index_path: str,
    new_embeddings: np.ndarray,
    id_map_path: str = "indexes/id_map.json"
) -> int:
    """
    Add new documents to existing index (incremental update).
    
    Args:
        index_path: Path to existing FAISS index
        new_embeddings: New embeddings to add (N x D)
        id_map_path: Path to ID mapping file
        
    Returns:
        Number of documents added
    """
    import json
    
    print(f"Loading existing index from {index_path}...")
    index = faiss.read_index(index_path)
    
    current_count = index.ntotal
    print(f"Current index size: {current_count}")
    
    # Add new embeddings
    new_embeddings = new_embeddings.astype('float32')
    index.add(new_embeddings)
    
    new_count = index.ntotal
    added = new_count - current_count
    print(f"Added {added} new documents. New total: {new_count}")
    
    # Update ID map
    id_map = {}
    if os.path.exists(id_map_path):
        with open(id_map_path, 'r') as f:
            id_map = json.load(f)
    
    # Add new IDs (simple sequential mapping)
    for i in range(added):
        new_id = current_count + i
        id_map[str(new_id)] = {
            "added_at": __import__('datetime').datetime.now().isoformat(),
            "original_index": new_id
        }
    
    with open(id_map_path, 'w') as f:
        json.dump(id_map, f, indent=2)
    
    # Save updated index
    faiss.write_index(index, index_path)
    print(f"Updated index saved to {index_path}")
    
    return added


def remove_documents(
    index_path: str,
    embedding_path: str,
    ids_to_remove: list,
    id_map_path: str = "indexes/id_map.json"
) -> int:
    """
    Remove documents from index by rebuilding without specified IDs.
    Note: FAISS HNSW doesn't support direct deletion, so we rebuild.
    
    Args:
        index_path: Path to FAISS index
        embedding_path: Path to embeddings file
        ids_to_remove: List of document IDs to remove
        id_map_path: Path to ID mapping file
        
    Returns:
        Number of documents removed
    """
    import json
    
    print(f"Removing {len(ids_to_remove)} documents from index...")
    
    # Load embeddings
    embeddings = np.load(embedding_path)
    
    # Create mask for documents to keep
    ids_to_remove_set = set(ids_to_remove)
    keep_mask = [i not in ids_to_remove_set for i in range(len(embeddings))]
    
    # Filter embeddings
    new_embeddings = embeddings[keep_mask]
    removed_count = len(embeddings) - len(new_embeddings)
    
    print(f"Kept {len(new_embeddings)} documents, removed {removed_count}")
    
    # Save filtered embeddings
    np.save(embedding_path, new_embeddings)
    
    # Rebuild index
    d = new_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, CONFIG["M"])
    index.hnsw.efConstruction = CONFIG["EF_CONSTRUCTION"]
    index.add(new_embeddings.astype('float32'))
    faiss.write_index(index, index_path)
    
    # Update ID map
    if os.path.exists(id_map_path):
        with open(id_map_path, 'r') as f:
            id_map = json.load(f)
        
        # Remove deleted IDs
        for id_str in list(id_map.keys()):
            if int(id_str) in ids_to_remove_set:
                del id_map[id_str]
        
        with open(id_map_path, 'w') as f:
            json.dump(id_map, f, indent=2)
    
    print(f"Index rebuilt. New size: {index.ntotal}")
    return removed_count


def get_index_info(index_path: str, id_map_path: str = "indexes/id_map.json") -> dict:
    """Get information about the current index."""
    import json
    
    info = {"index_path": index_path, "exists": os.path.exists(index_path)}
    
    if info["exists"]:
        index = faiss.read_index(index_path)
        info["total_vectors"] = index.ntotal
        info["dimension"] = index.d
    
    if os.path.exists(id_map_path):
        with open(id_map_path, 'r') as f:
            id_map = json.load(f)
        info["tracked_documents"] = len(id_map)
        if id_map:
            latest = max(id_map.values(), key=lambda x: x.get("added_at", ""))
            info["last_updated"] = latest.get("added_at", "unknown")
    
    return info

def main():
    parser = argparse.ArgumentParser(description="FAISS HNSW Indexing")
    parser.add_argument("--mode", type=str, choices=["build", "dry-run"], default="build", help="Mode: build (default) or dry-run")
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
