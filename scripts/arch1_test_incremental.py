"""
arch1_test_incremental.py - Incremental Update Validation Test
Tests that newly added documents are properly indexed and retrievable.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_incremental_update():
    """
    End-to-end test for incremental index update:
    1. Add new documents with unique content
    2. Update index incrementally
    3. Query for the new content
    4. Verify it comes from the new source
    """
    print("="*60)
    print("🧪 INCREMENTAL UPDATE TEST")
    print("="*60 + "\n")
    
    # Import modules
    import arch1_embeddings
    import arch1_faiss
    from arch1_retriever import RAGRetriever
    
    # 1. Create test documents with unique content
    print("Step 1: Creating unique test documents...")
    
    test_docs = [
        {
            "text": "INCREMENTAL_TEST_UNIQUE_12345: The quantum neural network achieved 99.9% accuracy on the benchmark dataset.",
            "doc_id": "test_new_1",
            "source": "test_incremental",
            "ingested_at": datetime.now().isoformat()
        },
        {
            "text": "INCREMENTAL_TEST_UNIQUE_67890: The revolutionary AI system can predict market trends with unprecedented precision.",
            "doc_id": "test_new_2", 
            "source": "test_incremental",
            "ingested_at": datetime.now().isoformat()
        }
    ]
    
    print(f"   Created {len(test_docs)} test documents with unique identifiers")
    
    # 2. Append to passages file
    print("\nStep 2: Appending to passages file...")
    
    passages_path = "indexes/passages.txt"
    metadata_path = "indexes/passages_metadata.jsonl"
    
    # Read current count
    with open(passages_path, 'r') as f:
        original_count = sum(1 for _ in f)
    
    # Append new documents
    with open(passages_path, 'a') as f:
        for doc in test_docs:
            f.write(doc["text"] + "\n")
    
    with open(metadata_path, 'a') as f:
        for i, doc in enumerate(test_docs):
            meta = {
                "doc_id": doc["doc_id"],
                "chunk_id": original_count + i,
                "source": doc["source"],
                "ingested_at": doc["ingested_at"],
                "section": "Test",
                "doc_version": "1.0"
            }
            f.write(json.dumps(meta) + "\n")
    
    print(f"   Original passages: {original_count}")
    print(f"   New passages added: {len(test_docs)}")
    print(f"   New total: {original_count + len(test_docs)}")
    
    # 3. Generate embeddings for new documents only
    print("\nStep 3: Generating embeddings for new documents...")
    
    from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
    model.eval()
    
    new_texts = [doc["text"] for doc in test_docs]
    with torch.no_grad():
        inputs = tokenizer(new_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        new_embeddings = model(**inputs).pooler_output.cpu().numpy()
    
    print(f"   Generated {new_embeddings.shape[0]} new embeddings")
    
    # 4. Add to index incrementally
    print("\nStep 4: Adding to index incrementally...")
    
    added = arch1_faiss.add_documents(
        index_path="indexes/nq_hnsw.index",
        new_embeddings=new_embeddings
    )
    
    print(f"   Added {added} documents to index")
    
    # 5. Query for the new content
    print("\nStep 5: Testing retrieval of new content...")
    
    # Initialize retriever with recency boost
    retriever = RAGRetriever(
        passages_path=passages_path,
        metadata_path=metadata_path,
        reranker_type="cross-encoder"  # Faster for test
    )
    
    # Test queries that should match new documents
    test_queries = [
        ("quantum neural network accuracy", "test_new_1"),
        ("AI system predict market trends", "test_new_2"),
    ]
    
    results = []
    for query, expected_source in test_queries:
        response = retriever.retrieve(
            query=query,
            top_k_rerank=5,
            apply_recency_boost=True  # Boost new documents
        )
        
        # Check if new document is in top results
        found_new = False
        found_position = -1
        
        for i, result in enumerate(response.reranked_results):
            if "INCREMENTAL_TEST_UNIQUE" in result.text:
                found_new = True
                found_position = i + 1
                break
        
        results.append({
            "query": query,
            "expected": expected_source,
            "found_new_doc": found_new,
            "position": found_position,
            "recency_boosted": any(r.metadata.get("recency_boosted") for r in response.reranked_results)
        })
        
        status = "✅ PASS" if found_new and found_position <= 3 else "❌ FAIL"
        print(f"   Query: '{query[:30]}...'")
        print(f"   {status} - New doc at position {found_position}")
    
    # 6. Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r["found_new_doc"] and r["position"] <= 3)
    total = len(results)
    
    print(f"   Passed: {passed}/{total}")
    print(f"   Recency boost working: {any(r['recency_boosted'] for r in results)}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - Incremental update working correctly!")
    else:
        print("\n⚠️ SOME TESTS FAILED - Check indexing or retrieval")
    
    return passed == total


if __name__ == "__main__":
    success = test_incremental_update()
    sys.exit(0 if success else 1)
