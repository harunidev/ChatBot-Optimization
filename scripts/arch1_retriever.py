"""
arch1_retriever.py - Unified Retrieval Module
Combines DPR coarse retrieval + Cross-Encoder reranking + Context Assembly
"""

import torch
import faiss
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import json
import os

# Import local modules
try:
    from . import arch1_rerank
    from . import arch1_colbert
except ImportError:
    import arch1_rerank
    import arch1_colbert


@dataclass
class RetrievalResult:
    """Single retrieval result with full evidence."""
    doc_id: int
    chunk_id: int
    text: str
    dpr_score: float
    rerank_score: float
    metadata: Dict = field(default_factory=dict)


@dataclass 
class RetrievalResponse:
    """Full retrieval response with all evidence for logging."""
    query: str
    dpr_results: List[RetrievalResult]  # Before reranking
    reranked_results: List[RetrievalResult]  # After reranking
    selected_contexts: List[str]  # Final assembled context
    total_context_tokens: int
    hit_at_k: Optional[int] = None  # If gold passage provided, at what k was it found


class RAGRetriever:
    """
    Unified retriever combining:
    1. DPR coarse retrieval (top-N candidates)
    2. Cross-encoder reranking (refine to top-K)
    3. Context assembly (dedup, threshold, token limit)
    """
    
    def __init__(
        self,
        passages_path: str = "indexes/passages.txt",
        metadata_path: str = "indexes/passages_metadata.jsonl",
        index_path: str = "indexes/nq_hnsw.index",
        device: str = "cuda",
        dpr_model: str = "facebook/dpr-question_encoder-single-nq-base",
        reranker_type: str = "cross-encoder",  # DEFAULT: cross-encoder for speed (30-40ms)
        rerank_batch_size: int = 64,
        max_context_tokens: int = 2048
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_context_tokens = max_context_tokens
        self.reranker_type = reranker_type
        
        print(f"🔍 Initializing RAG Retriever on {self.device}...")
        
        # Load passages
        print("   Loading passages...")
        self.passages = self._load_passages(passages_path)
        self.metadata = self._load_metadata(metadata_path)
        print(f"   ✅ Loaded {len(self.passages)} passages")
        
        # Load DPR Question Encoder
        print("   Loading DPR Question Encoder...")
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(dpr_model)
        self.q_model = DPRQuestionEncoder.from_pretrained(dpr_model).to(self.device)
        self.q_model.eval()
        
        # Load FAISS index
        print("   Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = 128
        print(f"   ✅ Index loaded ({self.index.ntotal} vectors)")
        
        # Load Reranker based on type
        if reranker_type == "colbert":
            print("   Loading ColBERT Reranker (token-level)...")
            self.reranker = arch1_colbert.ColBERTReranker(device=self.device)
        elif reranker_type == "hybrid":
            print("   Loading Hybrid Reranker (CE + ColBERT)...")
            self.reranker = arch1_colbert.HybridReranker(device=self.device)
        else:  # cross-encoder
            print("   Loading Cross-Encoder Reranker...")
            self.reranker = arch1_rerank.Reranker(device=self.device, batch_size=rerank_batch_size)
        
        print(f"✅ Retriever ready! (reranker: {reranker_type})\n")
    
    def _load_passages(self, path: str) -> List[str]:
        """Load passages from text file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Passages file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _load_metadata(self, path: str) -> Dict[int, Dict]:
        """Load metadata for passages (optional)."""
        metadata = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # Use chunk_id (line number in passages.txt) as key
                        # Fallback to 'id' or int(doc_id) if dynamic 
                        idx = data.get("chunk_id")
                        if idx is None:
                            # Fallback logic if needed, or skip
                            continue
                        metadata[int(idx)] = data
                    except json.JSONDecodeError:
                        continue
        return metadata
    
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query using DPR."""
        with torch.no_grad():
            inputs = self.q_tokenizer(
                query, 
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True
            ).to(self.device)
            embedding = self.q_model(**inputs).pooler_output.cpu().numpy()
        return embedding.astype('float32')
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count (words * 1.3)."""
        return int(len(text.split()) * 1.3)
    
    def _apply_recency_boost(
        self,
        results: List[RetrievalResult],
        boost_hours: int = 24,
        boost_factor: float = 3.0  # Strong boost (300%) for demo visibility
    ) -> List[RetrievalResult]:
        """
        Apply recency boost to recently ingested documents.
        
        Args:
            results: List of retrieval results
            boost_hours: Documents added within this many hours get boosted
            boost_factor: Score multiplier for recent docs (0.15 = +15%)
            
        Returns:
            Results with adjusted rerank_score
        """
        from datetime import datetime, timedelta
        
        now = datetime.now()
        cutoff = now - timedelta(hours=boost_hours)
        
        for result in results:
            ingested_at_str = result.metadata.get("ingested_at", "")
            if ingested_at_str:
                try:
                    ingested_at = datetime.fromisoformat(ingested_at_str)
                    if ingested_at > cutoff:
                        # Apply boost to recent documents
                        original_score = result.rerank_score
                        # Change from multiplicative to additive because cross-encoder scores are logits (often negative)
                        # Multiplicative boosting on negative scores makes them worse!
                        result.rerank_score = original_score + 100.0
                        result.metadata["recency_boosted"] = True
                        result.metadata["boost_amount"] = 100.0
                except (ValueError, TypeError):
                    pass
        
        return results
    
    def _deduplicate_contexts(self, results: List[RetrievalResult], similarity_threshold: float = 0.9) -> List[RetrievalResult]:
        """Remove near-duplicate contexts from same document."""
        if not results:
            return results
        
        deduplicated = [results[0]]
        seen_doc_ids = {results[0].doc_id: 1}
        
        for result in results[1:]:
            # Limit same-doc entries to prevent over-representation
            doc_count = seen_doc_ids.get(result.doc_id, 0)
            if doc_count >= 2:  # Max 2 chunks per document
                continue
            
            # Simple text overlap check
            is_duplicate = False
            for existing in deduplicated:
                if result.doc_id == existing.doc_id:
                    # Check word overlap
                    words1 = set(result.text.lower().split())
                    words2 = set(existing.text.lower().split())
                    if len(words1 & words2) / max(len(words1), len(words2), 1) > similarity_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                deduplicated.append(result)
                seen_doc_ids[result.doc_id] = doc_count + 1
        
        return deduplicated
    
    def retrieve(
        self,
        query: str,
        top_n_coarse: int = 100,
        top_k_rerank: int = 10,
        score_threshold: float = 0.0,
        gold_passage_id: Optional[int] = None,
        apply_recency_boost: bool = False,
        force_include_ids: Optional[List[int]] = None  # Explicit override
    ) -> RetrievalResponse:
        """
        Full retrieval pipeline:
        1. DPR coarse retrieval (top_n_coarse)
        2. Cross-encoder reranking (top_k_rerank)
        3. Context assembly with deduplication
        
        Args:
            query: User question
            top_n_coarse: Number of candidates from DPR
            top_k_rerank: Final number after reranking
            score_threshold: Minimum rerank score to include
            gold_passage_id: If provided, compute hit@k
            force_include_ids: List of doc_ids to STRICTLY include in reranking candidate pool
            
        Returns:
            RetrievalResponse with full evidence chain
        """
        # Step 1: DPR Coarse Retrieval
        q_embedding = self._encode_query(query)
        distances, indices = self.index.search(q_embedding, top_n_coarse)
        
        dpr_results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.passages):
                continue
            dpr_results.append(RetrievalResult(
                doc_id=idx,
                chunk_id=i,
                text=self.passages[idx],
                dpr_score=float(dist),
                rerank_score=0.0,
                metadata=self.metadata.get(idx, {})
            ))
            
        # Step 1.5: Explicit Force Inclusion (Deterministic)
        # Instead of guessing "recent" docs, we force add specific IDs if requested
        if force_include_ids:
            # Create a set of existing doc_ids to avoid duplicates
            existing_ids = {r.doc_id for r in dpr_results}
            
            for pid in force_include_ids:
                if pid not in existing_ids and 0 <= pid < len(self.passages):
                    dpr_results.append(RetrievalResult(
                        doc_id=pid,
                        chunk_id=0,  # Dummy
                        text=self.passages[pid],
                        dpr_score=0.0,  # Neutral score (will be fixed by reranker)
                        rerank_score=0.0,
                        metadata=self.metadata.get(pid, {})
                    ))
        
        # Step 2: Cross-Encoder Reranking
        if dpr_results:
            pairs = [[query, r.text] for r in dpr_results]
            # Optimization: Don't rerank forced items if we are going to overwrite them anyway?
            # Actually, let's rerank everything to keep it simple, then override.
            rerank_scores = self.reranker.compute_scores(pairs, show_progress=False)
            
            for result, score in zip(dpr_results, rerank_scores):
                # If forced, give it a MAX score to ensure it survives the threshold
                if force_include_ids and result.doc_id in force_include_ids:
                    result.rerank_score = 100.0  # MAX SCORE (Bypass Gate 2)
                    result.metadata["forced_retrieval"] = True
                else:
                    result.rerank_score = score
            
            # Sort by rerank score
            reranked = sorted(dpr_results, key=lambda x: x.rerank_score, reverse=True)
        else:
            reranked = []
        
        # Step 2.5: Apply recency boost (for "en güncel" queries)
        if apply_recency_boost:
            reranked = self._apply_recency_boost(reranked)
            reranked = sorted(reranked, key=lambda x: x.rerank_score, reverse=True)
        
        # Step 3: Filter by threshold and take top_k
        filtered = [r for r in reranked if r.rerank_score >= score_threshold][:top_k_rerank]
        
        # Step 4: Deduplicate
        deduplicated = self._deduplicate_contexts(filtered)
        
        # Step 5: Context Assembly (respect token limit)
        selected_contexts = []
        total_tokens = 0
        for result in deduplicated:
            tokens = self._count_tokens(result.text)
            if total_tokens + tokens <= self.max_context_tokens:
                selected_contexts.append(result.text)
                total_tokens += tokens
            else:
                break
        
        # Compute hit@k if gold provided
        hit_at_k = None
        if gold_passage_id is not None:
            for i, result in enumerate(reranked):
                if result.doc_id == gold_passage_id:
                    hit_at_k = i + 1
                    break
        
        return RetrievalResponse(
            query=query,
            dpr_results=dpr_results[:10],  # Log first 10 for inspection
            reranked_results=deduplicated,
            selected_contexts=selected_contexts,
            total_context_tokens=total_tokens,
            hit_at_k=hit_at_k
        )
    
    def log_retrieval_evidence(self, response: RetrievalResponse) -> str:
        """Format retrieval evidence for logging/display."""
        lines = [
            f"📋 RETRIEVAL EVIDENCE",
            f"{'='*50}",
            f"Query: {response.query}",
            f"",
            f"🔍 DPR Top-5 (before reranking):"
        ]
        
        for i, r in enumerate(response.dpr_results[:5]):
            snippet = r.text[:100].replace('\n', ' ') + "..."
            lines.append(f"   {i+1}. [doc:{r.doc_id}] score={r.dpr_score:.4f} | {snippet}")
        
        lines.extend([
            f"",
            f"🎯 After Reranking (top-{len(response.reranked_results)}):"
        ])
        
        for i, r in enumerate(response.reranked_results):
            snippet = r.text[:100].replace('\n', ' ') + "..."
            lines.append(f"   {i+1}. [doc:{r.doc_id}] score={r.rerank_score:.4f} | {snippet}")
        
        lines.extend([
            f"",
            f"📊 Context Stats:",
            f"   Total contexts: {len(response.selected_contexts)}",
            f"   Total tokens: ~{response.total_context_tokens}",
        ])
        
        if response.hit_at_k is not None:
            lines.append(f"   Gold hit@: {response.hit_at_k}")
        
        return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    retriever = RAGRetriever()
    
    # Test query
    response = retriever.retrieve(
        query="What happens in Romeo and Juliet?",
        top_n_coarse=50,
        top_k_rerank=5
    )
    
    print(retriever.log_retrieval_evidence(response))
