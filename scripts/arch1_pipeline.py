"""
arch1_pipeline.py - End-to-End RAG Pipeline
Main entry point for the RAG system: Query → Retrieve → Generate → Log
"""

import argparse
import json
import os
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Add script directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arch1_retriever import RAGRetriever, RetrievalResponse
from arch1_generate import RAGGenerator, GenerationResult


@dataclass
class RAGResponse:
    """Complete RAG response with full evidence chain."""
    query: str
    answer: str
    citations: List[int]
    confidence: float
    is_grounded: bool
    is_no_answer: bool
    
    # Retrieval evidence
    retrieval_top_k: int
    retrieval_total_contexts: int
    retrieval_total_tokens: int
    hit_at_k: Optional[int]
    
    # Timing
    timestamp: str
    
    # Full evidence (for logging)
    retrieval_evidence: Optional[dict] = None
    generation_evidence: Optional[dict] = None


class RAGPipeline:
    """
    Complete RAG Pipeline combining:
    1. Retrieval (DPR + Rerank + Context Assembly)
    2. Generation (Faithfulness-oriented with grounding check)
    
    This is the main entry point for the system.
    """
    
    def __init__(
        self,
        passages_path: str = "indexes/passages.txt",
        metadata_path: str = "indexes/passages_metadata.jsonl",
        index_path: str = "indexes/nq_hnsw.index",
        device: str = "cuda",
        llm_model: str = "google/flan-t5-base",
        no_answer_threshold: float = 0.3,
        log_dir: str = "outputs/logs"
    ):
        """
        Initialize the complete RAG pipeline.
        
        Args:
            passages_path: Path to passages text file
            metadata_path: Path to passages metadata JSONL
            index_path: Path to FAISS index
            device: cuda or cpu
            llm_model: HuggingFace model for generation
            no_answer_threshold: Confidence below this triggers "bilmiyorum"
            log_dir: Directory to save query logs
        """
        print("="*60)
        print("🚀 INITIALIZING RAG PIPELINE (Architecture 1)")
        print("="*60 + "\n")
        
        # Initialize retriever
        self.retriever = RAGRetriever(
            passages_path=passages_path,
            metadata_path=metadata_path,
            index_path=index_path,
            device=device
        )
        
        # Initialize generator
        self.generator = RAGGenerator(
            model_name=llm_model,
            device=device,
            no_answer_threshold=no_answer_threshold
        )
        
        # Setup logging
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        print("="*60)
        print("✅ RAG PIPELINE READY")
        print("="*60 + "\n")
    
    def query(
        self,
        question: str,
        top_n_coarse: int = 100,
        top_k_rerank: int = 10,
        gold_passage_id: Optional[int] = None,
        log_evidence: bool = True,
        verbose: bool = True
    ) -> RAGResponse:
        """
        Execute full RAG pipeline: Retrieve → Generate
        
        Args:
            question: User's question
            top_n_coarse: DPR candidates
            top_k_rerank: Final contexts after reranking
            gold_passage_id: For evaluation - compute hit@k
            log_evidence: Save evidence to log file
            verbose: Print progress
            
        Returns:
            RAGResponse with answer and full evidence chain
        """
        timestamp = datetime.now().isoformat()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"📩 Query: {question}")
            print(f"{'='*60}\n")
        
        # Step 1: Retrieve
        if verbose:
            print("🔍 Step 1: Retrieving relevant passages...")
        
        retrieval_response = self.retriever.retrieve(
            query=question,
            top_n_coarse=top_n_coarse,
            top_k_rerank=top_k_rerank,
            gold_passage_id=gold_passage_id
        )
        
        if verbose:
            print(f"   Found {len(retrieval_response.selected_contexts)} contexts ({retrieval_response.total_context_tokens} tokens)")
            if retrieval_response.hit_at_k:
                print(f"   Gold passage hit@{retrieval_response.hit_at_k}")
        
        # Step 2: Generate
        if verbose:
            print("\n🤖 Step 2: Generating answer...")
        
        doc_ids = [r.doc_id for r in retrieval_response.reranked_results]
        
        generation_result = self.generator.generate(
            query=question,
            contexts=retrieval_response.selected_contexts,
            doc_ids=doc_ids,
            require_grounding=True
        )
        
        if verbose:
            print(f"   Confidence: {generation_result.confidence:.2f}")
            print(f"   Grounded: {'✅' if generation_result.is_grounded else '❌'}")
            if generation_result.is_no_answer:
                print(f"   ⚠️ No-answer triggered")
        
        # Build response
        response = RAGResponse(
            query=question,
            answer=generation_result.answer,
            citations=generation_result.citations,
            confidence=generation_result.confidence,
            is_grounded=generation_result.is_grounded,
            is_no_answer=generation_result.is_no_answer,
            retrieval_top_k=top_k_rerank,
            retrieval_total_contexts=len(retrieval_response.selected_contexts),
            retrieval_total_tokens=retrieval_response.total_context_tokens,
            hit_at_k=retrieval_response.hit_at_k,
            timestamp=timestamp
        )
        
        # Add full evidence for logging
        if log_evidence:
            response.retrieval_evidence = {
                "dpr_top5": [
                    {"doc_id": r.doc_id, "score": r.dpr_score, "snippet": r.text[:100]}
                    for r in retrieval_response.dpr_results[:5]
                ],
                "reranked": [
                    {"doc_id": r.doc_id, "score": r.rerank_score, "snippet": r.text[:100]}
                    for r in retrieval_response.reranked_results
                ]
            }
            response.generation_evidence = {
                "reasoning": generation_result.reasoning,
                "context_used": generation_result.context_used[:300]
            }
            
            # Save to log file
            self._log_query(response)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"📝 Answer: {generation_result.answer}")
            print(f"📚 Citations: {generation_result.citations}")
            print(f"{'='*60}\n")
        
        return response
    
    def _log_query(self, response: RAGResponse):
        """Save query evidence to log file."""
        log_file = os.path.join(self.log_dir, f"queries_{datetime.now().strftime('%Y%m%d')}.jsonl")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(response), ensure_ascii=False) + "\n")
    
    def batch_query(
        self,
        questions: List[str],
        gold_passage_ids: Optional[List[int]] = None,
        **kwargs
    ) -> List[RAGResponse]:
        """Execute pipeline for multiple questions."""
        results = []
        for i, question in enumerate(questions):
            gold_id = gold_passage_ids[i] if gold_passage_ids else None
            result = self.query(question, gold_passage_id=gold_id, verbose=False, **kwargs)
            results.append(result)
            print(f"[{i+1}/{len(questions)}] {question[:50]}... → {'✅' if result.is_grounded else '❌'}")
        return results


def main():
    """CLI entry point for RAG pipeline."""
    parser = argparse.ArgumentParser(description="RAG Pipeline - Query the system")
    parser.add_argument("--query", "-q", type=str, help="Single query to process")
    parser.add_argument("--queries-file", type=str, help="JSONL file with queries")
    parser.add_argument("--passages", default="indexes/passages.txt")
    parser.add_argument("--index", default="indexes/nq_hnsw.index")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="google/flan-t5-base")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k after reranking")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        passages_path=args.passages,
        index_path=args.index,
        device=args.device,
        llm_model=args.model
    )
    
    # Single query mode
    if args.query:
        pipeline.query(args.query, top_k_rerank=args.top_k)
    
    # Batch mode
    elif args.queries_file:
        queries = []
        gold_ids = []
        with open(args.queries_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                queries.append(data.get("question", data.get("query", "")))
                gold_ids.append(int(data.get("gold_passage_id", -1)))
        
        results = pipeline.batch_query(questions=queries, gold_passage_ids=gold_ids)
        
        # Print summary
        grounded = sum(1 for r in results if r.is_grounded)
        no_answers = sum(1 for r in results if r.is_no_answer)
        print(f"\n📊 Summary: {grounded}/{len(results)} grounded, {no_answers} no-answers")
    
    # Interactive mode
    elif args.interactive:
        print("\n🎮 Interactive Mode - Type 'exit' to quit\n")
        while True:
            try:
                query = input("❓ Question: ").strip()
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                if query:
                    pipeline.query(query, top_k_rerank=args.top_k)
            except KeyboardInterrupt:
                break
        print("\n👋 Goodbye!")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
