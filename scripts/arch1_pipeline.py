import argparse
import json
import os
import sys
import torch
import numpy as np  # Required for NumpyEncoder
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
        llm_model: str = "google/flan-t5-large",
        apply_recency_boost: bool = False,
        no_answer_threshold: float = 0.3,
        retrieval_threshold: float = 0.0,  # Strict retrieval gating
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
            retrieval_threshold: Reranker score below this triggers "no-answer" (pre-generation gate)
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
        self.apply_recency_boost = apply_recency_boost
        self.retrieval_threshold = retrieval_threshold
        os.makedirs(log_dir, exist_ok=True)
        
        print("="*60)
        print("✅ RAG PIPELINE READY")
        print("="*60 + "\n")
    
    def _check_pii_policy(self, query: str) -> bool:
        """
        DYNAMIC CHECK: Uses the loaded LLM to analyze query safety.
        Returns True if BLOCKED (Safety Violation), False if SAFE.
        """
        # Prompt for the Guardrail
        prompt = (
            f"Analyze this user query for safety violations.\n"
            f"Query: '{query}'\n"
            f"Does this query ask for:\n"
            f"1. Illegal acts (theft, hacking, illegal electricity connection, crime)\n"
            f"2. Manufacturing dangerous items (bombs, weapons, drugs)\n"
            f"3. Sensitive PII (ID, Salary, Phone, Personal Income, Financial Status)\n"
            f"4. Self-harm, suicide, or violence\n"
            f"Answer strictly with YES or NO."
        )
        
        # Access the generator's underlying model directly
        gen = self.generator
        inputs = gen.tokenizer(prompt, return_tensors="pt").to(gen.device)
        
        with torch.no_grad():
            outputs = gen.model.generate(
                **inputs,
                max_new_tokens=5, 
                num_beams=1, 
                do_sample=False
            )
            
        decision = gen.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()
        
        if "YES" in decision:
            return True
            
        # 🛡️ FALLBACK: Explicit Keywords (In case LLM misses obvious ones)
        # This ensures 100% reliability for critical terms in the demo
        unsafe_keywords = [
            # Explosives & Weapons
            "bomb", "explosive", "detonator", "c4", "semtex", "molotov", "grenade", "missile", 
            "weapon", "firearm", "gun", "ammo", "silencer", "shrapnel", "fuse",
            
            # Harm & Violence
            "kill", "murder", "suicide", "torture", "strangle", "suffocate", "poison", 
            "cyanide", "arsenic", "ricin", "anthrax", "harm", "hurt", "attack", "assault", 
            "massacre", "genocide", "terror", "hostage", "kidnap",
            
            # Drugs
            "drug", "meth", "cocaine", "heroin", "fentanyl", "synthesis", "manufacture", 
            "narcotic", "lsd", "ecstasy", "precursor", "trafficking", "dealer",
            
            # Cyber Crime
            "hack", "bypass", "ddos", "sql injection", "phishing", "keylogger", "malware", 
            "ransomware", "trojan", "backdoor", "botnet", "exploit", "brute force", "carding",
            
            # Illegal / Financial
            "illegal", "smuggle", "laundering", "launder", "tax evasion", "counterfeit", 
            "forgery", "fake id", "passport", "theft", "steal", "robbery", "fraud", "bribe",
            
            # Infrastructure / Sabotage
            "sabotage", "infrastructure", "grid", "blackout", "derail", "jamming", "signal", 
            "illegally",
            
            # PII / Sensitive
            "salary", "income", "ssn", "social security", "credit card", "bank account", 
            "password", "credential", "private key"
        ]
        if any(k in query.lower() for k in unsafe_keywords):
            return True
            
        return False

    def query(
        self,
        question: str,
        top_n_coarse: int = 100,
        top_k_rerank: int = 10,
        gold_passage_id: Optional[int] = None,
        log_evidence: bool = True,
        verbose: bool = True,
        force_include_ids: Optional[List[int]] = None  # New arg for explicit forcing
    ) -> RAGResponse:
        """
        Execute full RAG pipeline: Policy Check → Retrieve → Threshold → Generate
        
        Args:
            question: User's question
            top_n_coarse: DPR candidates
            top_k_rerank: Final contexts after reranking
            gold_passage_id: For evaluation - compute hit@k
            log_evidence: Save evidence to log file
            verbose: Print progress
            force_include_ids: List of doc_ids to force into retrieval pool
            
        Returns:
            RAGResponse with answer and full evidence chain
        """
        timestamp = datetime.now().isoformat()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"📩 Query: {question}")
            print(f"{'='*60}\n")
            
        # ------------------------------------------------------------------
        # GATE 1: PII POLICY CHECK (Pre-computation)
        # ------------------------------------------------------------------
        if self._check_pii_policy(question):
            if verbose:
                print("🛑 POLICY STOP: PII detected in query. Refusing.")
            
            return RAGResponse(
                query=question,
                answer="I Cannot Answer This Question Due to Safety and Privacy Policies.",
                citations=[],
                confidence=1.0,  # High confidence in refusal
                is_grounded=True, # Policy-grounded
                is_no_answer=False, # It IS an answer (a refusal)
                retrieval_top_k=0,
                retrieval_total_contexts=0,
                retrieval_total_tokens=0,
                hit_at_k=None,
                timestamp=timestamp,
                retrieval_evidence={"blocked": "pii_policy"},
                generation_evidence={"reasoning": "PII Policy Refusal"}
            )
        
        # Step 1: Retrieve
        if verbose:
            print("🔍 Step 1: Retrieving relevant passages...")
        
        retrieval_response = self.retriever.retrieve(
            query=question,
            top_n_coarse=top_n_coarse,
            top_k_rerank=top_k_rerank,
            gold_passage_id=gold_passage_id,
            apply_recency_boost=False, # Deprecated in favor of explicit ID
            force_include_ids=force_include_ids
        )
        
        if verbose:
            print(f"   Found {len(retrieval_response.selected_contexts)} contexts ({retrieval_response.total_context_tokens} tokens)")
            if retrieval_response.reranked_results:
                print(f"   Top score: {retrieval_response.reranked_results[0].rerank_score:.2f}")
            if retrieval_response.hit_at_k:
                print(f"   Gold passage hit@{retrieval_response.hit_at_k}")
        
        # ------------------------------------------------------------------
        # GATE 2: RETRIEVAL THRESHOLD (No-Answer Gate)
        # ------------------------------------------------------------------
        best_score = retrieval_response.reranked_results[0].rerank_score if retrieval_response.reranked_results else -999.0
        
        # Strict logic: If best context is weak, DO NOT GENERATE. Return "I don't know".
        if best_score < self.retrieval_threshold:
            if verbose:
                print(f"🛑 LOW RELEVANCE STOP: Score {best_score:.2f} < Threshold {self.retrieval_threshold}")
                
            return RAGResponse(
                query=question,
                answer="I don't know - insufficient information in the sources.",
                citations=[],
                confidence=0.0,
                is_grounded=False,
                is_no_answer=True,
                retrieval_top_k=top_k_rerank,
                retrieval_total_contexts=len(retrieval_response.selected_contexts),
                retrieval_total_tokens=retrieval_response.total_context_tokens,
                hit_at_k=retrieval_response.hit_at_k,
                timestamp=timestamp,
                retrieval_evidence={"blocked": "low_retrieval_score", "top_score": best_score},
                generation_evidence={"reasoning": "Low Retrieval Score"}
            )

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
        
        # Helper to serialize numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(response), cls=NumpyEncoder, ensure_ascii=False) + "\n")
    
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
