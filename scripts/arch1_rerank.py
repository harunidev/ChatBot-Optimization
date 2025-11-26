import argparse
import json
import torch
from typing import List, Tuple
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from contextlib import nullcontext

CONFIG = {
    "RERANK_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2", # Efficient and good performance
    "MAX_LENGTH": 512,
    "BATCH_SIZE": 32
}

class Reranker:
    def __init__(self, device: str = "cuda", batch_size: int = CONFIG["BATCH_SIZE"]):
        self.device = device
        self.batch_size = max(1, batch_size)
        print(f"Loading Reranker: {CONFIG['RERANK_MODEL']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['RERANK_MODEL'])
        self.model = AutoModelForSequenceClassification.from_pretrained(CONFIG['RERANK_MODEL']).to(device)
        self.model.eval()

    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Reranks a list of candidate passages for a given query.
        Returns list of (passage, score) sorted by score descending.
        """
        pairs = [[query, doc] for doc in candidates]
        
        scores = []
        autocast_ctx = torch.cuda.amp.autocast if self.device.startswith("cuda") else nullcontext
        with torch.inference_mode():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=CONFIG["MAX_LENGTH"],
                    return_tensors="pt"
                ).to(self.device)
                
                with autocast_ctx():
                    logits = self.model(**inputs).logits
                batch_scores = logits.view(-1).float().cpu().numpy().tolist()
                scores.extend(batch_scores)
        
        # Combine candidates with scores
        results = list(zip(candidates, scores))
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results

    def compute_scores(self, pairs: List[List[str]], show_progress: bool = True) -> List[float]:
        """
        Computes scores for a list of [query, doc] pairs in batches.
        Efficient for bulk processing.
        """
        scores = []
        iterator = range(0, len(pairs), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Reranking Batches")
        
        autocast_ctx = torch.cuda.amp.autocast if self.device.startswith("cuda") else nullcontext
        with torch.inference_mode():
            for i in iterator:
                batch = pairs[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=CONFIG["MAX_LENGTH"],
                    return_tensors="pt"
                ).to(self.device)
                
                with autocast_ctx():
                    logits = self.model(**inputs).logits
                batch_scores = logits.view(-1).float().cpu().numpy().tolist()
                scores.extend(batch_scores)
        return scores

def main():
    # This script can be used standalone to rerank a specific file of candidates
    # But mostly it will be imported by arch1_eval.py
    parser = argparse.ArgumentParser(description="Rerank Candidates")
    parser.add_argument("--query", type=str, help="Query string (for single query mode)")
    parser.add_argument("--candidates", nargs="+", help="List of candidate strings")
    
    args = parser.parse_args()
    
    if args.query and args.candidates:
        reranker = Reranker(device="cuda" if torch.cuda.is_available() else "cpu")
        results = reranker.rerank(args.query, args.candidates)
        for doc, score in results:
            print(f"{score:.4f}: {doc[:50]}...")

if __name__ == "__main__":
    main()
