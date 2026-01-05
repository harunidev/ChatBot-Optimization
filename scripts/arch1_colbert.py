"""
arch1_colbert.py - ColBERT Token-Level Reranking
Implements late interaction (MaxSim) for token-level similarity scoring.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class ColBERTReranker:
    """
    ColBERT-style reranker using late interaction (MaxSim).
    
    ColBERT computes token-level embeddings and uses MaxSim:
    - Each query token finds its max similarity with any document token
    - Sum of max similarities = document score
    
    This gives more fine-grained matching than sequence-level cross-encoders.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",  # Can use colbert-ir/colbertv2.0 if available
        device: str = "cuda",
        dim: int = 128,  # ColBERT typically projects to 128-dim
        batch_size: int = 32
    ):
        """
        Initialize ColBERT reranker.
        
        Args:
            model_name: Base transformer model
            device: cuda or cpu
            dim: Projection dimension for token embeddings
            batch_size: Batch size for encoding
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dim = dim
        self.batch_size = batch_size
        
        print(f"🔷 Initializing ColBERT Reranker on {self.device}...")
        print(f"   Model: {model_name}")
        print(f"   Projection dim: {dim}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Linear projection layer (ColBERT projects to lower dim)
        hidden_size = self.model.config.hidden_size
        self.projection = torch.nn.Linear(hidden_size, dim).to(self.device)
        
        # Initialize projection with small weights
        torch.nn.init.xavier_uniform_(self.projection.weight)
        
        print("✅ ColBERT Reranker ready!")
    
    def _encode_tokens(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        """
        Encode texts to token-level embeddings.
        
        Args:
            texts: List of texts to encode
            is_query: If True, add [Q] marker (ColBERT convention)
            
        Returns:
            Token embeddings (batch, seq_len, dim)
        """
        # Add query/document markers (ColBERT convention)
        if is_query:
            texts = ["[Q] " + t for t in texts]
        else:
            texts = ["[D] " + t for t in texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get token embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state  # (batch, seq, hidden)
        
        # Project to lower dimension
        projected = self.projection(token_embeddings)  # (batch, seq, dim)
        
        # L2 normalize
        projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
        
        # Get attention mask for valid tokens
        mask = inputs.attention_mask  # (batch, seq)
        
        return projected, mask
    
    def _maxsim(
        self,
        query_emb: torch.Tensor,
        query_mask: torch.Tensor,
        doc_emb: torch.Tensor,
        doc_mask: torch.Tensor
    ) -> float:
        """
        Compute MaxSim score (ColBERT late interaction).
        
        For each query token, find max similarity with any document token.
        Sum these max similarities.
        
        Args:
            query_emb: Query token embeddings (1, q_len, dim)
            query_mask: Query attention mask (1, q_len)
            doc_emb: Document token embeddings (1, d_len, dim)
            doc_mask: Document attention mask (1, d_len)
            
        Returns:
            MaxSim score
        """
        # Compute similarity matrix (q_len, d_len)
        similarity = torch.matmul(query_emb.squeeze(0), doc_emb.squeeze(0).T)
        
        # Mask out padding tokens in document
        doc_mask_expanded = doc_mask.squeeze(0).unsqueeze(0).expand_as(similarity)
        similarity = similarity.masked_fill(~doc_mask_expanded.bool(), float('-inf'))
        
        # MaxSim: max similarity for each query token
        max_sim_per_query_token = similarity.max(dim=1).values  # (q_len,)
        
        # Mask out padding tokens in query and sum
        query_mask_flat = query_mask.squeeze(0).bool()
        score = max_sim_per_query_token[query_mask_flat].sum().item()
        
        return score
    
    def compute_scores(
        self,
        query: str,
        documents: List[str],
        show_progress: bool = True
    ) -> List[float]:
        """
        Compute ColBERT scores for query-document pairs.
        
        Args:
            query: Query text
            documents: List of document texts
            show_progress: Show progress bar
            
        Returns:
            List of MaxSim scores
        """
        # Encode query once
        query_emb, query_mask = self._encode_tokens([query], is_query=True)
        
        scores = []
        iterator = range(0, len(documents), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="ColBERT Scoring")
        
        for i in iterator:
            batch_docs = documents[i:i + self.batch_size]
            doc_embs, doc_masks = self._encode_tokens(batch_docs, is_query=False)
            
            # Compute MaxSim for each document
            for j in range(len(batch_docs)):
                doc_emb = doc_embs[j:j+1]
                doc_mask = doc_masks[j:j+1]
                score = self._maxsim(query_emb, query_mask, doc_emb, doc_mask)
                scores.append(score)
        
        return scores
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents using ColBERT MaxSim.
        
        Args:
            query: Query text
            documents: List of documents
            top_k: Return only top-k results
            
        Returns:
            List of (document, score) tuples sorted by score
        """
        scores = self.compute_scores(query, documents, show_progress=False)
        
        results = list(zip(documents, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results


class HybridReranker:
    """
    Hybrid reranker combining Cross-Encoder and ColBERT.
    Uses weighted combination of both scores.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        cross_encoder_weight: float = 0.5,
        colbert_weight: float = 0.5
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.ce_weight = cross_encoder_weight
        self.cb_weight = colbert_weight
        
        print("🔶 Initializing Hybrid Reranker...")
        
        # Import cross-encoder from existing module
        import arch1_rerank
        self.cross_encoder = arch1_rerank.Reranker(device=self.device)
        
        # Initialize ColBERT
        self.colbert = ColBERTReranker(device=self.device)
        
        print("✅ Hybrid Reranker ready!")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, dict]]:
        """
        Rerank using both Cross-Encoder and ColBERT.
        
        Returns:
            List of (document, combined_score, {ce_score, cb_score}) tuples
        """
        # Get Cross-Encoder scores
        ce_results = self.cross_encoder.rerank(query, documents)
        ce_scores = {doc: score for doc, score in ce_results}
        
        # Get ColBERT scores
        cb_results = self.colbert.rerank(query, documents)
        cb_scores = {doc: score for doc, score in cb_results}
        
        # Normalize scores to 0-1 range
        ce_vals = list(ce_scores.values())
        cb_vals = list(cb_scores.values())
        
        ce_min, ce_max = min(ce_vals), max(ce_vals)
        cb_min, cb_max = min(cb_vals), max(cb_vals)
        
        def normalize(val, vmin, vmax):
            if vmax == vmin:
                return 0.5
            return (val - vmin) / (vmax - vmin)
        
        # Combine scores
        results = []
        for doc in documents:
            ce_norm = normalize(ce_scores[doc], ce_min, ce_max)
            cb_norm = normalize(cb_scores[doc], cb_min, cb_max)
            combined = self.ce_weight * ce_norm + self.cb_weight * cb_norm
            
            results.append((doc, combined, {
                "cross_encoder": ce_scores[doc],
                "colbert": cb_scores[doc]
            }))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results


# Standalone test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ColBERT Reranker Test")
    print("="*60 + "\n")
    
    reranker = ColBERTReranker()
    
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France and its largest city.",
        "Berlin is the capital of Germany.",
        "France is a country in Western Europe.",
        "The Eiffel Tower is located in Paris, France."
    ]
    
    results = reranker.rerank(query, documents)
    
    print(f"Query: {query}\n")
    print("Results (token-level MaxSim):")
    for i, (doc, score) in enumerate(results):
        print(f"  {i+1}. [{score:.4f}] {doc}")
