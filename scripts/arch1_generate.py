"""
arch1_generate.py - Faithfulness-Oriented Answer Generation
Generates answers grounded in retrieved context with "I don't know" fallback.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re
import numpy as np


@dataclass
class GenerationResult:
    """Result of answer generation with evidence."""
    query: str
    answer: str
    citations: List[int]  # doc_ids used
    confidence: float
    is_grounded: bool  # True if answer is grounded in context
    is_no_answer: bool  # True if "bilmiyorum" was triggered
    context_used: str
    reasoning: str  # Why this decision was made


class RAGGenerator:
    """
    Faithfulness-oriented answer generator.
    - Generates answers grounded in retrieved context
    - Returns "I don't know" if context is insufficient
    - Provides citations to source documents
    - Optimized for high faithfulness (≥94%) and low hallucination (≤1%)
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",  # OPTIMIZED: base for speed (latency)
        device: str = "cuda",
        no_answer_threshold: float = 0.25,  # BALANCED: not too strict
        min_context_overlap: float = 0.15   # BALANCED: 15% minimum overlap
    ):
        """
        Args:
            model_name: HuggingFace model for generation (default: flan-t5-large)
            device: cuda or cpu
            no_answer_threshold: If confidence below this, return "bilmiyorum" (0.25 = balanced)
            min_context_overlap: Minimum overlap for grounding (0.15 = 15%)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.no_answer_threshold = no_answer_threshold
        self.min_context_overlap = min_context_overlap
        
        print(f"🤖 Initializing RAG Generator on {self.device}...")
        print(f"   Model: {model_name}")
        print(f"   No-answer threshold: {no_answer_threshold}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print("✅ Generator ready!\n")
    
    def _build_grounded_prompt(self, query: str, contexts: List[str], doc_ids: List[int]) -> str:
        """Build a prompt that enforces grounding in sources."""
        # Format contexts with citation markers
        context_parts = []
        for i, (ctx, doc_id) in enumerate(zip(contexts, doc_ids)):
            context_parts.append(f"[Source {i+1}, doc_{doc_id}]: {ctx}")
        
        context_str = "\n\n".join(context_parts)
        
        prompt = f"""You are an accurate assistant. Answer using ONLY the sources below.
        
INSTRUCTIONS:
1. If the answer is in the sources, copy it EXACTLY word-for-word.
2. If the sources do NOT contain the answer, you MUST say "Bilmiyorum - kaynaklarda yeterli bilgi yok".
3. Do NOT use outside knowledge.
4. Do NOT make up answers.

CRITICAL SAFETY RULES:
- NEVER reveal personal identification numbers (TC, SSN, ID numbers, passport numbers)
- NEVER share credit card, banking, or financial account information
- NEVER disclose phone numbers, addresses, or private contact information
- If asked for such information, respond: "I cannot provide personal identification information"

Sources:
{context_str}

Question: {query}

Answer (from sources only):"""
        
        return prompt
    
    def _compute_grounding_score(self, answer: str, contexts: List[str]) -> Tuple[float, bool]:
        """
        Check if answer is grounded in context.
        Returns (confidence_score, is_grounded)
        """
        if not answer or not contexts:
            return 0.0, False
        
        answer_lower = answer.lower()
        
        # Check for explicit no-answer
        no_answer_phrases = [
            "bilmiyorum", "i don't know", "yeterli bilgi yok",
            "kaynaklarda yok", "bulamadım", "cevaplayamam"
        ]
        for phrase in no_answer_phrases:
            if phrase in answer_lower:
                return 1.0, True  # Confident "no answer" is correct behavior
        
        # Compute word overlap with contexts
        answer_words = set(re.findall(r'\w+', answer_lower))
        if not answer_words:
            return 0.0, False
        
        all_context_words = set()
        for ctx in contexts:
            all_context_words.update(re.findall(r'\w+', ctx.lower()))
        
        # Calculate overlap
        overlap = answer_words & all_context_words
        overlap_ratio = len(overlap) / len(answer_words) if answer_words else 0
        
        # Also check for direct substring matches (phrases)
        phrase_grounded = False
        for ctx in contexts:
            # Check if significant phrases from answer appear in context
            answer_phrases = re.findall(r'\b\w+\s+\w+\s+\w+\b', answer_lower)
            for phrase in answer_phrases:
                if phrase in ctx.lower():
                    phrase_grounded = True
                    break
        
        # Combine signals
        is_grounded = overlap_ratio >= self.min_context_overlap or phrase_grounded
        confidence = min(1.0, overlap_ratio + (0.3 if phrase_grounded else 0))
        
        return confidence, is_grounded
    
    def _extract_citations(self, answer: str, num_sources: int) -> List[int]:
        """Extract cited source numbers from answer."""
        citations = []
        # Look for patterns like "Source 1", "[1]", "(doc_123)"
        patterns = [
            r'\[?[Ss]ource\s*(\d+)\]?',
            r'\[(\d+)\]',
            r'\(doc_(\d+)\)',
            r'kaynak\s*(\d+)'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, answer)
            for match in matches:
                try:
                    idx = int(match)
                    if 1 <= idx <= num_sources:
                        citations.append(idx - 1)  # Convert to 0-indexed
                except ValueError:
                    pass
        return list(set(citations)) if citations else [0]  # Default to first source
    
    def generate(
        self,
        query: str,
        contexts: List[str],
        doc_ids: List[int],
        max_length: int = 150,
        require_grounding: bool = True
    ) -> GenerationResult:
        """
        Generate an answer grounded in the provided contexts.
        
        Args:
            query: User question
            contexts: List of retrieved context texts
            doc_ids: Document IDs for each context (for citation)
            max_length: Maximum answer length
            require_grounding: If True, return "bilmiyorum" for ungrounded answers
            
        Returns:
            GenerationResult with answer, citations, and confidence
        """
        # Handle empty context
        if not contexts:
            return GenerationResult(
                query=query,
                answer="Bilmiyorum - hiç context bulunamadı.",
                citations=[],
                confidence=0.0,
                is_grounded=False,
                is_no_answer=True,
                context_used="",
                reasoning="No contexts provided to the generator."
            )
        
        # Build grounded prompt
        prompt = self._build_grounded_prompt(query, contexts, doc_ids)
        context_str = " ".join(contexts)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=1,  # Greedy for speed

                no_repeat_ngram_size=2,
                do_sample=False,  # Deterministic extraction
                repetition_penalty=1.2  # Avoid repetition
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Check grounding
        confidence, is_grounded = self._compute_grounding_score(answer, contexts)
        
        # Extract citations
        citations = self._extract_citations(answer, len(contexts))
        cited_doc_ids = [doc_ids[i] for i in citations if i < len(doc_ids)]
        
        # Determine if we should return "I don't know"
        is_no_answer = False
        reasoning = ""
        
        # Check for explicit no-answer in generated text
        if any(phrase in answer.lower() for phrase in ["bilmiyorum", "i don't know", "yeterli bilgi yok"]):
            is_no_answer = True
            is_grounded = True  # This is actually correct behavior
            reasoning = "Model explicitly stated it doesn't know based on sources."
        
        # If grounding failed and we require it, override with no-answer
        elif require_grounding and not is_grounded and confidence < self.no_answer_threshold:
            original_answer = answer
            answer = "Bilmiyorum - kaynaklarda bu soruya yeterli cevap bulunamadı."
            is_no_answer = True
            reasoning = f"Grounding failed (confidence={confidence:.2f} < {self.no_answer_threshold}). Original: '{original_answer[:100]}...'"
        else:
            reasoning = f"Answer grounded with confidence={confidence:.2f}"
        
        return GenerationResult(
            query=query,
            answer=answer,
            citations=cited_doc_ids,
            confidence=confidence,
            is_grounded=is_grounded,
            is_no_answer=is_no_answer,
            context_used=context_str[:500] + "..." if len(context_str) > 500 else context_str,
            reasoning=reasoning
        )
    
    def generate_batch(
        self,
        queries: List[str],
        contexts_list: List[List[str]],
        doc_ids_list: List[List[int]],
        batch_size: int = 4,
        **kwargs
    ) -> List[GenerationResult]:
        """Generate answers for multiple queries."""
        results = []
        for query, contexts, doc_ids in zip(queries, contexts_list, doc_ids_list):
            result = self.generate(query, contexts, doc_ids, **kwargs)
            results.append(result)
        return results
    
    def log_generation_evidence(self, result: GenerationResult) -> str:
        """Format generation evidence for logging/display."""
        lines = [
            f"📝 GENERATION EVIDENCE",
            f"{'='*50}",
            f"Query: {result.query}",
            f"",
            f"🤖 Answer: {result.answer}",
            f"",
            f"📊 Metrics:",
            f"   Confidence: {result.confidence:.2f}",
            f"   Is Grounded: {'✅' if result.is_grounded else '❌'}",
            f"   No-Answer: {'Yes' if result.is_no_answer else 'No'}",
            f"   Citations: {result.citations}",
            f"",
            f"💭 Reasoning: {result.reasoning}",
            f"",
            f"📄 Context used: {result.context_used[:200]}..."
        ]
        return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    generator = RAGGenerator()
    
    # Test with good context
    result = generator.generate(
        query="What is Paris known for?",
        contexts=["Paris is the capital of France. It is famous for the Eiffel Tower."],
        doc_ids=[42]
    )
    print(generator.log_generation_evidence(result))
    
    print("\n" + "="*60 + "\n")
    
    # Test with irrelevant context
    result2 = generator.generate(
        query="What is the capital of Japan?",
        contexts=["Paris is the capital of France. It is famous for the Eiffel Tower."],
        doc_ids=[42]
    )
    print(generator.log_generation_evidence(result2))
