"""
arch1_generate.py - Answer Generation Module
Generates answers from retrieved passages using an instruction-tuned LLM (FLAN-T5).
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict
import argparse

class AnswerGenerator:
    """
    Answer generator using FLAN-T5 or similar seq2seq LLM.
    """
    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "cuda"):
        """
        Initialize the answer generator.
        
        Args:
            model_name: HuggingFace model name (default: flan-t5-base for balance of quality/speed)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading answer generation model: {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
    
    def generate_answer(self, question: str, context: str, max_length: int = 128) -> str:
        """
        Generate an answer given a question and context.
        
        Args:
            question: User's question
            context: Retrieved passage/context
            max_length: Maximum answer length in tokens
            
        Returns:
            Generated answer string
        """
        # Format prompt for FLAN-T5
        # FLAN-T5 works well with simple instruction-style prompts
        prompt = f"Answer the following question based on the context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,  # Beam search for better quality
                early_stopping=True,
                no_repeat_ngram_size=3,  # Avoid repetition
                temperature=0.7,
                do_sample=False  # Deterministic for eval consistency
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
    
    def generate_batch(self, questions: List[str], contexts: List[str], 
                       max_length: int = 128, batch_size: int = 8) -> List[str]:
        """
        Generate answers for a batch of questions.
        
        Args:
            questions: List of questions
            contexts: List of contexts (must match length of questions)
            max_length: Maximum answer length
            batch_size: Processing batch size
            
        Returns:
            List of generated answers
        """
        assert len(questions) == len(contexts), "Questions and contexts must have same length"
        
        answers = []
        
        for i in range(0, len(questions), batch_size):
            batch_q = questions[i:i+batch_size]
            batch_c = contexts[i:i+batch_size]
            
            # Format prompts
            prompts = [
                f"Answer the following question based on the context.\n\nContext: {ctx}\n\nQuestion: {q}\n\nAnswer:"
                for q, ctx in zip(batch_q, batch_c)
            ]
            
            # Tokenize batch
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    temperature=0.7,
                    do_sample=False
                )
            
            # Decode
            batch_answers = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            answers.extend([ans.strip() for ans in batch_answers])
        
        return answers


def main():
    """Demo: Generate answers for sample questions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str, default="Paris is the capital of France. It is known for the Eiffel Tower.")
    parser.add_argument("--question", type=str, default="What is Paris known for?")
    parser.add_argument("--model", type=str, default="google/flan-t5-base",
                       help="Model name: flan-t5-small/base/large/xl")
    args = parser.parse_args()
    
    # Initialize generator
    generator = AnswerGenerator(model_name=args.model)
    
    # Generate answer
    print(f"\n{'='*60}")
    print("ANSWER GENERATION DEMO")
    print(f"{'='*60}\n")
    print(f"Context: {args.context}\n")
    print(f"Question: {args.question}\n")
    
    answer = generator.generate_answer(args.question, args.context)
    
    print(f"Generated Answer: {answer}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
