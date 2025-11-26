import argparse
import os
import numpy as np
import torch
from contextlib import nullcontext
from tqdm import tqdm
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)

CONFIG = {
    "CTX_MODEL": "facebook/dpr-ctx_encoder-single-nq-base",
    "Q_MODEL": "facebook/dpr-question_encoder-single-nq-base",
    "MAX_LENGTH": 256,
    "DEFAULT_BATCH_SIZE": 128,  # Optimized batch size for better performance on A100
}

# Enable TF32 for faster matmuls on A100 (safe for inference)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def load_passages(path: str) -> list[str]:
    """Loads passages from a plain text file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def encode_passages(passages: list[str], batch_size: int, device: str, output_path: str, precision: str = "auto"):
    """Encodes passages using DPR Context Encoder and saves to .npy."""
    print(f"Loading Context Encoder: {CONFIG['CTX_MODEL']}...")
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(CONFIG['CTX_MODEL'])
    model = DPRContextEncoder.from_pretrained(CONFIG['CTX_MODEL']).to(device)
    model.eval()

    use_fp16 = False
    if device.startswith("cuda"):
        if precision == "fp16":
            use_fp16 = True
        elif precision == "auto":
            use_fp16 = True  # default to fp16 on GPU for speed
    if precision == "fp32":
        use_fp16 = False

    if use_fp16:
        model = model.half()
        print("FP16 encoding enabled for faster throughput.")

    autocast_ctx = (
        torch.cuda.amp.autocast if (device.startswith("cuda") and use_fp16) else nullcontext
    )

    total = len(passages)
    hidden_size = model.config.hidden_size
    embeddings_buffer = np.empty((total, hidden_size), dtype=np.float32)

    print(f"Encoding {total} passages on {device} (batch size={batch_size})...")
    with torch.no_grad():
        for start in tqdm(range(0, total, batch_size), desc="Encoding passages", miniters=5):
            end = min(start + batch_size, total)
            batch = passages[start:end]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CONFIG["MAX_LENGTH"],
            ).to(device)

            with autocast_ctx():
                embeddings = model(**inputs).pooler_output

            embeddings_buffer[start:end] = embeddings.detach().float().cpu().numpy()

    print(f"Saving embeddings shape {embeddings_buffer.shape} to {output_path}...")
    np.save(output_path, embeddings_buffer)

def encode_query(query: str, device: str = "cuda") -> np.ndarray:
    """Helper function to encode a single query (for use in other scripts)."""
    # Note: In a real production setup, you'd load the model once globally or in a class.
    # This is a standalone helper for demonstration or simple imports.
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(CONFIG['Q_MODEL'])
    model = DPRQuestionEncoder.from_pretrained(CONFIG['Q_MODEL']).to(device)
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(query, return_tensors="pt").to(device)
        embedding = model(**inputs).pooler_output
    return embedding.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Generate DPR Embeddings")
    parser.add_argument("--passages-txt", type=str, required=True, help="Path to passages.txt")
    parser.add_argument("--output-embeddings", type=str, required=True, help="Path to output .npy file")
    parser.add_argument("--batch-size", type=int, default=CONFIG["DEFAULT_BATCH_SIZE"], help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["auto", "fp16", "fp32"],
        default="auto",
        help="Precision mode: auto (default), fp16, or fp32.",
    )
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_embeddings), exist_ok=True)
    
    passages = load_passages(args.passages_txt)
    encode_passages(passages, args.batch_size, args.device, args.output_embeddings, precision=args.precision)

if __name__ == "__main__":
    main()
