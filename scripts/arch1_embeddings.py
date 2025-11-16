from transformers import AutoTokenizer, AutoModel
import torch, json, numpy as np
import os

os.makedirs("indexes", exist_ok=True)

# MacBook için device seçimi: MPS (Apple Silicon) > CUDA > CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

MODEL_P = "facebook/dpr-ctx_encoder-single-nq-base"
tok = AutoTokenizer.from_pretrained(MODEL_P)
enc = AutoModel.from_pretrained(MODEL_P).eval().to(device)

def encode(texts, bs=32):
    out=[]
    for i in range(0, len(texts), bs):
        t = tok(texts[i:i+bs], padding=True, truncation=True,
                return_tensors="pt", max_length=256).to(device)
        with torch.no_grad():
            h = enc(**t).pooler_output
        out.append(h.detach().cpu().numpy())
    return np.vstack(out).astype("float32")

passages=[]
with open("data/passages_train.jsonl","r",encoding="utf-8") as f:
    for line in f:
        passages.append(json.loads(line)["text"])

emb = encode(passages)
np.save("indexes/passage_emb.npy", emb)
open("indexes/passages.txt","w",encoding="utf-8").write(
    "\n".join(p.replace("\n"," ") for p in passages)
)
print("OK: indexes/passage_emb.npy & indexes/passages.txt")
