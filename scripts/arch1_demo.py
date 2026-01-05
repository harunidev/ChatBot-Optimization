
import argparse
import sys
import os
import time
import torch
import numpy as np
import psutil
from typing import List, Optional

# Add the script directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import arch1_embeddings
import arch1_faiss
import arch1_rerank
import arch1_generate  # NEW: Answer generation module
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

# --- ANSI Renk Kodları ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}\n{text}\n{'='*60}{Colors.ENDC}")

def print_step(text):
    print(f"\n{Colors.CYAN}➤ {text}{Colors.ENDC}")

def print_result(label, value, color=Colors.GREEN):
    print(f"{Colors.BOLD}{label}:{Colors.ENDC} {color}{value}{Colors.ENDC}")

# --- Metrik Hesaplamaları (arch1_eval.py'den basitleştirilmiş) ---
def compute_metrics(prediction: str, ground_truth: Optional[str]):
    if not ground_truth:
        return None
    
    # Normalize
    pred_clean = prediction.lower().strip()
    gt_clean = ground_truth.lower().strip()
    
    # EM check
    em = pred_clean == gt_clean
    
    # F1
    pred_tokens = pred_clean.split()
    gt_tokens = gt_clean.split()
    common = set(pred_tokens) & set(gt_tokens)
    if not pred_tokens or not gt_tokens:
        f1 = 0.0
    else:
        prec = len(common) / len(pred_tokens)
        rec = len(common) / len(gt_tokens)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = scorer.score(gt_clean, pred_clean)['rougeL'].fmeasure

    # BLEU
    bleu = BLEU()
    try:
        bleu_score = bleu.sentence_score(pred_clean, [gt_clean]).score
    except:
        bleu_score = 0.0

    return {
        "EM": em,
        "F1": f1,
        "ROUGE-L": rouge_score,
        "BLEU": bleu_score
    }

class InteractiveDemo:
    def __init__(self, passages_path, index_path, device="cuda"):
        print_header("SİSTEM BAŞLATILIYOR...")
        self.device = device if torch.cuda.is_available() else "cpu"
        print_result("Cihaz", self.device, Colors.BLUE)

        # 1. Pasajları Yükle
        print_step("Pasajlar yükleniyor...")
        self.passages = arch1_embeddings.load_passages(passages_path)
        print_result("Toplam Pasaj", len(self.passages))

        # 2. Question Encoder Yükle
        print_step("Soru Encoder'ı yükleniyor...")
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.q_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(self.device)
        self.q_model.eval()

        # 3. FAISS Index Yükle
        print_step("FAISS Index yükleniyor...")
        self.index = faiss.read_index(index_path)
        self.index.hnsw.efSearch = 128 # Arama kalitesi
        print_result("Index Boyutu", self.index.ntotal)

        # 4. Reranker Yükle
        print_step("Reranker modeli yükleniyor...")
        self.reranker = arch1_rerank.Reranker(device=self.device)
        
        # 5. Answer Generator Yükle (NEW)
        print_step("Answer Generator (LLM) yükleniyor...")
        self.answer_generator = arch1_generate.AnswerGenerator(
            model_name="google/flan-t5-base",  # balanced size/quality
            device=self.device
        )
        
        print_header("SİSTEM HAZIR! SORGULARI GİREBİLİRSİNİZ.")

    def run_query(self, query: str, ground_truth: Optional[str] = None, k: int = 20):
        print(f"\n{Colors.BOLD}Sorgu:{Colors.ENDC} {query}")
        if ground_truth:
            print(f"{Colors.BOLD}Beklenen Cevap:{Colors.ENDC} {ground_truth}")

        start_time = time.time()

        # 1. Encode Query
        # print_step("Soru encode ediliyor...")
        with torch.no_grad():
            inputs = self.q_tokenizer(query, return_tensors="pt").to(self.device)
            q_emb = self.q_model(**inputs).pooler_output.cpu().numpy().astype('float32')

        # 2. Search Index
        # print_step(f"Index aranıyor (Top-{k})...")
        D, I = self.index.search(q_emb, k)
        retrieved_indices = I[0]
        
        # 3. Prepare Candidates
        candidates = []
        valid_indices = []
        for idx in retrieved_indices:
            if idx < len(self.passages):
                candidates.append(self.passages[idx])
                valid_indices.append(idx)
        
        if not candidates:
            print(f"{Colors.FAIL}Hata: Hiçbir pasaj bulunamadı!{Colors.ENDC}")
            return

        # 4. Rerank
        # print_step("Sonuçlar rerank ediliyor...")
        reranked_results = self.reranker.rerank(query, candidates)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # --- SONUÇLARI GÖSTER ---
        print_header(f"SONUÇLAR (Süre: {latency_ms:.1f} ms)")

        top_result = reranked_results[0] # (text, score)
        top_context = top_result[0]  # Retrieved passage
        top_score = top_result[1]

        # **NEW: Generate answer from context using LLM**
        print_step("LLM ile cevap üretiliyor...")
        generated_answer = self.answer_generator.generate_answer(query, top_context)

        print(f"{Colors.GREEN}{Colors.BOLD}🤖 ÜRETİLEN CEVAP:{Colors.ENDC}")
        print(f"{Colors.BOLD}{generated_answer}{Colors.ENDC}")
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}📄 Kaynak Pasaj (Skor: {top_score:.4f}):{Colors.ENDC}")
        print(f"{Colors.BLUE}{top_context[:200]}...{Colors.ENDC}")
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}--- Diğer Adaylar (Top-3) ---{Colors.ENDC}")
        for i in range(1, min(3, len(reranked_results))):
            res = reranked_results[i]
            print(f"{i+1}. [{res[1]:.4f}] {res[0][:100]}...")

        # --- METRİKLER ---
        if ground_truth:
            print_header("PERFORMANS METRİKLERİ")
            # Compare generated answer with ground truth
            metrics = compute_metrics(generated_answer, ground_truth)
            
            # Hallucination Check (Basit heuristic)
            is_match = metrics["F1"] > 0.1 # Eşik değeri
            
            print_result("Kesin Eşleşme (EM)", "EVET" if metrics["EM"] else "HAYIR", Colors.GREEN if metrics["EM"] else Colors.WARNING)
            print_result("F1 Skoru", f"{metrics['F1']:.4f}")
            print_result("ROUGE-L", f"{metrics['ROUGE-L']:.4f}")
            print_result("BLEU", f"{metrics['BLEU']:.4f}")
            
            # Pass/Fail
            if is_match:
                print(f"\n{Colors.GREEN}{Colors.BOLD}✅ BAŞARILI: Sistem bilgiyi buldu!{Colors.ENDC}")
            else:
                 print(f"\n{Colors.FAIL}{Colors.BOLD}❌ BAŞARISIZ: Bulunan pasaj beklenen cevapla eşleşmiyor.{Colors.ENDC}")
        else:
            print(f"\n{Colors.WARNING}Not: Beklenen cevap girilmediği için doğruluk metrikleri hesaplanmadı.{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description="Canlı RAG Demosu")
    parser.add_argument("--passages-txt", type=str, default="indexes/passages.txt", help="Pasaj dosyası yolu")
    parser.add_argument("--index-path", type=str, default="rag_arch1_colab/indexes/nq_hnsw.index", help="Index dosyası yolu")
    parser.add_argument("--device", type=str, default="cuda", help="Cihaz (cuda/cpu)")
    
    args = parser.parse_args()

    # Dosya kontrolü
    if not os.path.exists(args.passages_txt):
        print(f"{Colors.FAIL}Hata: Pasaj dosyası bulunamadı: {args.passages_txt}{Colors.ENDC}")
        print("Lütfen önce veri hazırlama adımını çalıştırın.")
        return
    
    if not os.path.exists(args.index_path):
        print(f"{Colors.FAIL}Hata: Index dosyası bulunamadı: {args.index_path}{Colors.ENDC}")
        print("Lütfen önce embedding ve indexleme adımlarını çalıştırın.")
        return

    demo = InteractiveDemo(args.passages_txt, args.index_path, args.device)

    while True:
        try:
            print(f"\n{Colors.HEADER}{'-'*40}{Colors.ENDC}")
            query = input(f"{Colors.BOLD}Soru girin (Çıkış için 'q'): {Colors.ENDC}")
            
            if query.lower() in ['q', 'exit', 'quit']:
                print("Çıkış yapılıyor...")
                break
                
            if not query.strip():
                continue

            ground_truth = input(f"{Colors.BOLD}Beklenen cevap (Opsiyonel, Boş geçmek için Enter): {Colors.ENDC}")
            
            demo.run_query(query, ground_truth if ground_truth.strip() else None)
            
        except KeyboardInterrupt:
            print("\nÇıkış yapılıyor...")
            break
        except Exception as e:
            print(f"\n{Colors.FAIL}Bir hata oluştu: {e}{Colors.ENDC}")

if __name__ == "__main__":
    main()
