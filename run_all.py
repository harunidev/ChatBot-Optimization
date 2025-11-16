#!/usr/bin/env python3
"""
Tüm pipeline'ı sırayla çalıştıran ana script.
MacBook için optimize edilmiştir.
"""

import os
import sys
import subprocess

def run_script(script_name, description):
    """Bir script'i çalıştırır ve hata kontrolü yapar."""
    print(f"\n{'='*60}")
    print(f"Adım: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}\n")
    
    script_path = os.path.join("scripts", script_name)
    if not os.path.exists(script_path):
        print(f"HATA: {script_path} bulunamadı!")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        print(f"\n✓ {description} tamamlandı.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ HATA: {description} başarısız oldu!")
        print(f"Çıkış kodu: {e.returncode}\n")
        return False

def main():
    """Ana pipeline fonksiyonu."""
    print("\n" + "="*60)
    print("ChatBot Optimization - Tüm Pipeline")
    print("MacBook için optimize edilmiştir")
    print("="*60)
    
    # Gerekli klasörleri oluştur
    os.makedirs("indexes", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Pipeline adımları
    steps = [
        ("arch1_prepare_data.py", "Veri Hazırlama"),
        ("arch1_embeddings.py", "Embedding Oluşturma"),
        ("arch1_faiss.py", "FAISS Index Oluşturma"),
        ("arch1_eval.py", "Değerlendirme")
    ]
    
    # Her adımı sırayla çalıştır
    for script_name, description in steps:
        if not run_script(script_name, description):
            print(f"\n❌ Pipeline {description} adımında durdu!")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ TÜM ADIMLAR BAŞARIYLA TAMAMLANDI!")
    print("="*60)
    print("\nSonuçlar:")
    print("  - Indexes: indexes/ klasöründe")
    print("  - Metrikler: outputs/metrics_report.txt")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

