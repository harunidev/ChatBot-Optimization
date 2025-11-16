# ChatBot Optimization Projesi

Bu proje, DPR (Dense Passage Retrieval) modeli kullanarak soru-cevap sistemi için embedding oluşturma, indeksleme ve reranking işlemlerini gerçekleştirir.

## Gereksinimler

- Python 3.8+
- macOS (MacBook için optimize edilmiştir)

## Kurulum

1. Python sanal ortamı oluşturun (önerilir):
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

### Adım 1: Veri Hazırlama
```bash
python scripts/arch1_prepare_data.py
```
Bu script, eğitim ve değerlendirme verilerini oluşturur.

### Adım 2: Embedding Oluşturma
```bash
python scripts/arch1_embeddings.py
```
Bu script, passage'lar için embedding'leri oluşturur ve `indexes/passage_emb.npy` dosyasına kaydeder.

### Adım 3: Index Oluşturma
İki seçenek var:

**FAISS Index (Hızlı, önerilen):**
```bash
python scripts/arch1_faiss.py
```

**Scikit-learn Index (Alternatif):**
```bash
python scripts/arch1_index_sklearn.py
```

### Adım 4: Değerlendirme
```bash
python scripts/arch1_eval.py
```
Bu script, model performansını değerlendirir ve aşağıdaki dosyaları üretir:
- `outputs/metrics_report.txt`
- `outputs/metrics_report.csv`
- `outputs/metrics_report.json`
- `outputs/emissions.csv` (CodeCarbon enerji/log çıktısı)

Raporlanan metrikler: EM, F1, Precision@5, Recall@5, Recall@20, Recall@100, MRR, ROUGE-L, BLEU, latency (ms), GPU+CPU bellek kullanımı (MB), enerji (kWh) ve karbon emisyonu (kg CO₂e).

## Tüm Adımları Tek Seferde Çalıştırma

```bash
python run_all.py
```

## MacBook Optimizasyonları

- **MPS (Metal Performance Shaders) Desteği**: Apple Silicon MacBook'larda GPU hızlandırması kullanılır
- **FAISS CPU Versiyonu**: MacBook uyumlu `faiss-cpu` paketi kullanılır
- **Otomatik Device Seçimi**: Scriptler otomatik olarak en uygun cihazı (MPS > CUDA > CPU) seçer

## Çıktılar

- `indexes/passage_emb.npy`: Passage embedding'leri
- `indexes/passages.txt`: Passage metinleri
- `indexes/nq_hnsw.index`: FAISS index dosyası (veya `indexes/sk_index.joblib` scikit-learn için)
- `outputs/metrics_report.*`: Performans metrikleri (txt/csv/json)
- `outputs/emissions.csv`: CodeCarbon tarafından kaydedilen enerji ve emisyon değerleri

## Notlar

- İlk çalıştırmada model dosyaları indirilecektir (internet bağlantısı gerekir)
- Model dosyaları `~/.cache/huggingface/` dizininde saklanır
- Apple Silicon MacBook'larda MPS kullanımı performansı önemli ölçüde artırır

