# Proje Çalıştırma Sırası ve Bağımlılıklar

## Çalıştırma Sırası

Proje aşağıdaki sırayla çalıştırılmalıdır:

### 1. Veri Hazırlama (`arch1_prepare_data.py`)
**Gereksinimler:** Yok (internet bağlantısı gerekir - model indirme için)

**Oluşturduğu Dosyalar:**
- `data/passages_train.jsonl` - Eğitim verisi (10,000 passage)
- `data/gold_eval.jsonl` - Değerlendirme verisi (500 soru-cevap çifti)

### 2. Embedding Oluşturma (`arch1_embeddings.py`)
**Gereksinimler:**
- `data/passages_train.jsonl` (Adım 1'den)

**Oluşturduğu Dosyalar:**
- `indexes/passage_emb.npy` - Passage embedding'leri
- `indexes/passages.txt` - Passage metinleri

### 3. Index Oluşturma
İki seçenek var:

#### 3a. FAISS Index (`arch1_faiss.py`) - **ÖNERİLEN**
**Gereksinimler:**
- `indexes/passage_emb.npy` (Adım 2'den)

**Oluşturduğu Dosyalar:**
- `indexes/nq_hnsw.index` - FAISS HNSW index dosyası

#### 3b. Scikit-learn Index (`arch1_index_sklearn.py`) - Alternatif
**Gereksinimler:**
- `indexes/passage_emb.npy` (Adım 2'den)

**Oluşturduğu Dosyalar:**
- `indexes/sk_index.joblib` - Scikit-learn index dosyası

**NOT:** `arch1_eval.py` sadece FAISS index kullanır, bu yüzden FAISS önerilir.

### 4. Değerlendirme (`arch1_eval.py`)
**Gereksinimler:**
- `data/gold_eval.jsonl` (Adım 1'den)
- `indexes/passages.txt` (Adım 2'den)
- `indexes/nq_hnsw.index` (Adım 3a'dan)
- `scripts/arch1_rerank.py` (modül olarak import edilir)

**Oluşturduğu Dosyalar:**
- `outputs/metrics_report.txt` - Metrikler (text format)
- `outputs/metrics_report.csv` - Metrikler (CSV format)
- `outputs/metrics_report.json` - Metrikler (JSON format)
- `outputs/diagnostics.json` - Detaylı tanılama bilgileri
- `outputs/emissions.csv` - Enerji ve karbon emisyonu verileri

## Bağımlılık Diyagramı

```
arch1_prepare_data.py
    ├──> data/passages_train.jsonl
    └──> data/gold_eval.jsonl
            │
            ├──> arch1_embeddings.py
            │       ├──> indexes/passage_emb.npy
            │       └──> indexes/passages.txt
            │               │
            │               ├──> arch1_faiss.py
            │               │       └──> indexes/nq_hnsw.index
            │               │               │
            │               └──> arch1_rerank.py (modül)
            │                       │
            │                       └──> arch1_eval.py
            │                               └──> outputs/*.txt, *.csv, *.json
            │
            └──> arch1_eval.py (doğrudan kullanır)
```

## Tek Komutla Çalıştırma

Tüm pipeline'ı tek seferde çalıştırmak için:

```bash
python run_all.py
```

Bu komut yukarıdaki adımları sırayla çalıştırır:
1. Veri Hazırlama
2. Embedding Oluşturma
3. FAISS Index Oluşturma
4. Değerlendirme

## Manuel Çalıştırma

Her adımı ayrı ayrı çalıştırmak için:

```bash
# Adım 1
python scripts/arch1_prepare_data.py

# Adım 2
python scripts/arch1_embeddings.py

# Adım 3a (FAISS - önerilen)
python scripts/arch1_faiss.py

# Adım 3b (Scikit-learn - alternatif)
# python scripts/arch1_index_sklearn.py

# Adım 4
python scripts/arch1_eval.py
```

## Önemli Notlar

1. **Çalışma Dizini:** Tüm script'ler proje root dizininden çalıştırılmalıdır (relative path'ler kullanıldığı için).

2. **Import Yolu:** `arch1_eval.py` otomatik olarak `scripts/` dizinini Python path'ine ekler, böylece `arch1_rerank` modülünü import edebilir.

3. **FAISS vs Scikit-learn:** `arch1_eval.py` sadece FAISS index kullanır. Scikit-learn index kullanmak isterseniz, `arch1_eval.py` ve `arch1_rerank.py` dosyalarını güncellemeniz gerekir.

4. **İlk Çalıştırma:** İlk çalıştırmada Hugging Face modelleri indirilecektir (internet bağlantısı gerekir).

5. **Platform Desteği:** 
   - Apple Silicon MacBook'larda MPS (Metal) kullanılır
   - CUDA destekli sistemlerde CUDA kullanılır
   - Diğer sistemlerde CPU kullanılır

