# AI-Generated Text Detection — Bahasa Indonesia

> Final Project AI 2026

Sistem deteksi teks berbahasa Indonesia yang ditulis AI vs manusia, menggunakan tiga pendekatan pemodelan: SVM + TF-IDF (baseline), BiLSTM + FastText, dan IndoBERT fine-tuning.

---

## Anggota Kelompok

| Nama | NIM | Tugas Utama |
|------|-----|-------------|
| [Nama A] | [NIM] | Data collection (Groq/Llama) · SVM baseline |
| [Nama B] | [NIM] | Data collection (GPT-OSS) · BiLSTM |
| [Nama C] | [NIM] | EDA · IndoBERT fine-tuning |
| [Nama D] | [NIM] | Evaluation · Report |

---

## Progress

| Milestone | Status |
|-----------|--------|
| Dataset collection | ✅ Selesai |
| Data pipeline (fetch → prepare → split) | ✅ Selesai |
| EDA | 🔄 In progress |
| Baseline SVM + TF-IDF | ✅ Selesai |
| Deep Learning BiLSTM + FastText | 🔄 In progress |
| Transformer IndoBERT fine-tuning | ⏳ Belum dimulai |
| Comparative analysis | ⏳ Menunggu semua model selesai |
| Robustness testing | ⏳ Menunggu semua model selesai |
| Error analysis | ⏳ Menunggu semua model selesai |
| Simple report | ⏳ Belum dimulai |

---

## Dataset

| Sumber | Tipe | Jumlah |
|--------|------|--------|
| [IndoNLI](https://github.com/ir-nlp-csui/indonli) (kolom `premise`) | Human | 914 |
| Llama 3.1-8B via Groq (controlled prompting) | AI-generated | 789 |
| GPT-OSS-20B (controlled prompting) | AI-generated | 211 |
| **Total** | | **1.914** |

**Split:** 70% train / 15% val / 15% test (stratified)

| Split | Total | Human | AI |
|-------|-------|-------|----|
| Train | 1.337 | 639 | 698 |
| Val | 287 | 137 | 150 |
| Test | 287 | 138 | 149 |

Dataset besar **tidak disimpan di repo** (gitignored). File ada di `data/raw/` lokal masing-masing anggota.

---

## Model

| Tier | Model | Library | Status |
|------|-------|---------|--------|
| Baseline (ML) | SVM + TF-IDF char n-gram (3-5) | `scikit-learn` | ✅ |
| Deep Learning | BiLSTM + FastText ID (cc.id.300) | `PyTorch` | 🔄 |
| Transformer | IndoBERT (`indobenchmark/indobert-base-p1`) | `transformers` | ⏳ |

---

## Struktur Repo

```
FP-AI-Detecting-AI-generated-text-Bahasa/
├── data/
│   ├── raw/            # gitignored — file besar
│   └── samples/        # tracked — sample kecil + stats
├── notebooks/
│   ├── 01_data_collection/
│   │   └── indonli_data_prep.ipynb
│   ├── 02_eda/
│   ├── 03_baseline/
│   ├── 04_deep_learning/
│   ├── 05_transformer/
│   └── 06_evaluation/
├── src/
│   ├── data/
│   │   ├── paths.py
│   │   ├── collect_gpt_oss.py
│   │   ├── collect_openai.py
│   │   └── prepare_dataset.py
│   └── models/
│       ├── baseline.py
│       ├── bilstm.py
│       └── indobert.py
├── results/
│   ├── metrics/
│   └── figures/
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### 1. Clone & Setup

```bash
git clone https://github.com/[username]/[repo-name].git
cd [repo-name]
pip install -r requirements.txt
```

### 2. Setup API Keys

```bash
cp .env.example .env
# Isi nilai berikut di .env:
# GROQ_API_KEY=...
# API_KEY_OPENAI=...
```

### 3. Siapkan Dataset

```bash
# Step 1 — Jalankan notebook untuk data human (sekali saja)
# Buka: notebooks/01_data_collection/indonli_data_prep.ipynb

# Step 2 — Kumpulkan AI data (dua terminal paralel)
python3 src/data/collect_gpt_oss.py   # Terminal 1
python3 src/data/collect_openai.py    # Terminal 2

# Step 3 — Buat processed splits
python3 run_pipeline.py
```

### 4. Training Model

```bash
python3 src/models/baseline.py
python3 src/models/bilstm.py
python3 src/models/indobert.py
```

---

## Hasil Evaluasi (sementara)

| Model | Accuracy | F1 Macro | Train Time | Inference/sample |
|-------|----------|----------|------------|------------------|
| SVM + TF-IDF | - | - | 0.21s | 0.14ms |
| BiLSTM + FastText | - | - | - | - |
| IndoBERT | - | - | - | - |

> **Catatan:** Terdapat potensi topic bias antara AI-generated data dan IndoNLI human data. Didokumentasikan sebagai limitasi dataset pada bagian error analysis.

---

## Referensi

- Koto et al. (2020). [IndoBERT](https://arxiv.org/abs/2011.00677)
- Wilie et al. (2020). [IndoNLU](https://arxiv.org/abs/2009.05387)
- Devlin et al. (2019). [BERT](https://arxiv.org/abs/1810.04805)
- Joulin et al. (2016). [FastText](https://arxiv.org/abs/1607.01759)