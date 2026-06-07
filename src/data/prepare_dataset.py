# ============================================================
# src/data/prepare_dataset.py
# Merge semua sumber data → train / val / test split
# Jalankan SETELAH collect_gemini.py dan collect_openai.py selesai.
#
# Lokasi di repo : src/data/prepare_dataset.py
# Input          : data/raw/*.jsonl  +  IndoNLI (HuggingFace)
# Output         : data/processed/train.csv | val.csv | test.csv   (→ Drive)
#                  data/samples/sample_50.csv  (→ repo, di-track git)
#                  data/samples/dataset_stats.json
# ============================================================

# ── CELL 1: Install ─────────────────────────────────────────
# !pip install datasets pandas scikit-learn python-dotenv -q

# ── CELL 2: Import & Path Setup ─────────────────────────────
import sys, os, json
from pathlib import Path

REPO_ROOT = Path("/content/FP-AI-Detecting-AI-generated-text-Bahasa") if os.path.exists("/content") else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import PATHS, ensure_dirs
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ensure_dirs()

# ── CELL 3: Load AI-Generated Data ──────────────────────────
def load_jsonl(filepath):
    records = []
    p = Path(filepath)
    if not p.exists():
        print(f"  ⚠ Tidak ditemukan: {filepath}")
        return records
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            records.append({
                "text":       d["text"],
                "label":      d["label"],
                "label_int":  d["label_int"],
                "source":     d["source_model"],
                "topic":      d.get("topic", ""),
                "word_count": d.get("word_count", len(d["text"].split())),
            })
    print(f"  ✓ {len(records):>5} records ← {p.name}")
    return records

print("=== [1/4] Loading AI-generated data ===")
ai_records = load_jsonl(PATHS["raw_gemini"]) + load_jsonl(PATHS["raw_openai"])
print(f"  Total AI samples: {len(ai_records)}")

# ── CELL 4: Load IndoNLI Human Data ─────────────────────────
def load_indonli(min_words=20, max_samples=1500):
    import random
    print("\n=== [2/4] Loading IndoNLI (human) ===")
    dataset = load_dataset("afaji/indonli")
    records, seen = [], set()
    for split_name in ["train", "validation", "test_lay", "test_expert"]:
        for item in dataset.get(split_name, []):
            text = item["premise"].strip()
            if text in seen:
                continue
            wc = len(text.split())
            if wc < min_words:
                continue
            records.append({
                "text":       text,
                "label":      "human",
                "label_int":  0,
                "source":     "IndoNLI",
                "topic":      "various",
                "word_count": wc,
            })
            seen.add(text)
    random.seed(42)
    random.shuffle(records)
    selected = records[:max_samples]
    print(f"  ✓ {len(selected):>5} samples (filtered ≥{min_words} kata, dari {len(records)} unique)")
    return selected

human_indonli = load_indonli(min_words=20, max_samples=1500)

# ── CELL 5: Load Self-Constructed Human Data ─────────────────
def load_self_human():
    print("\n=== [3/4] Loading self-constructed human data ===")
    p = PATHS["raw_self_human"]
    if not p.exists():
        print(f"  ⚠ {p} tidak ditemukan — skip")
        return []
    df = pd.read_csv(p)
    assert "text" in df.columns, "CSV self_human.csv harus punya kolom 'text'"
    records = []
    for _, row in df.iterrows():
        records.append({
            "text":       str(row["text"]).strip(),
            "label":      "human",
            "label_int":  0,
            "source":     row.get("source", "self_human"),
            "topic":      row.get("topic", "various"),
            "word_count": len(str(row["text"]).split()),
        })
    print(f"  ✓ {len(records):>5} self-constructed human samples")
    return records

self_human = load_self_human()

# ── CELL 6: Merge & Clean ────────────────────────────────────
print("\n=== [4/4] Merging & cleaning ===")
df = pd.DataFrame(ai_records + human_indonli + self_human)

before = len(df)
df = df.dropna(subset=["text"])
df["text"] = df["text"].str.strip()
df = df[df["text"].str.len() > 20]
df = df.drop_duplicates(subset=["text"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["id"] = [f"SAMPLE_{i:05d}" for i in range(len(df))]

print(f"  Cleaned: {before} → {len(df)} records ({before-len(df)} dihapus)")
print(f"\n  Label distribution:\n{df['label'].value_counts().to_string()}")
print(f"\n  Source distribution:\n{df['source'].value_counts().to_string()}")
print(f"\n  Word count: avg={df['word_count'].mean():.1f}, min={df['word_count'].min()}, max={df['word_count'].max()}")

# ── CELL 7: Train / Val / Test Split (stratified) ────────────
df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
df_val,  df_test  = train_test_split(df_temp, test_size=0.50, random_state=42, stratify=df_temp["label"])

print(f"\n  Train : {len(df_train):>5} | {df_train['label'].value_counts().to_dict()}")
print(f"  Val   : {len(df_val):>5} | {df_val['label'].value_counts().to_dict()}")
print(f"  Test  : {len(df_test):>5} | {df_test['label'].value_counts().to_dict()}")

# ── CELL 8: Save ke Drive (processed) ───────────────────────
print(f"\n  Saving processed splits → {PATHS['processed_dir']}")
df.to_csv(PATHS["full_dataset"], index=False)
df_train.to_csv(PATHS["train"], index=False)
df_val.to_csv(PATHS["val"],   index=False)
df_test.to_csv(PATHS["test"],  index=False)

CLEAN_COLS = ["id", "text", "label", "label_int"]
df_train[CLEAN_COLS].to_csv(PATHS["train_clean"], index=False)
df_val[CLEAN_COLS].to_csv(PATHS["val_clean"],   index=False)
df_test[CLEAN_COLS].to_csv(PATHS["test_clean"],  index=False)

# ── CELL 9: Save sample ke repo (tracked di git) ─────────────
print(f"\n  Saving sample → {PATHS['sample_50']}")
df.sample(50, random_state=42)[CLEAN_COLS].to_csv(PATHS["sample_50"], index=False)

# Simpan stats sebagai JSON (untuk README dan laporan)
import json as _json
stats = {
    "total":       int(len(df)),
    "train":       int(len(df_train)),
    "val":         int(len(df_val)),
    "test":        int(len(df_test)),
    "label_dist":  df["label"].value_counts().to_dict(),
    "source_dist": df["source"].value_counts().to_dict(),
    "avg_word_count": round(float(df["word_count"].mean()), 1),
}
with open(PATHS["dataset_stats"], "w", encoding="utf-8") as f:
    _json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"  Saving stats → {PATHS['dataset_stats']}")
print("\n✅ Semua output tersimpan:")
print(f"   Drive : {PATHS['processed_dir']}")
print(f"   Repo  : {PATHS['samples_dir']}  (commit ke git!)")