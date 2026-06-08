# ============================================================
# src/data/prepare_dataset.py
# Fungsi : Gabung semua raw data → clean → split → simpan
#
# Input  : data/raw/indonli_human.jsonl    ← output fetch_human.py
#          data/raw/ai_generated_gemini.jsonl ← output collect_gemini.py
#          data/raw/ai_generated_openai.jsonl ← output collect_openai.py
#          data/raw/self_human.csv          ← opsional, scraping manual
#
# Output : data/processed/train.csv | val.csv | test.csv
#          data/processed/train_clean.csv | val_clean.csv | test_clean.csv
#          data/samples/sample_50.csv      ← tracked di git
#          data/samples/dataset_stats.json ← tracked di git
#
# Jalankan SETELAH fetch_human.py + semua collect_*.py selesai.
# ============================================================

import sys, os, json
from pathlib import Path

# ── Path setup (Lokal & Colab) ───────────────────────────────
IN_COLAB  = os.path.exists("/content")
REPO_ROOT = (
    Path("/content/FP-AI-Detecting-AI-generated-text-Bahasa-")   # Colab
    if IN_COLAB
    else Path(__file__).resolve().parents[2]                       # Lokal / WSL
)
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import PATHS, ensure_dirs
import pandas as pd
from sklearn.model_selection import train_test_split

ensure_dirs()


# ── Helper: load JSONL ───────────────────────────────────────
def load_jsonl(filepath, required_keys=("text", "label", "label_int")):
    records = []
    p = Path(filepath)
    if not p.exists():
        print(f"  ⚠ Tidak ditemukan: {p.name} — skip")
        return records
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                # Pastikan field wajib ada
                if not all(k in d for k in required_keys):
                    continue
                records.append({
                    "text"      : str(d["text"]).strip(),
                    "label"     : d["label"],
                    "label_int" : int(d["label_int"]),
                    "source"    : d.get("source_model") or d.get("source", "unknown"),
                    "topic"     : d.get("topic", "various"),
                    "word_count": d.get("word_count", len(str(d["text"]).split())),
                })
            except Exception:
                continue
    print(f"  ✓ {len(records):>5} records ← {p.name}")
    return records


# ── [1/4] Load AI-generated data ─────────────────────────────
# Llama (collect_gpt_oss.py) : 789 rows → ai_generated_llama.jsonl
# GPT-OSS (teammate script)  : 211 rows → ai_generated_gpt_oss.jsonl
# Total AI                   : 1000 (tidak perlu cap)
print("=" * 55)
print("  PREPARE DATASET")
print("=" * 55)
print("\n[1/4] Loading AI-generated data...")

llama_records  = load_jsonl(PATHS["raw_dir"] / "ai_generated_llama.jsonl")
gptoss_records = load_jsonl(PATHS["raw_dir"] / "ai_generated_gpt_oss.jsonl")
ai_records     = llama_records + gptoss_records

print(f"  Llama   : {len(llama_records):>5} rows")
print(f"  GPT-OSS : {len(gptoss_records):>5} rows")
print(f"  Total AI: {len(ai_records)}")

if len(ai_records) == 0:
    print("  Tidak ada AI data.")
    exit(1)


# ── [2/4] Load Human data ────────────────────────────────────
print("\n[2/4] Loading human data...")

# Prioritas: local jsonl (dari fetch_human.py) → HuggingFace fallback
human_records = []
indonli_local = PATHS["raw_dir"] / "indonli_human.jsonl"

if indonli_local.exists():
    # Local file sudah ada — load langsung, tidak perlu internet
    raw_human = load_jsonl(indonli_local)
    # Cap 1000 sesuai kesepakatan kelompok
    import random as _rnd2; _rnd2.seed(42)
    human_records = _rnd2.sample(raw_human, min(1000, len(raw_human)))
    if len(raw_human) > 1000:
        print(f"    capped ke 1000 dari {len(raw_human)}")
        
    if len(human_records) < 1000:
        print(f"  ⚠ Human hanya {len(human_records)}/1000 — re-run notebook indonli_data_prep.ipynb")
    print(f"    dengan MIN_WORDS=10 untuk dapat lebih banyak sampel")
else:
    # Fallback: load dari HuggingFace datasets
    print("  indonli_human.jsonl tidak ditemukan → fallback ke HuggingFace...")
    try:
        from datasets import load_dataset
        dataset  = load_dataset("afaji/indonli")
        seen     = set()
        for split_name in ["train", "validation", "test_lay", "test_expert"]:
            # FIX: gunakan 'in' bukan .get() — DatasetDict tidak support .get()
            if split_name not in dataset:
                continue
            for item in dataset[split_name]:
                text = item["premise"].strip()
                if text in seen or len(text.split()) < 15:
                    continue
                human_records.append({
                    "text"      : text,
                    "label"     : "human",
                    "label_int" : 0,
                    "source"    : "IndoNLI",
                    "topic"     : "various",
                    "word_count": len(text.split()),
                })
                seen.add(text)
        import random
        random.seed(42)
        random.shuffle(human_records)
        human_records = human_records[:1500]
        print(f"  ✓ {len(human_records):>5} records ← HuggingFace IndoNLI")
    except Exception as e:
        print(f"  ✗ HuggingFace fallback gagal: {e}")
        print("    Jalankan fetch_human.py terlebih dahulu.")
        exit(1)

print(f"  Total human: {len(human_records)}")


# ── [3/4] Load self-constructed human (opsional) ─────────────
print("\n[3/4] Loading self-constructed human data...")
self_records = []
self_path    = PATHS["raw_self_human"]

if self_path.exists():
    df_self = pd.read_csv(self_path)
    if "text" not in df_self.columns:
        print(f"  ⚠ self_human.csv tidak punya kolom 'text' — skip")
    else:
        for _, row in df_self.iterrows():
            text = str(row["text"]).strip()
            if not text:
                continue
            self_records.append({
                "text"      : text,
                "label"     : "human",
                "label_int" : 0,
                "source"    : str(row.get("source", "self_human")),
                "topic"     : str(row.get("topic",  "various")),
                "word_count": len(text.split()),
            })
        print(f"  ✓ {len(self_records):>5} records ← self_human.csv")
else:
    print(f"  ⚠ self_human.csv tidak ditemukan — skip (opsional)")


# ── [4/4] Merge, clean, split ────────────────────────────────
print("\n[4/4] Merging, cleaning, splitting...")

df = pd.DataFrame(ai_records + human_records + self_records)

before = len(df)
df = df.dropna(subset=["text"])
df["text"] = df["text"].str.strip()
df = df[df["text"].str.len() > 20]         # minimal 20 karakter
df = df[df["word_count"] >= 5]              # minimal 5 kata
df = df.drop_duplicates(subset=["text"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["id"] = [f"SAMPLE_{i:05d}" for i in range(len(df))]

print(f"  Cleaned   : {before} → {len(df)} records ({before - len(df)} dihapus)")
print(f"\n  Label distribution:\n{df['label'].value_counts().to_string()}")
print(f"\n  Source distribution:\n{df['source'].value_counts().to_string()}")
print(f"\n  Word count: avg={df['word_count'].mean():.1f}, "
      f"min={df['word_count'].min()}, max={df['word_count'].max()}")

# Cek imbalance
label_counts = df["label"].value_counts()
ratio = label_counts.max() / label_counts.min()
if ratio > 2:
    print(f"\n  ⚠ Class imbalance terdeteksi (ratio {ratio:.1f}:1)")
    print(f"    Pertimbangkan oversampling atau class_weight saat training")

# Stratified split: 70% train, 15% val, 15% test
df_train, df_temp = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label"]
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, random_state=42, stratify=df_temp["label"]
)

print(f"\n  Train : {len(df_train):>5}  {df_train['label'].value_counts().to_dict()}")
print(f"  Val   : {len(df_val):>5}  {df_val['label'].value_counts().to_dict()}")
print(f"  Test  : {len(df_test):>5}  {df_test['label'].value_counts().to_dict()}")


# ── Save processed splits → Drive/local (gitignored) ─────────
print(f"\n  Saving → {PATHS['processed_dir']}")
df.to_csv(PATHS["full_dataset"], index=False)
df_train.to_csv(PATHS["train"], index=False)
df_val.to_csv(  PATHS["val"],   index=False)
df_test.to_csv( PATHS["test"],  index=False)

CLEAN = ["id", "text", "label", "label_int"]
df_train[CLEAN].to_csv(PATHS["train_clean"], index=False)
df_val[CLEAN].to_csv(  PATHS["val_clean"],   index=False)
df_test[CLEAN].to_csv( PATHS["test_clean"],  index=False)


# ── Save sample + stats → repo (tracked di git) ──────────────
df.sample(50, random_state=42)[CLEAN].to_csv(PATHS["sample_50"], index=False)
print(f"  Sample   → {PATHS['sample_50']}")

stats = {
    "total"          : int(len(df)),
    "train"          : int(len(df_train)),
    "val"            : int(len(df_val)),
    "test"           : int(len(df_test)),
    "label_dist"     : {k: int(v) for k, v in df["label"].value_counts().items()},
    "source_dist"    : {k: int(v) for k, v in df["source"].value_counts().items()},
    "avg_word_count" : round(float(df["word_count"].mean()), 1),
    "class_ratio"    : round(float(ratio), 2),
}
with open(PATHS["dataset_stats"], "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"  Stats    → {PATHS['dataset_stats']}")

print("\n✅ Dataset siap! File untuk modeling:")
print(f"   {PATHS['train_clean']}")
print(f"   {PATHS['val_clean']}")
print(f"   {PATHS['test_clean']}")