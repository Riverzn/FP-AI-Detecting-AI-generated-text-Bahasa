# ============================================================
# src/data/prepare_dataset.py
# Fungsi : Gabung semua raw data → clean → split → simpan
#
# Komposisi dataset (kesepakatan kelompok):
#   Human  : 700 IndoNLI + 300 scraped (Detik/Kompas/Tempo) = 1000
#   #AI    : 500 Llama + 100 GPT-OSS + 400 Qwen = 1000                      = 1000
#   Total  : 2000 (balanced 1:1)
#   Split  : 70% train / 15% val / 15% test (stratified)
#
# Input  : data/raw/indonli_human.jsonl
#          data/raw/self_human.json       ← scraped 300
#          data/raw/ai_generated_llama.jsonl
#          data/raw/ai_generated_gpt_oss.jsonl
#
# Output : data/processed/train|val|test_clean.csv
#          data/samples/sample_50.csv
#          data/samples/dataset_stats.json
# ============================================================

import sys, os, json, random
from pathlib import Path

IN_COLAB  = os.path.exists("/content")
REPO_ROOT = (
    Path("/content/FP-AI-Detecting-AI-generated-text-Bahasa-")
    if IN_COLAB
    else Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import PATHS, ensure_dirs
import pandas as pd
from sklearn.model_selection import train_test_split

ensure_dirs()
random.seed(42)

# ── Komposisi target ──────────────────────────────────────────
CAP_INDONLI = 700    # human dari IndoNLI
CAP_SCRAPED = 300    # human dari scraping (sudah fix 300, tidak perlu cap)
CAP_LLAMA   = 789    # semua llama (789 total)
CAP_GPTOSS  = 211    # semua gpt-oss (211 total)
# Total human = 1000, Total AI = 1000, Grand total = 2000


# ── Helper: load JSONL ───────────────────────────────────────
def load_jsonl(filepath):
    records = []
    p = Path(filepath)
    if not p.exists():
        print(f"  ⚠ Tidak ditemukan: {p.name} — skip")
        return records
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if not d.get("text"):
                    continue
                records.append({
                    "text"      : str(d["text"]).strip(),
                    "label"     : str(d.get("label", "unknown")),
                    "label_int" : int(d.get("label_int", -1)),
                    "source"    : d.get("source_model") or d.get("source", "unknown"),
                    "topic"     : d.get("topic", "various"),
                    "word_count": d.get("word_count", len(str(d["text"]).split())),
                })
            except Exception:
                continue
    print(f"  ✓ {len(records):>5} records ← {p.name}")
    return records


# ── Helper: load JSON (scraped) ───────────────────────────────
def load_json_scraped(filepath):
    records = []
    p = Path(filepath)
    if not p.exists():
        print(f"  ⚠ Tidak ditemukan: {p.name} — skip")
        return records
    try:
        raw = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"  ⚠ Gagal load {p.name}: {e}")
        return records

    for d in raw:
        text = str(d.get("text", "")).strip()
        if not text or len(text.split()) < 5:
            continue
        records.append({
            "text"      : text,
            "label"     : "human",
            "label_int" : 0,
            "source"    : f"scraped_{d.get('media', 'news')}",
            "topic"     : "news",
            "word_count": len(text.split()),
        })
    print(f"  ✓ {len(records):>5} records ← {p.name}")
    return records


# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  PREPARE DATASET")
print("=" * 55)
print(f"\n  Target: {CAP_INDONLI} IndoNLI + {CAP_SCRAPED} scraped"
      f" + {CAP_LLAMA} Llama + {CAP_GPTOSS} GPT-OSS = 2000 total")


# ── [1/4] AI data ─────────────────────────────────────────────
print("\n[1/4] Loading AI-generated data...")

ai_raw     = load_jsonl(PATHS["raw_dir"] / "dataset_ai_baru.jsonl")
ai_records = random.sample(ai_raw, min(1000, len(ai_raw)))
print(f"  Total AI: {len(ai_records)} / 1000")

if len(ai_records) == 0:
    print("  ✗ Tidak ada AI data. Jalankan collect_gpt_oss.py dulu.")
    exit(1)


# ── [2/4] Human — IndoNLI ─────────────────────────────────────
print("\n[2/4] Loading IndoNLI human data...")

indonli_path = PATHS["raw_dir"] / "indonli_human.jsonl"

if indonli_path.exists():
    indonli_raw   = load_jsonl(indonli_path)
    indonli_records = random.sample(indonli_raw, min(CAP_INDONLI, len(indonli_raw)))
    if len(indonli_raw) > CAP_INDONLI:
        print(f"    capped ke {CAP_INDONLI} dari {len(indonli_raw)}")
    print(f"  IndoNLI : {len(indonli_records):>4} / {CAP_INDONLI}")
else:
    # Fallback ke HuggingFace
    print("  indonli_human.jsonl tidak ada → fallback ke HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("afaji/indonli")
        seen, indonli_raw = set(), []
        for split_name in ["train", "validation", "test_lay", "test_expert"]:
            if split_name not in dataset:
                continue
            for item in dataset[split_name]:
                text = item["premise"].strip()
                if text in seen or len(text.split()) < 15:
                    continue
                indonli_raw.append({
                    "text": text, "label": "human", "label_int": 0,
                    "source": "IndoNLI", "topic": "various",
                    "word_count": len(text.split()),
                })
                seen.add(text)
        random.shuffle(indonli_raw)
        indonli_records = indonli_raw[:CAP_INDONLI]
        print(f"  ✓ {len(indonli_records)} records ← HuggingFace IndoNLI")
    except Exception as e:
        print(f"  ✗ Fallback gagal: {e} — jalankan notebook indonli_data_prep.ipynb dulu")
        exit(1)


# ── [3/4] Human — Scraped ─────────────────────────────────────
print("\n[3/4] Loading scraped human data...")

scraped_path = PATHS["raw_dir"] / "human_scraped_300.json"
scraped_records = load_json_scraped(scraped_path)

if len(scraped_records) == 0:
    print("  ⚠ Scraped data tidak ditemukan — lanjut tanpa scraped")
else:
    # Cap di 300
    scraped_records = random.sample(scraped_records, min(CAP_SCRAPED, len(scraped_records)))
    print(f"  Scraped : {len(scraped_records):>4} / {CAP_SCRAPED}")

human_records = indonli_records + scraped_records
print(f"  Total human: {len(human_records)}")


# ── [4/4] Merge, clean, split ─────────────────────────────────
print("\n[4/4] Merging, cleaning, splitting...")

all_records = ai_records + human_records
df = pd.DataFrame(all_records)

before = len(df)
df = df.dropna(subset=["text"])
df["text"] = df["text"].str.strip()
df = df[df["text"].str.len() > 20]
df = df[df["word_count"] >= 5]

# === MODIFIKASI DISINI (ANTI LENGTH BIAS BY CLAUDE) ===
df = df.drop_duplicates(subset=["text"])
df["text"] = df["text"].apply(lambda t: ' '.join(str(t).split()[:40]))  # truncate ke 40 kata
df["word_count"] = df["text"].str.split().str.len()                 # update word_count asli
# =======================================================

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["id"] = [f"SAMPLE_{i:05d}" for i in range(len(df))]

print(f"  Cleaned  : {before} → {len(df)} records ({before - len(df)} dihapus)")
print(f"\n  Label distribution:\n{df['label'].value_counts().to_string()}")
print(f"\n  Source distribution:\n{df['source'].value_counts().to_string()}")
print(f"\n  Word count: avg={df['word_count'].mean():.1f}, "
      f"min={df['word_count'].min()}, max={df['word_count'].max()}")

label_counts = df["label"].value_counts()
ratio = label_counts.max() / label_counts.min()
if ratio > 1.5:
    print(f"\n  ⚠ Class ratio {ratio:.2f}:1 — pertimbangkan class_weight saat training")
else:
    print(f"\n  ✓ Class ratio {ratio:.2f}:1 (balanced)")

# Stratified 70 / 15 / 15
df_train, df_temp = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label"]
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, random_state=42, stratify=df_temp["label"]
)

print(f"\n  Train : {len(df_train):>5}  {df_train['label'].value_counts().to_dict()}")
print(f"  Val   : {len(df_val):>5}  {df_val['label'].value_counts().to_dict()}")
print(f"  Test  : {len(df_test):>5}  {df_test['label'].value_counts().to_dict()}")


# ── Save processed → Drive/local (gitignored) ─────────────────
print(f"\n  Saving → {PATHS['processed_dir']}")
df.to_csv(PATHS["full_dataset"], index=False)
df_train.to_csv(PATHS["train"], index=False)
df_val.to_csv(  PATHS["val"],   index=False)
df_test.to_csv( PATHS["test"],   index=False)

CLEAN = ["id", "text", "label", "label_int"]
df_train[CLEAN].to_csv(PATHS["train_clean"], index=False)
df_val[CLEAN].to_csv(  PATHS["val_clean"],   index=False)
df_test[CLEAN].to_csv( PATHS["test_clean"],  index=False)


# ── Save sample + stats → repo (tracked di git) ───────────────
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
    "composition": {
    "indonli_human" : len(indonli_records),
    "scraped_human" : len(scraped_records),
    "ai_combined"   : len(ai_records),
    }
}
with open(PATHS["dataset_stats"], "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"  Stats    → {PATHS['dataset_stats']}")

print("\n✅ Dataset siap!")
print(f"   Train : {len(df_train)}")
print(f"   Val   : {len(df_val)}")
print(f"   Test  : {len(df_test)}")