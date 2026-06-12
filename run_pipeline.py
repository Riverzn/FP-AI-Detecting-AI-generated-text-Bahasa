#!/usr/bin/env python3
# ============================================================
# run_pipeline.py
# Master script: cek + jalankan semua tahap data pipeline
# sebelum mulai training IndoBERT.
#
# Letakkan di ROOT repo, jalankan dari root:
#   python run_pipeline.py
#
# Urutan:
#   [1] Cek raw AI data (collect_gemini / collect_openai)
#   [2] fetch_human.py  → data/raw/indonli_human.jsonl & human_scraped_300.json
#   [3] prepare_dataset.py → data/processed/train|val|test_clean.csv
#   [4] Verifikasi akhir → tampilkan stats siap IndoBERT
# ============================================================

import sys, json, subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from src.data.paths import PATHS, ensure_dirs

ensure_dirs()

# ── ANSI colors ───────────────────────────────────────────────
OK   = "\033[92m✓\033[0m"
WARN = "\033[93m⚠\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94m→\033[0m"

def run(script_path):
    """Jalankan script Python dan return True jika sukses."""
    print(f"\n  {INFO} Menjalankan: {script_path.relative_to(REPO_ROOT)}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT)
    )
    return result.returncode == 0

def count_jsonl(filepath):
    """Hitung jumlah baris valid di file JSONL."""
    p = Path(filepath)
    if not p.exists():
        return 0
    count = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count

def count_json(filepath):
    """Hitung jumlah item (array) di file JSON standar."""
    p = Path(filepath)
    if not p.exists():
        return 0
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
            return len(data) if isinstance(data, list) else 0
    except json.JSONDecodeError:
        return 0

def file_size_mb(filepath):
    p = Path(filepath)
    return p.stat().st_size / 1e6 if p.exists() else 0


# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 58)
print("  DATA PIPELINE CHECK — PRE-INDOBERT")
print("=" * 58)

# ── [1] Cek raw AI data ───────────────────────────────────────
print("\n[1/4] Status raw AI-generated data:")

llama_path  = PATHS["raw_dir"] / "ai_generated_llama.jsonl"
openai_path = PATHS["raw_dir"] / "ai_generated_gpt_oss.jsonl"

llama_count  = count_jsonl(llama_path)
openai_count = count_jsonl(openai_path)
total_ai     = llama_count + openai_count

print(f"  {OK if llama_count  > 0 else WARN} ai_generated_llama.jsonl (Llama) : {llama_count:>5} sampel")
print(f"  {OK if openai_count > 0 else WARN} ai_generated_gpt_oss.jsonl (GPT-OSS)    : {openai_count:>5} sampel")
print(f"\n  Total AI data: {total_ai} sampel")

if total_ai < 500:
    print(f"  {WARN} AI data masih kurang — pipeline tetap dilanjutkan.")
else:
    print(f"  {OK} AI data mencukupi")


# ── [2] Cek human data ────────────────────────────────────────
print("\n[2/4] Status human data:")

indonli_path  = PATHS["raw_dir"] / "indonli_human.jsonl"
scraped_path  = PATHS["raw_dir"] / "human_scraped_300.json"

indonli_count = count_jsonl(indonli_path)
scraped_count = count_json(scraped_path)
total_human   = indonli_count + scraped_count

print(f"  {OK if indonli_count > 0 else WARN} indonli_human.jsonl    : {indonli_count:>5} sampel")
print(f"  {OK if scraped_count > 0 else WARN} human_scraped_300.json : {scraped_count:>5} sampel")
print(f"\n  Total Human data: {total_human} sampel")

notebook_path = REPO_ROOT / "notebooks" / "01_data_collection" / "indonli_data_prep.ipynb"

if total_human >= 500:
    print(f"  {OK} Human data mencukupi")
else:
    print(f"  {FAIL} Human data belum ada atau kurang dari 500 sampel (ada: {total_human})")
    print(f"\n  Jalankan notebook ini terlebih dahulu:")
    print(f"  {notebook_path}")
    print(f"\n  Atau pastikan scraping berjalan dengan benar.")
    sys.exit(1)


# ── [3] Prepare dataset ───────────────────────────────────────
print("\n[3/4] Membuat processed splits...")

train_exists = PATHS["train_clean"].exists()
val_exists   = PATHS["val_clean"].exists()
test_exists  = PATHS["test_clean"].exists()

# Force re-run jika AI data atau Human data bertambah
if train_exists and val_exists and test_exists:
    import pandas as pd
    existing_total = len(pd.read_csv(PATHS["train_clean"])) + \
                     len(pd.read_csv(PATHS["val_clean"]))   + \
                     len(pd.read_csv(PATHS["test_clean"]))
    expected_total = total_ai + total_human
    gap = abs(expected_total - existing_total)

    if gap < 50:
        print(f"  {OK} Processed splits sudah up-to-date ({existing_total} total) — skip")
    else:
        print(f"  {WARN} Data bertambah/berubah ({gap} baris beda) — re-run prepare_dataset.py")
        run(REPO_ROOT / "src" / "data" / "prepare_dataset.py")
else:
    print(f"  {INFO} Processed splits belum ada — menjalankan prepare_dataset.py...")
    success = run(REPO_ROOT / "src" / "data" / "prepare_dataset.py")
    if not success:
        print(f"  {FAIL} prepare_dataset.py gagal.")
        sys.exit(1)


# ── [4] Final verification ────────────────────────────────────
print("\n[4/4] Verifikasi akhir:")

all_ok = True
for name, path in [
    ("train_clean.csv", PATHS["train_clean"]),
    ("val_clean.csv",   PATHS["val_clean"]),
    ("test_clean.csv",  PATHS["test_clean"]),
]:
    if path.exists():
        import pandas as pd
        df   = pd.read_csv(path)
        dist = df["label"].value_counts().to_dict()
        print(f"  {OK} {name:<20} {len(df):>5} rows  {dist}")
    else:
        print(f"  {FAIL} {name} tidak ditemukan!")
        all_ok = False

# Dataset stats
stats_path = PATHS["dataset_stats"]
if stats_path.exists():
    with open(stats_path) as f:
        stats = json.load(f)
    print(f"\n  Dataset stats:")
    print(f"    Total         : {stats.get('total', '?')}")
    print(f"    Train/Val/Test: {stats.get('train','?')} / {stats.get('val','?')} / {stats.get('test','?')}")
    print(f"    Label dist    : {stats.get('label_dist', {})}")
    print(f"    Avg word count: {stats.get('avg_word_count', '?')}")
    ratio = stats.get("class_ratio", 1)
    icon  = OK if ratio <= 1.5 else WARN
    print(f"  {icon} Class ratio: {ratio:.2f}:1 {'(balanced)' if ratio <= 1.5 else '(pertimbangkan class_weight)'}")

print("\n" + "=" * 58)
if all_ok:
    print("  ✅ PIPELINE SELESAI — Dataset siap untuk IndoBERT!")
    print("\n  Langkah berikutnya:")
    print(f"    python src/models/indobert.py")
    print(f"\n  Atau load di notebook:")
    print(f"    import pandas as pd")
    print(f"    df_train = pd.read_csv('{PATHS['train_clean']}')")
    print(f"    df_val   = pd.read_csv('{PATHS['val_clean']}')")
    print(f"    df_test  = pd.read_csv('{PATHS['test_clean']}')")
else:
    print("  ❌ Ada file yang belum siap. Cek log di atas.")
print("=" * 58 + "\n")