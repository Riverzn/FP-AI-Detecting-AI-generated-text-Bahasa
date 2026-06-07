# ============================================================
# src/data/paths.py
# Single source of truth untuk semua path di project.
# Import modul ini dari script manapun — jangan hardcode path!
#
# Cara pakai:
#   from src.data.paths import PATHS
#   df.to_csv(PATHS["train"])
# ============================================================

import os
from pathlib import Path

# ── Deteksi environment ──────────────────────────────────────
IN_COLAB = os.path.exists("/content")

# Root repo — otomatis detect, tidak perlu diubah
if IN_COLAB:
    # Asumsi: git clone ke /content/ai-text-detection-id
    REPO_ROOT = Path("/content/ai-text-detection-id")
    # Google Drive — file besar (gitignored) disimpan di sini
    DRIVE_ROOT = Path("/content/drive/MyDrive/FP_AI")
else:
    # Lokal (WSL / PC)
    REPO_ROOT  = Path(__file__).resolve().parents[2]  # 2 level up dari src/data/
    DRIVE_ROOT = REPO_ROOT  # lokal: simpan di dalam repo (folder gitignored)

# ── Folder definitions ───────────────────────────────────────
PATHS = {
    # ── DATA (file besar → Drive, gitignored di repo) ────────
    "raw_dir"        : DRIVE_ROOT / "data" / "raw",
    "processed_dir"  : DRIVE_ROOT / "data" / "processed",

    # File mentah per sumber
    "raw_gemini"     : DRIVE_ROOT / "data" / "raw" / "ai_generated_gemini.jsonl",
    "raw_openai"     : DRIVE_ROOT / "data" / "raw" / "ai_generated_openai.jsonl",
    "raw_self_human" : DRIVE_ROOT / "data" / "raw" / "self_human.csv",

    # Processed splits
    "train"          : DRIVE_ROOT / "data" / "processed" / "train.csv",
    "val"            : DRIVE_ROOT / "data" / "processed" / "val.csv",
    "test"           : DRIVE_ROOT / "data" / "processed" / "test.csv",
    "train_clean"    : DRIVE_ROOT / "data" / "processed" / "train_clean.csv",
    "val_clean"      : DRIVE_ROOT / "data" / "processed" / "val_clean.csv",
    "test_clean"     : DRIVE_ROOT / "data" / "processed" / "test_clean.csv",
    "full_dataset"   : DRIVE_ROOT / "data" / "processed" / "dataset_full.csv",

    # ── SAMPLES (kecil, di-track di repo) ────────────────────
    "samples_dir"    : REPO_ROOT / "data" / "samples",
    "sample_50"      : REPO_ROOT / "data" / "samples" / "sample_50.csv",
    "dataset_stats"  : REPO_ROOT / "data" / "samples" / "dataset_stats.json",

    # ── MODELS (besar → Drive, gitignored) ───────────────────
    "models_dir"     : DRIVE_ROOT / "models_saved",
    "svm_model"      : DRIVE_ROOT / "models_saved" / "svm_model.pkl",
    "bilstm_weights" : DRIVE_ROOT / "models_saved" / "bilstm_weights.pt",
    "indobert_ckpt"  : DRIVE_ROOT / "models_saved" / "indobert_checkpoint",

    # ── RESULTS (kecil → repo, di-track) ─────────────────────
    "results_dir"    : REPO_ROOT / "results",
    "metrics_dir"    : REPO_ROOT / "results" / "metrics",
    "figures_dir"    : REPO_ROOT / "results" / "figures",
    "robustness_dir" : REPO_ROOT / "results" / "robustness",
}

def ensure_dirs():
    """Buat semua folder yang belum ada. Panggil sekali di awal setiap script."""
    dirs = [
        PATHS["raw_dir"], PATHS["processed_dir"],
        PATHS["samples_dir"],
        PATHS["models_dir"],
        PATHS["metrics_dir"], PATHS["figures_dir"], PATHS["robustness_dir"],
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✓ Paths ready | Repo: {REPO_ROOT} | Drive: {DRIVE_ROOT}")

if __name__ == "__main__":
    ensure_dirs()
    print("\nSemua path:")
    for k, v in PATHS.items():
        print(f"  {k:<20} → {v}")