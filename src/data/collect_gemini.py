# ============================================================
# src/data/collect_gemini.py
# Pengumpulan AI-generated samples via Gemini API
# Topik 1–4 | Target: 500 sampel
#
# Lokasi di repo : src/data/collect_gemini.py
# Output data    : data/raw/ai_generated_gemini.jsonl  (→ Drive, gitignored)
#
# Cara run di Colab:
#   !pip install google-generativeai -q
#   from google.colab import drive; drive.mount('/content/drive')
#   %run src/data/collect_gemini.py
# ============================================================

# ── CELL 1: Install & Mount ─────────────────────────────────
# !pip install google-generativeai python-dotenv -q
# from google.colab import drive; drive.mount('/content/drive')

# ── CELL 2: Import & Path Setup ─────────────────────────────
import sys
import os
from pathlib import Path

# Tambahkan root repo ke sys.path agar bisa import src.*
REPO_ROOT = Path("/content/FP-AI-Detecting-AI-generated-text-Bahasa") if os.path.exists("/content") else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import PATHS, ensure_dirs
import google.generativeai as genai
import json, time, random
from datetime import datetime

# ── CELL 3: Konfigurasi ─────────────────────────────────────
# API key: simpan di .env atau set langsung di bawah (jangan commit!)
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

API_KEY      = os.getenv("API_KEY_GEMINI")
MODEL_NAME   = "gemini-2.5-flash-lite"   # free tier: 15 RPM, 1.500 req/hari
SOURCE_MODEL = "Gemini-1.5-Flash"
OUTPUT_FILE  = PATHS["raw_gemini"]  # → data/raw/ai_generated_gemini.jsonl (Drive)
TARGET       = 500
DELAY        = 2.5                  # detik antar request (batas free tier: 15 RPM)

ensure_dirs()
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)
print(f"Output → {OUTPUT_FILE}")

# ── CELL 4: Topic–Subtopic Matrix (Topik 1–4 untuk Gemini) ──
TOPICS = [
    {
        "topic": "Kesehatan",
        "subtopics": [
            "manfaat olahraga rutin terhadap kesehatan jantung",
            "dampak kurang tidur terhadap sistem imun tubuh",
            "peran gizi seimbang dalam mencegah penyakit kronis",
            "kesehatan mental remaja di era media sosial",
            "program jaminan kesehatan nasional di Indonesia",
        ],
    },
    {
        "topic": "Sains dan Alam",
        "subtopics": [
            "dampak perubahan iklim terhadap curah hujan di Indonesia",
            "keanekaragaman hayati hutan hujan tropis Kalimantan",
            "proses terbentuknya gempa bumi tektonik lempeng",
            "fenomena gerhana matahari total dan cara mengamatinya",
            "aktivitas vulkanik gunung berapi aktif di Indonesia",
        ],
    },
    {
        "topic": "Sejarah dan Budaya",
        "subtopics": [
            "peran Soekarno dalam proklamasi kemerdekaan Indonesia",
            "sistem pemerintahan kerajaan Majapahit",
            "tradisi dan upacara adat suku Jawa",
            "warisan budaya batik sebagai identitas nasional Indonesia",
            "sejarah perkembangan wayang kulit di Jawa",
        ],
    },
    {
        "topic": "Politik dan Hukum",
        "subtopics": [
            "sistem pemilihan umum langsung di Indonesia",
            "peran Mahkamah Konstitusi dalam demokrasi Indonesia",
            "kebijakan otonomi daerah dan dampaknya",
            "hubungan diplomatik Indonesia dengan negara ASEAN",
            "penegakan hukum terhadap tindak pidana korupsi",
        ],
    },
]

STYLES = [
    ("berita",         "Gunakan gaya penulisan jurnalistik seperti artikel berita Kompas."),
    ("ensiklopedia",   "Gunakan gaya penulisan formal seperti artikel Wikipedia bahasa Indonesia."),
    ("ilmiah_populer", "Tulis seperti bagian dari artikel ilmiah populer untuk pembaca umum."),
    ("opini",          "Gunakan gaya penulisan opini editorial dengan sudut pandang informatif."),
    ("narasi",         "Gunakan gaya penulisan naratif deskriptif yang mengalir."),
]

LENGTHS = [
    ("pendek",  "20–30 kata"),
    ("sedang",  "35–45 kata"),
    ("panjang", "45–55 kata"),
]

# ── CELL 5: Helper Functions ─────────────────────────────────
def build_prompt(subtopic, style_instr, length_instr):
    return (
        f"Tulis satu paragraf tentang {subtopic} dalam Bahasa Indonesia.\n\n"
        "Ketentuan:\n"
        f"- Panjang teks: {length_instr}\n"
        f"- {style_instr}\n"
        "- Jangan gunakan bullet points, daftar bernomor, atau heading\n"
        "- Tulis MAKSIMAL 55 kata, tidak lebih\n"
        "- Jangan tambahkan kalimat pembuka seperti \"Tentu,\" atau \"Berikut adalah\"\n"
        "- Langsung tulis paragrafnya saja"
    )

def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt).text.strip()
        except Exception as e:
            print(f"  ⚠ Error attempt {attempt+1}: {e}")
            time.sleep(45)
    return None

def load_existing_ids(filepath):
    ids = set()
    if Path(filepath).exists():
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["prompt_id"])
                except Exception:
                    pass
    return ids

# ── CELL 6: Build Prompt Combinations ───────────────────────
def build_combinations():
    combos = []
    for t in TOPICS:
        for sub in t["subtopics"]:
            for style_name, style_instr in STYLES:
                for length_name, length_instr in LENGTHS:
                    pid = f"{t['topic']}|{sub[:25]}|{style_name}|{length_name}"
                    combos.append({
                        "prompt_id":     pid,
                        "topic":         t["topic"],
                        "subtopic":      sub,
                        "style":         style_name,
                        "length_target": length_name,
                        "prompt":        build_prompt(sub, style_instr, length_instr),
                    })
    random.shuffle(combos)
    return combos

# ── CELL 7: Main Collection Loop ─────────────────────────────
def collect():
    existing_ids = load_existing_ids(OUTPUT_FILE)
    collected    = len(existing_ids)
    combos       = build_combinations()

    print(f"▶ Resume dari {collected} sampel | Target {TARGET} | Output: {OUTPUT_FILE}\n")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for combo in combos:
            if collected >= TARGET:
                break
            if combo["prompt_id"] in existing_ids:
                continue

            print(f"[{collected+1:>3}/{TARGET}] {combo['topic']:22s} | {combo['style']:15s} | {combo['length_target']}")

            text = generate_with_retry(combo["prompt"])
            if text is None:
                print("  ✗ Skipped")
                continue

            record = {
                "sample_id":     f"AI_GEM_{collected+1:04d}",
                "text":          text,
                "label":         "AI",
                "label_int":     1,
                "source_model":  SOURCE_MODEL,
                "topic":         combo["topic"],
                "subtopic":      combo["subtopic"],
                "style":         combo["style"],
                "length_target": combo["length_target"],
                "word_count":    len(text.split()),
                "prompt_id":     combo["prompt_id"],
                "collected_at":  datetime.now().isoformat(),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # simpan langsung, aman dari crash

            existing_ids.add(combo["prompt_id"])
            collected += 1

            if collected % 50 == 0:
                print(f"\n  ✓ Checkpoint: {collected}/{TARGET}\n")

            time.sleep(DELAY)

    print(f"\n✅ Selesai! {collected} sampel → {OUTPUT_FILE}")

collect()

# ── CELL 8: Quick Stats ──────────────────────────────────────
import pandas as pd

records = [json.loads(l) for l in open(OUTPUT_FILE, encoding="utf-8")]
df = pd.DataFrame(records)
print(f"\nTotal  : {len(df)}")
print(f"Avg wc : {df['word_count'].mean():.1f} kata")
print(f"\nTopik  :\n{df['topic'].value_counts()}")
print(f"\nGaya   :\n{df['style'].value_counts()}")