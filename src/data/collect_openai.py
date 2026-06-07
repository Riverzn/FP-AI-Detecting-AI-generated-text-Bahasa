# ============================================================
# src/data/collect_openai.py
# Pengumpulan AI-generated samples via OpenAI API
# Topik 5–8 | Target: 500 sampel
#
# Lokasi di repo : src/data/collect_openai.py
# Output data    : data/raw/ai_generated_openai.jsonl  (→ Drive, gitignored)
#
# Cara run di Colab:
#   !git clone https://github.com/[user]/ai-text-detection-id
#   %cd ai-text-detection-id
#   !pip install openai python-dotenv -q
#   from google.colab import drive; drive.mount('/content/drive')
#   %run src/data/collect_openai.py
# ============================================================

# ── CELL 1: Install & Mount ─────────────────────────────────
# !pip install openai python-dotenv -q
# from google.colab import drive; drive.mount('/content/drive')

# ── CELL 2: Import & Path Setup ─────────────────────────────
import sys, os
from pathlib import Path

REPO_ROOT = Path("/content/FP-AI-Detecting-AI-generated-text-Bahasa") if os.path.exists("/content") else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import PATHS, ensure_dirs
from openai import OpenAI
import json, time, random
from datetime import datetime

# ── CELL 3: Konfigurasi ─────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

API_KEY      = os.getenv("API_KEY_OPENAI")
MODEL_NAME   = "gpt-4o-mini"        # murah: estimasi ~$0.02–0.05 untuk 500 sampel
SOURCE_MODEL = "GPT-4o-mini"
OUTPUT_FILE  = PATHS["raw_openai"]  # → data/raw/ai_generated_openai.jsonl (Drive)
TARGET       = 500
DELAY        = 1.0                  # OpenAI tier lebih longgar

ensure_dirs()
client = OpenAI(api_key=API_KEY)
print(f"Output → {OUTPUT_FILE}")

# ── CELL 4: Topic–Subtopic Matrix (Topik 5–8 untuk OpenAI) ──
TOPICS = [
    {
        "topic": "Ekonomi dan Bisnis",
        "subtopics": [
            "pertumbuhan ekonomi Indonesia dan faktor pendorongnya",
            "peran UMKM dalam perekonomian nasional",
            "perkembangan pasar modal Indonesia",
            "perdagangan internasional Indonesia dalam era globalisasi",
            "transformasi layanan perbankan melalui teknologi fintech",
        ],
    },
    {
        "topic": "Teknologi",
        "subtopics": [
            "penerapan kecerdasan buatan dalam kehidupan sehari-hari",
            "ancaman keamanan siber dan cara mengatasinya",
            "perkembangan Internet of Things di Indonesia",
            "potensi energi surya sebagai energi terbarukan",
            "transformasi digital sektor pemerintahan Indonesia",
        ],
    },
    {
        "topic": "Sosial dan Pendidikan",
        "subtopics": [
            "ketimpangan sosial ekonomi di perkotaan dan pedesaan",
            "tantangan sistem pendidikan Indonesia di era digital",
            "masalah ketenagakerjaan dan pengangguran muda",
            "urbanisasi dan dampaknya terhadap kota besar",
            "kesetaraan gender dalam dunia kerja di Indonesia",
        ],
    },
    {
        "topic": "Lingkungan",
        "subtopics": [
            "deforestasi hutan Kalimantan dan dampak ekologisnya",
            "pengelolaan sampah plastik di wilayah perkotaan",
            "polusi udara dan dampaknya terhadap kesehatan masyarakat",
            "konservasi satwa langka di Indonesia",
            "pertanian organik sebagai pendekatan pertanian berkelanjutan",
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
    ("pendek",  "sekitar 50–70 kata"),
    ("sedang",  "sekitar 75–100 kata"),
    ("panjang", "sekitar 100–130 kata"),
]

# ── CELL 5: Helper Functions ─────────────────────────────────
def build_prompt(subtopic, style_instr, length_instr):
    return (
        f"Tulis satu paragraf tentang {subtopic} dalam Bahasa Indonesia.\n\n"
        "Ketentuan:\n"
        f"- Panjang teks: {length_instr}\n"
        f"- {style_instr}\n"
        "- Jangan gunakan bullet points, daftar bernomor, atau heading\n"
        "- Jangan tambahkan kalimat pembuka seperti \"Tentu,\" atau \"Berikut adalah\"\n"
        "- Langsung tulis paragrafnya saja"
    )

def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.8,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠ Error attempt {attempt+1}: {e}")
            time.sleep(10)
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

            print(f"[{collected+1:>3}/{TARGET}] {combo['topic']:25s} | {combo['style']:15s} | {combo['length_target']}")

            text = generate_with_retry(combo["prompt"])
            if text is None:
                print("  ✗ Skipped")
                continue

            record = {
                "sample_id":     f"AI_OAI_{collected+1:04d}",
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
            f.flush()

            existing_ids.add(combo["prompt_id"])
            collected += 1

            if collected % 50 == 0:
                print(f"\n  ✓ Checkpoint: {collected}/{TARGET}\n")

            time.sleep(DELAY)

    print(f"\n✅ Selesai! {collected} sampel → {OUTPUT_FILE}")

collect()