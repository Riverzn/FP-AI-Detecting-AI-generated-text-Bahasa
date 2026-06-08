import sys
import os
from pathlib import Path
import json
import time
import random
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

# Setup Path Root Repo
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import PATHS, ensure_dirs

# Load ENV & Inisialisasi Groq
load_dotenv(REPO_ROOT / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"  # Model gratis, kencang, & fasih Bahasa Indonesia
SOURCE_MODEL = "Llama-3.1-8B"
OUTPUT_FILE = PATHS["raw_gemini"]  # Menargetkan file .jsonl yang sama biar nimbrung
TARGET = 1000  # Target total data AI kelompok

ensure_dirs()

# ============================================================
# MATRIKS TOPIK (Disamakan persis dengan format kelompok lu)
# ============================================================
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
    ("pendek",  "sekitar 50–70 kata"),
    ("sedang",  "sekitar 75–100 kata"),
    ("panjang", "sekitar 100–130 kata"),
]

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
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL_NAME,
                temperature=0.7,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠ Error Groq (Attempt {attempt+1}): {e}")
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

def collect():
    existing_ids = load_existing_ids(OUTPUT_FILE)
    collected    = len(existing_ids)
    combos       = build_combinations()

    print(f"▶ Menyambung Data Kelompok | Terdeteksi: {collected} sampel awal | Target Baru: {TARGET}\n")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for combo in combos:
            if collected >= TARGET:
                break
            if combo["prompt_id"] in existing_ids:
                continue

            print(f"[{collected+1:>4}/{TARGET}] {combo['topic']:22s} | {combo['style']:15s} | {combo['length_target']}")

            text = generate_with_retry(combo["prompt"])
            if text is None:
                print("  ✗ Skipped")
                continue

            record = {
                "sample_id":     f"AI_LLM_{collected+1:04d}",
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
            
            # Jeda aman dari rate limit gratisan Groq
            time.sleep(1.5)

    print(f"\n✅ Selesai Total! {collected} sampel terkumpul di -> {OUTPUT_FILE}")

if __name__ == "__main__":
    collect()