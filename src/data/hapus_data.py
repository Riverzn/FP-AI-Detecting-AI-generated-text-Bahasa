import json
from pathlib import Path

# Sesuaikan dengan lokasi dan nama file jsonl lu
FILE_PATH = "data/raw/dataset_ai_baru.jsonl"  # <-- Ganti sesuai nama file lu

# 1. Baca semua data yang ada sekarang
all_records = []
with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            all_records.append(json.loads(line))

# 2. Filter: Buang data yang sample_id nya AI_HYBRID_0017, 0018, dan 0019
ids_to_remove = {"AI_HYBRID_0017", "AI_HYBRID_0018", "AI_HYBRID_0019"}
clean_records = [r for r in all_records if r["sample_id"] not in ids_to_remove]

# 3. Tulis ulang file jsonl dengan data yang udah bersih
with open(FILE_PATH, "w", encoding="utf-8") as f:
    for record in clean_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Selesai! Sampel 0017-0019 udah dihapus. Sisa data aktif: {len(clean_records)}")