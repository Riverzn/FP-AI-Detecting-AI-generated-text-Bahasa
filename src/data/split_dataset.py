import json
from pathlib import Path

# Setup Path langsung relatif aman
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = REPO_ROOT / "data" / "raw" / "ai_generated_gemini.jsonl"
LLAMA_OUTPUT = REPO_ROOT / "data" / "raw" / "ai_generated_llama.jsonl"
GPTOSS_OUTPUT = REPO_ROOT / "data" / "raw" / "ai_generated_gpt_oss.jsonl"

def split():
    print("✂️ Memulai proses pembelahan dataset master 1000 data...")
    
    if not INPUT_FILE.exists():
        print(f"❌ Eror: File master tidak ditemukan di {INPUT_FILE}")
        return

    llama_count = 0
    gptoss_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(LLAMA_OUTPUT, "w", encoding="utf-8") as f_llama, \
         open(GPTOSS_OUTPUT, "w", encoding="utf-8") as f_gptoss:
         
        for line in f_in:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Cek berdasarkan kolom source_model yang udah kita rekam
                model = data.get("source_model", "").upper()
                
                if "LLAMA" in model:
                    f_llama.write(json.dumps(data, ensure_ascii=False) + "\n")
                    llama_count += 1
                elif "GPT-OSS" in model or "GPTOSS" in model:
                    f_gptoss.write(json.dumps(data, ensure_ascii=False) + "\n")
                    gptoss_count += 1
                else:
                    # Jaga-jaga kalau ada format lain, masukkan ke llama kelompok
                    f_llama.write(json.dumps(data, ensure_ascii=False) + "\n")
                    llama_count += 1
            except Exception as e:
                print(f"⚠ Gagal memproses baris: {e}")

    print("\n⚡ PROSES BELAH DATA SELESAI!")
    print(f"📊 Total Data Llama    -> {llama_count} sampel sukses dipisah!")
    print(f"📊 Total Data GPT-OSS  -> {gptoss_count} sampel sukses dipisah!")
    print(f"📁 Output tersimpan di folder: data/raw/")

if __name__ == "__main__":
    split()