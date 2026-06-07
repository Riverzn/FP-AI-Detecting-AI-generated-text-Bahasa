from pathlib import Path
import pandas as pd

def main():
    print("🌐 Menghubungkan ke jalur resmi Hugging Face Parquet (Bypass Token)...")
    
    # Setup Path biar akurat nembak folder data/samples/
    REPO_ROOT = Path(__file__).resolve().parents[2]
    OUTPUT_PATH = REPO_ROOT / "data" / "samples" / "indonli_sample.csv"
    
    # Jalur resmi file parquet IndoNLI di server Hugging Face
    URL_RESMI = "https://huggingface.co/datasets/indonli/resolve/refs%2Fconvert%2Fparquet/indonli/train/0000.parquet"
    
    try:
        # Download langsung via pandas tanpa lib datasets, tanpa perlu login token
        df = pd.read_parquet(URL_RESMI)
        
        # IndoNLI aslinya pakai nama kolom 'premise' untuk teks utamanya
        if 'premise' in df.columns:
            df = df.rename(columns={'premise': 'text'})
            
        # Ambil kolom text saja, buang baris kosong atau duplikat
        df_clean = df[['text']].dropna().drop_duplicates().copy()
        
        # Ekspor dan timpa file indonli_sample.csv kelompok kalian
        df_clean.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ SUKSES BESAR, RAY! File indonli_sample.csv berhasil diperbarui jadi {len(df_clean)} data manusia asli!")

    except Exception as e:
        print(f"❌ Gagal mengambil data karena: {e}")
        print("💡 Pastikan koneksi internet lu stabil ya, Ray.")

if __name__ == "__main__":
    main()