import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Setup Path Root Repo
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

def main():
    print("🚀 [BASELINE - PEMBATASAN EKSTREM] Memulai Eksperimen Anti-Overfit...")

    # 1. PATH DATASET
    HUMAN_DATA_PATH = REPO_ROOT / "data" / "samples" / "indonli_sample.csv"
    AI_LLAMA_PATH = REPO_ROOT / "data" / "raw" / "ai_generated_llama.jsonl"
    AI_GPTOSS_PATH = REPO_ROOT / "data" / "raw" / "ai_generated_gpt_oss.jsonl"

    # 2. LOAD DATA MANUSIA
    df_human = pd.read_csv(HUMAN_DATA_PATH)
    if 'text' not in df_human.columns:
        for col in ['sentence1', 'premise', 'text']:
            if col in df_human.columns:
                df_human = df_human.rename(columns={col: 'text'})
                break
    df_human = df_human[['text']].dropna().copy()
    df_human['label'] = 0

    # 3. LOAD DATA AI
    df_llama = pd.read_json(AI_LLAMA_PATH, lines=True)
    df_gptoss = pd.read_json(AI_GPTOSS_PATH, lines=True)
    df_ai_combined = pd.concat([df_llama[['text']], df_gptoss[['text']]], ignore_index=True).dropna()
    df_ai_combined['label'] = 1

    # Normalisasi Panjang Karakter (Potong pendek banget biar adil)
    df_human['text'] = df_human['text'].astype(str).str.slice(0, 150)
    df_ai_combined['text'] = df_ai_combined['text'].astype(str).str.slice(0, 150)

    # 4. BALANCED SAMPLING (1000 VS 1000)
    n_samples = min(len(df_human), len(df_ai_combined))
    df_human_balanced = df_human.sample(n=n_samples, random_state=42)
    df_ai_balanced = df_ai_combined.sample(n=n_samples, random_state=42)
    df_final = pd.concat([df_human_balanced, df_ai_balanced], ignore_index=True)
    print(f"📊 Total Dataset: {len(df_final)} sampel ({n_samples} Human, {n_samples} AI)")

    # 5. SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        df_final['text'], df_final['label'], test_size=0.2, random_state=42, stratify=df_final['label']
    )

    # 🛠️ STRATEGI BARU: Stopwords Bahasa Indonesia Manual (Buat buang kata pemicu overfit)
    # Kita buang kata hubung dan kata pemicu bias topik IndoNLI (pria, wanita, anak, dll)
    stopwords_id = [
        'yang', 'di', 'dan', 'itu', 'dengan', 'untuk', 'tidak', 'ini', 'dari', 'dalam',
        'akan', 'pada', 'juga', 'ke', 'karena', 'bisa', 'ada', 'mereka', 'sebuah', 'atau',
        'pria', 'wanita', 'anak', 'orang', 'seorang', 'sedang', 'dua', 'tiga', 'sambil', 'berdiri'
    ]

   # 6. FEATURE ENGINEERING: TF-IDF UNTUK GAYA BAHASA (STYLETOMETRY)
    print("✨ Mengekstrak fitur gaya bahasa (Membuang kata topik)...")
    
    # Kita balik logikanya: gunakan max_df dan min_df yang ekstrem
    # untuk membuang kata-kata spesifik topik yang bikin overfit
    vectorizer = TfidfVectorizer(
        max_features=100,       # Batasi cuma 100 fitur teratas (biar cuma dapet kata hubung umum)
        ngram_range=(1, 2),     # Ambil frasa 1-2 kata biar dapet kata transisi kayak "oleh karena"
        lowercase=True,
        min_df=20,              # Kata yang terlalu jarang (kata benda spesifik) pasti kebuang
        max_df=0.6              # Kata yang terlalu dominan juga kebuang
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 7. TRAINING MODEL DENGAN REGULARISASI PALING PENAKUT (C=0.005)
    # Makin kecil C, model makin dipaksa membagi bobot koefisiennya secara rata (ga boleh condong ke 1-2 kata)
    model = LogisticRegression(C=0.005, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # 8. EVALUASI
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    print("\n================ EVALUASI MODEL BASELINE ================")
    print(f"🎯 Akurasi Akhir: {acc * 100:.2f}%")
    print("\n📋 Laporan Klasifikasi DETAIL:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    print("=========================================================\n")

    # CETAK 5 KATA KUNCI TERKUAT MASING-MASING KELAS
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    
    if len(feature_names) >= 5:
        print("🔍 5 Kata terkuat penentu AI:")
        top_ai = np.argsort(coef)[-5:]
        print("👉 AI:", feature_names[top_ai])
        
        print("\n🔍 5 Kata terkuat penentu Human:")
        top_human = np.argsort(coef)[:5]
        print("👉 Human:", feature_names[top_human])

if __name__ == "__main__":
    main()