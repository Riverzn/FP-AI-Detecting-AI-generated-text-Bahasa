import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def truncate_at_word(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0]


def load_and_rename(path: Path, lines: bool = True) -> pd.DataFrame:
    df = pd.read_json(path, lines=lines) if lines else pd.read_json(path)
    for col in ['text_human', 'sentence1', 'premise', 'sentence', 'text']:
        if col in df.columns:
            return df.rename(columns={col: 'text'})
    raise ValueError(f"Tidak ada kolom teks yang dikenali di {path}")


def safe_sample(df: pd.DataFrame, n: int, label: int, source_name: str, random_state: int = 42) -> pd.DataFrame:
    df = df[['text']].dropna().copy()
    df['text'] = df['text'].apply(clean_text)
    
    # Menghapus string kosong/corrupt di bawah 10 karakter
    df = df[df['text'].str.len() > 10]

    if len(df) < n:
        n = len(df)

    df = df.sample(n=n, random_state=random_state).reset_index(drop=True)
    df['label'] = label
    df['source'] = source_name
    return df


def main():
    print("[INFO] Menjalankan Pipeline Baseline SVM (AI vs Human)...")

    HUMAN_INDONLI_PATH = REPO_ROOT / "data" / "raw" / "indonli_human.jsonl"
    HUMAN_SCRAPED_PATH = REPO_ROOT / "data" / "raw" / "human_scraped_300.json"
    AI_LLAMA_PATH      = REPO_ROOT / "data" / "raw" / "ai_generated_llama.jsonl"
    AI_GPTOSS_PATH     = REPO_ROOT / "data" / "raw" / "ai_generated_gpt_oss.jsonl"

    df_indonli = load_and_rename(HUMAN_INDONLI_PATH, lines=True)
    df_scraped  = load_and_rename(HUMAN_SCRAPED_PATH, lines=False)
    df_llama    = load_and_rename(AI_LLAMA_PATH,      lines=True)
    df_gptoss   = load_and_rename(AI_GPTOSS_PATH,     lines=True)

    # Stratified sampling untuk menyeimbangkan sub-sumber data Human
    df_h1 = safe_sample(df_indonli, 700, label=0, source_name='indonli')
    df_h2 = safe_sample(df_scraped, 300, label=0, source_name='scraped')
    df_human = pd.concat([df_h1, df_h2], ignore_index=True)

    df_ai_pool = pd.concat([df_llama[['text']], df_gptoss[['text']]], ignore_index=True)
    df_ai = safe_sample(df_ai_pool, 1000, label=1, source_name='ai_combined')

    # Truncation berbasis median panjang data Human untuk mitigasi length leakage
    human_median = int(pd.concat([df_h1['text'], df_h2['text']]).str.len().median())
    TRUNC_LEN = max(human_median, 100)

    df_human['text'] = df_human['text'].apply(lambda t: truncate_at_word(t, TRUNC_LEN))
    df_ai['text']    = df_ai['text'].apply(lambda t: truncate_at_word(t, TRUNC_LEN))

    # Pembersihan pasca-truncation dan downsampling akhir untuk menjamin proporsi 50:50
    MIN_LEN = 50
    df_human = df_human[df_human['text'].str.len() >= MIN_LEN].reset_index(drop=True)
    df_ai    = df_ai[df_ai['text'].str.len() >= MIN_LEN].reset_index(drop=True)
    
    n_final  = min(len(df_human), len(df_ai))
    df_human = df_human.sample(n=n_final, random_state=42)
    df_ai    = df_ai.sample(n=n_final, random_state=42)

    df_final = pd.concat([df_human, df_ai], ignore_index=True).sample(
        frac=1, random_state=42).reset_index(drop=True)
    
    print(f"[INFO] Dataset Final Terbentuk: {len(df_final)} sampel ({n_final} Human, {n_final} AI)")

    X_train, X_test, y_train, y_test = train_test_split(
        df_final['text'], df_final['label'],
        test_size=0.2, random_state=42, stratify=df_final['label']
    )

    # Arsitektur Pipeline: Character-level TF-IDF + LinearSVC Regularisasi Ketat
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=300,
            max_df=0.80,
            min_df=3,
            lowercase=True,
            sublinear_tf=True,
        )),
        ('clf', LinearSVC(C=0.05, random_state=42, dual=False, max_iter=2000))
    ])

    print("[INFO] Mengevaluasi model dengan 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"  CV Scores per Fold : {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  CV Mean Accuracy  : {cv_scores.mean() * 100:.2f}% (± {cv_scores.std() * 100:.2f}%)")

    # Uji coba pada Hold-out Test Set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*20 + " LAPORAN EVALUASI BASELINE SVM " + "="*20)
    print(f"🎯 Test Set Accuracy : {acc * 100:.2f}%")
    print(f"📈 Gap CV vs Test    : {abs(cv_scores.mean() - acc):.4f}")
    print("\n📋 Klasifikasi Detail:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    print("=" * 71)

    # Analisis Fitur Diskriminatif (Stylometry)
    feature_names = np.array(pipeline.named_steps['tfidf'].get_feature_names_out())
    coef = pipeline.named_steps['clf'].coef_[0]
    n_top = 10

    print(f"\n[ANALYSIS] Top {n_top} Fitur AI   : {feature_names[np.argsort(coef)[-n_top:]].tolist()}")
    print(f"[ANALYSIS] Top {n_top} Fitur Human: {feature_names[np.argsort(coef)[:n_top]].tolist()}")


if __name__ == "__main__":
    main()