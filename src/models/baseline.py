import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score

# Setup root path project
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main():
    print("[INFO] Menjalankan Pipeline Baseline SVM (AI vs Human)...")

    # Path file dataset clean
    TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train_clean.csv"
    VAL_PATH   = REPO_ROOT / "data" / "processed" / "val_clean.csv"
    TEST_PATH  = REPO_ROOT / "data" / "processed" / "test_clean.csv"

    # Load dataset
    df_train = pd.read_csv(TRAIN_PATH)
    df_val   = pd.read_csv(VAL_PATH)
    df_test  = pd.read_csv(TEST_PATH)

    # Print info jumlah sampel dan sebaran kelas
    print(f"[INFO] Train : {len(df_train)} sampel "
          f"({(df_train['label_int']==1).sum()} AI, {(df_train['label_int']==0).sum()} Human)")
    print(f"[INFO] Val   : {len(df_val)} sampel "
          f"({(df_val['label_int']==1).sum()} AI, {(df_val['label_int']==0).sum()} Human)")
    print(f"[INFO] Test  : {len(df_test)} sampel "
          f"({(df_test['label_int']==1).sum()} AI, {(df_test['label_int']==0).sum()} Human)")

    # Pisahkan fitur teks dan label target
    X_train, y_train = df_train['text'].astype(str), df_train['label_int']
    X_val,   y_val   = df_val['text'].astype(str),   df_val['label_int']
    X_test,  y_test  = df_test['text'].astype(str),  df_test['label_int']

    # Definisikan pipeline: TF-IDF Karakter (3-5 n-gram) + Classifier LinearSVC
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
        ('clf', LinearSVC(C=0.05, class_weight='balanced', random_state=42, dual=False, max_iter=2000))
    ])

    # Training model pada data train
    pipeline.fit(X_train, y_train)

    # Evaluasi pada validation set
    y_val_pred = pipeline.predict(X_val)
    val_acc    = accuracy_score(y_val, y_val_pred)
    val_bacc   = balanced_accuracy_score(y_val, y_val_pred)
    print(f"\n[VAL] Accuracy          : {val_acc * 100:.2f}%")
    print(f"[VAL] Balanced Accuracy : {val_bacc * 100:.2f}%")
    print(classification_report(y_val, y_val_pred, target_names=['Human', 'AI']))

    # Evaluasi final pada test set
    y_test_pred = pipeline.predict(X_test)
    test_acc    = accuracy_score(y_test, y_test_pred)
    test_bacc   = balanced_accuracy_score(y_test, y_test_pred)

    print("\n" + "=" * 20 + " LAPORAN EVALUASI BASELINE SVM " + "=" * 20)
    print(f"  Test Accuracy          : {test_acc * 100:.2f}%")
    print(f"  Test Balanced Accuracy : {test_bacc * 100:.2f}%")
    print(f"  Gap Val vs Test        : {abs(val_acc - test_acc):.4f}")
    print("\n  Klasifikasi Detail:")
    print(classification_report(y_test, y_test_pred, target_names=['Human', 'AI']))
    print("=" * 71)

    # Ambil nama fitur TF-IDF dan bobot koefisien SVM
    feature_names = np.array(pipeline.named_steps['tfidf'].get_feature_names_out())
    coef          = pipeline.named_steps['clf'].coef_[0]
    n_top         = 10

    # Print 10 fitur n-gram terkuat untuk masing-masing kelas
    print(f"\n[ANALYSIS] Top {n_top} Fitur AI   : {feature_names[np.argsort(coef)[-n_top:]].tolist()}")
    print(f"[ANALYSIS] Top {n_top} Fitur Human: {feature_names[np.argsort(coef)[:n_top]].tolist()}")


if __name__ == "__main__":
    main()