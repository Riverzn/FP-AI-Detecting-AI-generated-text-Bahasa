# ============================================================
# src/models/bilstm.py
# Bidirectional LSTM untuk deteksi AI-generated text Bahasa Indonesia
#
# Cara run di Colab:
#   !pip install torch pandas scikit-learn -q
#   %run src/models/bilstm.py
# ============================================================

import sys, os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# ── Konfigurasi ───────────────────────────────────────────────
HUMAN_DATA_PATH = REPO_ROOT / "data" / "samples" / "indonli_sample.csv"
AI_LLAMA_PATH   = REPO_ROOT / "data" / "raw" / "ai_generated_llama.jsonl"
AI_GPTOSS_PATH  = REPO_ROOT / "data" / "raw" / "ai_generated_gpt_oss.jsonl"

MAX_VOCAB   = 20000
MAX_LEN     = 128
EMBED_DIM   = 128
HIDDEN_DIM  = 256
DROPOUT     = 0.5
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── Tokenizer sederhana ───────────────────────────────────────
def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

class Vocabulary:
    def __init__(self, max_size=MAX_VOCAB):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        for word, _ in counter.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_len=MAX_LEN):
        tokens = tokenize(text)[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)

# ── Dataset ───────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts  = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# ── Model ─────────────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm    = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                                  bidirectional=True, num_layers=2,
                                  dropout=dropout)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, (hn, _) = self.bilstm(emb)
        # Gabung hidden state terakhir dari kedua arah
        last = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.fc(self.dropout(last))

# ── Load data (sama persis dengan baseline.py) ────────────────
def load_data():
    df_human = pd.read_csv(HUMAN_DATA_PATH)
    if "text" not in df_human.columns:
        for col in ["premise", "sentence1"]:
            if col in df_human.columns:
                df_human = df_human.rename(columns={col: "text"})
                break
    df_human = df_human[["text"]].dropna().copy()
    df_human["label"] = 0

    df_llama  = pd.read_json(AI_LLAMA_PATH,  lines=True)
    df_gptoss = pd.read_json(AI_GPTOSS_PATH, lines=True)
    df_ai = pd.concat([df_llama[["text"]], df_gptoss[["text"]]], ignore_index=True).dropna()
    df_ai["label"] = 1

    n = min(len(df_human), len(df_ai))
    df = pd.concat([df_human.sample(n, random_state=42),
                    df_ai.sample(n, random_state=42)], ignore_index=True)
    return df

# ── Training loop ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for texts, labels in loader:
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(texts)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            logits = model(texts)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), correct / len(loader.dataset), all_preds, all_labels

# ── Main ──────────────────────────────────────────────────────
def main():
    print(f"🚀 BiLSTM | device: {DEVICE}")
    df = load_data()
    print(f"📊 Dataset: {len(df)} sampel ({df.label.value_counts().to_dict()})")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    vocab = Vocabulary()
    vocab.build(X_train)
    print(f"📖 Vocab size: {len(vocab)}")

    train_loader = DataLoader(TextDataset(X_train, y_train, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(X_val,   y_val,   vocab), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TextDataset(X_test,  y_test,  vocab), batch_size=BATCH_SIZE)

    model     = BiLSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state   = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step(vl_loss)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} "
              f"| val_loss={vl_loss:.4f} acc={vl_acc:.3f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, test_acc, preds, labels = eval_epoch(model, test_loader, criterion)

    print("\n================ EVALUASI BILSTM ================")
    print(f"🎯 Test Accuracy: {test_acc * 100:.2f}%")
    print(classification_report(labels, preds, target_names=["Human", "AI"]))
    print("=================================================")

if __name__ == "__main__":
    main()
    