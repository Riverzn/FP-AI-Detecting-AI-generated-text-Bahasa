# ============================================================
# src/models/cnn_bilstm.py
# CNN + BiLSTM Hybrid untuk deteksi AI-generated text
# CNN menangkap pola lokal (n-gram khas), BiLSTM tangkap konteks panjang
# ============================================================

import sys, os, re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# ── Konfigurasi ───────────────────────────────────────────────
HUMAN_DATA_PATH = REPO_ROOT / "data" / "samples" / "indonli_sample.csv"
AI_LLAMA_PATH   = REPO_ROOT / "data" / "raw" / "ai_generated_llama.jsonl"
AI_GPTOSS_PATH  = REPO_ROOT / "data" / "raw" / "ai_generated_gpt_oss.jsonl"

MAX_VOCAB    = 20000
MAX_LEN      = 128
EMBED_DIM    = 128
CNN_FILTERS  = 128
KERNEL_SIZES = [3, 5, 7]   # tangkap bigram, trigram, 4-gram
HIDDEN_DIM   = 128
DROPOUT      = 0.4
BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Vocabulary (sama seperti bilstm.py) ───────────────────────
def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

class Vocabulary:
    def __init__(self, max_size=MAX_VOCAB):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        for word, _ in counter.most_common(self.max_size - 2):
            self.word2idx[word] = len(self.word2idx)

    def encode(self, text, max_len=MAX_LEN):
        tokens = tokenize(text)[:max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]
        ids   += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.X = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ── Model ─────────────────────────────────────────────────────
class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters,
                 kernel_sizes, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)

        # CNN branch: satu Conv per kernel size, lalu max-over-time pooling
        self.convs = nn.ModuleList([
    nn.Conv1d(embed_dim, num_filters, k, padding=(k - 1) // 2)
    for k in kernel_sizes
])

        # BiLSTM branch: input = concat output CNN
        cnn_out_dim = num_filters * len(kernel_sizes)
        self.bilstm = nn.LSTM(
            cnn_out_dim, hidden_dim,
            batch_first=True, bidirectional=True,
            num_layers=1
        )

        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.dropout(self.embedding(x))          # (batch, seq, embed)
        emb_t = emb.permute(0, 2, 1)                   # (batch, embed, seq) — untuk Conv1d

        # CNN: tangkap fitur lokal per kernel size
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb_t))                    # (batch, filters, seq)
            conv_outs.append(c)

        # Concat semua output CNN → (batch, filters*n_kernels, seq)
        cnn_out = torch.cat(conv_outs, dim=1)
        cnn_out = cnn_out.permute(0, 2, 1)             # (batch, seq, filters*n_kernels)
        cnn_out = self.dropout(cnn_out)

        # BiLSTM: tangkap konteks sekuensial
        lstm_out, (hn, _) = self.bilstm(cnn_out)
        last = torch.cat([hn[-2], hn[-1]], dim=1)      # (batch, hidden*2)

        return self.fc(self.dropout(last))

# ── Load data ─────────────────────────────────────────────────
def load_data():
    df_human = pd.read_csv(HUMAN_DATA_PATH)
    if "text" not in df_human.columns:
        for col in ["premise", "sentence1"]:
            if col in df_human.columns:
                df_human = df_human.rename(columns={col: "text"})
                break
    df_human = df_human[["text"]].dropna().copy()
    df_human["label"] = 0

    df_ai = pd.concat([
        pd.read_json(AI_LLAMA_PATH,  lines=True)[["text"]],
        pd.read_json(AI_GPTOSS_PATH, lines=True)[["text"]]
    ], ignore_index=True).dropna()
    df_ai["label"] = 1

    n  = min(len(df_human), len(df_ai))
    df = pd.concat([df_human.sample(n, random_state=42),
                    df_ai.sample(n, random_state=42)], ignore_index=True)
    return df

# ── Training & eval loop ──────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if train:
                optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / len(loader.dataset), all_preds, all_labels

# ── Main ──────────────────────────────────────────────────────
def main():
    print(f"🚀 CNN+BiLSTM | device: {DEVICE}")
    df = load_data()
    print(f"📊 Dataset: {len(df)} sampel")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    vocab = Vocabulary()
    vocab.build(X_train)

    train_loader = DataLoader(TextDataset(X_train, y_train, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(X_val,   y_val,   vocab), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TextDataset(X_test,  y_test,  vocab), batch_size=BATCH_SIZE)

    model     = CNNBiLSTMClassifier(len(vocab), EMBED_DIM, CNN_FILTERS,
                                     KERNEL_SIZES, HIDDEN_DIM, DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = 0, None
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, optimizer, criterion, train=True)
        vl_loss, vl_acc, _, _ = run_epoch(model, val_loader,   optimizer, criterion, train=False)
        scheduler.step()
        print(f"Epoch {epoch:02d} | train={tr_acc:.3f} | val={vl_acc:.3f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, test_acc, preds, labels = run_epoch(model, test_loader, optimizer, criterion, train=False)

    print("\n================ EVALUASI CNN+BiLSTM ================")
    print(f"🎯 Test Accuracy: {test_acc * 100:.2f}%")
    print(classification_report(labels, preds, target_names=["Human", "AI"]))
    print("======================================================")

if __name__ == "__main__":
    main()