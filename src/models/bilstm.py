
import sys, os, re
from pathlib import Path 
import pandas as pd #csv
import numpy as np #sklearn
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report #evaluasi
from collections import Counter #vocab counting
import torch #deep learning
import torch.nn as nn #lstm
from torch.utils.data import Dataset, DataLoader #dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# hyperparameter 
PROCESSED_DIR  = REPO_ROOT / "data" / "processed" 
TRAIN_PATH     = PROCESSED_DIR / "train_clean.csv"   
VAL_PATH       = PROCESSED_DIR / "val_clean.csv"
TEST_PATH      = PROCESSED_DIR / "test_clean.csv"

MAX_VOCAB   = 10000   # dari 20000 — paksa model fokus ke kata umum
MAX_LEN     = 100     # dari 128 — hindari model baca "signature" panjang
EMBED_DIM   = 64      # dari 128 — model lebih kecil, lebih cepat
HIDDEN_DIM  = 128     # dari 256 -- cukup untuk pola sederhana
DROPOUT     = 0.6     
BATCH_SIZE  = 64      
EPOCHS      = 20
LR          = 5e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Augmentasi teks sederhana
import random

def augment_text(text, p=0.15): #teks secara acak dihapus 15% 
    """Random word dropout — paksa model tidak bergantung kata spesifik."""
    words = text.split()
    if len(words) < 5:
        return text
    words = [w for w in words if random.random() > p]
    return " ".join(words) if words else text

# Tokenizer
def tokenize(text): 
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split() #text split gimana sih

class Vocabulary:
    def __init__(self, max_size=MAX_VOCAB):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1} # memetakan kata ke angka

    def build(self, texts):  #Bangun vocab dari kumpulan teks, simpan hanya kata yang paling umum
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        # Buang kata yang terlalu jarang (< 3x) — hindari overfit ke rare word
        for word, count in counter.most_common(self.max_size - 2):
            if count < 3:
                break
            self.word2idx[word] = len(self.word2idx)

    def encode(self, text, max_len=MAX_LEN): #encoding dengan truncation dan padding
        tokens = tokenize(text)[:max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]
        ids   += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)

# Dataset
class TextDataset(Dataset): #Dataset PyTorch untuk teks, dengan opsi augmentasi
    def __init__(self, texts, labels, vocab, augment=False):
        self.vocab   = vocab
        self.augment = augment
        self.raw     = texts
        self.labels  = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.raw[idx]
        if self.augment:
            text = augment_text(text)
        return torch.tensor(self.vocab.encode(text), dtype=torch.long), self.labels[idx]

# Model
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0) 
        self.embed_drop = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            embed_dim, hidden_dim,
            batch_first=True, bidirectional=True,
            num_layers=2, dropout=dropout
        )

        # Mean pooling + max pooling → concat, lebih robust dari ambil hidden terakhir
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_dim * 4, hidden_dim)  # *4 karena bi + mean+max
        self.fc2     = nn.Linear(hidden_dim, 2) #layer output untuk klasifikasi 2 kelas
        self.relu    = nn.ReLU() 

    def forward(self, x):
        emb = self.embed_drop(self.embedding(x))          # (B, L, E)
        out, _ = self.bilstm(emb)                         # (B, L, H*2)

        # Gabung mean pooling dan max pooling 
        mask     = (x != 0).unsqueeze(-1).float()   
        mean_out = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        max_out  = out.masked_fill(mask == 0, -1e9).max(1).values

        pooled = torch.cat([mean_out, max_out], dim=1)   # (B, H*4)
        out    = self.relu(self.fc1(self.dropout(pooled)))
        return self.fc2(self.dropout(out))
 
# Load data
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    # Pastikan kolom konsisten
    for df in [train_df, val_df, test_df]:
        assert "text"  in df.columns, f"Kolom 'text' tidak ditemukan: {df.columns.tolist()}"
        assert "label" in df.columns, f"Kolom 'label' tidak ditemukan: {df.columns.tolist()}"

    train_df = train_df[["text", "label"]].dropna()
    val_df   = val_df[["text",   "label"]].dropna()
    test_df  = test_df[["text",  "label"]].dropna()

    label_map = {"human": 0, "AI": 1} #labeling ulang ke angka untuk klasifikasi
    for df in [train_df, val_df, test_df]:
        df["label"] = df["label"].map(label_map)
        # Cek kalau ada label aneh yang tidak terpetakan
        if df["label"].isna().any():
            unique = df["label"].unique().tolist()
            raise ValueError(f"Label tidak dikenal: {unique}. Sesuaikan label_map.")

    print(f"📊 Train : {len(train_df)} sampel {train_df.label.value_counts().to_dict()}")
    print(f"📊 Val   : {len(val_df)}   sampel {val_df.label.value_counts().to_dict()}")
    print(f"📊 Test  : {len(test_df)}  sampel {test_df.label.value_counts().to_dict()}")

    return train_df, val_df, test_df

# Training
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss   = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / len(loader.dataset), all_preds, all_labels

# Main
def main():
    print(f"🚀 BiLSTM | device: {DEVICE}")

    train_df, val_df, test_df = load_data()

    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_val,   y_val   = val_df["text"].tolist(),   val_df["label"].tolist()
    X_test,  y_test  = test_df["text"].tolist(),  test_df["label"].tolist()

    vocab = Vocabulary()
    vocab.build(X_train)
    print(f"📖 Vocab size: {len(vocab)}")

    train_loader = DataLoader(
        TextDataset(X_train, y_train, vocab, augment=True),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader  = DataLoader(TextDataset(X_val,  y_val,  vocab), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TextDataset(X_test, y_test, vocab), batch_size=BATCH_SIZE)


    model     = BiLSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3) #awalnya 1e-3
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing — hindari terlalu confident

    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0
    PATIENCE      = 5   # early stopping

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>9} | {'Val Loss':>8} {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc          = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _   = eval_epoch(model, val_loader, criterion)
        scheduler.step(vl_loss)

        gap = tr_acc - vl_acc
        flag = " ⚠️Overfit" if gap > 0.08 else ""
        print(f"{epoch:>5} | {tr_loss:>10.4f} {tr_acc:>9.3f} | {vl_loss:>8.4f} {vl_acc:>7.3f}{flag}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n⏹ Early stopping di epoch {epoch}")
                break

    model.load_state_dict(best_state)
    _, test_acc, preds, labels = eval_epoch(model, test_loader, criterion)

    print("\n================ EVALUASI BILSTM ================")
    print(f"🎯 Test Accuracy: {test_acc * 100:.2f}%")
    print(f"📉 Train-Val gap terakhir: {tr_acc - vl_acc:.3f} (ideal < 0.05)")
    print(classification_report(labels, preds, target_names=["Human", "AI"]))
    print("=================================================")

if __name__ == "__main__":
    main()