# ============================================================
# src/models/indobert.py
# IndoBERT fine-tuning untuk deteksi AI-generated text Bahasa Indonesia
#
# Cara run di Colab (wajib GPU):
#   !pip install transformers datasets torch -q
#   %run src/models/indobert.py
#
# Model: indobenchmark/indobert-base-p2
# ============================================================

import sys, os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# ── Konfigurasi ───────────────────────────────────────────────
HUMAN_DATA_PATH = REPO_ROOT / "data" / "samples" / "indonli_sample.csv"
AI_LLAMA_PATH   = REPO_ROOT / "data" / "raw" / "ai_generated_llama.jsonl"
AI_GPTOSS_PATH  = REPO_ROOT / "data" / "raw" / "ai_generated_gpt_oss.jsonl"

MODEL_NAME  = "indobenchmark/indobert-base-p2"
MAX_LEN     = 128
BATCH_SIZE  = 16     # turunkan ke 8 kalau OOM
EPOCHS      = 5
LR          = 2e-5   # learning rate standar fine-tuning BERT
WARMUP_RATIO = 0.1
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    print("⚠️  Disarankan pakai GPU di Colab (Runtime → Change runtime type → T4 GPU)")

# ── Dataset ───────────────────────────────────────────────────
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

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
    df = pd.concat([
        df_human.sample(n, random_state=42),
        df_ai.sample(n, random_state=42)
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ── Training & eval ───────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct = 0, 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids,
                        attention_mask=attn_mask,
                        labels=labels)
        loss    = outputs.loss
        logits  = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

def eval_epoch(model, loader):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids,
                            attention_mask=attn_mask,
                            labels=labels)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / len(loader.dataset), all_preds, all_labels

# ── Main ──────────────────────────────────────────────────────
def main():
    print("🚀 IndoBERT Fine-tuning")
    df = load_data()
    print(f"📊 Dataset: {len(df)} sampel ({df.label.value_counts().to_dict()})")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"📥 Loading tokenizer & model dari: {MODEL_NAME}")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2
    ).to(DEVICE)

    # Freeze semua layer kecuali 2 terakhir + classifier head
    # → menghindari catastrophic forgetting pada dataset kecil
    for name, param in model.bert.named_parameters():
        if not any(f"encoder.layer.{i}" in name for i in [10, 11]):
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Trainable params: {trainable:,}")

    train_ds = BERTDataset(X_train, y_train, tokenizer)
    val_ds   = BERTDataset(X_val,   y_val,   tokenizer)
    test_ds  = BERTDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_acc, best_state = 0, None
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler)
        vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} "
              f"| val_loss={vl_loss:.4f} acc={vl_acc:.3f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, test_acc, preds, labels = eval_epoch(model, test_loader)

    print("\n================ EVALUASI INDOBERT ================")
    print(f"🎯 Test Accuracy: {test_acc * 100:.2f}%")
    print(classification_report(labels, preds, target_names=["Human", "AI"]))
    print("====================================================")

    # Simpan model (opsional, uncomment kalau mau save)
    # MODEL_SAVE_PATH = REPO_ROOT / "models_saved" / "indobert_checkpoint"
    # model.save_pretrained(MODEL_SAVE_PATH)
    # tokenizer.save_pretrained(MODEL_SAVE_PATH)
    # print(f"💾 Model tersimpan di {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()