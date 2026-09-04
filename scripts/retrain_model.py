import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
from scipy.optimize import minimize_scalar

# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYER1_PATH = os.path.join(BASE_DIR, 'dataset', 'layer1_claudette', 'unfair_tos.csv')
MERGED_DIR = os.path.join(BASE_DIR, 'dataset', 'merged_training_set')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'backend', 'models', 'screening_classifier')

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LEN = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[+] Using device: {DEVICE}")

# PyTorch Dataset Definition
class ClauseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

def map_label(lbl):
    if str(lbl).strip() in ['High-risk', 'Medium-risk', '1']:
        return 1
    return 0

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_model(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return logits, labels

def main():
    print(f"[+] Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # -------------------------------------------------------------
    # STAGE 1: Pre-training on Layer 1 (CLAUDETTE unfair TOS dataset)
    # -------------------------------------------------------------
    print("\n==================================================")
    print("[+] STAGE 1: Pre-training on Layer 1 (CLAUDETTE Dataset)")
    print("==================================================")
    
    df_l1 = pd.read_csv(LAYER1_PATH)
    texts_l1 = df_l1['Clause / Finding Description'].values
    labels_l1 = [map_label(l) for l in df_l1['Label'].values]

    # 85/15 train/val split for Stage 1
    split_idx = int(0.85 * len(texts_l1))
    train_ds_l1 = ClauseDataset(texts_l1[:split_idx], labels_l1[:split_idx], tokenizer, MAX_LEN)
    val_ds_l1 = ClauseDataset(texts_l1[split_idx:], labels_l1[split_idx:], tokenizer, MAX_LEN)

    train_loader_l1 = DataLoader(train_ds_l1, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_l1 = DataLoader(val_ds_l1, batch_size=BATCH_SIZE, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    
    stage1_epochs = 3
    for epoch in range(stage1_epochs):
        loss = train_epoch(model, train_loader_l1, optimizer, None, DEVICE)
        logits, targets = eval_model(model, val_loader_l1, DEVICE)
        preds = np.argmax(logits, axis=1)
        f1 = f1_score(targets, preds, average='binary', zero_division=0)
        print(f"    Epoch {epoch+1}/{stage1_epochs} - Stage 1 Train Loss: {loss:.4f} | Stage 1 Val F1: {f1:.4f}")

    # -------------------------------------------------------------
    # STAGE 2: Fine-tuning on Merged Layer 2+3+4 Dataset
    # -------------------------------------------------------------
    print("\n==================================================")
    print("[+] STAGE 2: Fine-tuning on Merged Layer 2+3+4 Dataset")
    print("==================================================")

    train_df = pd.read_csv(os.path.join(MERGED_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(MERGED_DIR, 'val.csv'))

    # Prioritize/Filter Layer 2, 3, 4 for Stage 2 domain adaptation
    # We use all training data but ensure Layer 2, 3, 4 are present and fine-tuned
    train_l234 = train_df[train_df['layer'].isin([2, 3, 4])]
    val_l234 = val_df[val_df['layer'].isin([2, 3, 4])]

    print(f"    - Stage 2 Domain Train Rows (Layers 2, 3, 4): {len(train_l234)}")
    print(f"    - Stage 2 Domain Val Rows (Layers 2, 3, 4):   {len(val_l234)}")

    train_ds_l234 = ClauseDataset(train_l234['text'].values, [map_label(l) for l in train_l234['label'].values], tokenizer, MAX_LEN)
    val_ds_l234 = ClauseDataset(val_l234['text'].values, [map_label(l) for l in val_l234['label'].values], tokenizer, MAX_LEN)

    train_loader_l234 = DataLoader(train_ds_l234, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_l234 = DataLoader(val_ds_l234, batch_size=BATCH_SIZE, shuffle=False)

    optimizer_stage2 = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    best_f1 = 0.0
    best_model_state = None
    stage2_epochs = 4

    for epoch in range(stage2_epochs):
        loss = train_epoch(model, train_loader_l234, optimizer_stage2, None, DEVICE)
        logits, targets = eval_model(model, val_loader_l234, DEVICE)
        preds = np.argmax(logits, axis=1)
        f1 = f1_score(targets, preds, average='binary', zero_division=0)
        prec = precision_score(targets, preds, average='binary', zero_division=0)
        rec = recall_score(targets, preds, average='binary', zero_division=0)
        print(f"    Epoch {epoch+1}/{stage2_epochs} - Loss: {loss:.4f} | Val F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

        if f1 >= best_f1:
            best_f1 = f1
            best_model_state = model.state_dict().copy()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # -------------------------------------------------------------
    # STAGE 3: Temperature Scaling Confidence Calibration
    # -------------------------------------------------------------
    print("\n==================================================")
    print("[+] STAGE 3: Temperature Scaling Confidence Calibration")
    print("==================================================")

    # Get raw validation logits on full validation set
    full_val_ds = ClauseDataset(val_df['text'].values, [map_label(l) for l in val_df['label'].values], tokenizer, MAX_LEN)
    full_val_loader = DataLoader(full_val_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_logits, val_labels = eval_model(model, full_val_loader, DEVICE)

    def nll_loss_func(T):
        scaled_logits = val_logits / T
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0 - eps)
        loss = -np.mean(np.log(probs[np.arange(len(val_labels)), val_labels]))
        return loss

    res = minimize_scalar(nll_loss_func, bounds=(0.05, 5.0), method='bounded')
    calibrated_temperature = float(res.x)
    print(f"    - Optimized Temperature Parameter T*: {calibrated_temperature:.4f}")

    # Calibrated probabilities
    scaled_logits = val_logits / calibrated_temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    calibrated_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    high_risk_probs = calibrated_probs[:, 1]

    # -------------------------------------------------------------
    # STAGE 4: Threshold Selection from Precision-Recall Curve
    # -------------------------------------------------------------
    print("\n==================================================")
    print("[+] STAGE 4: Threshold Selection from Precision-Recall Curve")
    print("==================================================")

    precisions, recalls, thresholds = precision_recall_curve(val_labels, high_risk_probs)

    # Target high recall on High-risk class (target >= 90% recall, max F1)
    best_thresh = 0.5
    best_score = -1.0
    chosen_rec = 0.0
    chosen_prec = 0.0
    chosen_f1 = 0.0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        # Calculate F1 score at threshold t
        f1_at_t = (2 * p * r) / (p + r + 1e-8)
        # Prioritize recall >= 0.90 while maximizing F1
        if r >= 0.90:
            if f1_at_t > best_score:
                best_score = f1_at_t
                best_thresh = float(t)
                chosen_rec = float(r)
                chosen_prec = float(p)
                chosen_f1 = float(f1_at_t)

    # Fallback if no threshold gave >= 0.90 recall
    if best_score < 0:
        for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
            f1_at_t = (2 * p * r) / (p + r + 1e-8)
            if f1_at_t > best_score:
                best_score = f1_at_t
                best_thresh = float(t)
                chosen_rec = float(r)
                chosen_prec = float(p)
                chosen_f1 = float(f1_at_t)

    print(f"    - Chosen Escalation Threshold: {best_thresh:.4f}")
    print(f"    - Validation F1 at Threshold:   {chosen_f1:.4f}")
    print(f"    - Validation Recall:            {chosen_rec:.4f}")
    print(f"    - Validation Precision:         {chosen_prec:.4f}")
    print(f"    - Rationale: Threshold {best_thresh:.4f} prioritizes High-risk recall ({chosen_rec*100:.1f}%) to prevent false negatives on predatory terms while maintaining optimal F1 ({chosen_f1:.4f}).")

    # -------------------------------------------------------------
    # STAGE 5: Save Model Checkpoint, Tokenizer & Calibration Params
    # -------------------------------------------------------------
    print("\n==================================================")
    print(f"[+] Saving model, tokenizer, and calibration params to {MODEL_SAVE_DIR}...")
    print("==================================================")

    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    calibration_params = {
        "model_name": MODEL_NAME,
        "temperature": calibrated_temperature,
        "escalation_threshold": best_thresh,
        "val_f1": chosen_f1,
        "val_recall": chosen_rec,
        "val_precision": chosen_prec,
        "target_class": "High-risk",
        "rationale": f"Threshold {best_thresh:.4f} selected from Precision-Recall curve to achieve {chosen_rec*100:.1f}% recall on High-risk clauses with {chosen_f1:.4f} F1."
    }

    with open(os.path.join(MODEL_SAVE_DIR, 'calibration_params.json'), 'w') as f:
        json.dump(calibration_params, f, indent=2)

    print(f"[+] Model checkpoint successfully saved at {MODEL_SAVE_DIR}")
    print(f"[+] Final Validation F1: {chosen_f1:.4f}")
    print(f"[+] Chosen Threshold:  {best_thresh:.4f}")

if __name__ == '__main__':
    main()
