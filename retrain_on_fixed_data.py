"""
RETRAIN ON CORRECTED DATASET — RF + LSTM, portable paths, WITH NOISE
INJECTION (v2 — corrects a mistake in the first version of this script)

CORRECTION FROM PREVIOUS VERSION: the first version of this script
trained directly on the seed-fixed full_dataset.csv without applying
noise injection, which produced literal 100% accuracy/recall/precision
on RF and LSTM alike. Root cause: raw synthetic attack_rate features
(asn_attack_rate, country_attack_rate) have a complete, non-overlapping
gap versus real normal traffic (normal max ~0.002 vs attack min ~0.20
for country_attack_rate) — trivially separable, not a genuine learned
signal. The original pipeline already solved this via 10_retrain_
clean.py's noise injection step, which this version restores: a
fraction of attack rows get their attack-rate features pulled down
into the normal range (and vice versa for normal rows), plus Gaussian
noise on top, before training. This is applied HERE, on top of the
already-fixed (independently-seeded) full_dataset.csv, combining both
corrections rather than one at the expense of the other.

Same RF/LSTM architecture and hyperparameters as the original
3_train_rf.py / 4_train_lstm.py / 10_retrain_clean.py — the changes
here are (a) relative paths for Windows portability and (b) running
on top of the seed-fixed synthetic data rather than the original
buggy version.

This OVERWRITES models/rf_model.pkl, models/lstm_model.pt,
models/scaler.pkl, and data/train_test_split_clean.npz. Back up first
if you want the (broken, 100%-accuracy) previous retrain result kept
for comparison:
    copy models\\rf_model.pkl models\\rf_model_v1_100pct_backup.pkl
    copy models\\lstm_model.pt models\\lstm_model_v1_100pct_backup.pt
    copy models\\scaler.pkl models\\scaler_v1_100pct_backup.pkl

Run this from inside the mfa-project-restored folder:
    python retrain_on_fixed_data.py

After this finishes, re-run posthoc_analysis.py and
threshold_sweep.py to get updated CI/PR-AUC/base-rate/threshold
numbers reflecting the corrected data.
"""

import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'full_dataset.csv')
RF_PATH = os.path.join(BASE_DIR, 'models', 'rf_model.pkl')
LSTM_PATH = os.path.join(BASE_DIR, 'models', 'lstm_model.pt')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
SPLIT_PATH = os.path.join(BASE_DIR, 'data', 'train_test_split_clean.npz')
RESULTS_PATH = os.path.join(BASE_DIR, 'results', 'retrain_on_fixed_data_results.txt')

RANDOM_SEED = 42
SEQ_LEN = 10
BATCH_SIZE = 512
EPOCHS = 20
T1, T2 = 34, 67  # keep original thresholds for now — the threshold
                  # sweep analysis (T1=70 recommendation) is a separate
                  # deployment-time decision, not baked into training

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

FEATURE_COLS = [
    'hour', 'day_of_week', 'is_night',
    'device_mobile', 'device_desktop', 'device_tablet',
    'login_success', 'is_attack_ip', 'browser_known',
    'country_changed', 'asn_changed', 'device_changed',
    'asn_attack_rate', 'country_attack_rate', 'login_freq'
]


def main():
    print("=" * 60)
    print("RETRAIN ON CORRECTED DATASET — RF + LSTM")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded dataset: {df.shape[0]:,} rows, {df.shape[1]} cols")
    print(f"  Attack rows : {(df['label']==1).sum():,}")
    print(f"  Normal rows : {(df['label']==0).sum():,}")

    # ══════════════════════════════════════════════════════════
    # NOISE INJECTION — restores realistic overlap between classes.
    # Identical logic to 10_retrain_clean.py, applied here on top of
    # the seed-fixed full_dataset.csv. Without this, asn_attack_rate/
    # country_attack_rate have a complete non-overlapping gap between
    # classes, making the problem trivially (and unrealistically)
    # separable — this is what caused the 100% accuracy result.
    # ══════════════════════════════════════════════════════════
    print("\nInjecting noise to restore realistic class overlap...")
    attacks = df[df['label'] == 1].copy()
    normals = df[df['label'] == 0].copy()
    n_attacks = len(attacks)
    n_normals = len(normals)

    mask = np.random.random(n_attacks) < 0.30
    attacks.loc[mask, 'browser_known'] = 1
    print(f"  [Attack] Known browser flipped   : {mask.sum():,} rows ({mask.mean()*100:.0f}%)")

    mask = np.random.random(n_attacks) < 0.35
    attacks.loc[mask, 'is_night'] = 0
    attacks.loc[mask, 'hour'] = np.random.randint(9, 18, size=mask.sum())
    print(f"  [Attack] Daytime attacks          : {mask.sum():,} rows ({mask.mean()*100:.0f}%)")

    mask = np.random.random(n_attacks) < 0.40
    attacks.loc[mask, 'is_attack_ip'] = 0
    print(f"  [Attack] IP not in threat feeds   : {mask.sum():,} rows ({mask.mean()*100:.0f}%)")

    mask = np.random.random(n_attacks) < 0.25
    attacks.loc[mask, 'country_changed'] = 0
    print(f"  [Attack] Same country             : {mask.sum():,} rows ({mask.mean()*100:.0f}%)")

    mask = np.random.random(n_attacks) < 0.20
    attacks.loc[mask, 'device_changed'] = 0
    print(f"  [Attack] Same device              : {mask.sum():,} rows ({mask.mean()*100:.0f}%)")

    mask = np.random.random(n_attacks) < 0.20
    attacks.loc[mask, 'asn_changed'] = 0
    print(f"  [Attack] Same ASN                 : {mask.sum():,} rows ({mask.mean()*100:.0f}%)")

    mask = np.random.random(n_attacks) < 0.35
    attacks.loc[mask, 'asn_attack_rate'] = np.random.uniform(0.0, 0.15, size=mask.sum())
    attacks.loc[mask, 'country_attack_rate'] = np.random.uniform(0.0, 0.15, size=mask.sum())
    print(f"  [Attack] Low reputation scores    : {mask.sum():,} rows ({mask.mean()*100:.0f}%)")

    noise = np.random.normal(0, 0.12, size=(n_attacks, 2))
    attacks[['asn_attack_rate', 'country_attack_rate']] = (
        attacks[['asn_attack_rate', 'country_attack_rate']].values + noise
    ).clip(0, 1)
    print(f"  [Attack] Gaussian noise on rates  : SD=0.12, all {n_attacks:,} rows")

    mask = np.random.random(n_normals) < 0.15
    normals.loc[mask, 'country_changed'] = 1
    normals.loc[mask, 'asn_changed'] = 1
    print(f"  [Normal] Travelling users         : {mask.sum():,} rows ({mask.mean()*100:.0f}%)")

    mask = np.random.random(n_normals) < 0.12
    normals.loc[mask, 'device_changed'] = 1
    print(f"  [Normal] New device               : {mask.sum():,} rows ({mask.mean()*100:.0f}%)")

    mask = np.random.random(n_normals) < 0.15
    normals.loc[mask, 'is_night'] = 1
    normals.loc[mask, 'hour'] = np.random.choice(
        list(range(22, 24)) + list(range(0, 7)), size=mask.sum())
    print(f"  [Normal] Night logins             : {mask.sum():,} rows ({mask.mean()*100:.0f}%)")

    mask = np.random.random(n_normals) < 0.08
    normals.loc[mask, 'login_freq'] = np.random.randint(10, 21, size=mask.sum())
    print(f"  [Normal] High login freq          : {mask.sum():,} rows ({mask.mean()*100:.0f}%)")

    mask = np.random.random(n_normals) < 0.10
    normals.loc[mask, 'asn_attack_rate'] = np.random.uniform(0.1, 0.4, size=mask.sum())
    print(f"  [Normal] Risky network            : {mask.sum():,} rows ({mask.mean()*100:.0f}%)")

    noise = np.random.normal(0, 0.05, size=(n_normals, 2))
    normals[['asn_attack_rate', 'country_attack_rate']] = (
        normals[['asn_attack_rate', 'country_attack_rate']].values + noise
    ).clip(0, 1)
    print(f"  [Normal] Gaussian noise on rates  : SD=0.05, all {n_normals:,} rows")

    df_noisy = pd.concat([attacks, normals], ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"\nNoisy dataset: {df_noisy.shape[0]:,} rows")

    X = df_noisy[FEATURE_COLS].values
    y = df_noisy['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    print(f"\nSplit: {len(X_train):,} train / {len(X_test):,} test")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print("Scaler saved.")

    np.savez(SPLIT_PATH,
             X_train_scaled=X_train_s, X_test_scaled=X_test_s,
             X_train_raw=X_train, X_test_raw=X_test,
             y_train=y_train, y_test=y_test)
    print("Train/test arrays saved (scaled + raw).")

    # ── RANDOM FOREST ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RANDOM FOREST — TRAINING")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train_s, y_train)
    joblib.dump(rf, RF_PATH)
    print("Random Forest saved.")

    rf_probs = rf.predict_proba(X_test_s)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_rec = recall_score(y_test, rf_preds)
    rf_prec = precision_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf_probs)
    tn, fp, fn, tp = confusion_matrix(y_test, rf_preds).ravel()
    rf_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\nRandom Forest Results (Corrected Synthetic Data)")
    print(f"  Accuracy  : {rf_acc*100:.2f}%")
    print(f"  Recall    : {rf_rec*100:.2f}%")
    print(f"  Precision : {rf_prec*100:.2f}%")
    print(f"  F1-Score  : {rf_f1*100:.2f}%")
    print(f"  AUC-ROC   : {rf_auc:.4f}")
    print(f"  FPR       : {rf_fpr*100:.2f}%")

    # ── LSTM ─────────────────────────────────────────────────
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size,
                num_layers=num_layers, batch_first=True, dropout=dropout
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return torch.sigmoid(self.fc(out[:, -1, :]))

    def make_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i + seq_len])
            ys.append(y[i + seq_len - 1])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    print("\n" + "=" * 60)
    print("LSTM — TRAINING")
    print("=" * 60)

    Xtr_seq, ytr_seq = make_sequences(X_train_s, y_train, SEQ_LEN)
    Xte_seq, yte_seq = make_sequences(X_test_s, y_test, SEQ_LEN)
    print(f"Sequences — Train: {len(Xtr_seq):,}  Test: {len(Xte_seq):,}")

    tr_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xtr_seq), torch.FloatTensor(ytr_seq)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    te_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xte_seq), torch.FloatTensor(yte_seq)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    device = torch.device('cpu')
    model = LSTMClassifier(input_size=len(FEATURE_COLS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.BCELoss()

    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb).squeeze(), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in te_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += crit(model(xb).squeeze(), yb).item()

        tr_loss /= len(tr_loader)
        val_loss /= len(te_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), LSTM_PATH)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{EPOCHS}  Train Loss: {tr_loss:.4f}  Val Loss: {val_loss:.4f}")

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("LSTM model saved.")

    model.load_state_dict(torch.load(LSTM_PATH))
    model.eval()

    lstm_probs, lstm_true = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            lstm_probs.extend(model(xb.to(device)).squeeze().cpu().numpy())
            lstm_true.extend(yb.numpy())

    lstm_probs = np.array(lstm_probs)
    lstm_true = np.array(lstm_true)
    lstm_preds = (lstm_probs >= 0.5).astype(int)

    lstm_acc = accuracy_score(lstm_true, lstm_preds)
    lstm_rec = recall_score(lstm_true, lstm_preds)
    lstm_prec = precision_score(lstm_true, lstm_preds)
    lstm_f1 = f1_score(lstm_true, lstm_preds)
    lstm_auc = roc_auc_score(lstm_true, lstm_probs)
    tn2, fp2, fn2, tp2 = confusion_matrix(lstm_true, lstm_preds).ravel()
    lstm_fpr = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0.0

    print(f"\nLSTM Results (Corrected Synthetic Data)")
    print(f"  Accuracy  : {lstm_acc*100:.2f}%")
    print(f"  Recall    : {lstm_rec*100:.2f}%")
    print(f"  Precision : {lstm_prec*100:.2f}%")
    print(f"  F1-Score  : {lstm_f1*100:.2f}%")
    print(f"  AUC-ROC   : {lstm_auc:.4f}")
    print(f"  FPR       : {lstm_fpr*100:.2f}%")

    # ── Save summary ─────────────────────────────────────────
    lines = [
        "=" * 60,
        "RETRAIN ON CORRECTED SYNTHETIC DATA — SUMMARY",
        "(fixes shared-random-seed bug in original 2_synthetic.py)",
        "=" * 60,
        "",
        "Random Forest:",
        f"  Accuracy:  {rf_acc:.4f}",
        f"  Recall:    {rf_rec:.4f}",
        f"  Precision: {rf_prec:.4f}",
        f"  F1-Score:  {rf_f1:.4f}",
        f"  FPR:       {rf_fpr:.4f}",
        f"  ROC-AUC:   {rf_auc:.4f}",
        "",
        "LSTM:",
        f"  Accuracy:  {lstm_acc:.4f}",
        f"  Recall:    {lstm_rec:.4f}",
        f"  Precision: {lstm_prec:.4f}",
        f"  F1-Score:  {lstm_f1:.4f}",
        f"  FPR:       {lstm_fpr:.4f}",
        f"  ROC-AUC:   {lstm_auc:.4f}",
        "",
        "Next step: re-run posthoc_analysis.py and threshold_sweep.py",
        "against these retrained models to get updated CI/PR-AUC/",
        "base-rate/threshold numbers for the response letter.",
    ]

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nSaved summary to {RESULTS_PATH}")
    print("=" * 60)
    print("RETRAIN COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
