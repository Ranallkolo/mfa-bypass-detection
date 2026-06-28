import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score,
                             confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────
# NOTE: these use relative paths so the script works on any machine,
# not just the original VM. Adjust BASE_DIR only if you move the
# project root.
import os
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'full_dataset.csv')
RF_PATH     = os.path.join(BASE_DIR, 'models', 'rf_model.pkl')
LSTM_PATH   = os.path.join(BASE_DIR, 'models', 'lstm_model.pt')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
SPLIT_PATH  = os.path.join(BASE_DIR, 'data', 'train_test_split_clean.npz')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

FEATURE_COLS = [
    'hour', 'day_of_week', 'is_night',
    'device_mobile', 'device_desktop', 'device_tablet',
    'login_success', 'is_attack_ip', 'browser_known',
    'country_changed', 'asn_changed', 'device_changed',
    'asn_attack_rate', 'country_attack_rate', 'login_freq'
]
SEQ_LEN    = 10
BATCH_SIZE = 512
EPOCHS     = 20
T1, T2     = 34, 67  # decision thresholds (corrected values, used consistently)

print("=" * 60)
print("PHASE 9 (CLEAN) — RETRAIN WITH REALISTIC NOISE")
print("=" * 60)

# ── Load dataset ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} cols")
print(f"  Attack rows : {(df['label']==1).sum():,}")
print(f"  Normal rows : {(df['label']==0).sum():,}")

attacks = df[df['label'] == 1].copy()
normals = df[df['label'] == 0].copy()
n_attacks = len(attacks)
n_normals = len(normals)

# ══════════════════════════════════════════════════════════════
# NOISE INJECTION — ATTACK RECORDS
# ══════════════════════════════════════════════════════════════
print("\nInjecting noise into attack records...")

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

# ══════════════════════════════════════════════════════════════
# NOISE INJECTION — NORMAL RECORDS
# ══════════════════════════════════════════════════════════════
print("\nInjecting noise into normal records...")

mask = np.random.random(n_normals) < 0.15
normals.loc[mask, 'country_changed'] = 1
normals.loc[mask, 'asn_changed']     = 1
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

# ── Recombine and shuffle ──────────────────────────────────────
df_noisy = pd.concat([attacks, normals], ignore_index=True).sample(
    frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"\nNoisy dataset: {df_noisy.shape[0]:,} rows")

X = df_noisy[FEATURE_COLS].values
y = df_noisy['label'].values

# ── Train/test split (80/20 stratified) ───────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

print(f"\nSplit: {len(X_train):,} train / {len(X_test):,} test")

# ── Scale (for RF/LSTM input only) ─────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler, SCALER_PATH)
print("Scaler saved.")

# IMPORTANT: we save BOTH the scaled features (X_test_s, for RF/LSTM
# input) and the raw, pre-scaling features (X_test, for the
# contextual/behavioural fusion score, which expects 0-1 reputation
# rates rather than standardized values). Using scaled values for the
# fusion score was a bug identified during reproducibility testing —
# StandardScaler output is centered near 0 with values outside [0,1]
# and negative numbers, which silently corrupts the .clip(0,1) logic
# in fuse_risk_score() below.
np.savez(SPLIT_PATH,
         X_train_scaled=X_train_s, X_test_scaled=X_test_s,
         X_train_raw=X_train, X_test_raw=X_test,
         y_train=y_train, y_test=y_test)
print("Train/test arrays saved (scaled + raw).")

# ══════════════════════════════════════════════════════════════
# RANDOM FOREST — RETRAIN
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RANDOM FOREST — RETRAINING")
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

rf_acc  = accuracy_score(y_test, rf_preds)
rf_rec  = recall_score(y_test, rf_preds)
rf_prec = precision_score(y_test, rf_preds)
rf_f1   = f1_score(y_test, rf_preds)
rf_auc  = roc_auc_score(y_test, rf_probs)
tn, fp, fn, tp = confusion_matrix(y_test, rf_preds).ravel()
rf_fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0

print(f"\nRandom Forest Results (Noisy Data)")
print(f"  Accuracy  : {rf_acc*100:.2f}%")
print(f"  Recall    : {rf_rec*100:.2f}%")
print(f"  Precision : {rf_prec*100:.2f}%")
print(f"  F1-Score  : {rf_f1*100:.2f}%")
print(f"  AUC-ROC   : {rf_auc:.4f}")
print(f"  FPR       : {rf_fpr*100:.2f}%")
print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

importances = list(zip(FEATURE_COLS, rf.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
print("\nFeature Importances:")
for rank, (feat, imp) in enumerate(importances, 1):
    print(f"  {rank:2d}. {feat:<25s} {imp:.4f}")

# ══════════════════════════════════════════════════════════════
# LSTM — ARCHITECTURE & RETRAIN
# ══════════════════════════════════════════════════════════════
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
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
print("LSTM — RETRAINING")
print("=" * 60)

Xtr_seq, ytr_seq = make_sequences(X_train_s, y_train, SEQ_LEN)
Xte_seq, yte_seq = make_sequences(X_test_s,  y_test,  SEQ_LEN)
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
model  = LSTMClassifier(input_size=len(FEATURE_COLS)).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=0.001)
crit   = nn.BCELoss()

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

    tr_loss  /= len(tr_loader)
    val_loss /= len(te_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), LSTM_PATH)

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{EPOCHS}  "
              f"Train Loss: {tr_loss:.4f}  Val Loss: {val_loss:.4f}")

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
lstm_true  = np.array(lstm_true)
lstm_preds = (lstm_probs >= 0.5).astype(int)

lstm_acc  = accuracy_score(lstm_true, lstm_preds)
lstm_rec  = recall_score(lstm_true, lstm_preds)
lstm_prec = precision_score(lstm_true, lstm_preds)
lstm_f1   = f1_score(lstm_true, lstm_preds)
lstm_auc  = roc_auc_score(lstm_true, lstm_probs)
tn2, fp2, fn2, tp2 = confusion_matrix(lstm_true, lstm_preds).ravel()
lstm_fpr  = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0.0

print(f"\nLSTM Results (Noisy Data)")
print(f"  Accuracy  : {lstm_acc*100:.2f}%")
print(f"  Recall    : {lstm_rec*100:.2f}%")
print(f"  Precision : {lstm_prec*100:.2f}%")
print(f"  F1-Score  : {lstm_f1*100:.2f}%")
print(f"  AUC-ROC   : {lstm_auc:.4f}")
print(f"  FPR       : {lstm_fpr*100:.2f}%")
print(f"  TP={tp2}  FP={fp2}  TN={tn2}  FN={fn2}")

# ══════════════════════════════════════════════════════════════
# ENSEMBLE RISK SCORE — EVALUATION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ENSEMBLE RISK SCORE — EVALUATION")
print("=" * 60)

# Align RF probs to LSTM test indices (LSTM loses SEQ_LEN-1 rows from front)
rf_probs_aligned = rf_probs[SEQ_LEN - 1: SEQ_LEN - 1 + len(lstm_probs)]
y_test_aligned   = y_test[SEQ_LEN - 1: SEQ_LEN - 1 + len(lstm_probs)]

# FIX: use RAW (pre-scaling) features for contextual/behavioural score,
# not the scaled X_test_s. The fusion formula and its .clip(0,1) calls
# assume 0-1 reputation rates, which only the raw features provide.
X_test_raw_aligned = X_test[SEQ_LEN - 1: SEQ_LEN - 1 + len(lstm_probs)]

asn_rate_idx     = FEATURE_COLS.index('asn_attack_rate')
country_rate_idx = FEATURE_COLS.index('country_attack_rate')
login_freq_idx   = FEATURE_COLS.index('login_freq')
is_night_idx     = FEATURE_COLS.index('is_night')

contextual_score  = (X_test_raw_aligned[:, asn_rate_idx] +
                     X_test_raw_aligned[:, country_rate_idx]) / 2
contextual_score  = contextual_score.clip(0, 1)

behavioural_score = (X_test_raw_aligned[:, login_freq_idx] / 20 +
                     X_test_raw_aligned[:, is_night_idx]) / 2
behavioural_score = behavioural_score.clip(0, 1)

ml_score    = 0.6 * rf_probs_aligned + 0.4 * lstm_probs
risk_scores = (0.6 * ml_score + 0.2 * contextual_score +
               0.2 * behavioural_score) * 100
risk_scores = np.clip(risk_scores, 0, 100)

# Decision engine — corrected thresholds (34/67)
decisions = np.where(risk_scores < T1, 'ALLOW',
            np.where(risk_scores < T2, 'STEP_UP_MFA', 'BLOCK'))

# FIX: "Flagged" definition — BLOCK *and* STEP_UP_MFA both count as
# detected/positive, matching the thesis's "Full Framework (Flagged)"
# terminology and the binary_from_decision() logic used elsewhere.
# (Previously this only counted BLOCK, which understates recall.)
flagged_preds = (decisions != 'ALLOW').astype(int)

ens_acc  = accuracy_score(y_test_aligned, flagged_preds)
ens_rec  = recall_score(y_test_aligned, flagged_preds)
ens_prec = precision_score(y_test_aligned, flagged_preds)
ens_f1   = f1_score(y_test_aligned, flagged_preds)
ens_auc  = roc_auc_score(y_test_aligned, risk_scores / 100)
tn3, fp3, fn3, tp3 = confusion_matrix(y_test_aligned, flagged_preds).ravel()
ens_fpr  = fp3 / (fp3 + tn3) if (fp3 + tn3) > 0 else 0.0

normal_mask = y_test_aligned == 0
usr = (decisions[normal_mask] == 'STEP_UP_MFA').mean()

print(f"\nFull Framework Results (Noisy Data, Flagged Definition)")
print(f"  Accuracy  : {ens_acc*100:.2f}%")
print(f"  Recall    : {ens_rec*100:.2f}%")
print(f"  Precision : {ens_prec*100:.2f}%")
print(f"  F1-Score  : {ens_f1*100:.2f}%")
print(f"  AUC-ROC   : {ens_auc:.4f}")
print(f"  FPR       : {ens_fpr*100:.2f}%")
print(f"  USR       : {usr*100:.2f}%")
print(f"  TP={tp3}  FP={fp3}  TN={tn3}  FN={fn3}")

allow_n   = (decisions == 'ALLOW').sum()
stepup_n  = (decisions == 'STEP_UP_MFA').sum()
block_n   = (decisions == 'BLOCK').sum()
total_n   = len(decisions)
print(f"\nDecision Distribution:")
print(f"  ALLOW       : {allow_n:,}  ({allow_n/total_n*100:.1f}%)")
print(f"  STEP_UP_MFA : {stepup_n:,}  ({stepup_n/total_n*100:.1f}%)")
print(f"  BLOCK       : {block_n:,}  ({block_n/total_n*100:.1f}%)")

# ── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 9 (CLEAN) COMPLETE — SUMMARY")
print("=" * 60)
print(f"{'Model':<30} {'Acc':>8} {'Recall':>8} {'Prec':>8} "
      f"{'F1':>8} {'FPR':>8}")
print("-" * 60)
print(f"{'Random Forest (noisy)':<30} "
      f"{rf_acc*100:>7.2f}% {rf_rec*100:>7.2f}% "
      f"{rf_prec*100:>7.2f}% {rf_f1*100:>7.2f}% {rf_fpr*100:>7.2f}%")
print(f"{'LSTM (noisy)':<30} "
      f"{lstm_acc*100:>7.2f}% {lstm_rec*100:>7.2f}% "
      f"{lstm_prec*100:>7.2f}% {lstm_f1*100:>7.2f}% {lstm_fpr*100:>7.2f}%")
print(f"{'Full Framework (flagged)':<30} "
      f"{ens_acc*100:>7.2f}% {ens_rec*100:>7.2f}% "
      f"{ens_prec*100:>7.2f}% {ens_f1*100:>7.2f}% {ens_fpr*100:>7.2f}%")
print(f"\nUSR (Unnecessary Step-Up Rate): {usr*100:.2f}%")
print("\nFiles saved:")
print(f"  {RF_PATH}")
print(f"  {LSTM_PATH}")
print(f"  {SCALER_PATH}")
print(f"  {SPLIT_PATH}")
print("\nNext step: run 11_evaluate_clean.py to verify reproducibility.")