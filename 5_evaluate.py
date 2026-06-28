import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix, roc_auc_score)

SPLIT_PATH   = '/home/da_otcifithom/mfa-bypass-detection/data/train_test_split.npz'
RF_PATH      = '/home/da_otcifithom/mfa-bypass-detection/models/rf_model.pkl'
LSTM_PATH    = '/home/da_otcifithom/mfa-bypass-detection/models/lstm_model.pt'
SCALER_PATH  = '/home/da_otcifithom/mfa-bypass-detection/models/scaler.pkl'
DATA_PATH    = '/home/da_otcifithom/mfa-bypass-detection/data/full_dataset.csv'
RESULTS_PATH = '/home/da_otcifithom/mfa-bypass-detection/results/evaluation_results.txt'
SEQ_LEN      = 10
RANDOM_SEED  = 42

print("=" * 60)
print("PHASE 6 — RISK FUSION & EVALUATION")
print("=" * 60)

# ── Load models ───────────────────────────────────────────────
rf_model = joblib.load(RF_PATH)
scaler   = joblib.load(SCALER_PATH)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc      = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).squeeze()

data       = np.load(SPLIT_PATH)
X_test     = data['X_test'].astype(np.float32)
y_test     = data['y_test'].astype(np.float32)
INPUT_SIZE = X_test.shape[1]

lstm_model = LSTMClassifier(INPUT_SIZE)
lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location='cpu'))
lstm_model.eval()
print("Models loaded successfully")

# ── Helper: get LSTM probabilities ────────────────────────────
def get_lstm_probs(X, seq_len=SEQ_LEN):
    Xs = []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
    Xs = np.array(Xs, dtype=np.float32)
    ds = TensorDataset(torch.tensor(Xs))
    dl = DataLoader(ds, batch_size=512, shuffle=False)
    probs = []
    with torch.no_grad():
        for (Xb,) in dl:
            p = lstm_model(Xb).cpu().numpy()
            probs.extend(p)
    pad = [np.mean(probs)] * (seq_len - 1)
    return np.array(pad + probs)

# ── Risk score fusion ─────────────────────────────────────────
def fuse_risk_score(p_rf, p_lstm, X_raw):
    ml_score          = 0.6 * p_rf + 0.4 * p_lstm
    contextual_score  = (X_raw[:, 12] + X_raw[:, 13]) / 2
    behavioural_score = (X_raw[:, 14] / 20.0 + X_raw[:, 2]) / 2
    R = (0.6 * ml_score + 0.2 * contextual_score + 0.2 * behavioural_score) * 100
    return np.clip(R, 0, 100)

# ── Decision engine ───────────────────────────────────────────
def decision(R, t1=33, t2=66):
    d = []
    for r in R:
        if r < t1:
            d.append('ALLOW')
        elif r < t2:
            d.append('STEP_UP_MFA')
        else:
            d.append('BLOCK')
    return np.array(d)

def binary_from_decision(decisions):
    return np.array([0 if d == 'ALLOW' else 1 for d in decisions])

def print_metrics(label, y_true, y_pred, y_proba=None):
    acc  = accuracy_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
    auc  = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    lines = [
        f"\n{'=' * 40}",
        f"{label}",
        f"{'=' * 40}",
        f"Accuracy:  {acc:.4f}  (target >= 0.95)",
        f"Recall:    {rec:.4f}  (target >= 0.90)",
        f"Precision: {prec:.4f} (target >= 0.92)",
        f"F1-Score:  {f1:.4f}  (target >= 0.91)",
        f"FPR:       {fpr:.4f}  (target <= 0.05)",
    ]
    if auc is not None:
        lines.append(f"ROC-AUC:   {auc:.4f}")
    lines += [
        f"Confusion Matrix:",
        f"  TN={tn}  FP={fp}",
        f"  FN={fn}  TP={tp}",
    ]
    for l in lines:
        print(l)
    return lines

results_log = []

# ════════════════════════════════════════════════════════════
# EVALUATION 1 — IN-DISTRIBUTION TEST SET
# ════════════════════════════════════════════════════════════
print("\n[1/5] In-distribution evaluation...")
X_test_sc = scaler.transform(X_test)
p_rf      = rf_model.predict_proba(X_test_sc)[:, 1]
p_lstm    = get_lstm_probs(X_test_sc)

# align lengths
min_len        = min(len(p_rf), len(p_lstm))
p_rf           = p_rf[:min_len]
p_lstm         = p_lstm[:min_len]
X_test_aligned = X_test[:min_len]
y_true         = y_test.astype(int)[:min_len]

R         = fuse_risk_score(p_rf, p_lstm, X_test_aligned)
decisions = decision(R)
y_pred    = binary_from_decision(decisions)

lines = print_metrics("1. IN-DISTRIBUTION (Full Framework)", y_true, y_pred, p_rf)
results_log.extend(lines)

# USR
legit_mask = y_true == 0
usr_count  = np.sum(decisions[legit_mask] == 'STEP_UP_MFA')
usr        = usr_count / np.sum(legit_mask)
usr_line   = f"USR:       {usr:.4f}  (target <= 0.08)"
print(usr_line)
results_log.append(usr_line)

print("\nDecision distribution:")
for d in ['ALLOW', 'STEP_UP_MFA', 'BLOCK']:
    print(f"  {d}: {np.sum(decisions == d):,}")

# ════════════════════════════════════════════════════════════
# EVALUATION 2 — RF ONLY (ablation)
# ════════════════════════════════════════════════════════════
print("\n[2/5] RF-only ablation...")
y_pred_rf = (p_rf >= 0.5).astype(int)
lines = print_metrics("2. RF ONLY (Ablation)", y_true, y_pred_rf, p_rf)
results_log.extend(lines)

# ════════════════════════════════════════════════════════════
# EVALUATION 3 — LSTM ONLY (ablation)
# ════════════════════════════════════════════════════════════
print("\n[3/5] LSTM-only ablation...")
y_pred_lstm = (p_lstm >= 0.5).astype(int)
lines = print_metrics("3. LSTM ONLY (Ablation)", y_true, y_pred_lstm, p_lstm)
results_log.extend(lines)

# ════════════════════════════════════════════════════════════
# EVALUATION 4 — STATIC MFA BASELINE
# ════════════════════════════════════════════════════════════
print("\n[4/5] Static MFA baseline...")
static_pred = X_test_aligned[:, 7].astype(int)
lines = print_metrics("4. STATIC MFA BASELINE", y_true, static_pred)
results_log.extend(lines)

# ════════════════════════════════════════════════════════════
# EVALUATION 5 — OOD TEST SET
# ════════════════════════════════════════════════════════════
print("\n[5/5] Out-of-distribution evaluation...")
np.random.seed(999)
N_OOD = 10000

def make_ood_attacks(n_each):
    rows = []
    for atype in ['sim_swap', 'aitm_phishing', 'session_hijacking', 'mfa_fatigue']:
        for _ in range(n_each):
            r = {
                'hour':                np.random.randint(0, 24),
                'day_of_week':         np.random.randint(0, 7),
                'is_night':            np.random.choice([0, 1]),
                'device_mobile':       np.random.choice([0, 1]),
                'device_desktop':      np.random.choice([0, 1]),
                'device_tablet':       0,
                'login_success':       1,
                'is_attack_ip':        np.random.choice([0, 1], p=[0.3, 0.7]),
                'browser_known':       np.random.choice([0, 1]),
                'country_changed':     np.random.choice([0, 1], p=[0.2, 0.8]),
                'asn_changed':         np.random.choice([0, 1], p=[0.2, 0.8]),
                'device_changed':      np.random.choice([0, 1], p=[0.3, 0.7]),
                'asn_attack_rate':     np.random.uniform(0.1, 0.9),
                'country_attack_rate': np.random.uniform(0.1, 0.9),
                'login_freq':          np.random.randint(1, 20),
                'label':               1,
                'attack_type':         atype
            }
            rows.append(r)
    return pd.DataFrame(rows)

ood_attacks = make_ood_attacks(N_OOD // 4)
ood_normals = pd.DataFrame({
    'hour':                np.random.randint(8, 20, N_OOD),
    'day_of_week':         np.random.randint(0, 5, N_OOD),
    'is_night':            np.zeros(N_OOD, dtype=int),
    'device_mobile':       np.random.choice([0, 1], N_OOD),
    'device_desktop':      np.random.choice([0, 1], N_OOD),
    'device_tablet':       np.zeros(N_OOD, dtype=int),
    'login_success':       np.ones(N_OOD, dtype=int),
    'is_attack_ip':        np.zeros(N_OOD, dtype=int),
    'browser_known':       np.ones(N_OOD, dtype=int),
    'country_changed':     np.zeros(N_OOD, dtype=int),
    'asn_changed':         np.zeros(N_OOD, dtype=int),
    'device_changed':      np.zeros(N_OOD, dtype=int),
    'asn_attack_rate':     np.random.uniform(0, 0.1, N_OOD),
    'country_attack_rate': np.random.uniform(0, 0.1, N_OOD),
    'login_freq':          np.random.randint(1, 5, N_OOD),
    'label':               np.zeros(N_OOD, dtype=int),
    'attack_type':         'none'
})

ood_df = pd.concat([ood_attacks, ood_normals], ignore_index=True).sample(
    frac=1, random_state=999).reset_index(drop=True)

FEATURE_COLS = [
    'hour', 'day_of_week', 'is_night',
    'device_mobile', 'device_desktop', 'device_tablet',
    'login_success', 'is_attack_ip', 'browser_known',
    'country_changed', 'asn_changed', 'device_changed',
    'asn_attack_rate', 'country_attack_rate', 'login_freq'
]

X_ood    = ood_df[FEATURE_COLS].values.astype(np.float32)
y_ood    = ood_df['label'].values.astype(int)
X_ood_sc = scaler.transform(X_ood)

p_rf_ood   = rf_model.predict_proba(X_ood_sc)[:, 1]
p_lstm_ood = get_lstm_probs(X_ood_sc)

# align
min_ood      = min(len(p_rf_ood), len(p_lstm_ood))
p_rf_ood     = p_rf_ood[:min_ood]
p_lstm_ood   = p_lstm_ood[:min_ood]
X_ood_al     = X_ood[:min_ood]
y_ood_al     = y_ood[:min_ood]
ood_df_al    = ood_df.iloc[:min_ood]

R_ood      = fuse_risk_score(p_rf_ood, p_lstm_ood, X_ood_al)
dec_ood    = decision(R_ood)
y_pred_ood = binary_from_decision(dec_ood)

lines = print_metrics("5. OOD TEST SET (Primary Reported Result)",
                      y_ood_al, y_pred_ood, p_rf_ood)
results_log.extend(lines)

print("\nOOD Per-Attack-Type Recall:")
for atype in ['sim_swap', 'aitm_phishing', 'session_hijacking', 'mfa_fatigue']:
    mask = (ood_df_al['attack_type'] == atype).values
    if mask.sum() > 0:
        rec_t = recall_score(y_ood_al[mask], y_pred_ood[mask], zero_division=0)
        print(f"  {atype}: recall={rec_t:.4f} (n={mask.sum()})")

legit_ood = y_ood_al == 0
usr_ood   = np.sum(dec_ood[legit_ood] == 'STEP_UP_MFA') / np.sum(legit_ood)
print(f"OOD USR: {usr_ood:.4f}  (target <= 0.08)")

# ── Save results ──────────────────────────────────────────────
with open(RESULTS_PATH, 'w') as f:
    f.write('\n'.join(results_log))
print(f"\nResults saved to {RESULTS_PATH}")
print("=" * 60)
print("PHASE 6 COMPLETE")
print("=" * 60)