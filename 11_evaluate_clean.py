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
import os

# ── Paths (relative, portable across machines) ─────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SPLIT_PATH   = os.path.join(BASE_DIR, 'data', 'train_test_split_clean.npz')
RF_PATH      = os.path.join(BASE_DIR, 'models', 'rf_model.pkl')
LSTM_PATH    = os.path.join(BASE_DIR, 'models', 'lstm_model.pt')
SCALER_PATH  = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
RESULTS_PATH = os.path.join(BASE_DIR, 'results', 'evaluation_results_clean.txt')
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

SEQ_LEN     = 10
RANDOM_SEED = 42
T1, T2      = 34, 67  # must match the thresholds used in 10_retrain_clean.py

FEATURE_COLS = [
    'hour', 'day_of_week', 'is_night',
    'device_mobile', 'device_desktop', 'device_tablet',
    'login_success', 'is_attack_ip', 'browser_known',
    'country_changed', 'asn_changed', 'device_changed',
    'asn_attack_rate', 'country_attack_rate', 'login_freq'
]

print("=" * 60)
print("PHASE 11 (CLEAN) — RISK FUSION & EVALUATION")
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


# ── Load data — BOTH scaled (for model input) and raw (for fusion) ──
data         = np.load(SPLIT_PATH)
X_test_s     = data['X_test_scaled'].astype(np.float32)   # for RF/LSTM
X_test_raw   = data['X_test_raw'].astype(np.float32)      # for fusion score
y_test       = data['y_test'].astype(np.float32)
INPUT_SIZE   = X_test_s.shape[1]

lstm_model = LSTMClassifier(INPUT_SIZE)
lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location='cpu'))
lstm_model.eval()
print("Models loaded successfully")


# ── Helper: get LSTM probabilities, trimming (not padding) the front ──
def get_lstm_probs_trimmed(X, seq_len=SEQ_LEN):
    """Returns real LSTM predictions only (no fake mean-fill padding).
    Output length is len(X) - seq_len, aligned to start at index
    seq_len - 1 of the original array (i.e. caller should slice their
    other aligned arrays the same way: arr[seq_len - 1:])."""
    Xs = []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
    Xs = np.array(Xs, dtype=np.float32)
    ds = TensorDataset(torch.tensor(Xs))
    dl = DataLoader(ds, batch_size=512, shuffle=False)
    probs = []
    with torch.no_grad():
        for (Xb,) in dl:
            p = lstm_model(Xb).cpu().numpy()
            probs.extend(p)
    return np.array(probs)


# ── Risk score fusion (expects RAW 0-1 features, not scaled) ──────
def fuse_risk_score(p_rf, p_lstm, X_raw):
    ml_score          = 0.6 * p_rf + 0.4 * p_lstm
    contextual_score  = (X_raw[:, 12] + X_raw[:, 13]) / 2
    contextual_score  = contextual_score.clip(0, 1)
    behavioural_score = (X_raw[:, 14] / 20.0 + X_raw[:, 2]) / 2
    behavioural_score = behavioural_score.clip(0, 1)
    R = (0.6 * ml_score + 0.2 * contextual_score + 0.2 * behavioural_score) * 100
    return np.clip(R, 0, 100)


# ── Decision engine ───────────────────────────────────────────
def decision(R, t1=T1, t2=T2):
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
    """'Flagged' definition: BLOCK or STEP_UP_MFA both count as
    detected/positive. Matches the thesis's 'Full Framework (Flagged)'
    terminology."""
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
print("\n[1/8] In-distribution evaluation...")

p_rf_full   = rf_model.predict_proba(X_test_s)[:, 1]
p_lstm_real = get_lstm_probs_trimmed(X_test_s)  # length = len(X_test_s) - SEQ_LEN

# Align everything to the LSTM's trimmed output. The LSTM consumes
# SEQ_LEN rows to produce one prediction, so its first real
# prediction corresponds to original index SEQ_LEN - 1 (0-indexed).
n = len(p_lstm_real)
p_rf            = p_rf_full[SEQ_LEN - 1: SEQ_LEN - 1 + n]
p_lstm          = p_lstm_real
X_test_aligned  = X_test_raw[SEQ_LEN - 1: SEQ_LEN - 1 + n]
y_true          = y_test.astype(int)[SEQ_LEN - 1: SEQ_LEN - 1 + n]

R         = fuse_risk_score(p_rf, p_lstm, X_test_aligned)
decisions = decision(R)
y_pred    = binary_from_decision(decisions)

lines = print_metrics("1. IN-DISTRIBUTION (Full Framework, Flagged)", y_true, y_pred, p_rf)
results_log.extend(lines)

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
print("\n[2/8] RF-only ablation...")
y_pred_rf = (p_rf >= 0.5).astype(int)
lines = print_metrics("2. RF ONLY (Ablation)", y_true, y_pred_rf, p_rf)
results_log.extend(lines)

# ════════════════════════════════════════════════════════════
# EVALUATION 3 — LSTM ONLY (ablation)
# ════════════════════════════════════════════════════════════
print("\n[3/8] LSTM-only ablation...")
y_pred_lstm = (p_lstm >= 0.5).astype(int)
lines = print_metrics("3. LSTM ONLY (Ablation)", y_true, y_pred_lstm, p_lstm)
results_log.extend(lines)

# ════════════════════════════════════════════════════════════
# EVALUATION 4 — STATIC MFA BASELINE
# ════════════════════════════════════════════════════════════
print("\n[4/8] Static MFA baseline...")
static_pred = X_test_aligned[:, 7].astype(int)  # is_attack_ip column
lines = print_metrics("4. STATIC MFA BASELINE", y_true, static_pred)
results_log.extend(lines)

# ════════════════════════════════════════════════════════════
# EVALUATION 5 — ABLATION: ML FUSION ONLY (no contextual/behavioural)
# ════════════════════════════════════════════════════════════
print("\n[5/8] ML fusion only ablation (no contextual/behavioural)...")
ml_only_score = (0.6 * p_rf + 0.4 * p_lstm) * 100
ml_only_score = np.clip(ml_only_score, 0, 100)
dec_ml_only   = decision(ml_only_score)
y_pred_ml_only = binary_from_decision(dec_ml_only)
lines = print_metrics("5. ML FUSION ONLY (Ablation, no ctx/beh)",
                      y_true, y_pred_ml_only, p_rf)
results_log.extend(lines)
legit_mask_ml = y_true == 0
usr_ml = np.sum(dec_ml_only[legit_mask_ml] == 'STEP_UP_MFA') / np.sum(legit_mask_ml)
usr_ml_line = f"USR:       {usr_ml:.4f}"
print(usr_ml_line)
results_log.append(usr_ml_line)

# ════════════════════════════════════════════════════════════
# EVALUATION 6 — ABLATION: BEHAVIOURAL ONLY (no reputation)
# ════════════════════════════════════════════════════════════
print("\n[6/8] Behavioural only ablation (no reputation)...")
# NOTE: the exact original implementation for this ablation row could
# not be located in the codebase (checked all scripts, notebooks, and
# checkpoints). This is a reconstruction attempt based on the reported
# pattern (recall ~100%, precision ~76%, FPR ~31%), which suggests an
# OR-based flagging rule rather than an averaged score: flag as
# suspicious if EITHER is_night=1 OR login_freq is elevated.
is_night_col   = X_test_aligned[:, 2]
login_freq_col = X_test_aligned[:, 14]
LOGIN_FREQ_THRESHOLD = 10  # matches the "high login freq" noise injection range (10-20)

flagged_behav = (is_night_col == 1) | (login_freq_col >= LOGIN_FREQ_THRESHOLD)
y_pred_behav  = flagged_behav.astype(int)

lines = print_metrics("6. BEHAVIOURAL ONLY (Ablation, no reputation, OR-rule reconstruction)",
                      y_true, y_pred_behav)
results_log.extend(lines)
legit_mask_bh = y_true == 0
usr_bh = np.sum(y_pred_behav[legit_mask_bh] == 1) / np.sum(legit_mask_bh)
usr_bh_line = f"USR (proxy, = FPR for this binary rule): {usr_bh:.4f}"
print(usr_bh_line)
results_log.append(usr_bh_line)

# ════════════════════════════════════════════════════════════
# EVALUATION 7 — OUT-OF-DISTRIBUTION TEST SET
# ════════════════════════════════════════════════════════════
print("\n[7/8] Out-of-distribution evaluation...")
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

X_ood_raw = ood_df[FEATURE_COLS].values.astype(np.float32)
y_ood     = ood_df['label'].values.astype(int)
X_ood_sc  = scaler.transform(X_ood_raw)  # OOD data starts raw, so scaling here is correct

p_rf_ood_full   = rf_model.predict_proba(X_ood_sc)[:, 1]
p_lstm_ood_real = get_lstm_probs_trimmed(X_ood_sc)

n_ood = len(p_lstm_ood_real)
p_rf_ood   = p_rf_ood_full[SEQ_LEN - 1: SEQ_LEN - 1 + n_ood]
p_lstm_ood = p_lstm_ood_real
X_ood_al   = X_ood_raw[SEQ_LEN - 1: SEQ_LEN - 1 + n_ood]  # raw, for fusion
y_ood_al   = y_ood[SEQ_LEN - 1: SEQ_LEN - 1 + n_ood]
ood_df_al  = ood_df.iloc[SEQ_LEN - 1: SEQ_LEN - 1 + n_ood].reset_index(drop=True)

R_ood      = fuse_risk_score(p_rf_ood, p_lstm_ood, X_ood_al)
dec_ood    = decision(R_ood)
y_pred_ood = binary_from_decision(dec_ood)

lines = print_metrics("7. OOD TEST SET (Primary Reported Result)",
                      y_ood_al, y_pred_ood, p_rf_ood)
results_log.extend(lines)

print("\n[8/8] OOD per-attack-type recall...")
print("\nOOD Per-Attack-Type Recall:")
ood_lines = ["\nOOD Per-Attack-Type Recall:"]
for atype in ['sim_swap', 'aitm_phishing', 'session_hijacking', 'mfa_fatigue']:
    mask = (ood_df_al['attack_type'] == atype).values
    if mask.sum() > 0:
        rec_t = recall_score(y_ood_al[mask], y_pred_ood[mask], zero_division=0)
        line = f"  {atype}: recall={rec_t:.4f} (n={mask.sum()})"
        print(line)
        ood_lines.append(line)
results_log.extend(ood_lines)

legit_ood = y_ood_al == 0
usr_ood   = np.sum(dec_ood[legit_ood] == 'STEP_UP_MFA') / np.sum(legit_ood)
ood_usr_line = f"OOD USR: {usr_ood:.4f}  (target <= 0.08)"
print(ood_usr_line)
results_log.append(ood_usr_line)

# ── Save results ──────────────────────────────────────────────
with open(RESULTS_PATH, 'w') as f:
    f.write('\n'.join(results_log))
print(f"\nResults saved to {RESULTS_PATH}")
print("=" * 60)
print("PHASE 11 (CLEAN) COMPLETE")
print("=" * 60)
