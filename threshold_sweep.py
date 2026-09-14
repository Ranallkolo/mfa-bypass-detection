"""
THRESHOLD SWEEP — finding a better operating point at realistic prevalence

Purpose: the post-hoc base-rate analysis showed precision collapsing at
low attack prevalence (e.g. 2.8% precision at 0.1% prevalence) even
though FPR stays around 3.5%. This script checks whether raising the
decision threshold (T1, the ALLOW/STEP_UP_MFA cutoff) trades a small
amount of recall for a large improvement in precision at realistic
prevalence — giving a concrete, evidence-based deployment
recommendation instead of a vague "recalibrate thresholds" statement.

No retraining — reuses the same models, same risk-fusion formula, same
in-distribution test set as posthoc_analysis.py, just sweeps T1 across
a range of values before resampling to each target prevalence.

Run this from inside the mfa-project-restored folder:
    python threshold_sweep.py

Output: results/threshold_sweep_results.txt
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_PATH = os.path.join(BASE_DIR, 'data', 'train_test_split_clean.npz')
RF_PATH = os.path.join(BASE_DIR, 'models', 'rf_model.pkl')
LSTM_PATH = os.path.join(BASE_DIR, 'models', 'lstm_model.pt')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
OUT_PATH = os.path.join(BASE_DIR, 'results', 'threshold_sweep_results.txt')

SEQ_LEN = 10
RANDOM_SEED = 42

# Sweep candidate T1 values (ALLOW/STEP_UP cutoff on the 0-100 risk
# score). Original was T1=34. We sweep higher values only, since the
# goal is fewer false positives (raising the bar to flag something).
T1_CANDIDATES = [34, 40, 50, 60, 70, 80, 85, 90]
T2 = 67  # keep STEP_UP/BLOCK cutoff fixed; only tune the ALLOW cutoff,
         # since that's what drives the flagged-vs-not-flagged decision
         # used for recall/precision/FPR

TARGET_PREVALENCES = [0.01, 0.001]

FEATURE_COLS = [
    'hour', 'day_of_week', 'is_night',
    'device_mobile', 'device_desktop', 'device_tablet',
    'login_success', 'is_attack_ip', 'browser_known',
    'country_changed', 'asn_changed', 'device_changed',
    'asn_attack_rate', 'country_attack_rate', 'login_freq'
]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).squeeze()


def get_lstm_probs_trimmed(model, X, seq_len=SEQ_LEN):
    Xs = []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
    Xs = np.array(Xs, dtype=np.float32)
    ds = TensorDataset(torch.tensor(Xs))
    dl = DataLoader(ds, batch_size=512, shuffle=False)
    probs = []
    with torch.no_grad():
        for (Xb,) in dl:
            p = model(Xb).cpu().numpy()
            probs.extend(p)
    return np.array(probs)


def fuse_risk_score(p_rf, p_lstm, X_raw):
    ml_score = 0.6 * p_rf + 0.4 * p_lstm
    contextual_score = (X_raw[:, 12] + X_raw[:, 13]) / 2
    contextual_score = contextual_score.clip(0, 1)
    behavioural_score = (X_raw[:, 14] / 20.0 + X_raw[:, 2]) / 2
    behavioural_score = behavioural_score.clip(0, 1)
    R = (0.6 * ml_score + 0.2 * contextual_score + 0.2 * behavioural_score) * 100
    return np.clip(R, 0, 100)


def binary_from_threshold(R, t1):
    """Flagged (STEP_UP or BLOCK) = anything >= t1."""
    return (R >= t1).astype(int)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {'accuracy': acc, 'recall': rec, 'precision': prec, 'f1': f1,
            'fpr': fpr, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


def resample_to_prevalence(y_true, target_prev, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    attack_idx = np.where(y_true == 1)[0]
    normal_idx = np.where(y_true == 0)[0]
    n_normals_available = len(normal_idx)
    n_attacks_available = len(attack_idx)
    n_attacks_target = int(round(
        (target_prev * n_normals_available) / (1 - target_prev)
    ))
    if n_attacks_target <= n_attacks_available:
        chosen_attack_idx = rng.choice(attack_idx, size=n_attacks_target, replace=False)
        used_repl = False
    else:
        chosen_attack_idx = rng.choice(attack_idx, size=n_attacks_target, replace=True)
        used_repl = True
    combined_idx = np.concatenate([chosen_attack_idx, normal_idx])
    rng.shuffle(combined_idx)
    return combined_idx, used_repl


def main():
    print("=" * 60)
    print("THRESHOLD SWEEP — FINDING A BETTER OPERATING POINT")
    print("=" * 60)

    rf_model = joblib.load(RF_PATH)
    data = np.load(SPLIT_PATH)
    X_test_s = data['X_test_scaled'].astype(np.float32)
    X_test_raw = data['X_test_raw'].astype(np.float32)
    y_test = data['y_test'].astype(np.float32)
    INPUT_SIZE = X_test_s.shape[1]

    lstm_model = LSTMClassifier(INPUT_SIZE)
    lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location='cpu'))
    lstm_model.eval()

    p_rf_full = rf_model.predict_proba(X_test_s)[:, 1]
    p_lstm_real = get_lstm_probs_trimmed(lstm_model, X_test_s)

    n = len(p_lstm_real)
    p_rf = p_rf_full[SEQ_LEN - 1: SEQ_LEN - 1 + n]
    p_lstm = p_lstm_real
    X_test_aligned = X_test_raw[SEQ_LEN - 1: SEQ_LEN - 1 + n]
    y_true = y_test.astype(int)[SEQ_LEN - 1: SEQ_LEN - 1 + n]

    R = fuse_risk_score(p_rf, p_lstm, X_test_aligned)

    report_lines = []
    report_lines.append(
        "Threshold sweep on the fused risk score R (0-100 scale)."
    )
    report_lines.append(
        "T1 = ALLOW/flagged cutoff (original thesis value: 34)."
    )
    report_lines.append(
        "'Flagged' = STEP_UP_MFA or BLOCK, i.e. R >= T1."
    )
    report_lines.append("")

    for target_prev in TARGET_PREVALENCES:
        idx, used_repl = resample_to_prevalence(y_true, target_prev)
        y_true_r = y_true[idx]
        R_r = R[idx]

        report_lines.append("=" * 60)
        report_lines.append(f"TARGET PREVALENCE: {target_prev*100:.2g}%")
        report_lines.append("=" * 60)
        if used_repl:
            report_lines.append(
                "  (attack rows sampled with replacement to hit this prevalence)"
            )
        report_lines.append(
            f"{'T1':>5} | {'Recall':>8} | {'Precision':>10} | {'F1':>8} | {'FPR':>8} | {'FP/million':>12}"
        )
        report_lines.append("-" * 66)

        for t1 in T1_CANDIDATES:
            y_pred = binary_from_threshold(R_r, t1)
            m = compute_metrics(y_true_r, y_pred)
            fp_per_million = (m['fp'] / len(y_true_r)) * 1_000_000
            report_lines.append(
                f"{t1:>5} | {m['recall']:>8.4f} | {m['precision']:>10.4f} | "
                f"{m['f1']:>8.4f} | {m['fpr']:>8.4f} | {fp_per_million:>12,.1f}"
            )
        report_lines.append("")

    report_lines.append(
        "Interpretation: look for the lowest T1 (fewest recall points"
    )
    report_lines.append(
        "sacrificed) that still delivers a meaningfully lower FPR /"
    )
    report_lines.append(
        "FP-per-million and a materially better precision than the"
    )
    report_lines.append(
        "original T1=34 operating point, before deciding what to"
    )
    report_lines.append(
        "recommend as the realistic-prevalence deployment threshold."
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        f.write('\n'.join(report_lines))

    print('\n'.join(report_lines))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == '__main__':
    main()
