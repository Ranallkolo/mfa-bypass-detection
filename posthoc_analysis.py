"""
POST-HOC ANALYSIS — Confidence intervals, PR-AUC, realistic base-rate

Purpose: address reviewer points 3 (realistic class imbalance) and 8
(statistical validation). Uses ONLY the already-trained, already-saved
models and test data — no retraining.

Run this from inside the mfa-project-restored folder:
    python posthoc_analysis.py

Output: results/posthoc_analysis_results.txt

Notes on methodology (also written into the output file):
- Confidence intervals are computed via bootstrap resampling (n=1000)
  of the existing in-distribution test set predictions.
- PR-AUC is computed on the same predictions.
- Realistic base-rate re-evaluation resamples the existing test set's
  raw probabilities to target attack prevalences of 1% and 0.1%,
  since these are commonly cited real-world MFA-bypass rates.
  IMPORTANT CAVEAT (also in output): the "normal" pool available for
  this resampling is itself only a 1% sample of real normal traffic
  from the original RBA dataset (see 1_preprocess.py,
  NORMAL_SAMPLE_RATE=0.01), not the full real-world population. This
  is disclosed explicitly rather than presented as a limitless
  resampling.
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
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_PATH = os.path.join(BASE_DIR, 'data', 'train_test_split_clean.npz')
RF_PATH = os.path.join(BASE_DIR, 'models', 'rf_model.pkl')
LSTM_PATH = os.path.join(BASE_DIR, 'models', 'lstm_model.pt')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
OUT_PATH = os.path.join(BASE_DIR, 'results', 'posthoc_analysis_results.txt')

SEQ_LEN = 10
RANDOM_SEED = 42
T1, T2 = 34, 67
N_BOOTSTRAP = 1000
TARGET_PREVALENCES = [0.01, 0.001]  # 1% and 0.1%

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


def decision(R, t1=T1, t2=T2):
    d = np.where(R < t1, 'ALLOW', np.where(R < t2, 'STEP_UP_MFA', 'BLOCK'))
    return d


def binary_from_decision(decisions):
    return (decisions != 'ALLOW').astype(int)


def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    out = {'accuracy': acc, 'recall': rec, 'precision': prec,
           'f1': f1, 'fpr': fpr, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out['roc_auc'] = roc_auc_score(y_true, y_proba)
        out['pr_auc'] = average_precision_score(y_true, y_proba)
    return out


def bootstrap_ci(y_true, y_pred, y_proba, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Bootstrap resample (with replacement) over the test set to get
    95% confidence intervals for each metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metrics_list = {k: [] for k in ['accuracy', 'recall', 'precision', 'f1', 'fpr', 'roc_auc', 'pr_auc']}

    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue  # skip degenerate resamples
        m = compute_metrics(yt, yp, ypr)
        for k in metrics_list:
            if k in m:
                metrics_list[k].append(m[k])

    ci = {}
    for k, vals in metrics_list.items():
        if len(vals) == 0:
            continue
        vals = np.array(vals)
        ci[k] = {
            'mean': np.mean(vals),
            'lower_2.5': np.percentile(vals, 2.5),
            'upper_97.5': np.percentile(vals, 97.5),
        }
    return ci


def resample_to_prevalence(y_true, y_proba_positive_class, target_prev, seed=RANDOM_SEED):
    """Resample the existing test set (with replacement for the
    minority class if needed) to hit a target attack prevalence,
    preserving each sample's original predicted probability."""
    rng = np.random.RandomState(seed)
    attack_idx = np.where(y_true == 1)[0]
    normal_idx = np.where(y_true == 0)[0]

    n_attacks_available = len(attack_idx)
    # Keep all available normals, compute how many attacks that implies
    # at the target prevalence: prevalence = n_attack / (n_attack + n_normal)
    n_normals_available = len(normal_idx)
    n_attacks_target = int(round(
        (target_prev * n_normals_available) / (1 - target_prev)
    ))

    if n_attacks_target <= n_attacks_available:
        chosen_attack_idx = rng.choice(attack_idx, size=n_attacks_target, replace=False)
    else:
        # not enough real attack samples to hit target without replacement
        chosen_attack_idx = rng.choice(attack_idx, size=n_attacks_target, replace=True)

    combined_idx = np.concatenate([chosen_attack_idx, normal_idx])
    rng.shuffle(combined_idx)
    return combined_idx, (n_attacks_target > n_attacks_available)


def main():
    print("=" * 60)
    print("POST-HOC ANALYSIS — CIs, PR-AUC, REALISTIC BASE-RATE")
    print("=" * 60)

    rf_model = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)

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
    decisions = decision(R)
    y_pred = binary_from_decision(decisions)
    y_proba = R / 100.0

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"1. BOOTSTRAP CONFIDENCE INTERVALS (95%), n_boot={N_BOOTSTRAP}")
    report_lines.append("   Full Framework, in-distribution test set")
    report_lines.append("=" * 60)

    point_metrics = compute_metrics(y_true, y_pred, y_proba)
    ci = bootstrap_ci(y_true, y_pred, y_proba)

    for k in ['accuracy', 'recall', 'precision', 'f1', 'fpr', 'roc_auc', 'pr_auc']:
        if k in ci:
            pm = point_metrics.get(k, ci[k]['mean'])
            report_lines.append(
                f"  {k:12s}: point={pm:.4f}  95% CI=[{ci[k]['lower_2.5']:.4f}, {ci[k]['upper_97.5']:.4f}]"
            )

    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("2. PR-AUC (point estimate, imbalanced-deployment relevant)")
    report_lines.append("=" * 60)
    report_lines.append(f"  PR-AUC: {point_metrics.get('pr_auc', float('nan')):.4f}")
    report_lines.append(f"  ROC-AUC: {point_metrics.get('roc_auc', float('nan')):.4f}")
    report_lines.append(
        "  Note: PR-AUC is generally more informative than ROC-AUC"
    )
    report_lines.append(
        "  under class imbalance, which is why it is reported"
    )
    report_lines.append(
        "  alongside ROC-AUC here per the reviewer's request."
    )

    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("3. REALISTIC BASE-RATE RE-EVALUATION")
    report_lines.append("=" * 60)
    report_lines.append(
        "IMPORTANT CAVEAT: the normal-class pool available here is"
    )
    report_lines.append(
        "itself only a 1% sample of real normal traffic from the"
    )
    report_lines.append(
        "original RBA dataset (see 1_preprocess.py,"
    )
    report_lines.append(
        "NORMAL_SAMPLE_RATE=0.01), not the full real-world population."
    )
    report_lines.append(
        "Results below should be read as indicative of the trend,"
    )
    report_lines.append(
        "not as a substitute for evaluation on the full unsampled"
    )
    report_lines.append("normal-traffic population.")
    report_lines.append("")

    for target_prev in TARGET_PREVALENCES:
        idx, used_replacement = resample_to_prevalence(y_true, y_proba, target_prev)
        y_true_r = y_true[idx]
        y_pred_r = y_pred[idx]
        y_proba_r = y_proba[idx]

        m = compute_metrics(y_true_r, y_pred_r, y_proba_r)
        n_total = len(idx)
        n_attacks = int(y_true_r.sum())
        fp = m['fp']
        # False positives per million authentications
        fp_per_million = (fp / n_total) * 1_000_000 if n_total > 0 else float('nan')

        report_lines.append(f"--- Target prevalence: {target_prev*100:.2g}% ---")
        report_lines.append(f"  Total events (resampled): {n_total:,}  (attacks: {n_attacks:,})")
        if used_replacement:
            report_lines.append(
                "  NOTE: target required sampling attack rows WITH"
            )
            report_lines.append(
                "  replacement (not enough unique attack rows available"
            )
            report_lines.append(
                "  at this prevalence without repeats)."
            )
        report_lines.append(f"  Accuracy:  {m['accuracy']:.4f}")
        report_lines.append(f"  Recall:    {m['recall']:.4f}")
        report_lines.append(f"  Precision: {m['precision']:.4f}")
        report_lines.append(f"  F1-Score:  {m['f1']:.4f}")
        report_lines.append(f"  FPR:       {m['fpr']:.4f}")
        report_lines.append(f"  FP per million authentications: {fp_per_million:,.1f}")
        if 'pr_auc' in m:
            report_lines.append(f"  PR-AUC:    {m['pr_auc']:.4f}")
        report_lines.append("")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        f.write('\n'.join(report_lines))

    print('\n'.join(report_lines))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == '__main__':
    main()
