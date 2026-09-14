"""
ABLATION RETRAIN v2 — Reputation-features-excluded Random Forest
(now trained on the noise-injected, seed-corrected dataset for
consistency with the main framework's retrained models)

Purpose: replace the fabricated "Behavioural Only (OR-rule reconstruction)"
row with a genuine trained model. CORRECTION FROM v1: the first version
of this script trained on the raw full_dataset.csv (pre-seed-fix,
no noise injection), which is not on equal footing with the main RF/
LSTM models (retrain_on_fixed_data.py), which DO use noise-injected
data. This version applies the identical noise injection logic before
training, so the ablation result is directly comparable to the full
framework's reported numbers.

Run this from inside the mfa-project-restored folder, AFTER running
2_synthetic_fixed.py (so full_dataset.csv is the corrected version):
    python ablation_retrain.py

Output: results/ablation_no_reputation_results.txt
"""

import os
import numpy as np
import pandas as pd
import joblib
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
OUT_PATH = os.path.join(BASE_DIR, 'results', 'ablation_no_reputation_results.txt')
MODEL_OUT_PATH = os.path.join(BASE_DIR, 'models', 'rf_no_reputation_ablation.pkl')

RANDOM_SEED = 42

# Full feature set, as used everywhere else in the pipeline
FULL_FEATURE_COLS = [
    'hour', 'day_of_week', 'is_night',
    'device_mobile', 'device_desktop', 'device_tablet',
    'login_success', 'is_attack_ip', 'browser_known',
    'country_changed', 'asn_changed', 'device_changed',
    'asn_attack_rate', 'country_attack_rate',
    'login_freq'
]

# Ablated set — the two reputation/network features removed
REPUTATION_FEATURES = ['asn_attack_rate', 'country_attack_rate']
ABLATED_FEATURE_COLS = [c for c in FULL_FEATURE_COLS if c not in REPUTATION_FEATURES]


def inject_noise(df):
    """Identical noise injection logic to 10_retrain_clean.py /
    retrain_on_fixed_data.py, so this ablation is trained under the
    same realistic-overlap conditions as the main framework."""
    attacks = df[df['label'] == 1].copy()
    normals = df[df['label'] == 0].copy()
    n_attacks = len(attacks)
    n_normals = len(normals)

    mask = np.random.random(n_attacks) < 0.30
    attacks.loc[mask, 'browser_known'] = 1

    mask = np.random.random(n_attacks) < 0.35
    attacks.loc[mask, 'is_night'] = 0
    attacks.loc[mask, 'hour'] = np.random.randint(9, 18, size=mask.sum())

    mask = np.random.random(n_attacks) < 0.40
    attacks.loc[mask, 'is_attack_ip'] = 0

    mask = np.random.random(n_attacks) < 0.25
    attacks.loc[mask, 'country_changed'] = 0

    mask = np.random.random(n_attacks) < 0.20
    attacks.loc[mask, 'device_changed'] = 0

    mask = np.random.random(n_attacks) < 0.20
    attacks.loc[mask, 'asn_changed'] = 0

    mask = np.random.random(n_attacks) < 0.35
    attacks.loc[mask, 'asn_attack_rate'] = np.random.uniform(0.0, 0.15, size=mask.sum())
    attacks.loc[mask, 'country_attack_rate'] = np.random.uniform(0.0, 0.15, size=mask.sum())

    noise = np.random.normal(0, 0.12, size=(n_attacks, 2))
    attacks[['asn_attack_rate', 'country_attack_rate']] = (
        attacks[['asn_attack_rate', 'country_attack_rate']].values + noise
    ).clip(0, 1)

    mask = np.random.random(n_normals) < 0.15
    normals.loc[mask, 'country_changed'] = 1
    normals.loc[mask, 'asn_changed'] = 1

    mask = np.random.random(n_normals) < 0.12
    normals.loc[mask, 'device_changed'] = 1

    mask = np.random.random(n_normals) < 0.15
    normals.loc[mask, 'is_night'] = 1
    normals.loc[mask, 'hour'] = np.random.choice(
        list(range(22, 24)) + list(range(0, 7)), size=mask.sum())

    mask = np.random.random(n_normals) < 0.08
    normals.loc[mask, 'login_freq'] = np.random.randint(10, 21, size=mask.sum())

    mask = np.random.random(n_normals) < 0.10
    normals.loc[mask, 'asn_attack_rate'] = np.random.uniform(0.1, 0.4, size=mask.sum())

    noise = np.random.normal(0, 0.05, size=(n_normals, 2))
    normals[['asn_attack_rate', 'country_attack_rate']] = (
        normals[['asn_attack_rate', 'country_attack_rate']].values + noise
    ).clip(0, 1)

    df_noisy = pd.concat([attacks, normals], ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df_noisy


def main():
    print("=" * 60)
    print("ABLATION RETRAIN v2 — REPUTATION FEATURES EXCLUDED")
    print("(noise-injected, consistent with main framework)")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded: {df.shape}")

    print("Applying noise injection (same logic as main retrain)...")
    df = inject_noise(df)
    print(f"Noisy dataset: {df.shape}")

    print(f"Full feature set    : {FULL_FEATURE_COLS}")
    print(f"Ablated feature set : {ABLATED_FEATURE_COLS}")
    print(f"Removed             : {REPUTATION_FEATURES}")

    X = df[ABLATED_FEATURE_COLS].values
    y = df['label'].values

    # Same split strategy (stratified 80/20, same seed) as 3_train_rf.py,
    # so results are comparable to the full-feature model.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTrain size: {len(X_train):,}")
    print(f"Test size:  {len(X_test):,}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Same hyperparameters as the best config typically selected in
    # 3_train_rf.py (n_estimators=100, max_depth=15) rather than
    # re-running the full 5-fold grid search, to keep this fast.
    # If you want the identical model-selection process, swap this
    # for the same StratifiedKFold loop used in 3_train_rf.py.
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight='balanced'
    )

    print("\nTraining Random Forest on ablated feature set...")
    rf.fit(X_train_sc, y_train)

    y_pred = rf.predict(X_test_sc)
    y_proba = rf.predict_proba(X_test_sc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    lines = [
        "=" * 60,
        "ABLATION: REPUTATION FEATURES EXCLUDED (asn_attack_rate,",
        "country_attack_rate removed) — REAL TRAINED MODEL",
        "=" * 60,
        "",
        "This replaces the previous 'Behavioural Only (OR-rule",
        "reconstruction)' row, which was a hand-coded approximation,",
        "not a trained model (see 11_evaluate_clean.py Eval 6 comment).",
        "",
        f"Accuracy:  {acc:.4f}",
        f"Recall:    {rec:.4f}",
        f"Precision: {prec:.4f}",
        f"F1-Score:  {f1:.4f}",
        f"FPR:       {fpr:.4f}",
        f"ROC-AUC:   {auc:.4f}",
        "Confusion Matrix:",
        f"  TN={tn}  FP={fp}",
        f"  FN={fn}  TP={tp}",
        "",
        "Feature importances (ablated model):",
    ]

    importances = sorted(
        zip(ABLATED_FEATURE_COLS, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for rank, (feat, imp) in enumerate(importances, 1):
        lines.append(f"  {rank:2d}. {feat:<25s} {imp:.4f}")

    lines += [
        "",
        "For comparison, cite the full-feature model's reported",
        "metrics from results/evaluation_results_clean.txt (RF ONLY",
        "row) alongside this table in the response letter and paper.",
    ]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        f.write('\n'.join(lines))

    joblib.dump(rf, MODEL_OUT_PATH)

    print('\n'.join(lines))
    print(f"\nSaved results to {OUT_PATH}")
    print(f"Saved model to   {MODEL_OUT_PATH}")


if __name__ == '__main__':
    main()
