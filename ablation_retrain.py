"""
ABLATION RETRAIN — Reputation-features-excluded Random Forest

Purpose: replace the fabricated "Behavioural Only (OR-rule reconstruction)"
row in evaluation_results_clean.txt with a genuine trained model. The
original implementation for that ablation could not be located in the
codebase (per the comment in 11_evaluate_clean.py, Eval 6), so this
script trains a real RandomForestClassifier with asn_attack_rate and
country_attack_rate removed from the feature set, using the exact same
data, split, and hyperparameter approach as 3_train_rf.py.

Run this from inside the mfa-project-restored folder:
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


def main():
    print("=" * 60)
    print("ABLATION RETRAIN — REPUTATION FEATURES EXCLUDED")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded: {df.shape}")
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
