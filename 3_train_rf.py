import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler

DATA_PATH  = '/home/da_otcifithom/mfa-bypass-detection/data/full_dataset.csv'
MODEL_PATH = '/home/da_otcifithom/mfa-bypass-detection/models/rf_model.pkl'
SCALER_PATH= '/home/da_otcifithom/mfa-bypass-detection/models/scaler.pkl'
SPLIT_PATH = '/home/da_otcifithom/mfa-bypass-detection/data/train_test_split.npz'
RANDOM_SEED = 42

print("=" * 60)
print("PHASE 4 — RANDOM FOREST TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded: {df.shape}")

FEATURE_COLS = [
    'hour', 'day_of_week', 'is_night',
    'device_mobile', 'device_desktop', 'device_tablet',
    'login_success', 'is_attack_ip', 'browser_known',
    'country_changed', 'asn_changed', 'device_changed',
    'asn_attack_rate', 'country_attack_rate',
    'login_freq'
]

X = df[FEATURE_COLS].values
y = df['label'].values

# -- stratified train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"Train size: {len(X_train):,}")
print(f"Test size:  {len(X_test):,}")

# -- save split for LSTM to use same data
np.savez(SPLIT_PATH,
         X_train=X_train, X_test=X_test,
         y_train=y_train, y_test=y_test)
print(f"Split saved to {SPLIT_PATH}")

# -- scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved to {SCALER_PATH}")

# -- 5-fold cross validation to select best params
print("\nRunning 5-fold cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

configs = [
    {'n_estimators': 100, 'max_depth': 15},
    {'n_estimators': 200, 'max_depth': 20},
    {'n_estimators': 300, 'max_depth': 25},
]

best_f1     = 0
best_config = None

for cfg in configs:
    f1_scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_sc, y_train)):
        Xtr, Xval = X_train_sc[tr_idx], X_train_sc[val_idx]
        ytr, yval = y_train[tr_idx], y_train[val_idx]

        rf = RandomForestClassifier(
            n_estimators=cfg['n_estimators'],
            max_depth=cfg['max_depth'],
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        rf.fit(Xtr, ytr)
        preds = rf.predict(Xval)
        f1 = f1_score(yval, preds)
        f1_scores.append(f1)

    mean_f1 = np.mean(f1_scores)
    print(f"  Config {cfg} -> mean F1: {mean_f1:.4f}")

    if mean_f1 > best_f1:
        best_f1     = mean_f1
        best_config = cfg

print(f"\nBest config: {best_config} (F1={best_f1:.4f})")

# -- train final model on full training set
print("\nTraining final Random Forest on full training set...")
rf_final = RandomForestClassifier(
    n_estimators=best_config['n_estimators'],
    max_depth=best_config['max_depth'],
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf_final.fit(X_train_sc, y_train)

# -- evaluate on held-out test set
y_pred  = rf_final.predict(X_test_sc)
y_proba = rf_final.predict_proba(X_test_sc)[:, 1]

acc  = accuracy_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr  = fp / (fp + tn)

print("\n" + "=" * 40)
print("RANDOM FOREST — TEST SET RESULTS")
print("=" * 40)
print(f"Accuracy:  {acc:.4f}  (target >= 0.95)")
print(f"Recall:    {rec:.4f}  (target >= 0.90)")
print(f"Precision: {prec:.4f} (target >= 0.92)")
print(f"F1-Score:  {f1:.4f}  (target >= 0.91)")
print(f"ROC-AUC:   {auc:.4f}")
print(f"FPR:       {fpr:.4f}  (target <= 0.05)")
print(f"\nConfusion Matrix:")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")

# -- feature importance
print("\nTop 10 Feature Importances:")
importances = rf_final.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(min(10, len(FEATURE_COLS))):
    print(f"  {i+1}. {FEATURE_COLS[indices[i]]}: {importances[indices[i]]:.4f}")

joblib.dump(rf_final, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
print("=" * 60)
print("PHASE 4 COMPLETE")
print("=" * 60)