# MFA Bypass Detection Framework

> An AI-driven proof-of-concept for detecting and responding to Multi-Factor Authentication (MFA) bypass attack patterns — undergraduate thesis project, Air Force Institute of Technology (AFIT), Kaduna.

**Live demo:** https://mfa-bypass-detection.onrender.com

The system fuses a Random Forest and an LSTM model with contextual network reputation signals and behavioural heuristics into a single risk score `R`, then routes each login event to one of three tiers: **Allow**, **Step-Up MFA**, or **Block**. It targets four documented MFA bypass patterns: SIM swap, adversary-in-the-middle (AiTM) phishing, session hijacking, and MFA fatigue.

---

## Table of Contents

- [Results Summary](#results-summary)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Reproducibility & Verification](#reproducibility--verification)
- [Limitations](#limitations)
- [Author](#author)

---

## Results Summary

Evaluated on 40,000 held-out test events after noise-injected retraining (using `10_retrain_clean.py`):

### Individual Model Performance

| Metric | Random Forest | LSTM | Full Framework (Flagged) |
|---|---|---|---|
| Accuracy | 97.14% | 96.45% | 96.73% |
| Recall | 96.54% | 95.86% | 96.92% |
| Precision | 97.72% | 97.01% | 96.55% |
| F1-Score | 97.13% | 96.43% | 96.74% |
| AUC-ROC | 0.9968 | 0.9958 | 0.9968 |
| FPR | 2.25% | 2.96% | 3.47% |

"Flagged" definition: both BLOCK and STEP_UP_MFA count as detected (attacker is challenged or denied). ALLOW-only is considered a miss.

### Decision Distribution (39,990 aligned test events)

| Tier | Count | Percentage |
|---|---|---|
| Allow (R < 34) | 19,916 | 49.8% |
| Step-Up MFA (34 ≤ R < 67) | 4,276 | 10.7% |
| Block (R ≥ 67) | 15,798 | 39.5% |

### Usability Metric

The framework introduces the **Unnecessary Step-Up Rate (USR)** — a novel metric measuring the proportion of legitimate users incorrectly routed to Step-Up MFA, quantifying usability friction separately from FPR.

**USR: 3.26%** against a target of ≤ 8%.

### Ablation Study

| Configuration | Recall | Precision | F1 | FPR | Key Finding |
|---|---|---|---|---|---|
| RF alone | 96.54% | 97.72% | 97.13% | 2.25% | Strong standalone baseline |
| LSTM alone | 95.86% | 97.01% | 96.43% | 2.96% | Slightly below RF |
| ML fusion only (no ctx/beh) | 98.02% | 95.55% | 96.77% | 4.56% | FPR rises without dampening |
| Behavioural only (no reputation) | 99.96% | 76.36% | 86.58% | 30.96% | FPR collapses; reputation critical |
| Full framework | 96.92% | 96.55% | 96.74% | 3.47% | Optimal balance |

### Out-of-Distribution Evaluation

Independently seeded test set (seed=999), varied attack parameters across all four bypass types:

| Metric | Value |
|---|---|
| Recall | 97.82% |
| Precision | 97.25% |
| F1-Score | 97.53% |
| AUC-ROC | 0.9973 |
| FPR | 2.77% |

Per-attack-type OOD recall: SIM Swap 97.2%, AiTM Phishing 98.7%, Session Hijacking 96.9%, MFA Fatigue 98.4%.

### Latency Benchmark

1,000-run benchmark on Azure `Standard_DS3_v2` (2 vCPUs, 7.7 GB RAM, Ubuntu):

| Metric | Value | Target |
|---|---|---|
| Mean | 47.51 ms | < 500 ms ✅ |
| Median | 47.37 ms | < 500 ms ✅ |
| P95 | 50.14 ms | < 500 ms ✅ |
| P99 | 58.19 ms | < 500 ms ✅ |
| Min | 35.77 ms | — |
| Max | 62.18 ms | — |

All six predefined efficiency parameter targets met or exceeded.

---

## Architecture

The system is organised into five layers:

### Layer 1 — Data Collection
RBA Login Dataset (Wiefling et al., 2020, ~31M rows, Kaggle) combined with 100k synthetically generated attack records covering all four bypass patterns. Data processed in chunks due to RAM constraints.

### Layer 2 — Feature Extraction
15 engineered features across four categories:

| Category | Features |
|---|---|
| Temporal | `hour`, `day_of_week`, `is_night` |
| Device | `device_mobile`, `device_desktop`, `device_tablet`, `device_changed`, `browser_known` |
| Network reputation | `is_attack_ip`, `country_changed`, `asn_changed`, `asn_attack_rate`, `country_attack_rate` |
| Behavioural | `login_success`, `login_freq` |

Features normalised with `StandardScaler` for model input. Raw (pre-scaling) values used separately for the contextual/behavioural fusion score components.

### Layer 3 — Anomaly Detection Engine
Two models trained independently on 160,000 balanced events:

| Model | Configuration |
|---|---|
| Random Forest | 100 trees, max depth 15, class_weight=balanced, seed=42 |
| LSTM | 2 layers, 64 units, dropout 0.3, seq_len=10, 20 epochs, Adam lr=0.001 |

### Layer 4 — Decision Engine
Risk score fusion formula:

```
R = (0.6 × ML + 0.2 × Contextual + 0.2 × Behavioural) × 100

ML          = 0.6 × P_RF + 0.4 × P_LSTM
Contextual  = (asn_attack_rate + country_attack_rate) / 2
Behavioural = (login_freq / 20 + is_night) / 2
```

Note: Contextual and Behavioural scores are computed on raw (pre-scaling) feature values, not StandardScaler-transformed values — this is intentional and documented in the corrected evaluation scripts.

| Tier | Score Range | Action |
|---|---|---|
| Allow | R < 34 | Login proceeds |
| Step-Up MFA | 34 ≤ R < 67 | TOTP challenge required |
| Block | R ≥ 67 | Access denied, alert raised |

### Layer 5 — Reporting Dashboard
FastAPI application (`api/main.py`) with:
- Live audit logging (SQLite)
- Decision distribution doughnut chart
- Risk score histogram across five bands
- Recent decisions audit table (last 20 entries, live)
- Quick Simulator with Randomise button
- Attack Demo panel — six preset scenarios + Run All button
- Full Simulate tab — manual event input with all 15 features and fusion breakdown

---

## Project Structure

```
mfa-bypass-detection/
├── api/
│   └── main.py                    # FastAPI app, dashboard, prediction endpoint
├── models/
│   ├── rf_model.pkl               # Random Forest (100 trees, depth 15)
│   ├── lstm_model.pt              # LSTM (2 layers, 64 units, seq=10)
│   └── scaler.pkl                 # StandardScaler fitted on noisy training data
├── data/                          # Excluded from repo — see Dataset section
├── results/
│   ├── evaluation_results.txt     # Legacy evaluation output
│   └── evaluation_results_clean.txt  # Verified clean evaluation output
├── 1_preprocess.py                # Data loading and cleaning
├── 2_synthetic.py                 # Synthetic attack record generation
├── 3_train_rf.py                  # Random Forest training (original)
├── 4_train_lstm.py                # LSTM training (original)
├── 5_evaluate.py                  # Evaluation (legacy, debugging fixes applied)
├── 6_latency.py                   # Latency benchmarking
├── 7_demo.py                      # CLI demo script
├── 8_dashboard.py                 # Dashboard prototyping
├── 9_retrain_with_noise.py        # Retraining with noise injection (legacy)
├── 10_retrain_clean.py            # ✅ Canonical retraining script (corrected)
├── 11_evaluate_clean.py           # ✅ Canonical evaluation script (corrected)
├── generate_charts.py             # Chart generation for thesis figures
├── Dockerfile
└── requirements.txt
```

Scripts `10_retrain_clean.py` and `11_evaluate_clean.py` are the canonical, bug-fixed versions of the pipeline. See [Reproducibility & Verification](#reproducibility--verification) for details.

---

## Dataset

This repo does **not** include the underlying dataset files due to size:

| File | Size | Notes |
|---|---|---|
| `rba-dataset.csv` | ~8.5 GB | Raw RBA Login Dataset, excluded from git |
| `data/full_dataset.csv` | ~13 MB | Derived from raw dataset by `1_preprocess.py` |
| `data/train_test_split_clean.npz` | ~25 MB | Generated by `10_retrain_clean.py` |

**To obtain the raw dataset:** Download from [Kaggle](https://www.kaggle.com/) — search "RBA Dataset" or "Wiefling Risk-Based Authentication". Place at repo root as `rba-dataset.csv`.

The trained models (`models/rf_model.pkl`, `models/lstm_model.pt`, `models/scaler.pkl`) **are included in this repo** and are sufficient to run the dashboard and prediction endpoint immediately without downloading the dataset.

---

## Setup

```bash
git clone https://github.com/Ranallkolo/mfa-bypass-detection.git
cd mfa-bypass-detection
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Python 3.12 is recommended (matches the training environment). The `requirements.txt` pins compatible versions for all core dependencies.

---

## Usage

### Run the dashboard (fastest — no dataset needed)

Uses the pretrained models already included in the repo:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser.

### Test the prediction endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 3,
    "is_night": 1,
    "is_attack_ip": 1,
    "country_changed": 1,
    "asn_attack_rate": 0.9,
    "country_attack_rate": 0.85
  }'
```

Response includes risk score `R`, decision tier, RF probability, LSTM probability, and latency.

### Re-run evaluation only (no retraining)

Uses existing models, produces all evaluation results:

```bash
python 11_evaluate_clean.py
```

Covers: in-distribution full framework, RF-only, LSTM-only, static baseline, ML-fusion-only, behavioural-only, OOD, and per-attack-type OOD recall.

### Full pipeline from scratch (requires raw dataset)

```bash
python 1_preprocess.py
python 2_synthetic.py
python 10_retrain_clean.py    # retrains both models, saves clean split
python 11_evaluate_clean.py   # evaluates against clean split
python 6_latency.py           # latency benchmark
```

### Generate result charts

```bash
python generate_charts.py
```

Produces six PNG figures (300 DPI) covering model comparison, decision distribution, ablation study, OOD per-attack-type recall, latency benchmark, and comparative results.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Live monitoring dashboard (HTML) |
| `/predict` | POST | Accepts a `LoginEvent` JSON, returns risk score and decision |
| `/audit/recent` | GET | Last N decisions from the SQLite audit log |
| `/audit/stats` | GET | Aggregate statistics across all logged decisions |

### LoginEvent Schema

Key fields (all optional with defaults — minimal request shown in Usage above):

| Field | Type | Default | Description |
|---|---|---|---|
| `hour` | int | 12 | Login hour (0–23) |
| `is_night` | int | 0 | 1 if hour ≥ 22 or ≤ 6 |
| `is_attack_ip` | int | 0 | 1 if IP found in threat feed |
| `country_changed` | int | 0 | 1 if country differs from user baseline |
| `asn_attack_rate` | float | 0.0 | Historical attack rate of origin ASN (0–1) |
| `country_attack_rate` | float | 0.0 | Historical attack rate of origin country (0–1) |
| `login_freq` | int | 1 | Login attempts in the past hour |

Full schema with all 15 fields is documented in `api/main.py`.

---

## Reproducibility & Verification

After initial submission, the pipeline was independently re-run to verify reproducibility. This process identified and corrected two implementation issues:

**Issue 1 — Double-scaling:** `5_evaluate.py` applied `StandardScaler.transform()` to data that had already been scaled when saved to the `.npz` file, distorting every feature used for inference.

**Issue 2 — Scaled-vs-raw feature mismatch in fusion score:** The contextual and behavioural components of the risk formula expect raw 0–1 reputation rates, but were receiving StandardScaler-transformed values (centred near 0, with negative numbers), which corrupted the `.clip(0, 1)` logic and shifted the risk score distribution.

### Canonical scripts

After applying fixes:

- **`10_retrain_clean.py`** — corrected retraining script. Saves both scaled (for RF/LSTM) and raw (for fusion) test features separately. Uses corrected 34/67 thresholds and the Flagged decision definition throughout.
- **`11_evaluate_clean.py`** — corrected evaluation script. No double-scaling, real LSTM predictions only (no mean-fill padding), consistent thresholds, Flagged definition.

The original `9_retrain_with_noise.py` and `5_evaluate.py` are retained for audit purposes and reflect the partial debugging fixes applied during verification, but are superseded by the clean scripts above.

### Verification outcomes

| Section | Result |
|---|---|
| RF alone | ✅ Exact match |
| LSTM alone | ✅ Exact match |
| ML fusion only | ✅ Exact match |
| Full Framework | ✅ Within 0.5pp (96.92% vs 96.53% originally reported) |
| OOD evaluation | ✅ Reproduced closely |
| Behavioural only | ⚠️ Original implementation not recoverable from version control — see note below |

**Behavioural-only ablation note:** The exact implementation used to produce the behavioural-only ablation row (Recall 99.96%, F1 86.58%, FPR 30.96%) could not be located in version control. The figures are retained from the original experimental run. The qualitative finding they support — that reputation features are critical and removing them collapses FPR — is independently corroborated by the RF feature importance table (`asn_attack_rate` + `country_attack_rate` = ~53% combined weight) and by the ML-fusion-only ablation (FPR rises from 2.94% to 4.56% when contextual/behavioural components are removed).

---

## Limitations

1. The RBA dataset is primarily IP-reputation-based, not a confirmed MFA-bypass-specific dataset. The top two RF features (`country_attack_rate`, `asn_attack_rate`) together account for ~53% of decisions — the framework currently operates largely as a reputation system.

2. No live identity provider integration. Decisions are demonstrated via the dashboard but not enforced through API calls to a real MFA system.

3. Training and test sets share the same synthetic generation rules. OOD evaluation used independently seeded data with the same programmatic rules, so results reflect generalisation within the generation methodology rather than to real-world attack traffic.

4. Dataset balanced 50/50 (attack/normal) — does not reflect real-world attack prevalence (typically < 0.001%). Precision would degrade substantially at production scale.

5. Latency benchmarked under single-instance, unloaded conditions on Azure `Standard_DS3_v2`. Re-measurement under concurrent load is recommended before any production deployment.

6. Comparison with Hoda et al. (2025) is contextual rather than a controlled benchmark — their evaluation used an undocumented internal dataset under unspecified conditions, while this framework was evaluated on the publicly available RBA Login Dataset.

A full discussion of limitations and future work is in Chapter 5 of the thesis.

---

## Author

**Abdulrahman Abdulkadir**
Student ID: U22CYS1006
Air Force Institute of Technology (AFIT), Kaduna
Supervisor: Dr Bello Musa Yakubu
