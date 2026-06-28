import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROCESSED_PATH = '/home/da_otcifithom/mfa-bypass-detection/data/processed.csv'
OUT_PATH       = '/home/da_otcifithom/mfa-bypass-detection/data/full_dataset.csv'
ATTACKS_PER_TYPE = 25000
NORMAL_CAP       = 100000
RANDOM_SEED      = 42
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("PHASE 3 — SYNTHETIC ATTACK GENERATION")
print("=" * 60)

df = pd.read_csv(PROCESSED_PATH)
print(f"\nLoaded processed data: {df.shape}")

# Use normal rows as the statistical base for generation
normals = df[df['label'] == 0].copy()
print(f"Normal rows available: {len(normals):,}")

def sample_base(n):
    """Sample n rows from normals as a base."""
    return normals.sample(n=n, replace=True, random_state=RANDOM_SEED).copy()

# ── Attack Type 1: SIM Swap ───────────────────────────────────
# Signature: device changed, ASN changed, country changed,
#            successful login, attack IP, odd hours
print("\nGenerating SIM Swap attacks...")
sim = sample_base(ATTACKS_PER_TYPE)
sim['device_changed']      = 1
sim['asn_changed']         = 1
sim['country_changed']     = np.random.choice([0, 1], ATTACKS_PER_TYPE, p=[0.3, 0.7])
sim['login_success']       = 1
sim['is_attack_ip']        = np.random.choice([0, 1], ATTACKS_PER_TYPE, p=[0.2, 0.8])
sim['is_night']            = np.random.choice([0, 1], ATTACKS_PER_TYPE, p=[0.3, 0.7])
sim['hour']                = np.random.choice(list(range(0, 6)) + list(range(22, 24)), ATTACKS_PER_TYPE)
sim['asn_attack_rate']     = np.random.uniform(0.4, 1.0, ATTACKS_PER_TYPE)
sim['country_attack_rate'] = np.random.uniform(0.3, 0.9, ATTACKS_PER_TYPE)
sim['device_mobile']       = 1
sim['device_desktop']      = 0
sim['device_tablet']       = 0
sim['login_freq']          = np.random.randint(1, 5, ATTACKS_PER_TYPE)
sim['attack_type']         = 'sim_swap'
sim['label']               = 1
print(f"  SIM Swap rows: {len(sim):,}")

# ── Attack Type 2: AiTM Phishing ─────────────────────────────
# Signature: country changed, high login freq in short window,
#            successful login, attack IP, browser unknown
print("Generating AiTM Phishing attacks...")
aitm = sample_base(ATTACKS_PER_TYPE)
aitm['country_changed']     = 1
aitm['asn_changed']         = np.random.choice([0, 1], ATTACKS_PER_TYPE, p=[0.2, 0.8])
aitm['device_changed']      = np.random.choice([0, 1], ATTACKS_PER_TYPE, p=[0.4, 0.6])
aitm['login_success']       = 1
aitm['is_attack_ip']        = np.random.choice([0, 1], ATTACKS_PER_TYPE, p=[0.1, 0.9])
aitm['browser_known']       = 0
aitm['login_freq']          = np.random.randint(8, 20, ATTACKS_PER_TYPE)
aitm['asn_attack_rate']     = np.random.uniform(0.5, 1.0, ATTACKS_PER_TYPE)
aitm['country_attack_rate'] = np.random.uniform(0.4, 1.0, ATTACKS_PER_TYPE)
aitm['hour']                = np.random.randint(0, 24, ATTACKS_PER_TYPE)
aitm['is_night']            = (
    (aitm['hour'] >= 22) | (aitm['hour'] <= 6)
).astype(int)
aitm['attack_type']         = 'aitm_phishing'
aitm['label']               = 1
print(f"  AiTM rows: {len(aitm):,}")

# ── Attack Type 3: Session Hijacking ─────────────────────────
# Signature: device changed, ASN changed, successful login,
#            attack IP, no prior failed attempts
print("Generating Session Hijacking attacks...")
sess = sample_base(ATTACKS_PER_TYPE)
sess['device_changed']      = 1
sess['asn_changed']         = 1
sess['country_changed']     = np.random.choice([0, 1], ATTACKS_PER_TYPE, p=[0.4, 0.6])
sess['login_success']       = 1
sess['is_attack_ip']        = np.random.choice([0, 1], ATTACKS_PER_TYPE, p=[0.15, 0.85])
sess['login_freq']          = np.random.randint(1, 3, ATTACKS_PER_TYPE)
sess['asn_attack_rate']     = np.random.uniform(0.3, 1.0, ATTACKS_PER_TYPE)
sess['country_attack_rate'] = np.random.uniform(0.2, 0.8, ATTACKS_PER_TYPE)
sess['browser_known']       = np.random.choice([0, 1], ATTACKS_PER_TYPE, p=[0.6, 0.4])
sess['hour']                = np.random.randint(0, 24, ATTACKS_PER_TYPE)
sess['is_night']            = (
    (sess['hour'] >= 22) | (sess['hour'] <= 6)
).astype(int)
sess['attack_type']         = 'session_hijacking'
sess['label']               = 1
print(f"  Session Hijacking rows: {len(sess):,}")

# ── Attack Type 4: MFA Fatigue ────────────────────────────────
# Signature: high login frequency, odd hours, repeated attempts,
#            eventual success after failures
print("Generating MFA Fatigue attacks...")
fatigue = sample_base(ATTACKS_PER_TYPE)
fatigue['login_freq']          = np.random.randint(8, 20, ATTACKS_PER_TYPE)
fatigue['is_night']            = 1
fatigue['hour']                = np.random.choice(list(range(22, 24)) + list(range(0, 6)), ATTACKS_PER_TYPE)
fatigue['login_success']       = 1
fatigue['device_changed']      = np.random.choice([0, 1], ATTACKS_PER_TYPE, p=[0.5, 0.5])
fatigue['asn_changed']         = np.random.choice([0, 1], ATTACKS_PER_TYPE, p=[0.6, 0.4])
fatigue['country_changed']     = np.random.choice([0, 1], ATTACKS_PER_TYPE, p=[0.7, 0.3])
fatigue['is_attack_ip']        = np.random.choice([0, 1], ATTACKS_PER_TYPE, p=[0.3, 0.7])
fatigue['asn_attack_rate']     = np.random.uniform(0.2, 0.8, ATTACKS_PER_TYPE)
fatigue['country_attack_rate'] = np.random.uniform(0.2, 0.7, ATTACKS_PER_TYPE)
fatigue['attack_type']         = 'mfa_fatigue'
fatigue['label']               = 1
print(f"  MFA Fatigue rows: {len(fatigue):,}")

# ── Combine all attacks ───────────────────────────────────────
all_attacks = pd.concat([sim, aitm, sess, fatigue], ignore_index=True)
print(f"\nTotal synthetic attacks: {len(all_attacks):,}")

# ── Sample normal rows ────────────────────────────────────────
normal_sample = normals.sample(n=NORMAL_CAP, random_state=RANDOM_SEED).copy()
normal_sample['attack_type'] = 'none'
print(f"Normal rows sampled:     {len(normal_sample):,}")

# ── Combine and balance ───────────────────────────────────────
FEATURE_COLS = [
    'hour', 'day_of_week', 'is_night',
    'device_mobile', 'device_desktop', 'device_tablet',
    'login_success', 'is_attack_ip', 'browser_known',
    'country_changed', 'asn_changed', 'device_changed',
    'asn_attack_rate', 'country_attack_rate',
    'login_freq', 'attack_type', 'label'
]

df_full = pd.concat([all_attacks, normal_sample], ignore_index=True)
df_full = df_full[FEATURE_COLS].dropna()
df_full = df_full.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"\nFinal combined dataset: {df_full.shape}")
print(f"Label distribution:\n{df_full['label'].value_counts()}")
print(f"Attack type distribution:\n{df_full['attack_type'].value_counts()}")

df_full.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
print("=" * 60)
print("PHASE 3 COMPLETE")
print("=" * 60)