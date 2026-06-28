import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/home/da_otcifithom/mfa-bypass-detection/rba-dataset.csv'
OUT_PATH  = '/home/da_otcifithom/mfa-bypass-detection/data/processed.csv'
CHUNK_SIZE = 500_000
NORMAL_SAMPLE_RATE = 0.01

print("=" * 60)
print("PHASE 2 — DATA LOADING & PREPROCESSING")
print("=" * 60)

# Global rate tracking
asn_attack_counts     = {}
asn_total_counts      = {}
country_attack_counts = {}
country_total_counts  = {}

# Global user baselines
user_country_map = {}
user_asn_map     = {}
user_device_map  = {}
user_login_counts = {}

def process_chunk(chunk):
    global asn_attack_counts, asn_total_counts
    global country_attack_counts, country_total_counts

    # -- drop unused columns
    chunk = chunk.drop(
        columns=['index', 'Round-Trip Time [ms]', 'User Agent String'],
        errors='ignore'
    )

    # -- fill missing
    chunk['Region']  = chunk['Region'].fillna('Unknown')
    chunk['City']    = chunk['City'].fillna('Unknown')
    chunk['Country'] = chunk['Country'].fillna('XX')

    # -- timestamp features
    chunk['Login Timestamp'] = pd.to_datetime(chunk['Login Timestamp'], errors='coerce')
    chunk['hour']        = chunk['Login Timestamp'].dt.hour.fillna(0).astype(int)
    chunk['day_of_week'] = chunk['Login Timestamp'].dt.dayofweek.fillna(0).astype(int)
    chunk['is_night']    = ((chunk['hour'] >= 22) | (chunk['hour'] <= 6)).astype(int)

    # -- device encoding
    chunk['device_mobile']  = (chunk['Device Type'] == 'mobile').astype(int)
    chunk['device_desktop'] = (chunk['Device Type'] == 'desktop').astype(int)
    chunk['device_tablet']  = (chunk['Device Type'] == 'tablet').astype(int)
    chunk['login_success']  = chunk['Login Successful'].astype(int)
    chunk['is_attack_ip']   = chunk['Is Attack IP'].astype(int)

    # -- browser
    chunk['browser'] = chunk['Browser Name and Version'].str.split().str[0].str.lower()
    top_browsers = ['chrome', 'firefox', 'safari', 'edge', 'opera']
    chunk['browser_known'] = chunk['browser'].isin(top_browsers).astype(int)

    # -- label
    chunk['label'] = chunk['Is Account Takeover'].astype(int)

    # -- user baselines (vectorised — first seen value per user)
    new_users = chunk.groupby('User ID').first().reset_index()
    for _, row in new_users.iterrows():
        uid = row['User ID']
        if uid not in user_country_map:
            user_country_map[uid] = row['Country']
        if uid not in user_asn_map:
            user_asn_map[uid] = row['ASN']
        if uid not in user_device_map:
            user_device_map[uid] = row['Device Type']

    # -- deviation features (vectorised)
    chunk['country_changed'] = chunk.apply(
        lambda r: int(user_country_map.get(r['User ID'], r['Country']) != r['Country']),
        axis=1
    )
    chunk['asn_changed'] = chunk.apply(
        lambda r: int(user_asn_map.get(r['User ID'], r['ASN']) != r['ASN']),
        axis=1
    )
    chunk['device_changed'] = chunk.apply(
        lambda r: int(user_device_map.get(r['User ID'], r['Device Type']) != r['Device Type']),
        axis=1
    )

    # -- attack rate features (vectorised using cumulative counts)
    # update counts using groupby aggregation
    asn_grp = chunk.groupby('ASN')['label'].agg(['sum', 'count'])
    for asn, row in asn_grp.iterrows():
        asn_attack_counts[asn] = asn_attack_counts.get(asn, 0) + row['sum']
        asn_total_counts[asn]  = asn_total_counts.get(asn, 0) + row['count']

    country_grp = chunk.groupby('Country')['label'].agg(['sum', 'count'])
    for c, row in country_grp.iterrows():
        country_attack_counts[c] = country_attack_counts.get(c, 0) + row['sum']
        country_total_counts[c]  = country_total_counts.get(c, 0) + row['count']

    chunk['asn_attack_rate'] = chunk['ASN'].map(
        lambda a: asn_attack_counts.get(a, 0) / max(asn_total_counts.get(a, 1), 1)
    )
    chunk['country_attack_rate'] = chunk['Country'].map(
        lambda c: country_attack_counts.get(c, 0) / max(country_total_counts.get(c, 1), 1)
    )

    # -- login frequency (vectorised)
    freq_map = chunk.groupby('User ID').cumcount() + 1
    chunk['login_freq'] = freq_map.clip(upper=20)

    return chunk

# -- main loop
print(f"\nLoading in chunks of {CHUNK_SIZE:,} rows...")
print(f"Keeping ALL attack rows + {NORMAL_SAMPLE_RATE*100:.0f}% of normal rows\n")

chunks_out   = []
total_attack = 0
total_normal = 0
chunk_num    = 0

FEATURE_COLS = [
    'hour', 'day_of_week', 'is_night',
    'device_mobile', 'device_desktop', 'device_tablet',
    'login_success', 'is_attack_ip', 'browser_known',
    'country_changed', 'asn_changed', 'device_changed',
    'asn_attack_rate', 'country_attack_rate',
    'login_freq', 'label'
]

reader = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE)

for chunk in reader:
    chunk_num += 1
    print(f"  Chunk {chunk_num}...", end=' ', flush=True)

    chunk = process_chunk(chunk)

    attacks = chunk[chunk['label'] == 1][FEATURE_COLS]
    normals = chunk[chunk['label'] == 0][FEATURE_COLS].sample(
        frac=NORMAL_SAMPLE_RATE, random_state=42
    )

    total_attack += len(attacks)
    total_normal += len(normals)
    chunks_out.append(pd.concat([attacks, normals]))

    print(f"attacks={len(attacks)}, normals={len(normals)}")

print("\nCombining chunks...")
df_final = pd.concat(chunks_out, ignore_index=True).dropna()

print(f"\nFinal shape:       {df_final.shape}")
print(f"Total attacks:     {total_attack:,}")
print(f"Total normals:     {total_normal:,}")
print(f"Label distribution:\n{df_final['label'].value_counts()}")

df_final.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
print("=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)