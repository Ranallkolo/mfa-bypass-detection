import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

RF_PATH     = '/home/da_otcifithom/mfa-bypass-detection/models/rf_model.pkl'
LSTM_PATH   = '/home/da_otcifithom/mfa-bypass-detection/models/lstm_model.pt'
SCALER_PATH = '/home/da_otcifithom/mfa-bypass-detection/models/scaler.pkl'
SEQ_LEN     = 10
N_RUNS      = 1000

print("=" * 60)
print("PHASE 7 — LATENCY BENCHMARKING")
print("=" * 60)

rf_model = joblib.load(RF_PATH)
scaler   = joblib.load(SCALER_PATH)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.3)
        self.fc      = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).squeeze()

INPUT_SIZE = 15
lstm_model = LSTMClassifier(INPUT_SIZE)
lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location='cpu'))
lstm_model.eval()

def predict_single(features_raw):
    """Full pipeline for one login event."""
    # scale
    x_sc = scaler.transform(features_raw.reshape(1, -1))

    # RF probability
    p_rf = rf_model.predict_proba(x_sc)[0, 1]

    # LSTM probability (single sequence)
    seq = np.tile(x_sc, (SEQ_LEN, 1)).astype(np.float32)
    seq_t = torch.tensor(seq).unsqueeze(0)
    with torch.no_grad():
        p_lstm = lstm_model(seq_t).item()

    # risk fusion
    ml_score          = 0.6 * p_rf + 0.4 * p_lstm
    contextual_score  = (features_raw[12] + features_raw[13]) / 2
    behavioural_score = (features_raw[14] / 20.0 + features_raw[2]) / 2
    R = (0.6 * ml_score + 0.2 * contextual_score + 0.2 * behavioural_score) * 100
    R = float(np.clip(R, 0, 100))

    # decision
    if R < 33:
        dec = 'ALLOW'
    elif R < 66:
        dec = 'STEP_UP_MFA'
    else:
        dec = 'BLOCK'

    return R, dec

# generate test events
np.random.seed(42)
test_events = np.random.rand(N_RUNS, INPUT_SIZE).astype(np.float32)

# warmup
for i in range(10):
    predict_single(test_events[i])

# benchmark
print(f"\nRunning {N_RUNS} predictions...")
latencies = []
for i in range(N_RUNS):
    start = time.perf_counter()
    predict_single(test_events[i])
    end = time.perf_counter()
    latencies.append((end - start) * 1000)

latencies = np.array(latencies)

print("\n" + "=" * 40)
print("LATENCY RESULTS (milliseconds)")
print("=" * 40)
print(f"Mean:    {np.mean(latencies):.2f} ms  (target < 500ms)")
print(f"Median:  {np.median(latencies):.2f} ms")
print(f"P95:     {np.percentile(latencies, 95):.2f} ms")
print(f"P99:     {np.percentile(latencies, 99):.2f} ms")
print(f"Min:     {np.min(latencies):.2f} ms")
print(f"Max:     {np.max(latencies):.2f} ms")
print(f"Std:     {np.std(latencies):.2f} ms")

target_met = np.mean(latencies) < 500
print(f"\nTarget < 500ms: {'✅ MET' if target_met else '❌ NOT MET'}")

# save
results = (
    f"LATENCY BENCHMARK RESULTS\n"
    f"Runs: {N_RUNS}\n"
    f"Mean:   {np.mean(latencies):.2f} ms\n"
    f"Median: {np.median(latencies):.2f} ms\n"
    f"P95:    {np.percentile(latencies, 95):.2f} ms\n"
    f"P99:    {np.percentile(latencies, 99):.2f} ms\n"
    f"Min:    {np.min(latencies):.2f} ms\n"
    f"Max:    {np.max(latencies):.2f} ms\n"
)
with open('/home/da_otcifithom/mfa-bypass-detection/results/latency_results.txt', 'w') as f:
    f.write(results)

print("=" * 60)
print("PHASE 7 COMPLETE")
print("=" * 60)