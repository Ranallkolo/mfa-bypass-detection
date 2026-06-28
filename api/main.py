from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
import joblib
import sqlite3
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

RF_PATH     = '/home/da_otcifithom/mfa-bypass-detection/models/rf_model.pkl'
LSTM_PATH   = '/home/da_otcifithom/mfa-bypass-detection/models/lstm_model.pt'
SCALER_PATH = '/home/da_otcifithom/mfa-bypass-detection/models/scaler.pkl'
DB_PATH     = '/home/da_otcifithom/mfa-bypass-detection/logs/audit.db'
SEQ_LEN     = 10
INPUT_SIZE  = 15

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

print("Loading models...")
rf_model   = joblib.load(RF_PATH)
scaler     = joblib.load(SCALER_PATH)
lstm_model = LSTMClassifier(INPUT_SIZE)
lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location='cpu'))
lstm_model.eval()
print("Models loaded.")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT,
            user_id       TEXT,
            ip_address    TEXT,
            risk_score    REAL,
            p_rf          REAL,
            p_lstm        REAL,
            decision      TEXT,
            latency_ms    REAL,
            attack_type   TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_decision(user_id, ip_address, risk_score, p_rf,
                 p_lstm, decision, latency_ms, attack_type='manual'):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO audit_log
        (timestamp,user_id,ip_address,risk_score,
         p_rf,p_lstm,decision,latency_ms,attack_type)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (datetime.utcnow().isoformat(), user_id, ip_address,
          round(risk_score,4), round(p_rf,4), round(p_lstm,4),
          decision, round(latency_ms,2), attack_type))
    conn.commit()
    conn.close()

init_db()

app = FastAPI(title="MFA Bypass Detection Framework")

class LoginEvent(BaseModel):
    user_id:             str   = "user_001"
    ip_address:          str   = "0.0.0.0"
    attack_type:         str   = "manual"
    hour:                int   = 12
    day_of_week:         int   = 0
    is_night:            int   = 0
    device_mobile:       int   = 0
    device_desktop:      int   = 1
    device_tablet:       int   = 0
    login_success:       int   = 1
    is_attack_ip:        int   = 0
    browser_known:       int   = 1
    country_changed:     int   = 0
    asn_changed:         int   = 0
    device_changed:      int   = 0
    asn_attack_rate:     float = 0.0
    country_attack_rate: float = 0.0
    login_freq:          int   = 1

def predict(event: LoginEvent):
    features = np.array([[
        event.hour, event.day_of_week, event.is_night,
        event.device_mobile, event.device_desktop, event.device_tablet,
        event.login_success, event.is_attack_ip, event.browser_known,
        event.country_changed, event.asn_changed, event.device_changed,
        event.asn_attack_rate, event.country_attack_rate, event.login_freq
    ]], dtype=np.float32)

    x_sc   = scaler.transform(features)
    p_rf   = float(rf_model.predict_proba(x_sc)[0, 1])
    p_lstm = p_rf  # LSTM needs true sequence; RF-only for single-event API
    ml_score          = p_rf
    contextual_score  = (event.asn_attack_rate + event.country_attack_rate) / 2
    behavioural_score = (event.login_freq / 20.0 + event.is_night) / 2
    R = float(np.clip(
        (0.6*ml_score + 0.2*contextual_score + 0.2*behavioural_score)*100,
        0, 100
    ))

    if R < 34:
        dec = 'ALLOW'
        msg = 'Login proceeds normally. JWT issued.'
    elif R < 67:
        dec = 'STEP_UP_MFA'
        msg = 'Suspicious activity detected. TOTP verification required.'
    else:
        dec = 'BLOCK'
        msg = 'High risk detected. Access denied. Security alert raised.'

    return R, dec, msg, p_rf, p_lstm

@app.get("/")
def root():
    return HTMLResponse(get_html())

@app.post("/predict")
def predict_endpoint(event: LoginEvent):
    start = time.perf_counter()
    try:
        R, dec, msg, p_rf, p_lstm = predict(event)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = (time.perf_counter() - start) * 1000
    log_decision(event.user_id, event.ip_address, R,
                 p_rf, p_lstm, dec, latency_ms, event.attack_type)
    return {
        "user_id": event.user_id, "risk_score": round(R,2),
        "decision": dec, "message": msg,
        "p_rf": round(p_rf,4), "p_lstm": round(p_lstm,4),
        "latency_ms": round(latency_ms,2)
    }

@app.get("/audit/recent")
def get_recent(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(f"""
        SELECT timestamp,user_id,ip_address,risk_score,
               decision,latency_ms,attack_type
        FROM audit_log ORDER BY id DESC LIMIT {limit}
    """).fetchall()
    conn.close()
    return {"count": len(rows), "records": [
        {"timestamp":r[0],"user_id":r[1],"ip_address":r[2],
         "risk_score":r[3],"decision":r[4],
         "latency_ms":r[5],"attack_type":r[6]} for r in rows
    ]}

@app.get("/audit/stats")
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    total   = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    allows  = conn.execute("SELECT COUNT(*) FROM audit_log WHERE decision='ALLOW'").fetchone()[0]
    stepups = conn.execute("SELECT COUNT(*) FROM audit_log WHERE decision='STEP_UP_MFA'").fetchone()[0]
    blocks  = conn.execute("SELECT COUNT(*) FROM audit_log WHERE decision='BLOCK'").fetchone()[0]
    avg_lat = conn.execute("SELECT AVG(latency_ms) FROM audit_log").fetchone()[0]
    avg_risk= conn.execute("SELECT AVG(risk_score) FROM audit_log").fetchone()[0]
    conn.close()
    return {
        "total_requests": total,
        "decisions": {"ALLOW":allows,"STEP_UP_MFA":stepups,"BLOCK":blocks},
        "avg_latency_ms": round(avg_lat,2) if avg_lat else 0,
        "avg_risk_score": round(avg_risk,2) if avg_risk else 0
    }

def get_html():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MFA Bypass Detection Framework</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Segoe UI',sans-serif; background:#0f1117; color:#e2e8f0; }
  nav { background:#1a1d27; border-bottom:1px solid #2d3148;
        padding:0 2rem; display:flex; align-items:center;
        justify-content:space-between; height:60px; position:sticky; top:0; z-index:100; }
  .nav-brand { font-size:1.1rem; font-weight:700; color:#6366f1; }
  .nav-brand span { color:#e2e8f0; }
  .nav-tabs { display:flex; gap:0.5rem; }
  .tab-btn { background:none; border:none; color:#94a3b8; padding:0.5rem 1rem;
             border-radius:6px; cursor:pointer; font-size:0.9rem; transition:all 0.2s; }
  .tab-btn:hover { background:#2d3148; color:#e2e8f0; }
  .tab-btn.active { background:#6366f1; color:#fff; }
  .status-dot { width:8px; height:8px; border-radius:50%; background:#22c55e;
                display:inline-block; margin-right:6px; animation:pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  .page { display:none; padding:2rem; max-width:1400px; margin:0 auto; }
  .page.active { display:block; }
  .card { background:#1a1d27; border:1px solid #2d3148; border-radius:12px; padding:1.5rem; }
  .card-title { font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;
                color:#94a3b8; margin-bottom:0.5rem; }
  .card-value { font-size:2rem; font-weight:700; }
  .grid-4 { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem; }
  .grid-2 { display:grid; grid-template-columns:repeat(2,1fr); gap:1rem; margin-bottom:1.5rem; }
  .grid-3 { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem; }
  .green { color:#22c55e; } .yellow { color:#f59e0b; }
  .red { color:#ef4444; }   .purple { color:#6366f1; }
  table { width:100%; border-collapse:collapse; font-size:0.85rem; }
  th { text-align:left; padding:0.75rem 1rem; color:#94a3b8;
       border-bottom:1px solid #2d3148; font-weight:500; }
  td { padding:0.75rem 1rem; border-bottom:1px solid #1e2235; }
  tr:hover td { background:#1e2235; }
  .badge { padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
  .badge-allow { background:#052e16; color:#22c55e; border:1px solid #166534; }
  .badge-step  { background:#451a03; color:#f59e0b; border:1px solid #92400e; }
  .badge-block { background:#450a0a; color:#ef4444; border:1px solid #991b1b; }
  .risk-bar-bg { background:#2d3148; border-radius:4px; height:8px; width:120px;
                 display:inline-block; vertical-align:middle; }
  .risk-bar-fill { height:8px; border-radius:4px; transition:width 0.3s; }
  .form-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; }
  .form-group { display:flex; flex-direction:column; gap:0.4rem; }
  label { font-size:0.8rem; color:#94a3b8; }
  input, select { background:#0f1117; border:1px solid #2d3148; border-radius:8px;
                  color:#e2e8f0; padding:0.6rem 0.8rem; font-size:0.9rem; width:100%; }
  input:focus, select:focus { outline:none; border-color:#6366f1; }
  .btn { background:#6366f1; color:#fff; border:none; border-radius:8px;
         padding:0.75rem 2rem; font-size:0.95rem; cursor:pointer;
         font-weight:600; transition:all 0.2s; }
  .btn:hover { background:#4f46e5; }
  .btn-outline { background:none; border:1px solid #6366f1; color:#6366f1; }
  .btn-outline:hover { background:#6366f1; color:#fff; }
  .btn-green { background:#16a34a; }
  .btn-green:hover { background:#15803d; }
  .btn-yellow { background:#d97706; }
  .btn-yellow:hover { background:#b45309; }
  .btn-red { background:#dc2626; }
  .btn-red:hover { background:#b91c1c; }
  .gauge-wrap { text-align:center; margin:1rem 0; }
  .gauge-score { font-size:3rem; font-weight:700; }
  .gauge-label { font-size:0.85rem; color:#94a3b8; }
  .scenario-card { background:#1a1d27; border:1px solid #2d3148; border-radius:12px;
                   padding:1.2rem; cursor:pointer; transition:all 0.2s; }
  .scenario-card:hover { border-color:#6366f1; transform:translateY(-2px); }
  .scenario-name { font-weight:600; margin-bottom:0.3rem; }
  .scenario-desc { font-size:0.8rem; color:#94a3b8; }
  .scenario-tag  { font-size:0.7rem; padding:2px 8px; border-radius:10px;
                   background:#2d3148; color:#94a3b8; display:inline-block; margin-top:0.5rem; }
  .section-title { font-size:1.1rem; font-weight:600; margin-bottom:1rem; color:#e2e8f0; }
  .demo-log { background:#0f1117; border-radius:8px; padding:1rem;
              font-family:monospace; font-size:0.8rem; max-height:300px;
              overflow-y:auto; border:1px solid #2d3148; }
  .log-line { padding:3px 0; border-bottom:1px solid #1a1d27; }
  .log-allow { color:#22c55e; } .log-step { color:#f59e0b; } .log-block { color:#ef4444; }
  .spinner { display:none; width:20px; height:20px; border:2px solid #6366f1;
             border-top-color:transparent; border-radius:50%;
             animation:spin 0.8s linear infinite; margin:0 auto; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .info-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.3rem; font-size:0.75rem; }
  .info-row { display:flex; justify-content:space-between; padding:4px 0;
              border-bottom:1px solid #1e2235; }
  .fusion-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.5rem; }
  .fusion-box { background:#1a1d27; border-radius:6px; padding:0.5rem; text-align:center; }

  /* Quick simulator panel on dashboard */
  .quick-sim { background:#1a1d27; border:1px solid #2d3148; border-radius:12px;
               padding:1.5rem; margin-bottom:1.5rem; }
  .quick-sim-title { font-size:1rem; font-weight:600; margin-bottom:1rem;
                     display:flex; align-items:center; gap:0.5rem; }
  .quick-btns { display:flex; gap:0.75rem; flex-wrap:wrap; margin-bottom:1rem; }
  .quick-result { display:none; background:#0f1117; border-radius:10px; padding:1.2rem;
                  border:1px solid #2d3148; }
  .quick-result-inner { display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap; }
  .quick-score { font-size:2.5rem; font-weight:700; min-width:60px; text-align:center; }
  .quick-info { flex:1; min-width:200px; }
  .quick-info-name { font-weight:600; margin-bottom:0.2rem; }
  .quick-info-msg  { font-size:0.82rem; color:#94a3b8; margin-bottom:0.5rem; }
  .quick-metrics { display:flex; gap:1rem; font-size:0.78rem; flex-wrap:wrap; }
  .quick-metric  { background:#1a1d27; border-radius:6px; padding:4px 10px;
                   color:#94a3b8; border:1px solid #2d3148; }
  .quick-metric span { color:#e2e8f0; font-weight:600; }
  .quick-spinner { display:none; width:18px; height:18px; border:2px solid #6366f1;
                   border-top-color:transparent; border-radius:50%;
                   animation:spin 0.8s linear infinite; }

  @media(max-width:768px) {
    .grid-4,.grid-3 { grid-template-columns:repeat(2,1fr); }
    .grid-2,.form-grid { grid-template-columns:1fr; }
    .quick-btns { flex-direction:column; }
  }
</style>
</head>
<body>

<nav>
  <div class="nav-brand">MFA <span>Bypass Detection</span></div>
  <div class="nav-tabs">
    <button class="tab-btn active" onclick="showPage(\'dashboard\',event)">📊 Dashboard</button>
    <button class="tab-btn" onclick="showPage(\'simulate\',event)">🔬 Simulate</button>
    <button class="tab-btn" onclick="showPage(\'demo\',event)">⚡ Attack Demo</button>
  </div>
  <div><span class="status-dot"></span><span style="font-size:0.8rem;color:#94a3b8">API Live</span></div>
</nav>

<!-- ═══════════ DASHBOARD ═══════════ -->
<div id="page-dashboard" class="page active">
  <div style="margin-bottom:1.5rem">
    <h1 style="font-size:1.4rem;font-weight:700">Security Dashboard</h1>
    <p style="color:#94a3b8;font-size:0.85rem">Real-time MFA bypass detection monitoring</p>
  </div>

  <div class="grid-4">
    <div class="card"><div class="card-title">Total Requests</div><div class="card-value purple" id="stat-total">—</div></div>
    <div class="card"><div class="card-title">Blocked</div><div class="card-value red" id="stat-blocks">—</div></div>
    <div class="card"><div class="card-title">Step-Up MFA</div><div class="card-value yellow" id="stat-stepups">—</div></div>
    <div class="card"><div class="card-title">Avg Latency</div><div class="card-value green" id="stat-latency">—</div></div>
  </div>

  <!-- ── QUICK SIMULATOR ── -->
  <div class="quick-sim">
    <div class="quick-sim-title">
      ⚡ Quick Simulator
      <span style="font-size:0.75rem;color:#94a3b8;font-weight:400">
        — fire a login event directly from the dashboard
      </span>
      <div class="quick-spinner" id="qs-spinner"></div>
    </div>
    <div class="quick-btns">
      <button class="btn btn-green"  onclick="quickSim(\'normal\')">✅ Normal Login</button>
      <button class="btn btn-yellow" onclick="quickSim(\'borderline\')">⚠️ Borderline</button>
      <button class="btn btn-red"    onclick="quickSim(\'sim_swap\')">📱 SIM Swap</button>
      <button class="btn btn-red"    onclick="quickSim(\'aitm\')">🎣 AiTM Phishing</button>
      <button class="btn btn-red"    onclick="quickSim(\'session\')">🔑 Session Hijack</button>
      <button class="btn btn-red"    onclick="quickSim(\'fatigue\')">😴 MFA Fatigue</button>
      <button class="btn" onclick="randomSim()" style="background:#6366f1;">🎲 Randomise</button>
    </div>
    <div class="quick-result" id="qs-result">
      <div class="quick-result-inner">
        <div class="quick-score" id="qs-score">—</div>
        <div class="quick-info">
          <div class="quick-info-name" id="qs-name"></div>
          <div style="margin-bottom:0.4rem" id="qs-badge"></div>
          <div class="quick-info-msg"  id="qs-msg"></div>
          <div class="quick-metrics">
            <div class="quick-metric">RF <span id="qs-rf">—</span></div>
            <div class="quick-metric">LSTM <span id="qs-lstm">—</span></div>
            <div class="quick-metric">Latency <span id="qs-lat">—</span></div>
            <div class="quick-metric">Context <span id="qs-ctx">—</span></div>
            <div class="quick-metric">Behaviour <span id="qs-beh">—</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card"><div class="section-title">Decision Distribution</div><canvas id="chart-decisions" height="200"></canvas></div>
    <div class="card"><div class="section-title">Risk Score Distribution</div><canvas id="chart-risk" height="200"></canvas></div>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">
      <div class="section-title" style="margin:0">Recent Decisions</div>
      <button class="btn btn-outline" onclick="loadDashboard()" style="padding:0.4rem 1rem;font-size:0.8rem">↻ Refresh</button>
    </div>
    <table>
      <thead><tr><th>Timestamp</th><th>User</th><th>IP Address</th><th>Risk Score</th><th>Decision</th><th>Type</th><th>Latency</th></tr></thead>
      <tbody id="audit-table"><tr><td colspan="7" style="text-align:center;color:#94a3b8">Loading...</td></tr></tbody>
    </table>
  </div>
</div>

<!-- ═══════════ SIMULATE ═══════════ -->
<div id="page-simulate" class="page">
  <div style="margin-bottom:1.5rem">
    <h1 style="font-size:1.4rem;font-weight:700">Login Event Simulator</h1>
    <p style="color:#94a3b8;font-size:0.85rem">Manually configure a login event and see the AI decision in real time</p>
  </div>
  <div class="grid-2">
    <div>
      <div class="card" style="margin-bottom:1rem">
        <div class="section-title">Login Event Parameters</div>
        <div style="margin-bottom:1rem">
          <div class="form-grid">
            <div class="form-group">
              <label>User ID</label>
              <input type="text" id="s-user" value="user_test">
            </div>
            <div class="form-group">
              <label>IP Address</label>
              <select id="s-ip-select" onchange="updateIP()">
                <optgroup label="✅ Legitimate IPs">
                  <option value="192.168.1.1">192.168.1.1 — Private/Home</option>
                  <option value="82.45.12.100">82.45.12.100 — Residential UK</option>
                  <option value="8.8.8.8">8.8.8.8 — Google DNS (Clean)</option>
                  <option value="203.0.113.1">203.0.113.1 — Documentation Range</option>
                </optgroup>
                <optgroup label="⚠️ Suspicious IPs">
                  <option value="185.220.101.45">185.220.101.45 — Tor Exit Node</option>
                  <option value="194.165.16.77">194.165.16.77 — Known Proxy/VPN</option>
                  <option value="91.108.4.200">91.108.4.200 — Known Attack Range</option>
                  <option value="45.142.212.100">45.142.212.100 — Malicious ASN</option>
                  <option value="198.54.117.200">198.54.117.200 — Bulletproof Hosting</option>
                </optgroup>
                <optgroup label="✏️ Custom">
                  <option value="custom">Enter custom IP...</option>
                </optgroup>
              </select>
              <input type="text" id="s-ip" value="192.168.1.1"
                     style="margin-top:0.4rem;display:none" placeholder="Enter custom IP">
            </div>
            <div class="form-group">
              <label>Login Hour (0-23)</label>
              <input type="number" id="s-hour" value="9" min="0" max="23">
            </div>
          </div>
        </div>
        <div style="margin-bottom:1rem">
          <div class="form-grid">
            <div class="form-group">
              <label>Device Type</label>
              <select id="s-device">
                <option value="desktop">Desktop</option>
                <option value="mobile">Mobile</option>
                <option value="tablet">Tablet</option>
              </select>
            </div>
            <div class="form-group">
              <label>Is Attack IP?</label>
              <select id="s-attackip">
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>
            <div class="form-group">
              <label>Browser Known?</label>
              <select id="s-browser">
                <option value="1">Yes</option>
                <option value="0">No</option>
              </select>
            </div>
          </div>
        </div>
        <div style="margin-bottom:1rem">
          <div class="form-grid">
            <div class="form-group">
              <label>Country Changed?</label>
              <select id="s-country">
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>
            <div class="form-group">
              <label>ASN Changed?</label>
              <select id="s-asn">
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>
            <div class="form-group">
              <label>Device Changed?</label>
              <select id="s-devchg">
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>
          </div>
        </div>
        <div style="margin-bottom:1.5rem">
          <div class="form-grid">
            <div class="form-group">
              <label>ASN Attack Rate (0-1)</label>
              <input type="number" id="s-asnrate" value="0.01" min="0" max="1" step="0.01">
            </div>
            <div class="form-group">
              <label>Country Attack Rate (0-1)</label>
              <input type="number" id="s-cntrate" value="0.01" min="0" max="1" step="0.01">
            </div>
            <div class="form-group">
              <label>Login Frequency (1h)</label>
              <input type="number" id="s-freq" value="1" min="1" max="20">
            </div>
          </div>
        </div>
        <button class="btn" onclick="runSimulation()" style="width:100%">
          🔍 Analyse Login Event
        </button>
      </div>
    </div>

    <div>
      <div class="card" style="text-align:center;min-height:200px">
        <div class="section-title">AI Decision</div>
        <div id="sim-spinner" class="spinner" style="margin-top:2rem"></div>
        <div id="sim-placeholder" style="color:#94a3b8;margin-top:3rem">
          Configure parameters and click Analyse
        </div>
        <div id="sim-result" style="display:none">
          <div class="gauge-wrap">
            <div class="gauge-score" id="sim-score">—</div>
            <div class="gauge-label">Risk Score / 100</div>
          </div>
          <div id="sim-badge-wrap" style="margin-bottom:1rem"></div>
          <div id="sim-message" style="color:#94a3b8;font-size:0.9rem;margin-bottom:1.5rem"></div>
          <div style="background:#0f1117;border-radius:8px;padding:1rem;margin-bottom:1rem;text-align:left">
            <div style="font-size:0.75rem;color:#6366f1;font-weight:600;margin-bottom:0.6rem;
                        text-transform:uppercase;letter-spacing:1px">Model Outputs</div>
            <div class="fusion-grid">
              <div class="fusion-box">
                <div style="font-size:1rem;font-weight:700;color:#6366f1" id="sim-prf">—</div>
                <div style="font-size:0.7rem;color:#94a3b8">RF Probability</div>
              </div>
              <div class="fusion-box">
                <div style="font-size:1rem;font-weight:700;color:#6366f1" id="sim-plstm">—</div>
                <div style="font-size:0.7rem;color:#94a3b8">LSTM Probability</div>
              </div>
              <div class="fusion-box">
                <div style="font-size:1rem;font-weight:700;color:#22c55e" id="sim-lat">—</div>
                <div style="font-size:0.7rem;color:#94a3b8">Latency (ms)</div>
              </div>
            </div>
          </div>
          <div style="background:#0f1117;border-radius:8px;padding:1rem;text-align:left">
            <div style="font-size:0.75rem;color:#6366f1;font-weight:600;margin-bottom:0.6rem;
                        text-transform:uppercase;letter-spacing:1px">
              Risk Fusion: R = 0.6×ML + 0.2×Context + 0.2×Behaviour
            </div>
            <div class="fusion-grid">
              <div class="fusion-box">
                <div style="font-size:0.95rem;font-weight:700;color:#f59e0b" id="sim-ml">—</div>
                <div style="font-size:0.7rem;color:#94a3b8">ML Score (60%)</div>
                <div style="font-size:0.65rem;color:#64748b">0.6×RF + 0.4×LSTM</div>
              </div>
              <div class="fusion-box">
                <div style="font-size:0.95rem;font-weight:700;color:#f59e0b" id="sim-ctx">—</div>
                <div style="font-size:0.7rem;color:#94a3b8">Contextual (20%)</div>
                <div style="font-size:0.65rem;color:#64748b">ASN + Country rates</div>
              </div>
              <div class="fusion-box">
                <div style="font-size:0.95rem;font-weight:700;color:#f59e0b" id="sim-beh">—</div>
                <div style="font-size:0.7rem;color:#94a3b8">Behavioural (20%)</div>
                <div style="font-size:0.65rem;color:#64748b">Login freq + night</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="card" style="margin-top:1rem">
        <div class="section-title">Decision Thresholds</div>
        <div style="margin-bottom:0.8rem">
          <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px">
            <span class="green">ALLOW (0–33)</span><span>Low risk — JWT issued</span>
          </div>
          <div style="background:#052e16;border-radius:4px;height:8px;width:33%"></div>
        </div>
        <div style="margin-bottom:0.8rem">
          <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px">
            <span class="yellow">STEP-UP MFA (34–66)</span><span>Medium risk — TOTP required</span>
          </div>
          <div style="background:#451a03;border-radius:4px;height:8px;width:33%"></div>
        </div>
        <div>
          <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px">
            <span class="red">BLOCK (67–100)</span><span>High risk — access denied</span>
          </div>
          <div style="background:#450a0a;border-radius:4px;height:8px;width:33%"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════ DEMO ═══════════ -->
<div id="page-demo" class="page">
  <div style="margin-bottom:1.5rem">
    <h1 style="font-size:1.4rem;font-weight:700">Attack Scenario Demo</h1>
    <p style="color:#94a3b8;font-size:0.85rem">Pre-configured attack scenarios demonstrating all four MFA bypass patterns</p>
  </div>
  <div class="grid-3" id="scenario-grid"></div>
  <div class="grid-2" style="margin-top:0">
    <div class="card" style="overflow-y:auto;max-height:700px">
      <div class="section-title">Selected Scenario Result</div>
      <div id="demo-result-wrap" style="text-align:center;color:#94a3b8;padding:2rem">
        Click a scenario above to run it
      </div>
    </div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem">
        <div class="section-title" style="margin:0">Execution Log</div>
        <button class="btn btn-outline" onclick="runAllScenarios()"
                style="padding:0.4rem 1rem;font-size:0.8rem">▶ Run All</button>
      </div>
      <div class="demo-log" id="demo-log">
        <div style="color:#94a3b8">Waiting for scenarios to run...</div>
      </div>
    </div>
  </div>
</div>

<script>
const SUSPICIOUS_IPS = [
  "185.220.101.45","194.165.16.77",
  "91.108.4.200","45.142.212.100","198.54.117.200"
];

const SCENARIOS = [
  {
    id:"normal", name:"Normal Login",
    desc:"Regular user, known device, business hours",
    tag:"Benign", tagColor:"#22c55e", icon:"✅",
    payload:{user_id:"alice",ip_address:"82.45.12.100",attack_type:"normal",
      hour:9,day_of_week:1,is_night:0,device_mobile:0,device_desktop:1,device_tablet:0,
      login_success:1,is_attack_ip:0,browser_known:1,country_changed:0,asn_changed:0,
      device_changed:0,asn_attack_rate:0.01,country_attack_rate:0.02,login_freq:1}
  },
  {
    id:"sim_swap", name:"SIM Swap Attack",
    desc:"New device + new country + 3am + attack IP",
    tag:"SIM Swap", tagColor:"#ef4444", icon:"📱",
    payload:{user_id:"bob",ip_address:"185.220.101.45",attack_type:"sim_swap",
      hour:3,day_of_week:6,is_night:1,device_mobile:1,device_desktop:0,device_tablet:0,
      login_success:1,is_attack_ip:1,browser_known:0,country_changed:1,asn_changed:1,
      device_changed:1,asn_attack_rate:0.92,country_attack_rate:0.85,login_freq:2}
  },
  {
    id:"aitm", name:"AiTM Phishing",
    desc:"Relayed session via proxy — unknown browser, high freq",
    tag:"AiTM", tagColor:"#ef4444", icon:"🎣",
    payload:{user_id:"carol",ip_address:"194.165.16.77",attack_type:"aitm_phishing",
      hour:14,day_of_week:2,is_night:0,device_mobile:0,device_desktop:1,device_tablet:0,
      login_success:1,is_attack_ip:1,browser_known:0,country_changed:1,asn_changed:1,
      device_changed:1,asn_attack_rate:0.88,country_attack_rate:0.72,login_freq:15}
  },
  {
    id:"session", name:"Session Hijacking",
    desc:"Stolen token replayed from different device",
    tag:"Session Hijack", tagColor:"#ef4444", icon:"🔑",
    payload:{user_id:"dave",ip_address:"45.142.212.100",attack_type:"session_hijacking",
      hour:16,day_of_week:3,is_night:0,device_mobile:1,device_desktop:0,device_tablet:0,
      login_success:1,is_attack_ip:1,browser_known:0,country_changed:1,asn_changed:1,
      device_changed:1,asn_attack_rate:0.81,country_attack_rate:0.69,login_freq:1}
  },
  {
    id:"fatigue", name:"MFA Fatigue",
    desc:"12 push notifications at 2am — spam until approved",
    tag:"MFA Fatigue", tagColor:"#ef4444", icon:"😴",
    payload:{user_id:"eve",ip_address:"91.108.4.200",attack_type:"mfa_fatigue",
      hour:2,day_of_week:5,is_night:1,device_mobile:1,device_desktop:0,device_tablet:0,
      login_success:1,is_attack_ip:1,browser_known:0,country_changed:0,asn_changed:1,
      device_changed:0,asn_attack_rate:0.78,country_attack_rate:0.65,login_freq:12}
  },
  {
    id:"borderline", name:"Borderline Suspicious",
    desc:"Travelling user — unusual location but legitimate",
    tag:"Ambiguous", tagColor:"#f59e0b", icon:"⚠️",
    payload:{user_id:"frank",ip_address:"78.92.14.55",attack_type:"manual",
      hour:21,day_of_week:4,is_night:0,device_mobile:1,device_desktop:0,device_tablet:0,
      login_success:1,is_attack_ip:0,browser_known:1,country_changed:0,asn_changed:1,
      device_changed:1,asn_attack_rate:0.35,country_attack_rate:0.28,login_freq:3}
  }
];

let chartDecisions=null, chartRisk=null;

function initCharts() {
  chartDecisions = new Chart(
    document.getElementById("chart-decisions").getContext("2d"),
    { type:"doughnut",
      data:{ labels:["ALLOW","STEP_UP_MFA","BLOCK"],
             datasets:[{data:[1,1,1],
               backgroundColor:["#22c55e","#f59e0b","#ef4444"],
               borderColor:"#1a1d27",borderWidth:3}] },
      options:{ plugins:{legend:{labels:{color:"#94a3b8"}}},
                responsive:true,maintainAspectRatio:true } }
  );
  chartRisk = new Chart(
    document.getElementById("chart-risk").getContext("2d"),
    { type:"bar",
      data:{ labels:["0-20","21-40","41-60","61-80","81-100"],
             datasets:[{label:"Count",data:[0,0,0,0,0],
               backgroundColor:["#22c55e","#84cc16","#f59e0b","#f97316","#ef4444"],
               borderRadius:6}] },
      options:{ plugins:{legend:{display:false}},
                scales:{ x:{ticks:{color:"#94a3b8"},grid:{color:"#2d3148"}},
                         y:{ticks:{color:"#94a3b8"},grid:{color:"#2d3148"}} },
                responsive:true,maintainAspectRatio:true } }
  );
}

async function loadDashboard() {
  try {
    const [stats,recent] = await Promise.all([
      fetch("/audit/stats").then(r=>r.json()),
      fetch("/audit/recent?limit=20").then(r=>r.json())
    ]);
    document.getElementById("stat-total").textContent   = stats.total_requests;
    document.getElementById("stat-blocks").textContent  = stats.decisions.BLOCK;
    document.getElementById("stat-stepups").textContent = stats.decisions.STEP_UP_MFA;
    document.getElementById("stat-latency").textContent = stats.avg_latency_ms+"ms";
    chartDecisions.data.datasets[0].data = [
      stats.decisions.ALLOW, stats.decisions.STEP_UP_MFA, stats.decisions.BLOCK
    ];
    chartDecisions.update();
    const bins=[0,0,0,0,0];
    recent.records.forEach(r=>{ bins[Math.min(Math.floor(r.risk_score/20),4)]++; });
    chartRisk.data.datasets[0].data = bins;
    chartRisk.update();
    const tbody = document.getElementById("audit-table");
    if (!recent.records.length) {
      tbody.innerHTML=\'<tr><td colspan="7" style="text-align:center;color:#94a3b8">No data yet</td></tr>\';
      return;
    }
    tbody.innerHTML = recent.records.map(r=>{
      const badge = r.decision==="ALLOW"
        ? `<span class="badge badge-allow">ALLOW</span>`
        : r.decision==="STEP_UP_MFA"
        ? `<span class="badge badge-step">STEP UP</span>`
        : `<span class="badge badge-block">BLOCK</span>`;
      const rc = r.risk_score<34?"#22c55e":r.risk_score<67?"#f59e0b":"#ef4444";
      return `<tr>
        <td style="color:#94a3b8;font-size:0.8rem">${r.timestamp.substring(0,19)}</td>
        <td>${r.user_id}</td>
        <td style="font-family:monospace;font-size:0.8rem">${r.ip_address}</td>
        <td><div style="display:flex;align-items:center;gap:8px">
          <div class="risk-bar-bg">
            <div class="risk-bar-fill" style="width:${Math.round(r.risk_score*1.2)}px;background:${rc}"></div>
          </div>
          <span style="color:${rc};font-weight:600">${r.risk_score}</span>
        </div></td>
        <td>${badge}</td>
        <td style="color:#94a3b8;font-size:0.8rem">${r.attack_type||"—"}</td>
        <td style="color:#94a3b8">${r.latency_ms}ms</td>
      </tr>`;
    }).join("");
  } catch(e){ console.error(e); }
}

// ── RANDOM SIMULATOR ──────────────────────────────────────────
async function randomSim() {
  const spinner = document.getElementById("qs-spinner");
  const result  = document.getElementById("qs-result");
  spinner.style.display="inline-block";
  result.style.display="none";

  // Generate random parameters
  const hour = Math.floor(Math.random() * 24);
  const devices = ["desktop","mobile","tablet"];
  const dev = devices[Math.floor(Math.random()*3)];
  const payload = {
    user_id:             "random_user_" + Math.floor(Math.random()*9999),
    ip_address:          `${Math.floor(Math.random()*255)}.${Math.floor(Math.random()*255)}.${Math.floor(Math.random()*255)}.${Math.floor(Math.random()*255)}`,
    attack_type:         "random",
    hour,
    day_of_week:         Math.floor(Math.random()*7),
    is_night:            (hour>=22||hour<=6)?1:0,
    device_mobile:       dev==="mobile"?1:0,
    device_desktop:      dev==="desktop"?1:0,
    device_tablet:       dev==="tablet"?1:0,
    login_success:       Math.random()>0.2?1:0,
    is_attack_ip:        Math.random()>0.7?1:0,
    browser_known:       Math.random()>0.3?1:0,
    country_changed:     Math.random()>0.6?1:0,
    asn_changed:         Math.random()>0.6?1:0,
    device_changed:      Math.random()>0.7?1:0,
    asn_attack_rate:     parseFloat((Math.random()).toFixed(3)),
    country_attack_rate: parseFloat((Math.random()).toFixed(3)),
    login_freq:          Math.floor(Math.random()*20)+1
  };

  try {
    const res  = await fetch("/predict",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    const data = await res.json();
    const color = data.decision==="ALLOW"?"#22c55e":data.decision==="STEP_UP_MFA"?"#f59e0b":"#ef4444";
    const bcls  = data.decision==="ALLOW"?"badge-allow":data.decision==="STEP_UP_MFA"?"badge-step":"badge-block";
    const ctx = ((payload.asn_attack_rate+payload.country_attack_rate)/2).toFixed(3);
    const beh = ((payload.login_freq/20.0+payload.is_night)/2).toFixed(3);
    document.getElementById("qs-score").textContent = data.risk_score;
    document.getElementById("qs-score").style.color = color;
    document.getElementById("qs-name").textContent  = "🎲 Random Login Event";
    document.getElementById("qs-msg").textContent   = data.message + " | Hour: " + hour + ":00 | Device: " + dev + " | Login freq: " + payload.login_freq;
    document.getElementById("qs-badge").innerHTML   = `<span class="badge ${bcls}" style="font-size:0.85rem;padding:4px 14px">${data.decision}</span>`;
    document.getElementById("qs-rf").textContent    = data.p_rf;
    document.getElementById("qs-lstm").textContent  = data.p_lstm;
    document.getElementById("qs-lat").textContent   = data.latency_ms+"ms";
    document.getElementById("qs-ctx").textContent   = ctx;
    document.getElementById("qs-beh").textContent   = beh;
    result.style.display="block";
    loadDashboard();
  } catch(e) {
    document.getElementById("qs-name").textContent = "Error: "+e.message;
    result.style.display="block";
  }
  spinner.style.display="none";
}

// ── QUICK SIMULATOR (dashboard) ────────────────────────────────
async function quickSim(scenarioId) {
  const s = SCENARIOS.find(x=>x.id===scenarioId);
  if (!s) return;
  const spinner = document.getElementById("qs-spinner");
  const result  = document.getElementById("qs-result");
  spinner.style.display="inline-block";
  result.style.display="none";
  try {
    const res  = await fetch("/predict",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(s.payload)});
    const data = await res.json();
    const color = data.decision==="ALLOW"?"#22c55e":data.decision==="STEP_UP_MFA"?"#f59e0b":"#ef4444";
    const bcls  = data.decision==="ALLOW"?"badge-allow":data.decision==="STEP_UP_MFA"?"badge-step":"badge-block";
    const ctx = ((s.payload.asn_attack_rate+s.payload.country_attack_rate)/2).toFixed(3);
    const beh = ((s.payload.login_freq/20.0+s.payload.is_night)/2).toFixed(3);
    document.getElementById("qs-score").textContent = data.risk_score;
    document.getElementById("qs-score").style.color = color;
    document.getElementById("qs-name").textContent  = s.icon+" "+s.name;
    document.getElementById("qs-msg").textContent   = data.message;
    document.getElementById("qs-badge").innerHTML   = `<span class="badge ${bcls}" style="font-size:0.85rem;padding:4px 14px">${data.decision}</span>`;
    document.getElementById("qs-rf").textContent    = data.p_rf;
    document.getElementById("qs-lstm").textContent  = data.p_lstm;
    document.getElementById("qs-lat").textContent   = data.latency_ms+"ms";
    document.getElementById("qs-ctx").textContent   = ctx;
    document.getElementById("qs-beh").textContent   = beh;
    result.style.display="block";
    loadDashboard();
  } catch(e) {
    document.getElementById("qs-name").textContent = "Error: "+e.message;
    result.style.display="block";
  }
  spinner.style.display="none";
}

// IP selector
function updateIP() {
  const sel = document.getElementById("s-ip-select").value;
  const inp = document.getElementById("s-ip");
  if (sel==="custom") {
    inp.style.display="block"; inp.value=""; inp.focus();
  } else {
    inp.style.display="none"; inp.value=sel;
    if (SUSPICIOUS_IPS.includes(sel)) {
      document.getElementById("s-attackip").value="1";
      document.getElementById("s-asnrate").value="0.75";
      document.getElementById("s-cntrate").value="0.65";
      document.getElementById("s-country").value="1";
      document.getElementById("s-asn").value="1";
      document.getElementById("s-browser").value="0";
    } else {
      document.getElementById("s-attackip").value="0";
      document.getElementById("s-asnrate").value="0.01";
      document.getElementById("s-cntrate").value="0.01";
      document.getElementById("s-country").value="0";
      document.getElementById("s-asn").value="0";
      document.getElementById("s-browser").value="1";
    }
  }
}

async function runSimulation() {
  const hour = parseInt(document.getElementById("s-hour").value);
  const dev  = document.getElementById("s-device").value;
  const selIP= document.getElementById("s-ip-select").value;
  const ip   = selIP==="custom"
    ? document.getElementById("s-ip").value
    : selIP;
  const asnRate = parseFloat(document.getElementById("s-asnrate").value);
  const cntRate = parseFloat(document.getElementById("s-cntrate").value);
  const freq    = parseInt(document.getElementById("s-freq").value);
  const payload = {
    user_id:             document.getElementById("s-user").value,
    ip_address:          ip, attack_type:"manual",
    hour, day_of_week:0,
    is_night:            (hour>=22||hour<=6)?1:0,
    device_mobile:       dev==="mobile"?1:0,
    device_desktop:      dev==="desktop"?1:0,
    device_tablet:       dev==="tablet"?1:0,
    login_success:       1,
    is_attack_ip:        parseInt(document.getElementById("s-attackip").value),
    browser_known:       parseInt(document.getElementById("s-browser").value),
    country_changed:     parseInt(document.getElementById("s-country").value),
    asn_changed:         parseInt(document.getElementById("s-asn").value),
    device_changed:      parseInt(document.getElementById("s-devchg").value),
    asn_attack_rate:     asnRate, country_attack_rate: cntRate, login_freq: freq
  };
  document.getElementById("sim-result").style.display="none";
  document.getElementById("sim-placeholder").style.display="none";
  document.getElementById("sim-spinner").style.display="block";
  try {
    const res  = await fetch("/predict",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    const data = await res.json();
    document.getElementById("sim-spinner").style.display="none";
    document.getElementById("sim-result").style.display="block";
    const color = data.decision==="ALLOW"?"#22c55e":data.decision==="STEP_UP_MFA"?"#f59e0b":"#ef4444";
    const cls   = data.decision==="ALLOW"?"badge-allow":data.decision==="STEP_UP_MFA"?"badge-step":"badge-block";
    document.getElementById("sim-score").textContent   = data.risk_score;
    document.getElementById("sim-score").style.color   = color;
    document.getElementById("sim-message").textContent = data.message;
    document.getElementById("sim-prf").textContent     = data.p_rf;
    document.getElementById("sim-plstm").textContent   = data.p_lstm;
    document.getElementById("sim-lat").textContent     = data.latency_ms+"ms";
    const ml  = (0.6*data.p_rf + 0.4*data.p_lstm).toFixed(4);
    const ctx = ((asnRate+cntRate)/2).toFixed(4);
    const beh = ((freq/20.0 + ((hour>=22||hour<=6)?1:0))/2).toFixed(4);
    document.getElementById("sim-ml").textContent  = ml;
    document.getElementById("sim-ctx").textContent = ctx;
    document.getElementById("sim-beh").textContent = beh;
    document.getElementById("sim-badge-wrap").innerHTML =
      `<span class="badge ${cls}" style="font-size:1rem;padding:6px 20px">${data.decision}</span>`;
  } catch(e) {
    document.getElementById("sim-spinner").style.display="none";
    document.getElementById("sim-placeholder").style.display="block";
    document.getElementById("sim-placeholder").textContent="Error: "+e.message;
  }
}

function buildScenarioCards() {
  document.getElementById("scenario-grid").innerHTML = SCENARIOS.map(s=>`
    <div class="scenario-card" onclick="runScenario(\'${s.id}\')">
      <div style="font-size:1.5rem;margin-bottom:0.4rem">${s.icon}</div>
      <div class="scenario-name">${s.name}</div>
      <div class="scenario-desc">${s.desc}</div>
      <span class="scenario-tag" style="color:${s.tagColor};background:${s.tagColor}22;border:1px solid ${s.tagColor}44">
        ${s.tag}
      </span>
    </div>`).join("");
}

function addLog(text,cls="") {
  const log=document.getElementById("demo-log");
  const line=document.createElement("div");
  line.className=`log-line ${cls}`;
  line.textContent=`[${new Date().toLocaleTimeString()}] ${text}`;
  log.insertBefore(line,log.firstChild);
}

function paramRow(label, val, good) {
  const color = good ? "#22c55e" : "#ef4444";
  const icon  = good ? "✓" : "⚠";
  return `<div class="info-row">
    <span style="color:#94a3b8">${label}</span>
    <span style="color:${color};font-weight:600">${val} ${icon}</span>
  </div>`;
}

async function runScenario(id) {
  const s = SCENARIOS.find(x=>x.id===id);
  if (!s) return;
  addLog(`Running: ${s.name}...`);
  try {
    const res  = await fetch("/predict",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(s.payload)});
    const data = await res.json();
    const color= data.decision==="ALLOW"?"#22c55e":data.decision==="STEP_UP_MFA"?"#f59e0b":"#ef4444";
    const cls  = data.decision==="ALLOW"?"log-allow":data.decision==="STEP_UP_MFA"?"log-step":"log-block";
    const bcls = data.decision==="ALLOW"?"badge-allow":data.decision==="STEP_UP_MFA"?"badge-step":"badge-block";
    addLog(`${s.name} → ${data.decision} (risk=${data.risk_score}, ${data.latency_ms}ms)`,cls);
    const ml  = (0.6*data.p_rf + 0.4*data.p_lstm).toFixed(4);
    const ctx = ((s.payload.asn_attack_rate+s.payload.country_attack_rate)/2).toFixed(4);
    const beh = ((s.payload.login_freq/20.0+s.payload.is_night)/2).toFixed(4);
    document.getElementById("demo-result-wrap").innerHTML = `
      <div style="font-size:2rem;margin-bottom:0.4rem">${s.icon}</div>
      <div style="font-size:1.2rem;font-weight:700;margin-bottom:0.2rem">${s.name}</div>
      <div style="color:#94a3b8;font-size:0.8rem;margin-bottom:0.8rem">${s.desc}</div>
      <div style="font-size:2.8rem;font-weight:700;color:${color};margin-bottom:0.2rem">${data.risk_score}</div>
      <div style="color:#94a3b8;font-size:0.75rem;margin-bottom:0.6rem">Risk Score / 100</div>
      <span class="badge ${bcls}" style="font-size:0.95rem;padding:5px 18px">${data.decision}</span>
      <div style="color:#94a3b8;margin:0.8rem 0;font-size:0.8rem">${data.message}</div>
      <div style="background:#0f1117;border-radius:8px;padding:0.8rem;margin-bottom:0.8rem;text-align:left">
        <div style="font-size:0.7rem;color:#6366f1;font-weight:600;margin-bottom:0.5rem;
                    text-transform:uppercase;letter-spacing:1px">Model Outputs</div>
        <div class="fusion-grid">
          <div class="fusion-box">
            <div style="font-size:0.95rem;font-weight:700;color:#6366f1">${data.p_rf}</div>
            <div style="font-size:0.65rem;color:#94a3b8">RF Probability</div>
          </div>
          <div class="fusion-box">
            <div style="font-size:0.95rem;font-weight:700;color:#6366f1">${data.p_lstm}</div>
            <div style="font-size:0.65rem;color:#94a3b8">LSTM Probability</div>
          </div>
          <div class="fusion-box">
            <div style="font-size:0.95rem;font-weight:700;color:#22c55e">${data.latency_ms}ms</div>
            <div style="font-size:0.65rem;color:#94a3b8">Latency</div>
          </div>
        </div>
      </div>
      <div style="background:#0f1117;border-radius:8px;padding:0.8rem;margin-bottom:0.8rem;text-align:left">
        <div style="font-size:0.7rem;color:#6366f1;font-weight:600;margin-bottom:0.5rem;
                    text-transform:uppercase;letter-spacing:1px">
          Risk Fusion: R = 0.6×ML + 0.2×Context + 0.2×Behaviour
        </div>
        <div class="fusion-grid">
          <div class="fusion-box">
            <div style="font-size:0.95rem;font-weight:700;color:#f59e0b">${ml}</div>
            <div style="font-size:0.65rem;color:#94a3b8">ML Score (60%)</div>
            <div style="font-size:0.6rem;color:#64748b">0.6×RF + 0.4×LSTM</div>
          </div>
          <div class="fusion-box">
            <div style="font-size:0.95rem;font-weight:700;color:#f59e0b">${ctx}</div>
            <div style="font-size:0.65rem;color:#94a3b8">Contextual (20%)</div>
            <div style="font-size:0.6rem;color:#64748b">ASN + Country rates</div>
          </div>
          <div class="fusion-box">
            <div style="font-size:0.95rem;font-weight:700;color:#f59e0b">${beh}</div>
            <div style="font-size:0.65rem;color:#94a3b8">Behavioural (20%)</div>
            <div style="font-size:0.6rem;color:#64748b">Freq + night signal</div>
          </div>
        </div>
      </div>
      <div style="background:#0f1117;border-radius:8px;padding:0.8rem;text-align:left">
        <div style="font-size:0.7rem;color:#6366f1;font-weight:600;margin-bottom:0.5rem;
                    text-transform:uppercase;letter-spacing:1px">Input Parameters Considered</div>
        <div class="info-grid">
          ${paramRow("Is Attack IP",    s.payload.is_attack_ip?"YES":"NO",    !s.payload.is_attack_ip)}
          ${paramRow("Country Changed", s.payload.country_changed?"YES":"NO", !s.payload.country_changed)}
          ${paramRow("Device Changed",  s.payload.device_changed?"YES":"NO",  !s.payload.device_changed)}
          ${paramRow("ASN Changed",     s.payload.asn_changed?"YES":"NO",     !s.payload.asn_changed)}
          ${paramRow("Browser Known",   s.payload.browser_known?"YES":"NO",   !!s.payload.browser_known)}
          ${paramRow("Is Night",        s.payload.is_night?"YES":"NO",        !s.payload.is_night)}
          ${paramRow("Login Hour",      s.payload.hour+":00",                 s.payload.hour>=7&&s.payload.hour<=21)}
          ${paramRow("Login Freq/hr",   s.payload.login_freq+"x",             s.payload.login_freq<8)}
          ${paramRow("ASN Attack Rate", (s.payload.asn_attack_rate*100).toFixed(0)+"%", s.payload.asn_attack_rate<0.3)}
          ${paramRow("Country Atk Rate",(s.payload.country_attack_rate*100).toFixed(0)+"%",s.payload.country_attack_rate<0.3)}
        </div>
      </div>`;
    loadDashboard();
  } catch(e){ addLog(`Error: ${e.message}`,"log-block"); }
}

async function runAllScenarios() {
  addLog("Running all scenarios...");
  for (const s of SCENARIOS) {
    await runScenario(s.id);
    await new Promise(r=>setTimeout(r,600));
  }
  addLog("All scenarios complete.","log-allow");
}

function showPage(name,e) {
  document.querySelectorAll(".page").forEach(p=>p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b=>b.classList.remove("active"));
  document.getElementById("page-"+name).classList.add("active");
  if(e&&e.target) e.target.classList.add("active");
  if(name==="dashboard") loadDashboard();
}

initCharts();
buildScenarioCards();
loadDashboard();
setInterval(loadDashboard,10000);
</script>
</body>
</html>'''