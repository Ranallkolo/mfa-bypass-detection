import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import time

DB_PATH  = '/home/da_otcifithom/mfa-bypass-detection/logs/audit.db'
BASE_URL = 'http://localhost:8000'

def clear():
    print("\033[2J\033[H", end="")

def get_audit_data():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM audit_log ORDER BY id DESC", conn)
    conn.close()
    return df

def draw_bar(value, max_val, width=30, char='█'):
    filled = int((value / max_val) * width) if max_val > 0 else 0
    return char * filled + '░' * (width - filled)

def dashboard():
    print("Starting dashboard... Make sure FastAPI is running.")
    print("Press Ctrl+C to exit.\n")
    time.sleep(1)

    while True:
        try:
            clear()
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

            print("╔══════════════════════════════════════════════════════════╗")
            print("║     MFA BYPASS DETECTION FRAMEWORK — LIVE DASHBOARD     ║")
            print(f"║  {now}                              ║")
            print("╚══════════════════════════════════════════════════════════╝")

            # fetch stats
            try:
                stats = requests.get(f"{BASE_URL}/audit/stats", timeout=2).json()
                total    = stats['total_requests']
                allows   = stats['decisions']['ALLOW']
                stepups  = stats['decisions']['STEP_UP_MFA']
                blocks   = stats['decisions']['BLOCK']
                avg_lat  = stats['avg_latency_ms']
            except:
                print("\n  ⚠️  API not reachable. Start uvicorn first.\n")
                time.sleep(3)
                continue

            # decision distribution
            print("\n  DECISION DISTRIBUTION")
            print(f"  {'─' * 54}")
            if total > 0:
                print(f"  🟢 ALLOW       [{draw_bar(allows,  total)}] {allows:>4} ({allows/total*100:.1f}%)")
                print(f"  🟡 STEP_UP_MFA [{draw_bar(stepups, total)}] {stepups:>4} ({stepups/total*100:.1f}%)")
                print(f"  🔴 BLOCK       [{draw_bar(blocks,  total)}] {blocks:>4} ({blocks/total*100:.1f}%)")
            print(f"  Total requests: {total}")

            # latency
            print(f"\n  PERFORMANCE")
            print(f"  {'─' * 54}")
            print(f"  Avg API Latency:  {avg_lat:.1f}ms  (target < 500ms) ✅")

            # recent decisions
            df = get_audit_data()
            if len(df) > 0:
                print(f"\n  RECENT DECISIONS (last 10)")
                print(f"  {'─' * 54}")
                print(f"  {'Timestamp':<22} {'User':<12} {'Risk':>6} {'Decision':<14} {'ms':>6}")
                print(f"  {'─' * 54}")
                for _, row in df.head(10).iterrows():
                    ts  = row['timestamp'][:19]
                    uid = str(row['user_id'])[:11]
                    rs  = row['risk_score']
                    dec = row['decision']
                    lat = row['latency_ms']
                    icon = '🟢' if dec == 'ALLOW' else '🟡' if dec == 'STEP_UP_MFA' else '🔴'
                    print(f"  {ts:<22} {uid:<12} {rs:>6.1f} {icon} {dec:<12} {lat:>6.1f}")

                # risk score distribution
                print(f"\n  RISK SCORE DISTRIBUTION")
                print(f"  {'─' * 54}")
                low  = len(df[df['risk_score'] < 33])
                mid  = len(df[(df['risk_score'] >= 33) & (df['risk_score'] < 66)])
                high = len(df[df['risk_score'] >= 66])
                n    = len(df)
                print(f"  Low  (0-33):   [{draw_bar(low,  n, 20)}] {low}")
                print(f"  Mid  (34-66):  [{draw_bar(mid,  n, 20)}] {mid}")
                print(f"  High (67-100): [{draw_bar(high, n, 20)}] {high}")

                # attack detection rate
                blocked_pct = (blocks / total * 100) if total > 0 else 0
                print(f"\n  SECURITY METRICS")
                print(f"  {'─' * 54}")
                print(f"  Threats blocked:     {blocks} ({blocked_pct:.1f}%)")
                print(f"  Step-up enforced:    {stepups}")
                print(f"  Clean logins:        {allows}")

            print(f"\n  {'─' * 54}")
            print(f"  Dashboard refreshes every 5 seconds. Ctrl+C to exit.")

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")
            break

if __name__ == "__main__":
    dashboard()