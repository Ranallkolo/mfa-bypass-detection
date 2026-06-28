import requests
import time
import json

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("MFA BYPASS DETECTION FRAMEWORK — LIVE DEMONSTRATION")
print("=" * 60)

scenarios = [
    {
        "name": "SCENARIO 1 — Normal Login (Expected: ALLOW)",
        "description": "Regular user logging in from known device, daytime, known location",
        "payload": {
            "user_id": "user_alice",
            "ip_address": "82.45.12.100",
            "hour": 9,
            "day_of_week": 1,
            "is_night": 0,
            "device_mobile": 0,
            "device_desktop": 1,
            "device_tablet": 0,
            "login_success": 1,
            "is_attack_ip": 0,
            "browser_known": 1,
            "country_changed": 0,
            "asn_changed": 0,
            "device_changed": 0,
            "asn_attack_rate": 0.01,
            "country_attack_rate": 0.02,
            "login_freq": 1
        }
    },
    {
        "name": "SCENARIO 2 — SIM Swap Attack (Expected: BLOCK)",
        "description": "Login from new device + new country at 3am from known attack IP",
        "payload": {
            "user_id": "user_bob",
            "ip_address": "185.220.101.45",
            "hour": 3,
            "day_of_week": 6,
            "is_night": 1,
            "device_mobile": 1,
            "device_desktop": 0,
            "device_tablet": 0,
            "login_success": 1,
            "is_attack_ip": 1,
            "browser_known": 0,
            "country_changed": 1,
            "asn_changed": 1,
            "device_changed": 1,
            "asn_attack_rate": 0.92,
            "country_attack_rate": 0.85,
            "login_freq": 2
        }
    },
    {
        "name": "SCENARIO 3 — MFA Fatigue Attack (Expected: BLOCK)",
        "description": "12 login attempts in 1 hour at 2am — classic push notification spam",
        "payload": {
            "user_id": "user_carol",
            "ip_address": "91.108.4.200",
            "hour": 2,
            "day_of_week": 5,
            "is_night": 1,
            "device_mobile": 1,
            "device_desktop": 0,
            "device_tablet": 0,
            "login_success": 1,
            "is_attack_ip": 1,
            "browser_known": 0,
            "country_changed": 0,
            "asn_changed": 1,
            "device_changed": 0,
            "asn_attack_rate": 0.78,
            "country_attack_rate": 0.65,
            "login_freq": 12
        }
    },
    {
        "name": "SCENARIO 4 — AiTM Phishing (Expected: BLOCK)",
        "description": "Successful login relayed via proxy — new country, attack IP, unknown browser",
        "payload": {
            "user_id": "user_dave",
            "ip_address": "194.165.16.77",
            "hour": 14,
            "day_of_week": 2,
            "is_night": 0,
            "device_mobile": 0,
            "device_desktop": 1,
            "device_tablet": 0,
            "login_success": 1,
            "is_attack_ip": 1,
            "browser_known": 0,
            "country_changed": 1,
            "asn_changed": 1,
            "device_changed": 1,
            "asn_attack_rate": 0.88,
            "country_attack_rate": 0.72,
            "login_freq": 15
        }
    },
    {
        "name": "SCENARIO 5 — Session Hijacking (Expected: BLOCK)",
        "description": "Session resumed from completely different device and ASN",
        "payload": {
            "user_id": "user_eve",
            "ip_address": "45.142.212.100",
            "hour": 16,
            "day_of_week": 3,
            "is_night": 0,
            "device_mobile": 1,
            "device_desktop": 0,
            "device_tablet": 0,
            "login_success": 1,
            "is_attack_ip": 1,
            "browser_known": 0,
            "country_changed": 1,
            "asn_changed": 1,
            "device_changed": 1,
            "asn_attack_rate": 0.81,
            "country_attack_rate": 0.69,
            "login_freq": 1
        }
    },
    {
        "name": "SCENARIO 6 — Borderline Suspicious (Expected: STEP_UP_MFA)",
        "description": "Login from slightly unusual location — needs extra verification",
        "payload": {
            "user_id": "user_frank",
            "ip_address": "78.92.14.55",
            "hour": 21,
            "day_of_week": 4,
            "is_night": 0,
            "device_mobile": 1,
            "device_desktop": 0,
            "device_tablet": 0,
            "login_success": 1,
            "is_attack_ip": 0,
            "browser_known": 1,
            "country_changed": 0,
            "asn_changed": 1,
            "device_changed": 1,
            "asn_attack_rate": 0.35,
            "country_attack_rate": 0.28,
            "login_freq": 3
        }
    }
]

results = []

for scenario in scenarios:
    print(f"\n{'─' * 60}")
    print(f"  {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"{'─' * 60}")

    response = requests.post(
        f"{BASE_URL}/predict",
        json=scenario['payload']
    )
    result = response.json()

    decision  = result['decision']
    risk      = result['risk_score']
    latency   = result['latency_ms']
    p_rf      = result['p_rf']
    p_lstm    = result['p_lstm']
    message   = result['message']

    # visual risk bar
    bar_len  = int(risk / 5)
    bar      = '█' * bar_len + '░' * (20 - bar_len)
    color    = '🟢' if decision == 'ALLOW' else '🟡' if decision == 'STEP_UP_MFA' else '🔴'

    print(f"  Risk Score:  [{bar}] {risk:.1f}/100")
    print(f"  Decision:    {color} {decision}")
    print(f"  Message:     {message}")
    print(f"  RF Prob:     {p_rf:.4f}")
    print(f"  LSTM Prob:   {p_lstm:.4f}")
    print(f"  Latency:     {latency:.1f}ms")

    results.append({
        "scenario": scenario['name'],
        "decision": decision,
        "risk_score": risk,
        "latency_ms": latency
    })

    time.sleep(0.5)

# summary
print(f"\n{'=' * 60}")
print("DEMONSTRATION SUMMARY")
print(f"{'=' * 60}")
print(f"{'Scenario':<45} {'Decision':<15} {'Risk':>6}")
print(f"{'─' * 70}")
for r in results:
    name = r['scenario'].split('—')[1].strip().split('(')[0].strip()
    print(f"  {name:<43} {r['decision']:<15} {r['risk_score']:>6.1f}")

# final stats
print(f"\n{'─' * 60}")
stats = requests.get(f"{BASE_URL}/audit/stats").json()
print(f"Total API calls logged: {stats['total_requests']}")
print(f"ALLOW:       {stats['decisions']['ALLOW']}")
print(f"STEP_UP_MFA: {stats['decisions']['STEP_UP_MFA']}")
print(f"BLOCK:       {stats['decisions']['BLOCK']}")
print(f"Avg Latency: {stats['avg_latency_ms']}ms")
print(f"{'=' * 60}")
print("DEMONSTRATION COMPLETE")
print(f"{'=' * 60}")