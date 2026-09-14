import matplotlib.pyplot as plt
import numpy as np

# ── Color scheme (monochrome, matches academic thesis style) ──
COLORS = ['#1a1a2e', '#16213e', '#0f3460', '#533483', '#e94560']
GREY   = '#888888'

# ══════════════════════════════════════════════════════════════
# CHART 1 — Individual Model Performance (Grouped Bar Chart)
# ══════════════════════════════════════════════════════════════
metrics = ['Recall', 'Precision', 'F1-Score', 'AUC-ROC']
rf      = [96.54, 97.72, 97.13, 99.68]
lstm    = [95.86, 97.01, 96.43, 99.58]
full    = [96.92, 96.55, 96.74, 99.68]

x   = np.arange(len(metrics))
w   = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - w, rf,   w, label='Random Forest', color=COLORS[0])
ax.bar(x,     lstm, w, label='LSTM',          color=COLORS[2])
ax.bar(x + w, full, w, label='Full Framework',color=COLORS[4])

ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Figure 4.1 — Individual Model Performance', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(93, 101)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('fig4_1_model_performance.png', dpi=300)
plt.show()
print("Saved: fig4_1_model_performance.png")

# ══════════════════════════════════════════════════════════════
# CHART 2 — Decision Distribution (Doughnut Chart)
# ══════════════════════════════════════════════════════════════
labels = ['Allow\n49.8%', 'Step-Up MFA\n10.7%', 'Block\n39.5%']
sizes  = [49.8, 10.7, 39.5]
colors = [COLORS[0], COLORS[2], COLORS[4]]

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts = ax.pie(
    sizes, labels=labels, colors=colors,
    startangle=90, pctdistance=0.85,
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
)
for t in texts:
    t.set_fontsize(12)
ax.set_title('Figure 4.2 — Decision Distribution\n(39,990 test events)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_2_decision_distribution.png', dpi=300)
plt.show()
print("Saved: fig4_2_decision_distribution.png")

# ══════════════════════════════════════════════════════════════
# CHART 3 — Ablation Study (Grouped Bar Chart)
# ══════════════════════════════════════════════════════════════
configs  = ['RF alone', 'LSTM alone', 'ML Fusion\nOnly', 'Behavioural\nOnly', 'Full\nFramework']
recall   = [96.53, 95.86, 98.02, 99.96, 96.92]
f1       = [97.13, 96.43, 96.77, 86.58, 96.74]
fpr      = [2.25,  2.96,  4.56,  30.96, 3.47]

x = np.arange(len(configs))
w = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - w, recall, w, label='Recall (%)',    color=COLORS[0])
ax.bar(x,     f1,     w, label='F1-Score (%)',  color=COLORS[2])
ax.bar(x + w, fpr,    w, label='FPR (%)',       color=COLORS[4])

ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Figure 4.3 — Ablation Study Results', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('fig4_3_ablation_study.png', dpi=300)
plt.show()
print("Saved: fig4_3_ablation_study.png")

# ══════════════════════════════════════════════════════════════
# CHART 4 — OOD Per-Attack-Type Recall (Horizontal Bar Chart)
# ══════════════════════════════════════════════════════════════
attack_types = ['MFA Fatigue', 'Session Hijacking', 'AiTM Phishing', 'SIM Swap']
ood_recall   = [98.4, 96.9, 98.7, 97.2]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(attack_types, ood_recall, color=COLORS[2], edgecolor='white')

for bar, val in zip(bars, ood_recall):
    ax.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height()/2,
            f'{val}%', va='center', ha='right', color='white',
            fontsize=11, fontweight='bold')

ax.set_xlabel('Recall (%)', fontsize=12)
ax.set_title('Figure 4.4 — OOD Evaluation: Per-Attack-Type Recall', fontsize=13, fontweight='bold')
ax.set_xlim(90, 100)
ax.xaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('fig4_4_ood_per_attack_type.png', dpi=300)
plt.show()
print("Saved: fig4_4_ood_per_attack_type.png")

# ══════════════════════════════════════════════════════════════
# CHART 5 — Latency Distribution (Bar Chart)
# ══════════════════════════════════════════════════════════════
lat_labels = ['Min', 'Mean', 'Median', 'P95', 'P99', 'Max']
lat_values = [35.77, 47.51, 47.37, 50.14, 58.19, 62.18]
target     = 500

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(lat_labels, lat_values, color=COLORS[0], width=0.5)

ax.axhline(y=target, color=COLORS[4], linestyle='--',
           linewidth=1.5, label=f'Target < {target}ms')

for bar, val in zip(bars, lat_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val}ms', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Latency (ms)', fontsize=12)
ax.set_title('Figure 4.5 — Latency Benchmark Results (1,000 runs)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('fig4_5_latency_benchmark.png', dpi=300)
plt.show()
print("Saved: fig4_5_latency_benchmark.png")

# ══════════════════════════════════════════════════════════════
# CHART 6 — Framework vs Static Baseline vs Hoda (Radar/Spider)
# ══════════════════════════════════════════════════════════════
from matplotlib.patches import FancyArrowPatch

categories = ['Recall', 'Precision', 'F1-Score', 'AUC-ROC\n(×100)']
proposed   = [96.92, 96.55, 96.74, 99.68]
hoda       = [83.00, 89.00, 86.00, 92.00]
static     = [0.00,  0.00,  0.00,  0.00]

x   = np.arange(len(categories))
w   = 0.25

fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - w, proposed, w, label='Proposed Framework', color=COLORS[0])
ax.bar(x,     hoda,     w, label='Hoda et al. (2025)†', color=COLORS[2])
ax.bar(x + w, static,   w, label='Static MFA Baseline', color=COLORS[4])

ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Figure 4.6 — Comparative Results\n(Proposed Framework vs. Hoda et al. (2025) vs. Static Baseline)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# Add non-comparability footnote
fig.text(0.01, 0.01,
         '† Hoda et al. (2025) evaluated on a different dataset under different conditions. '
         'Figures are contextual, not a controlled benchmark.',
         fontsize=8, color=GREY, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('fig4_6_comparative_results.png', dpi=300)
plt.show()
print("Saved: fig4_6_comparative_results.png")

print("\nAll 6 charts generated and saved.")