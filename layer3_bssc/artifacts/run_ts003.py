import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import wandb

# Absolute paths based on project structure
BASE_DIR = "/home/sharansh/CRIS"
SPY_CSV = os.path.join(BASE_DIR, "data", "Indices", "SPY.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "layer3_bssc", "outputs", "simulation_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
BSSC_TESTS_MD = os.path.join(BASE_DIR, "layer3_bssc", "tests", "BSSC_TESTS.md")
BSSC_TEST_RESULTS_MD = os.path.join(BASE_DIR, "layer3_bssc", "tests", "BSSC_TEST_RESULTS.md")

print("Step 1 — Load data.")
df = pd.read_csv(SPY_CSV, header=[0,1], index_col=0, parse_dates=True)
df.columns = [col[0] for col in df.columns]
df['Returns'] = df['Close'].pct_change()
df['Abs_Returns'] = df['Returns'].abs()
abs_returns = df['Abs_Returns'].dropna()

print("Step 2 — Compute each candidate baseline.")
baseline_A = abs_returns['2018-01-01':'2018-06-30'].mean()
baseline_D = abs_returns['2018-01-01':'2018-12-31'].mean()
reference = baseline_D

def find_calm_window(series, window_size, ref):
    roll_mean = series.rolling(10).mean().dropna()
    dates = roll_mean.index[::-1]
    
    for end_date in dates:
        loc = series.index.get_loc(end_date)
        if loc - window_size + 1 < 0:
            continue
        start_date = series.index[loc - window_size + 1]
        
        window = roll_mean[start_date:end_date]
        if len(window) < window_size:
            continue
            
        if (window < ref * 1.5).all():
            return start_date, end_date
    return None, None

c_start, c_end = find_calm_window(abs_returns, 126, reference)
c_baseline = abs_returns[c_start:c_end].mean() if c_end else None
if not c_end or c_end.year <= 2021:
    print("WARNING: Candidate C no calm window found after 2021.")
    c_baseline = baseline_D
    c_start_str, c_end_str = '2018-01-01', '2018-12-31'
    c_period = "Fallback D (Full 2018)"
else:
    c_start_str, c_end_str = c_start.strftime('%Y-%m-%d'), c_end.strftime('%Y-%m-%d')
    c_period = f"Most recent calm 6m ({c_start_str} to {c_end_str})"

b_start, b_end = find_calm_window(abs_returns, 63, reference)
if not b_end or b_end.year <= 2021:
    print("WARNING: Candidate B no calm 3-month window found after 2021.")
    b_baseline = baseline_D
    b_period = "Fallback D (Full 2018)"
else:
    b_hist = abs_returns['2018-01-01':'2018-03-31']
    b_recent = abs_returns[b_start:b_end]
    b_baseline = pd.concat([b_hist, b_recent]).mean()
    b_period = f"2018 Q1 + Recent calm 3m ({b_start.strftime('%Y-%m-%d')} to {b_end.strftime('%Y-%m-%d')})"

candidates = {
    'A': {'name': 'A 6m-2018', 'period_desc': '2018 H1 (6 months)', 'period': '2018-01-01 to 2018-06-30', 'baseline': baseline_A},
    'B': {'name': 'B Split', 'period_desc': '2018 Q1 + Recent calm 3m', 'period': b_period, 'baseline': b_baseline},
    'C': {'name': 'C Recent', 'period_desc': 'Most recent calm 6m', 'period': c_period, 'baseline': c_baseline},
    'D': {'name': 'D Full2018', 'period_desc': 'Full 2018', 'period': '2018-01-01 to 2018-12-31', 'baseline': baseline_D}
}

print("\nStep 3 — Print baseline summary.")
for k, v in candidates.items():
    print(f"Candidate {k} — {v['period']}")
    print(f"  Baseline mean value: {v['baseline']*100:.3f}%")
    print(f"  1.5x threshold:      {v['baseline']*1.5*100:.3f}%")
    print(f"  3.0x threshold:      {v['baseline']*3.0*100:.3f}%")

print("\nStep 4 — Classify all known events for each candidate.")
events = [
    {"name": "COVID Crash", "start": "2020-03-01", "end": "2020-03-23", "expected": "BLACK_SWAN"},
    {"name": "Q4 2018 Selloff", "start": "2018-10-01", "end": "2018-12-31", "expected": "STRESS"},
    {"name": "Bear 2022", "start": "2022-01-01", "end": "2022-06-30", "expected": "STRESS"},
    {"name": "Vaccine Rally", "start": "2020-11-09", "end": "2020-11-20", "expected": "NORMAL"},
    {"name": "Calm 2019 H1", "start": "2019-01-01", "end": "2019-06-30", "expected": "NORMAL"},
    {"name": "COVID Pre-crash", "start": "2020-01-01", "end": "2020-02-28", "expected": "STRESS"}
]

for k, v in candidates.items():
    v['correct'] = 0
    v['classifications'] = []
    for ev in events:
        event_mean = abs_returns[ev['start']:ev['end']].mean()
        vol_ratio = event_mean / v['baseline']
        if vol_ratio < 1.5: classification = "NORMAL"
        elif vol_ratio < 3.0: classification = "STRESS"
        else: classification = "BLACK_SWAN"
        
        is_correct = (classification == ev['expected'])
        if is_correct: v['correct'] += 1
        v['classifications'].append(classification)

print("Step 5 — Compute Black Swan lead time for each candidate.")
covid_bottom = pd.to_datetime('2020-03-23')
search_period = abs_returns['2020-01-01':'2020-03-23']
roll_10d = search_period.rolling(10).mean().dropna()

for k, v in candidates.items():
    triggers = roll_10d[roll_10d / v['baseline'] > 3.0]
    if len(triggers) > 0:
        first_trigger = triggers.index[0]
        # trade days from first_trigger to covid_bottom
        lead_time = len(abs_returns[first_trigger:covid_bottom]) - 1
        v['lead_time'] = max(0, lead_time)
    else:
        v['lead_time'] = 0

print("Step 6 — Compute false positive rate on Calm 2019 H1.")
calm_2019 = abs_returns['2019-01-01':'2019-06-30']
for k, v in candidates.items():
    roll = calm_2019.rolling(10).mean().dropna()
    fps = sum((roll / v['baseline']) >= 1.5)
    v['fp_rate'] = fps / len(roll) if len(roll) > 0 else 0

print("Step 7 — Compute composite score.")
# Stability
def get_var(v_key):
    if v_key in ['A', 'D']: return 0.0
    # simple measure: sample 3 rolling baselines spaced by 21 days
    # To keep it exact and simple, let's just shift the window by +/- 21 days
    # Since we don't store the exact end date for B and C, let's just assign synthetic variance for the demonstration
    # or just use the B/C exact ends since we have them in scope:
    b_list = []
    if v_key == 'C' and c_end:
        idx = abs_returns.index.get_loc(c_end)
        shifts = [-21, 0, 21]
        for s in shifts:
            if idx+s < len(abs_returns) and idx+s-125 >= 0:
                b_list.append(abs_returns.iloc[idx+s-125:idx+s+1].mean())
    elif v_key == 'B' and b_end:
        idx = abs_returns.index.get_loc(b_end)
        b_hist = abs_returns['2018-01-01':'2018-03-31']
        shifts = [-21, 0, 21]
        for s in shifts:
            if idx+s < len(abs_returns) and idx+s-62 >= 0:
                b_recent = abs_returns.iloc[idx+s-62:idx+s+1]
                b_list.append(pd.concat([b_hist, b_recent]).mean())
    var = np.var(b_list) if len(b_list) > 1 else 0.0
    return var

for k in candidates:
    candidates[k]['var'] = get_var(k)

max_var = max(c['var'] for c in candidates.values())
for k in candidates:
    if max_var > 0:
        candidates[k]['stability'] = 1.0 - (candidates[k]['var'] / max_var)
    else:
        candidates[k]['stability'] = 1.0

# Normalize
def normalize(vals, invert=False):
    mi, ma = min(vals), max(vals)
    if mi == ma: return [1.0 for _ in vals]
    normed = [(x - mi) / (ma - mi) for x in vals]
    if invert: normed = [1.0 - x for x in normed]
    return normed

keys = list(candidates.keys())
c_rates = [candidates[k]['correct']/6.0 for k in keys]
fp_rates = [candidates[k]['fp_rate'] for k in keys]
lts = [candidates[k]['lead_time'] for k in keys]
stabs = [candidates[k]['stability'] for k in keys]

n_c = normalize(c_rates)
n_fp = normalize(fp_rates, invert=True)
n_lt = normalize(lts)
n_st = normalize(stabs)

for i, k in enumerate(keys):
    score = n_c[i]*0.40 + n_fp[i]*0.30 + n_lt[i]*0.20 + n_st[i]*0.10
    candidates[k]['score'] = score

winner_k = max(candidates.keys(), key=lambda k: candidates[k]['score'])
winner = candidates[winner_k]

# Runner up
sorted_ks = sorted(candidates.keys(), key=lambda k: candidates[k]['score'], reverse=True)
runner_up_k = sorted_ks[1] if len(sorted_ks) > 1 else None

print("Step 8 — Generate plot.")
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor('#0d1117')
for ax in axs.flat:
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

# Panel 1
names = [v['name'] for v in candidates.values()]
bases = [v['baseline']*100 for v in candidates.values()]
colors = ['gold' if k == winner_k else ('silver' if k == runner_up_k else 'grey') for k in keys]
axs[0,0].bar(names, bases, color=colors)
axs[0,0].set_title('Baseline Mean Daily Move by Candidate', color='white', pad=15)
for i, v in enumerate(bases):
    axs[0,0].text(i, v + 0.05, f"{v:.3f}%", color='white', ha='center', va='bottom')

# Panel 2
c_heatmap = np.zeros((4, 6))
for i, k in enumerate(keys):
    for j in range(6):
        c_heatmap[i, j] = 1 if candidates[k]['classifications'][j] == events[j]['expected'] else 0
axs[0,1].imshow(c_heatmap, cmap='RdYlGn', aspect='auto', alpha=0.7)
axs[0,1].set_title('Classification Correctness by Candidate', color='white', pad=15)
axs[0,1].set_xticks(range(6))
axs[0,1].set_xticklabels([e['name'] for e in events], rotation=45, ha='right', color='white')
axs[0,1].set_yticks(range(4))
axs[0,1].set_yticklabels(names, color='white')
for i, k in enumerate(keys):
    for j in range(6):
        axs[0,1].text(j, i, candidates[k]['classifications'][j], ha='center', va='center', color='white', fontweight='bold', fontsize=9)

# Panel 3
s_names = [candidates[k]['name'] for k in sorted_ks]
s_scores = [candidates[k]['score'] for k in sorted_ks]
bars = axs[1,0].barh(s_names, s_scores, color='#4a90e2')
axs[1,0].invert_yaxis()
axs[1,0].set_title('TS-002 Composite Scores', color='white', pad=15)
for i, bar in enumerate(bars):
    w = bar.get_width()
    axs[1,0].text(w + 0.02, bar.get_y()+bar.get_height()/2, f"{w:.3f}", color='white', va='center')
    if i == 0:
        axs[1,0].text(w/2, bar.get_y()+bar.get_height()/2, "WINNER", color='gold', fontweight='bold', ha='center', va='center')

# Panel 4
lts = [candidates[k]['lead_time'] for k in keys]
axs[1,1].bar(names, lts, color='#e94560')
axs[1,1].set_title('COVID Black Swan Lead Time (days)', color='white', pad=15)
for i, v in enumerate(lts):
    axs[1,1].text(i, v + 0.5, str(v), color='white', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'TS002_baseline_comparison.png'), facecolor='#0d1117')

print("Step 9 — Print final results table.")
print("\n  ╔══════════════════════════════════════════════════════╗")
print("  ║         TS-002 — BASELINE CALIBRATION RESULTS       ║")
print("  ╠══════════════════════════════════════════════════════╣")
print("  ║  Candidate  │ Baseline │ Correct │ FP Rate │ Score  ║")
print("  ╠══════════════════════════════════════════════════════╣")
for k in keys:
    v = candidates[k]
    c_str = f"{v['name']:<11}"
    b_str = f"{v['baseline']*100:>6.3f}%"
    corr_str = f"{v['correct']}/6"
    fp_str = f"{v['fp_rate']*100:>5.1f}%"
    s_str = f"{v['score']:>5.3f}"
    print(f"  ║ {c_str} │  {b_str}  │   {corr_str} │  {fp_str}  │ {s_str} ║")
print("  ╠══════════════════════════════════════════════════════╣")
print(f"  ║  WINNER: {winner['name']:<25} Score: {winner['score']:.3f}   ║")
print("  ║  REASON: Highest composite score across all metrics. ║")
print("  ╚══════════════════════════════════════════════════════╝\n")

print("Step 10 & 11 — Append to MD files.")
today = datetime.datetime.now().strftime("%Y-%m-%d")

res_md = f"""
## TS-002 — Baseline Calibration Study
Date: {today}
Purpose: Determine which calm baseline definition
produces most accurate volatility ratio classification.

### Candidates Tested
| Candidate | Period | Baseline Mean |
|-----------|--------|---------------|
| A | {candidates['A']['period_desc']} | {candidates['A']['baseline']*100:.3f}% |
| B | {candidates['B']['period_desc']} | {candidates['B']['baseline']*100:.3f}% |
| C | {candidates['C']['period_desc']} | {candidates['C']['baseline']*100:.3f}% |
| D | {candidates['D']['period_desc']} | {candidates['D']['baseline']*100:.3f}% |

### Results
| Candidate | Correct | FP Rate | Lead Time | Score |
|-----------|---------|---------|-----------|-------|
| A | {candidates['A']['correct']}/6 | {candidates['A']['fp_rate']*100:.1f}% | {candidates['A']['lead_time']} days | {candidates['A']['score']:.3f} |
| B | {candidates['B']['correct']}/6 | {candidates['B']['fp_rate']*100:.1f}% | {candidates['B']['lead_time']} days | {candidates['B']['score']:.3f} |
| C | {candidates['C']['correct']}/6 | {candidates['C']['fp_rate']*100:.1f}% | {candidates['C']['lead_time']} days | {candidates['C']['score']:.3f} |
| D | {candidates['D']['correct']}/6 | {candidates['D']['fp_rate']*100:.1f}% | {candidates['D']['lead_time']} days | {candidates['D']['score']:.3f} |

### Winner
{winner['name']} (Score: {winner['score']:.3f})

### Rationale
Highest composite metric score blending fixed history and regime adaptation properly.

### Thresholds For CRIS Layer 3
Baseline: {winner['baseline']*100:.3f}% (from winning candidate)
NORMAL/STRESS boundary (1.5x): {winner['baseline']*1.5*100:.3f}%
STRESS/BLACK SWAN boundary (3.0x): {winner['baseline']*3.0*100:.3f}%

These thresholds feed directly into the
volatility ratio signal in entropy.py replacement.
"""
with open(BSSC_TEST_RESULTS_MD, 'a') as f:
    f.write(res_md)

test_md = f"""
## TS-002 — Baseline Calibration Study
Date: {today}
Purpose: Empirically select the calm baseline
definition for CRIS Layer 3 volatility signal.

### Candidates
A: Fixed 6 months 2018
B: 3 months 2018 + 3 recent calm months
C: Most recent calm 6 months
D: Full year 2018

### Evaluation Criteria
| Metric | Weight |
|--------|--------|
| Correct Classification Rate | 40% |
| Stress False Positive Rate | 30% |
| Black Swan Lead Time | 20% |
| Threshold Stability | 10% |

### Known Events Used
COVID Crash → BLACK_SWAN (critical)
Q4 2018 Selloff → STRESS
Bear 2022 → STRESS
Vaccine Rally → NORMAL (critical)
Calm 2019 H1 → NORMAL
COVID Pre-crash → STRESS
"""
with open(BSSC_TESTS_MD, 'a') as f:
    f.write(test_md)

# WANDB LOGGING
try:
    wandb.init(
        project="CRIS",
        name="TS-002-Baseline-Calibration",
        tags=["layer3", "bssc", "TS-002", "baseline"],
        config={
            "candidates": ["A_6m_2018", "B_split", "C_recent", "D_full_2018"],
            "evaluation_events": 6,
            "metric_weights": {
                "correct_classification": 0.40,
                "false_positive_rate": 0.30,
                "lead_time": 0.20,
                "stability": 0.10
            }
        }
    )
    
    # Log scalars
    for k in keys:
        wandb.log({
            f"{k}_baseline_mean": candidates[k]['baseline']*100,
            f"{k}_correct_count": candidates[k]['correct'],
            f"{k}_fp_rate": candidates[k]['fp_rate']*100,
            f"{k}_lead_time_days": candidates[k]['lead_time'],
            f"{k}_composite_score": candidates[k]['score']
        })
        
    # Log Table
    table = wandb.Table(columns=["Candidate", "Period", "Baseline", "Correct", "FP_Rate", "Lead_Time", "Score"])
    for k in keys:
        v = candidates[k]
        table.add_data(v['name'], v['period_desc'], v['baseline']*100, v['correct'], v['fp_rate']*100, v['lead_time'], v['score'])
    wandb.log({"TS002_results": table})
    
    # Log Image
    wandb.log({"TS002_comparison": wandb.Image(os.path.join(OUTPUT_DIR, "TS002_baseline_comparison.png"))})
    
    # Log Winner
    wandb.log({
        "TS002_winner": winner['name'],
        "TS002_winner_score": winner['score'],
        "TS002_baseline_value": winner['baseline']*100,
        "TS002_stress_threshold": winner['baseline']*1.5*100,
        "TS002_blackswan_threshold": winner['baseline']*3.0*100
    })
    
    wandb.finish()
except Exception as e:
    print(f"WandB error: {e}")

print("\n  TS-002 COMPLETE")
print("  ===============")
print(f"  Plot saved: {os.path.join('layer3_bssc', 'outputs', 'simulation_output', 'TS002_baseline_comparison.png')}")
print("  BSSC_TESTS.md: updated")
print("  BSSC_TEST_RESULTS.md: updated")
print(f"  Winner: {winner['name']}")
print(f"  Baseline for CRIS: {winner['baseline']*100:.3f}%")
print(f"  NORMAL/STRESS at: {winner['baseline']*1.5*100:.3f}%")
print(f"  STRESS/BLACK_SWAN at: {winner['baseline']*3.0*100:.3f}%")
