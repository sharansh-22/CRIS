import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1
csv_path = '/home/sharansh/CRIS/data/Indices/SPY.csv'
df = pd.read_csv(csv_path, header=[0,1], index_col=0, parse_dates=True)
df.columns = [col[0] for col in df.columns]
df['Returns'] = df['Close'].pct_change()
df['Abs_Returns'] = df['Returns'].abs()
abs_rets = df['Abs_Returns'].dropna()

# Step 2
baseline_A = abs_rets['2018-01-01':'2018-06-30'].mean()
baseline_D = abs_rets['2018-01-01':'2018-12-31'].mean()

def find_calm_window(series, window_size, ref_baseline):
    roll_mean = series.rolling(10).mean()
    # No restriction backward except what's available
    search_space = roll_mean
    dates = search_space.index[::-1]
    
    for end_date in dates:
        loc = series.index.get_loc(end_date)
        if loc - window_size + 1 < 0: continue
        start_loc = loc - window_size + 1
        start_date = series.index[start_loc]
        
        window = roll_mean[start_date:end_date]
        if len(window) < window_size: continue
        
        if (window < ref_baseline * 1.5).all():
            return start_date, end_date
    return None, None

c_start, c_end = find_calm_window(abs_rets, 126, baseline_D)
if c_start:
    baseline_C = abs_rets[c_start:c_end].mean()
    c_period = f"{c_start.strftime('%Y-%m-%d')} to {c_end.strftime('%Y-%m-%d')}"
else:
    print("WARNING: No calm 126-day window found. Falling back to Candidate D.")
    baseline_C = baseline_D
    c_period = "Fallback to Full 2018"

b_start, b_end = find_calm_window(abs_rets, 63, baseline_D)
if b_start:
    b_hist_series = abs_rets['2018-01-01':'2018-03-31']
    b_recent_series = abs_rets[b_start:b_end]
    baseline_B = pd.concat([b_hist_series, b_recent_series]).mean()
    b_period = f"2018-01-01 to 2018-03-31 + {b_start.strftime('%Y-%m-%d')} to {b_end.strftime('%Y-%m-%d')}"
else:
    print("WARNING: No calm 63-day window found. Falling back to Candidate D.")
    baseline_B = baseline_D
    b_period = "Fallback to Full 2018"

candidates = {
    'A': {'name': 'A 6m-2018', 'period': '2018-01-01 to 2018-06-30', 'baseline': baseline_A},
    'B': {'name': 'B Split', 'period': b_period, 'baseline': baseline_B},
    'C': {'name': 'C Recent', 'period': c_period, 'baseline': baseline_C},
    'D': {'name': 'D Full2018', 'period': '2018-01-01 to 2018-12-31', 'baseline': baseline_D}
}

# Step 3
print("\n--- Baseline Summary ---")
for k, v in candidates.items():
    print(f"Candidate {k} ({v['name']}):")
    print(f"  Period: {v['period']}")
    print(f"  Baseline: {v['baseline']*100:.3f}%")
    print(f"  1.5x (STRESS): {v['baseline']*1.5*100:.3f}%")
    print(f"  3.0x (BLACK SWAN): {v['baseline']*3.0*100:.3f}%\n")

# Step 4
events = [
    {"name": "COVID Crash", "start": "2020-03-01", "end": "2020-03-23", "expected": "BLACK_SWAN"},
    {"name": "Q4 2018 Selloff", "start": "2018-10-01", "end": "2018-12-31", "expected": "STRESS"},
    {"name": "Bear 2022", "start": "2022-01-01", "end": "2022-06-30", "expected": "STRESS"},
    {"name": "Vaccine Rally", "start": "2020-11-09", "end": "2020-11-20", "expected": "NORMAL"},
    {"name": "Calm 2019 H1", "start": "2019-01-01", "end": "2019-06-30", "expected": "NORMAL"},
    {"name": "COVID Pre-crash", "start": "2020-01-01", "end": "2020-02-28", "expected": "STRESS"}
]

def classify(vol_ratio):
    if vol_ratio < 1.5: return "NORMAL"
    elif vol_ratio < 3.0: return "STRESS"
    else: return "BLACK_SWAN"

for k, v in candidates.items():
    correct_count = 0
    classifications = []
    for ev in events:
        event_mean = abs_rets[ev['start']:ev['end']].mean()
        vol_ratio = event_mean / v['baseline']
        cl = classify(vol_ratio)
        correct = (cl == ev['expected'])
        if correct: correct_count += 1
        classifications.append(cl)
    v['correct'] = correct_count
    v['classifications'] = classifications

# Step 5 - Lead time
for k, v in candidates.items():
    search_period = abs_rets['2020-01-01':'2020-03-23']
    roll = search_period.rolling(10).mean()
    trigger_dates = roll[roll / v['baseline'] > 3.0].dropna()
    if not trigger_dates.empty:
        trigger_date = trigger_dates.index[0]
        # count trading days
        lead_time = len(abs_rets[trigger_date:'2020-03-23']) - 1 # excluding trigger date itself? let's do simply count of days between 
        v['lead_time'] = max(0, lead_time)
    else:
        v['lead_time'] = 0

# Step 6 - False positive rate
for k, v in candidates.items():
    calm_h1 = abs_rets['2019-01-01':'2019-06-30']
    roll = calm_h1.rolling(10).mean()
    roll = roll.dropna()
    fps = sum((roll / v['baseline']) >= 1.5)
    v['fp_rate'] = fps / len(roll) if len(roll) > 0 else 0

# Threshold Stability
# Measure how much baseline varies if we shift the calm window by +/- 21 trading days (approx 1 month)
v_stability = {}
v_stability['A'] = 0 # prefect
v_stability['D'] = 0 # perfect

def get_shifted_baseline(shift_days, window_size):
    ref_b = baseline_D
    roll_mean = abs_rets.rolling(10).mean()
    start, end = find_calm_window(abs_rets, window_size, ref_b)
    if not start: return None
    # shift window
    loc_end = abs_rets.index.get_loc(end) + shift_days
    # limit check
    if loc_end >= len(abs_rets): loc_end = len(abs_rets) - 1
    if loc_end - window_size + 1 < 0: return None
    new_end = abs_rets.index[loc_end]
    new_start = abs_rets.index[loc_end - window_size + 1]
    return abs_rets[new_start:new_end].mean()

# Instead of actual shift, we can just use fixed stability scores as implied: candidates A/D 1.0. B/C vary. Let's calculate actual variance.
# To properly calculate variance, we can just shift the 3 month and 6 month windows back and forth 1 month, and take var of the 3 values.
baselines_C = []
baselines_C.append(baseline_C)
if c_end:
    idx = abs_rets.index.get_loc(c_end)
    if idx + 21 < len(abs_rets): baselines_C.append(abs_rets.iloc[(idx+21)-126+1 : idx+21+1].mean())
    if idx - 21 >= 125: baselines_C.append(abs_rets.iloc[(idx-21)-126+1 : idx-21+1].mean())
v_stability['C'] = np.var(baselines_C) / (np.mean(baselines_C)**2) if len(baselines_C)>0 else 1.0

baselines_B = []
baselines_B.append(baseline_B)
if b_end:
    idx = abs_rets.index.get_loc(b_end)
    def b_val(e_idx):
        return pd.concat([abs_rets['2018-01-01':'2018-03-31'], abs_rets.iloc[e_idx-63+1 : e_idx+1]]).mean()
    if idx + 21 < len(abs_rets): baselines_B.append(b_val(idx+21))
    if idx - 21 >= 62: baselines_B.append(b_val(idx-21))
v_stability['B'] = np.var(baselines_B) / (np.mean(baselines_B)**2) if len(baselines_B)>0 else 1.0

# convert variances to stability scores: lower variance -> 1.0. A and D are 0 variance.
max_var = max(v_stability.values()) if max(v_stability.values()) > 0 else 1.0
st_scores = {k: 1.0 - (v / max_var) for k, v in v_stability.items()}
st_scores['A'] = 1.0
st_scores['D'] = 1.0

for k in candidates:
    candidates[k]['stability'] = st_scores[k]

# Normalize metrics
c_rates = [v['correct']/6 for v in candidates.values()]
fp_rates = [v['fp_rate'] for v in candidates.values()]
lead_times = [v['lead_time'] for v in candidates.values()]
stabs = [v['stability'] for v in candidates.values()]

min_c, max_c = min(c_rates), max(c_rates)
min_fp, max_fp = min(fp_rates), max(fp_rates)
min_lt, max_lt = min(lead_times), max(lead_times)
min_st, max_st = min(stabs), max(stabs)

def norm(val, mi, ma, invert=False):
    if ma == mi: return 1.0 # all the same
    n = (val - mi) / (ma - mi)
    return 1.0 - n if invert else n

for idx, (k, v) in enumerate(candidates.items()):
    n_c = norm(c_rates[idx], min_c, max_c)
    n_fp = norm(fp_rates[idx], min_fp, max_fp, invert=True)
    n_lt = norm(lead_times[idx], min_lt, max_lt)
    n_st = norm(stabs[idx], min_st, max_st)
    score = (n_c * 0.4) + (n_fp * 0.3) + (n_lt * 0.2) + (n_st * 0.1)
    v['score'] = score

winner_k = max(candidates.keys(), key=lambda k: candidates[k]['score'])
sorted_scores = sorted([(k, v['score']) for k, v in candidates.items()], key=lambda x: x[1], reverse=True)
runner_up_k = sorted_scores[1][0] if len(sorted_scores) > 1 else None

# Dump info needed for later steps to a json string? Let's just create the plot and print results directly.

os.makedirs('/home/sharansh/CRIS/layer3_bssc/outputs/simulation_output', exist_ok=True)
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor('#0d1117')
for ax in axs.flat:
    ax.set_facecolor('#0d1117')

# Panel 1
names = [v['name'] for v in candidates.values()]
bases = [v['baseline']*100 for v in candidates.values()]
colors = ['gold' if k == winner_k else 'silver' if k == runner_up_k else 'grey' for k in candidates.keys()]
axs[0,0].bar(names, bases, color=colors)
axs[0,0].set_title('Baseline Mean Daily Move by Candidate')
for i, v in enumerate(bases):
    axs[0,0].text(i, v, f"{v:.3f}%", ha='center', va='bottom')

# Panel 2
ev_names = [ev['name'] for ev in events]
ev_expected = [ev['expected'] for ev in events]
# matrix of correct (1) / incorrect (0)
mat = np.zeros((4, 6))
class_labels = []
for i, (k, v) in enumerate(candidates.items()):
    cls_list = v['classifications']
    mat[i, :] = [1 if cls_list[j] == ev_expected[j] else 0 for j in range(6)]
    class_labels.append(cls_list)

axs[0,1].imshow(mat, cmap='RdYlGn', aspect='auto', alpha=0.6)
axs[0,1].set_xticks(np.arange(6))
axs[0,1].set_yticks(np.arange(4))
axs[0,1].set_xticklabels(ev_names, rotation=45, ha='right')
axs[0,1].set_yticklabels(names)
for i in range(4):
    for j in range(6):
        axs[0,1].text(j, i, class_labels[i][j], ha='center', va='center', color='white', fontweight='bold', fontsize=8)
axs[0,1].set_title('Classification Correctness by Candidate')

# Panel 3
s_names = [candidates[k]['name'] for k in [s[0] for s in sorted_scores]]
s_vals = [s[1] for s in sorted_scores]
bars = axs[1,0].barh(s_names, s_vals, color='steelblue')
axs[1,0].invert_yaxis()
for bar in bars:
    w = bar.get_width()
    axs[1,0].text(w, bar.get_y() + bar.get_height()/2, f" {w:.3f}", va='center')
# WINNER annotation
axs[1,0].text(s_vals[0]/2, 0, 'WINNER', color='gold', fontweight='bold', fontsize=12, ha='center', va='center')
axs[1,0].set_title('TS-002 Composite Scores')

# Panel 4
lts = [v['lead_time'] for v in candidates.values()]
axs[1,1].bar(names, lts, color='purple')
axs[1,1].set_title('COVID Black Swan Lead Time (days)')
for i, v in enumerate(lts):
    axs[1,1].text(i, v, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('/home/sharansh/CRIS/layer3_bssc/outputs/simulation_output/TS002_baseline_comparison.png')

print("╔══════════════════════════════════════════════════════╗")
print("║         TS-002 — BASELINE CALIBRATION RESULTS       ║")
print("╠══════════════════════════════════════════════════════╣")
print("║  Candidate  │ Baseline │ Correct │ FP Rate │ Score  ║")
print("╠══════════════════════════════════════════════════════╣")
for k, v in candidates.items():
    name_str = f"{v['name'][:10]:<11}"
    b_str = f"{v['baseline']*100:5.3f}%"
    c_str = f"{v['correct']}/6  "
    f_str = f"{v['fp_rate']*100:5.1f}%"
    s_str = f"{v['score']:.3f}"
    print(f"║ {name_str} │  {b_str}  │  {c_str} │  {f_str}  │ {s_str} ║")
print("╠══════════════════════════════════════════════════════╣")
w_name = candidates[winner_k]['name']
w_score = candidates[winner_k]['score']
print(f"║  WINNER: {w_name:<20} Score: {w_score:.3f}             ║")
print(f"║  REASON: Best composite score                         ║")
print("╚══════════════════════════════════════════════════════╝")

# Export json variables for bash to inject into README? Let's just output text that we can extract.
print("\n--- EXTRACT INFO ---")
print(f"WINNER_NAME={w_name}")
print(f"WINNER_SCORE={w_score:.3f}")
for k, v in candidates.items():
    print(f"CAND_{k}_BASELINE={v['baseline']*100:.3f}")
    print(f"CAND_{k}_CORRECT={v['correct']}")
    print(f"CAND_{k}_FP={v['fp_rate']*100:.1f}")
    print(f"CAND_{k}_LT={v['lead_time']}")
    print(f"CAND_{k}_SCORE={v['score']:.3f}")
print("--- END EXTRACT ---")
