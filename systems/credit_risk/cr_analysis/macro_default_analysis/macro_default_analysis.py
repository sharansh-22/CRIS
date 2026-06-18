import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, ttest_ind
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, roc_curve
from lightgbm import LGBMClassifier

# Configure project root
PROJECT_ROOT = Path("/home/sharansh/CRIS")
sys.path.append(str(PROJECT_ROOT))
from configs.credit_config import SEED, OUTPUT_DIR, MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("CRA.MacroDefaultAnalysis")

# Setup output folders
AN_DIR = PROJECT_ROOT / "systems" / "credit_risk" / "cr_analysis" / "macro_default_analysis"
FIGURES_DIR = AN_DIR / "outputs" / "figures"
TABLES_DIR = AN_DIR / "outputs" / "tables"
DATA_DIR = AN_DIR / "outputs" / "data"

for d in [FIGURES_DIR, TABLES_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Colors for plotting (rich dark theme style, premium palette)
BG_COLOR = "#0d1117"
AXES_COLOR = "#161b22"
GRID_COLOR = "#30363d"
PRIMARY_COLOR = "#58a6ff"  # Sleek blue
STEALTH_COLOR = "#f0883e"  # Vibrant orange
DEFAULT_COLOR = "#da3637"  # Deep red
GREEN_COLOR = "#3fb950"    # Rich green

def setup_plot_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        "axes.facecolor": AXES_COLOR,
        "figure.facecolor": BG_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "text.color": "#c9d1d9",
        "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "axes.edgecolor": "#30363d"
    })

def load_lending_club_data():
    logger.info("Loading LendingClub engineered data and predictions...")
    engineered_path = OUTPUT_DIR / "engineered_data.parquet"
    if not engineered_path.exists():
        raise FileNotFoundError(f"Missing LendingClub engineered data: {engineered_path}")
    df_all = pd.read_parquet(engineered_path)
    df_all['issue_d'] = pd.to_datetime(df_all['issue_d'])
    df_all['year'] = df_all['issue_d'].dt.year
    df_all['month'] = df_all['issue_d'].dt.to_period('M').dt.to_timestamp()
    
    # Load model and scaler to generate predictions for all loans
    full_model = joblib.load(MODEL_DIR / "lightgbm.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    
    features_spaces = list(scaler.feature_names_in_)
    features_underscores = [c.replace(' ', '_') for c in features_spaces]
    
    # Get test set (year >= 2018) to calculate optimal F1 threshold
    test_all = df_all[df_all["year"] >= 2018]
    test_df = test_all.sample(min(50000, len(test_all)), random_state=SEED).copy()
    X_test_spaces = test_df[features_spaces].fillna(0)
    X_test_full = X_test_spaces.copy()
    X_test_full.columns = features_underscores
    
    test_probs = full_model.predict_proba(X_test_full)[:, 1]
    y_test = test_df["target"].values
    
    prec, rec, thrs = precision_recall_curve(y_test, test_probs)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    opt_idx = np.argmax(f1_scores)
    opt_thr = thrs[opt_idx]
    logger.info(f"Optimal F1 Decision Threshold calculated on test set: {opt_thr:.5f}")
    
    # Generate predictions on the full dataset
    logger.info("Generating predictions on the full dataset...")
    # Predict in batches to be memory safe
    batch_size = 200000
    all_probs = np.zeros(len(df_all))
    for i in range(0, len(df_all), batch_size):
        batch_df = df_all.iloc[i : i + batch_size]
        X_batch = batch_df[features_spaces].fillna(0)
        X_batch.columns = features_underscores
        all_probs[i : i + batch_size] = full_model.predict_proba(X_batch)[:, 1]
        
    df_all["pred_pd"] = all_probs
    df_all["pred_target"] = (all_probs >= opt_thr).astype(int)
    
    # Identify stealth defaults (target == 1 and predicted PD < opt_thr)
    df_all["stealth_default"] = ((df_all["target"] == 1) & (df_all["pred_target"] == 0)).astype(int)
    df_all["captured_default"] = ((df_all["target"] == 1) & (df_all["pred_target"] == 1)).astype(int)
    
    return df_all, opt_thr, features_underscores

def load_and_preprocess_macro_variables():
    logger.info("Loading and preprocessing macroeconomic data sources...")
    macro_dir = PROJECT_ROOT / "data" / "macro"
    
    # 1. Load CRIS macro stress states
    cris_states_path = OUTPUT_DIR / "phase2_layer3_macro_states.csv"
    if not cris_states_path.exists():
        raise FileNotFoundError(f"Missing CRIS Layer 3 macro states: {cris_states_path}")
    df_cris = pd.read_csv(cris_states_path)
    df_cris['month'] = pd.to_datetime(df_cris['issue_month'])
    df_cris = df_cris[['month', 'macro_stress_score', 'uncertainty_pressure', 'liquidity_disruption']]
    
    # 2. Load FRED variables
    unrate = pd.read_csv(macro_dir / "UNRATE.csv")
    unrate['month'] = pd.to_datetime(unrate['observation_date'])
    unrate['unemployment_rate'] = pd.to_numeric(unrate['UNRATE'], errors='coerce')
    unrate = unrate[['month', 'unemployment_rate']]
    
    fedfunds = pd.read_csv(macro_dir / "FEDFUNDS.csv")
    fedfunds['month'] = pd.to_datetime(fedfunds['observation_date'])
    fedfunds['fed_funds_rate'] = pd.to_numeric(fedfunds['FEDFUNDS'], errors='coerce')
    fedfunds = fedfunds[['month', 'fed_funds_rate']]
    
    cpi = pd.read_csv(macro_dir / "CPIAUCSL.csv")
    cpi['month'] = pd.to_datetime(cpi['observation_date'])
    cpi['CPIAUCSL'] = pd.to_numeric(cpi['CPIAUCSL'], errors='coerce')
    cpi['cpi_inflation'] = cpi['CPIAUCSL'].pct_change(12) * 100  # YoY Inflation %
    cpi = cpi[['month', 'cpi_inflation']]
    
    t10y2y = pd.read_csv(macro_dir / "T10Y2Y.csv")
    t10y2y['Date'] = pd.to_datetime(t10y2y['observation_date'])
    t10y2y['spread_val'] = pd.to_numeric(t10y2y['T10Y2Y'], errors='coerce')
    t10y2y = t10y2y.dropna(subset=['spread_val'])
    # Resample daily to monthly average
    t10y2y_monthly = t10y2y.groupby(t10y2y['Date'].dt.to_period('M'))['spread_val'].mean().reset_index()
    t10y2y_monthly['month'] = t10y2y_monthly['Date'].dt.to_timestamp()
    t10y2y_monthly = t10y2y_monthly.rename(columns={'spread_val': 'treasury_spread'})[['month', 'treasury_spread']]
    
    usrec = pd.read_csv(macro_dir / "USREC.csv")
    usrec['month'] = pd.to_datetime(usrec['observation_date'])
    usrec = usrec.rename(columns={'USREC': 'recession_indicator'})[['month', 'recession_indicator']]
    
    # 3. Load Yahoo Finance market return and volatility
    spy = pd.read_csv(macro_dir / "SPY.csv", skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
    spy['Date'] = pd.to_datetime(spy['Date'])
    spy['Close'] = pd.to_numeric(spy['Close'], errors='coerce')
    spy = spy.dropna(subset=['Close']).sort_values(by='Date')
    spy['Return'] = spy['Close'].pct_change()
    
    spy_monthly_ret = spy.groupby(spy['Date'].dt.to_period('M'))['Return'].apply(lambda x: (1 + x).prod() - 1).reset_index()
    spy_monthly_ret['month'] = spy_monthly_ret['Date'].dt.to_timestamp()
    spy_monthly_ret = spy_monthly_ret.rename(columns={'Return': 'spy_monthly_return'})[['month', 'spy_monthly_return']]
    
    spy_monthly_vol = spy.groupby(spy['Date'].dt.to_period('M'))['Return'].std().reset_index()
    spy_monthly_vol['month'] = spy_monthly_vol['Date'].dt.to_timestamp()
    spy_monthly_vol = spy_monthly_vol.rename(columns={'Return': 'spy_monthly_vol'})[['month', 'spy_monthly_vol']]
    spy_monthly_vol['spy_monthly_vol'] = spy_monthly_vol['spy_monthly_vol'] * np.sqrt(252) # Annualized
    
    vix = pd.read_csv(macro_dir / "VIX.csv", skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
    vix['Date'] = pd.to_datetime(vix['Date'])
    vix['Close'] = pd.to_numeric(vix['Close'], errors='coerce')
    vix = vix.dropna(subset=['Close'])
    
    vix_monthly = vix.groupby(vix['Date'].dt.to_period('M'))['Close'].mean().reset_index()
    vix_monthly['month'] = vix_monthly['Date'].dt.to_timestamp()
    vix_monthly = vix_monthly.rename(columns={'Close': 'vix_monthly_mean'})[['month', 'vix_monthly_mean']]
    
    # Merge all macro tables
    dfs = [df_cris, unrate, fedfunds, cpi, t10y2y_monthly, usrec, spy_monthly_ret, spy_monthly_vol, vix_monthly]
    df_macro = dfs[0]
    for d in dfs[1:]:
        df_macro = pd.merge(df_macro, d, on='month', how='outer')
        
    df_macro = df_macro.sort_values(by='month').reset_index(drop=True)
    # Forward fill/backward fill missing values if any
    df_macro = df_macro.ffill().bfill()
    
    df_macro.to_csv(DATA_DIR / "processed_macro_monthly.csv", index=False)
    logger.info(f"Macro variables loaded: {len(df_macro)} months. Saved to {DATA_DIR / 'processed_macro_monthly.csv'}")
    return df_macro

def aggregate_monthly_loan_metrics(df_all):
    logger.info("Aggregating LendingClub loan performance metrics monthly...")
    monthly_stats = df_all.groupby('month').agg(
        total_loans=('target', 'count'),
        total_defaults=('target', 'sum'),
        stealth_defaults=('stealth_default', 'sum'),
        captured_defaults=('captured_default', 'sum')
    ).reset_index()
    
    monthly_stats['realized_default_rate'] = monthly_stats['total_defaults'] / monthly_stats['total_loans']
    monthly_stats['stealth_default_rate'] = monthly_stats['stealth_defaults'] / monthly_stats['total_defaults']
    
    # Clean stealth default rate if defaults == 0
    monthly_stats.loc[monthly_stats['total_defaults'] == 0, 'stealth_default_rate'] = 0.0
    
    monthly_stats.to_csv(DATA_DIR / "monthly_loan_performance.csv", index=False)
    logger.info(f"Monthly loan performance metrics computed for {len(monthly_stats)} months.")
    return monthly_stats

def run_correlation_analysis(merged_monthly):
    logger.info("Question 1 & 2: Correlation analysis between default rates and macro variables...")
    macro_cols = [
        "macro_stress_score", "uncertainty_pressure", "liquidity_disruption",
        "unemployment_rate", "fed_funds_rate", "cpi_inflation",
        "treasury_spread", "recession_indicator", "spy_monthly_return",
        "spy_monthly_vol", "vix_monthly_mean"
    ]
    
    target_rates = ["realized_default_rate", "stealth_default_rate"]
    
    records = []
    
    for m_col in macro_cols:
        row = {"Macro_Variable": m_col}
        for rate in target_rates:
            # Drop NaNs
            sub_df = merged_monthly[[m_col, rate]].dropna()
            x = sub_df[m_col].values
            y = sub_df[rate].values
            
            p_coef, p_pval = pearsonr(x, y)
            s_coef, s_pval = spearmanr(x, y)
            
            # Bootstrap CI for Pearson correlation
            n_boot = 1000
            boot_pearsons = []
            boot_spearmans = []
            rng = np.random.RandomState(SEED)
            for _ in range(n_boot):
                idx = rng.choice(len(sub_df), size=len(sub_df), replace=True)
                x_b, y_b = x[idx], y[idx]
                if len(np.unique(x_b)) > 1 and len(np.unique(y_b)) > 1:
                    boot_pearsons.append(pearsonr(x_b, y_b)[0])
                    boot_spearmans.append(spearmanr(x_b, y_b)[0])
                    
            p_ci = np.percentile(boot_pearsons, [2.5, 97.5]) if boot_pearsons else [np.nan, np.nan]
            s_ci = np.percentile(boot_spearmans, [2.5, 97.5]) if boot_spearmans else [np.nan, np.nan]
            
            row[f"{rate}_Pearson_Coef"] = p_coef
            row[f"{rate}_Pearson_Pval"] = p_pval
            row[f"{rate}_Pearson_CI_Lower"] = p_ci[0]
            row[f"{rate}_Pearson_CI_Upper"] = p_ci[1]
            
            row[f"{rate}_Spearman_Coef"] = s_coef
            row[f"{rate}_Spearman_Pval"] = s_pval
            row[f"{rate}_Spearman_CI_Lower"] = s_ci[0]
            row[f"{rate}_Spearman_CI_Upper"] = s_ci[1]
            
        records.append(row)
        
    df_corr = pd.DataFrame(records)
    df_corr.to_csv(TABLES_DIR / "macro_correlation_analysis.csv", index=False)
    logger.info(f"Correlation analysis results saved to {TABLES_DIR / 'macro_correlation_analysis.csv'}")
    
    # ── PLOT 2: Correlation Heatmap ──
    plt.figure(figsize=(10, 8))
    corr_matrix_data = merged_monthly[target_rates + macro_cols].corr()
    sns.heatmap(corr_matrix_data, annot=True, cmap="coolwarm_r", fmt=".2f",
                linewidths=0.5, linecolor=GRID_COLOR, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Correlation Matrix: Default Rates vs Macro Indicators", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
    
    # ── PLOT 3: Rolling Correlation ──
    # Calculate 12-month rolling correlations with macro_stress_score
    merged_monthly = merged_monthly.sort_values(by='month').reset_index(drop=True)
    merged_monthly['rolling_pearson_default'] = merged_monthly['realized_default_rate'].rolling(12).corr(merged_monthly['macro_stress_score'])
    merged_monthly['rolling_pearson_stealth'] = merged_monthly['stealth_default_rate'].rolling(12).corr(merged_monthly['macro_stress_score'])
    
    plt.figure(figsize=(10, 5))
    plt.plot(merged_monthly['month'], merged_monthly['rolling_pearson_default'], label="Rolling Corr (Default Rate & Macro Stress)", color=PRIMARY_COLOR, linewidth=2)
    plt.plot(merged_monthly['month'], merged_monthly['rolling_pearson_stealth'], label="Rolling Corr (Stealth Default Rate & Macro Stress)", color=STEALTH_COLOR, linewidth=2, linestyle='--')
    plt.axhline(0, color="#8b949e", linestyle=":", alpha=0.5)
    plt.title("12-Month Rolling Pearson Correlation with CRIS Macro Stress Score", fontsize=12, fontweight="bold")
    plt.xlabel("Month of Loan Issuance")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.legend()
    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rolling_correlation.png", dpi=150)
    plt.close()
    
    return df_corr

def run_predictive_model_comparison(df_all, df_macro, opt_thr, features_underscores):
    logger.info("Question 3: Building and comparing Explanatory Models (Model A vs Model B)...")
    
    # Exclude lender variables
    group_b_patterns = ["int_rate", "term_months", "installment", "grade"]
    features_borrower = [f for f in features_underscores if not any(pat in f for pat in group_b_patterns)]
    
    # Load scaler to get features_spaces
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    features_spaces = list(scaler.feature_names_in_)
    
    # Merge macro variables to df_all
    df_merged = pd.merge(df_all, df_macro, on='month', how='inner')
    
    macro_cols = [
        "unemployment_rate", "fed_funds_rate", "cpi_inflation",
        "treasury_spread", "recession_indicator", "spy_monthly_return",
        "spy_monthly_vol", "vix_monthly_mean", "macro_stress_score",
        "uncertainty_pressure", "liquidity_disruption"
    ]
    
    # Target is stealth default (1 if defaulted and PD < opt_thr, 0 otherwise)
    y_stealth = df_merged["stealth_default"].values
    
    train_idx = df_merged[df_merged["year"] <= 2015].index
    test_idx = df_merged[df_merged["year"] >= 2018].index
    
    # Sample training and test sets to maintain standard sizes
    np.random.seed(SEED)
    train_sample = np.random.choice(train_idx, size=min(100000, len(train_idx)), replace=False)
    test_sample = np.random.choice(test_idx, size=min(50000, len(test_idx)), replace=False)
    
    # Build X_train and X_test using original features_spaces first, then rename columns to features_underscores
    X_train_spaces = df_merged.loc[train_sample, features_spaces].fillna(0)
    X_test_spaces = df_merged.loc[test_sample, features_spaces].fillna(0)
    
    X_train_full = X_train_spaces.copy()
    X_train_full.columns = features_underscores
    X_test_full = X_test_spaces.copy()
    X_test_full.columns = features_underscores
    
    # Filter to borrower features only
    X_train_borrower = X_train_full[features_borrower]
    X_test_borrower = X_test_full[features_borrower]
    
    # Construct macro feature sets by copying borrower sets and appending macro variables
    X_train_macro = X_train_borrower.copy()
    X_test_macro = X_test_borrower.copy()
    for col in macro_cols:
        X_train_macro[col] = df_merged.loc[train_sample, col].values
        X_test_macro[col] = df_merged.loc[test_sample, col].values
    
    y_train_stealth = df_merged.loc[train_sample, "stealth_default"].values
    y_test_stealth = df_merged.loc[test_sample, "stealth_default"].values
    
    logger.info(f"Model comparison train set size: {len(X_train_borrower)} (Stealth default rate: {y_train_stealth.mean():.4%})")
    logger.info(f"Model comparison test set size: {len(X_test_borrower)} (Stealth default rate: {y_test_stealth.mean():.4%})")
    
    # Train Model A (Borrower-only)
    logger.info("Training Model A (Borrower-only)...")
    model_a = LGBMClassifier(
        n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=-1, verbosity=-1
    )
    model_a.fit(X_train_borrower, y_train_stealth)
    probs_a = model_a.predict_proba(X_test_borrower)[:, 1]
    
    # Train Model B (Borrower + Macro)
    logger.info("Training Model B (Borrower + Macro)...")
    model_b = LGBMClassifier(
        n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=-1, verbosity=-1
    )
    model_b.fit(X_train_macro, y_train_stealth)
    probs_b = model_b.predict_proba(X_test_macro)[:, 1]
    
    # Evaluate Model A
    auc_a = roc_auc_score(y_test_stealth, probs_a)
    pr_auc_a = average_precision_score(y_test_stealth, probs_a)
    
    prec_a, rec_a, thrs_a = precision_recall_curve(y_test_stealth, probs_a)
    f1_a = 2 * (prec_a * rec_a) / (prec_a + rec_a + 1e-8)
    opt_idx_a = np.argmax(f1_a)
    opt_thr_a = thrs_a[opt_idx_a] if opt_idx_a < len(thrs_a) else 0.5
    y_pred_a = (probs_a >= opt_thr_a).astype(int)
    tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_test_stealth, y_pred_a).ravel()
    rec_val_a = tp_a / (tp_a + fn_a + 1e-8)
    f1_val_a = f1_a[opt_idx_a]
    
    # Evaluate Model B
    auc_b = roc_auc_score(y_test_stealth, probs_b)
    pr_auc_b = average_precision_score(y_test_stealth, probs_b)
    
    prec_b, rec_b, thrs_b = precision_recall_curve(y_test_stealth, probs_b)
    f1_b = 2 * (prec_b * rec_b) / (prec_b + rec_b + 1e-8)
    opt_idx_b = np.argmax(f1_b)
    opt_thr_b = thrs_b[opt_idx_b] if opt_idx_b < len(thrs_b) else 0.5
    y_pred_b = (probs_b >= opt_thr_b).astype(int)
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test_stealth, y_pred_b).ravel()
    rec_val_b = tp_b / (tp_b + fn_b + 1e-8)
    f1_val_b = f1_b[opt_idx_b]
    
    # Bootstrap CIs & Significance tests
    logger.info("Running bootstrap validation (50 trials) for Model A vs Model B...")
    n_boot = 50
    boot_diffs = []
    boot_aucs_a = []
    boot_aucs_b = []
    
    rng = np.random.RandomState(SEED)
    for _ in range(n_boot):
        idx = rng.choice(len(y_test_stealth), size=len(y_test_stealth), replace=True)
        y_b = y_test_stealth[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_a_b = roc_auc_score(y_b, probs_a[idx])
        auc_b_b = roc_auc_score(y_b, probs_b[idx])
        boot_diffs.append(auc_b_b - auc_a_b)
        boot_aucs_a.append(auc_a_b)
        boot_aucs_b.append(auc_b_b)
        
    diff_ci = np.percentile(boot_diffs, [2.5, 97.5])
    ci_a = np.percentile(boot_aucs_a, [2.5, 97.5])
    ci_b = np.percentile(boot_aucs_b, [2.5, 97.5])
    
    # Calculate a p-value for the difference (fraction of bootstrap trials where difference <= 0)
    p_val_diff = np.mean(np.array(boot_diffs) <= 0)
    
    logger.info(f"Model A (Borrower-only) AUC: {auc_a:.5f} (95% CI: [{ci_a[0]:.5f}, {ci_a[1]:.5f}])")
    logger.info(f"Model B (Borrower + Macro) AUC: {auc_b:.5f} (95% CI: [{ci_b[0]:.5f}, {ci_b[1]:.5f}])")
    logger.info(f"AUC Difference (Model B - Model A): {auc_b - auc_a:.5f} (95% CI: [{diff_ci[0]:.5f}, {diff_ci[1]:.5f}], p-value: {p_val_diff:.4f})")
    
    comparison_df = pd.DataFrame({
        "Model": ["Model A (Borrower-Only)", "Model B (Borrower + Macro)", "Difference (B - A)"],
        "ROC-AUC": [auc_a, auc_b, auc_b - auc_a],
        "PR-AUC": [pr_auc_a, pr_auc_b, pr_auc_b - pr_auc_a],
        "Recall": [rec_val_a, rec_val_b, rec_val_b - rec_val_a],
        "F1_Score": [f1_val_a, f1_val_b, f1_val_b - f1_val_a],
        "ROC_AUC_CI_Lower": [ci_a[0], ci_b[0], diff_ci[0]],
        "ROC_AUC_CI_Upper": [ci_a[1], ci_b[1], diff_ci[1]],
        "P_Value": [np.nan, np.nan, p_val_diff]
    })
    
    comparison_df.to_csv(TABLES_DIR / "model_comparison_results.csv", index=False)
    
    # ── PLOT 5: ROC & PR Curves ──
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curves
    fpr_a, tpr_a, _ = roc_curve(y_test_stealth, probs_a)
    fpr_b, tpr_b, _ = roc_curve(y_test_stealth, probs_b)
    
    axs[0].plot(fpr_a, tpr_a, label=f"Model A (Borrower-only) AUC = {auc_a:.4f}", color=PRIMARY_COLOR, linewidth=2)
    axs[0].plot(fpr_b, tpr_b, label=f"Model B (Borrower + Macro) AUC = {auc_b:.4f}", color=STEALTH_COLOR, linewidth=2, linestyle='--')
    axs[0].plot([0, 1], [0, 1], color="#8b949e", linestyle=":", alpha=0.5)
    axs[0].set_title("Stealth Classifier: ROC Curves Comparison", fontsize=11, fontweight="bold")
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].legend(loc="lower right")
    axs[0].grid(True, alpha=0.15)
    
    # PR Curves
    p_curve_a, r_curve_a, _ = precision_recall_curve(y_test_stealth, probs_a)
    p_curve_b, r_curve_b, _ = precision_recall_curve(y_test_stealth, probs_b)
    
    axs[1].plot(r_curve_a, p_curve_a, label=f"Model A PR-AUC = {pr_auc_a:.4f}", color=PRIMARY_COLOR, linewidth=2)
    axs[1].plot(r_curve_b, p_curve_b, label=f"Model B PR-AUC = {pr_auc_b:.4f}", color=STEALTH_COLOR, linewidth=2, linestyle='--')
    axs[1].set_title("Stealth Classifier: Precision-Recall Curves", fontsize=11, fontweight="bold")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].legend(loc="lower left")
    axs[1].grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_curves.png", dpi=150)
    plt.close()
    
    return comparison_df

def run_regime_analysis(merged_monthly):
    logger.info("Question 4: Regime analysis of credit performance...")
    # Calculate thresholds for regimes based on macro_stress_score (full history)
    stress = merged_monthly['macro_stress_score'].values
    q25 = np.percentile(stress, 25)
    q75 = np.percentile(stress, 75)
    
    logger.info(f"Macro Stress Score regime cutoffs: Low (< {q25:.5f}), Medium ({q25:.5f} - {q75:.5f}), High (> {q75:.5f})")
    
    merged_monthly['regime'] = 'Medium'
    merged_monthly.loc[merged_monthly['macro_stress_score'] < q25, 'regime'] = 'Low'
    merged_monthly.loc[merged_monthly['macro_stress_score'] >= q75, 'regime'] = 'High'
    
    # Compute stats per regime
    regime_stats = merged_monthly.groupby('regime').agg(
        num_months=('month', 'count'),
        total_loans=('total_loans', 'sum'),
        total_defaults=('total_defaults', 'sum'),
        stealth_defaults=('stealth_defaults', 'sum'),
        macro_stress_mean=('macro_stress_score', 'mean'),
        unemployment_mean=('unemployment_rate', 'mean')
    ).reset_index()
    
    # Calculate rates over the pooled regime populations
    regime_stats['realized_default_rate'] = regime_stats['total_defaults'] / regime_stats['total_loans']
    regime_stats['stealth_default_rate'] = regime_stats['stealth_defaults'] / regime_stats['total_defaults']
    
    # Sort for plotting: Low -> Medium -> High
    regime_stats['regime'] = pd.Categorical(regime_stats['regime'], categories=['Low', 'Medium', 'High'], ordered=True)
    regime_stats = regime_stats.sort_values(by='regime').reset_index(drop=True)
    regime_stats.to_csv(TABLES_DIR / "stress_regime_performance.csv", index=False)
    
    # Run statistical tests (Mann-Whitney U) between monthly default rates in Low vs High
    low_months = merged_monthly[merged_monthly['regime'] == 'Low']
    high_months = merged_monthly[merged_monthly['regime'] == 'High']
    
    u_def, p_def = mannwhitneyu(low_months['realized_default_rate'], high_months['realized_default_rate'])
    u_sth, p_sth = mannwhitneyu(low_months['stealth_default_rate'], high_months['stealth_default_rate'])
    
    test_records = pd.DataFrame({
        "Performance_Metric": ["Realized Default Rate", "Stealth Default Rate"],
        "Low_Stress_Regime_Mean": [low_months['realized_default_rate'].mean(), low_months['stealth_default_rate'].mean()],
        "High_Stress_Regime_Mean": [high_months['realized_default_rate'].mean(), high_months['stealth_default_rate'].mean()],
        "Mann_Whitney_U_Stat": [u_def, u_sth],
        "MW_P_Value": [p_def, p_sth],
        "Significant_Difference_05": [p_def < 0.05, p_sth < 0.05]
    })
    test_records.to_csv(TABLES_DIR / "regime_significance_tests.csv", index=False)
    logger.info("Regime significance tests completed and saved.")
    
    # ── PLOT 6: Regime Analysis Bar Chart ──
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Default Rate
    axs[0].bar(regime_stats['regime'].astype(str), regime_stats['realized_default_rate'] * 100, color=[PRIMARY_COLOR, GREEN_COLOR, DEFAULT_COLOR], edgecolor=GRID_COLOR)
    axs[0].set_title("Realized Default Rate by Macro Stress Regime", fontsize=11, fontweight="bold")
    axs[0].set_ylabel("Pooled Realized Default Rate (%)")
    axs[0].grid(axis="y", alpha=0.15)
    
    # Stealth Rate
    axs[1].bar(regime_stats['regime'].astype(str), regime_stats['stealth_default_rate'] * 100, color=[PRIMARY_COLOR, GREEN_COLOR, STEALTH_COLOR], edgecolor=GRID_COLOR)
    axs[1].set_title("Stealth Default Rate by Macro Stress Regime", fontsize=11, fontweight="bold")
    axs[1].set_ylabel("Stealth Default Rate (FN / Defaults) (%)")
    axs[1].grid(axis="y", alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "regime_performance_comparison.png", dpi=150)
    plt.close()
    
    return regime_stats

def run_economic_shock_sensitivity(merged_monthly):
    logger.info("Question 5: Economic shock sensitivity profiling...")
    
    # Environment 1: Great Financial Crisis (GFC) Aftermath (2008-01-01 to 2010-12-31)
    gfc_df = merged_monthly[(merged_monthly['month'] >= "2008-01-01") & (merged_monthly['month'] <= "2010-12-31")].sort_values(by='month')
    gfc_df.to_csv(TABLES_DIR / "gfc_shock_sensitivity.csv", index=False)
    
    # Environment 2: Fed Interest Rate Hikings (2015-12-01 to 2018-12-01)
    hike_df = merged_monthly[(merged_monthly['month'] >= "2015-12-01") & (merged_monthly['month'] <= "2018-12-01")].sort_values(by='month')
    hike_df.to_csv(TABLES_DIR / "hiking_shock_sensitivity.csv", index=False)
    
    # ── PLOT 7: GFC Aftermath Sensitivity ──
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    ax1.plot(gfc_df['month'], gfc_df['realized_default_rate'] * 100, label="Realized Default Rate", color=DEFAULT_COLOR, linewidth=2)
    ax1.plot(gfc_df['month'], gfc_df['stealth_default_rate'] * 100, label="Stealth Default Rate", color=STEALTH_COLOR, linewidth=2, linestyle='--')
    ax2.plot(gfc_df['month'], gfc_df['unemployment_rate'], label="Unemployment Rate (Right Axis)", color=PRIMARY_COLOR, linewidth=2, alpha=0.7)
    
    ax1.set_title("GFC Aftermath Sensitivity Profiling (2008–2010)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Month of Loan Issuance")
    ax1.set_ylabel("Credit Defaults (%)")
    ax2.set_ylabel("Unemployment Rate (%)")
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gfc_shock_sensitivity.png", dpi=150)
    plt.close()
    
    # ── PLOT 8: Interest Rate Hikings Sensitivity ──
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    ax1.plot(hike_df['month'], hike_df['realized_default_rate'] * 100, label="Realized Default Rate", color=DEFAULT_COLOR, linewidth=2)
    ax1.plot(hike_df['month'], hike_df['stealth_default_rate'] * 100, label="Stealth Default Rate", color=STEALTH_COLOR, linewidth=2, linestyle='--')
    ax2.plot(hike_df['month'], hike_df['fed_funds_rate'], label="Federal Funds Rate (Right Axis)", color=PRIMARY_COLOR, linewidth=2, alpha=0.7)
    
    ax1.set_title("Interest Rate Hiking Cycle Sensitivity (2015–2018)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Month of Loan Issuance")
    ax1.set_ylabel("Credit Defaults (%)")
    ax2.set_ylabel("Federal Funds Rate (%)")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hiking_shock_sensitivity.png", dpi=150)
    plt.close()
    
    logger.info("Shock sensitivity profiling complete.")

def generate_remaining_figures(merged_monthly):
    logger.info("Generating remaining visualizations...")
    
    # ── PLOT 1: Realized Default Rate vs Macro Stress (Time Series) ──
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    ax1.plot(merged_monthly['month'], merged_monthly['realized_default_rate'] * 100, label="Realized Default Rate", color=DEFAULT_COLOR, linewidth=2)
    ax2.plot(merged_monthly['month'], merged_monthly['macro_stress_score'], label="CRIS Macro Stress Score (Right Axis)", color=PRIMARY_COLOR, linewidth=2, alpha=0.7)
    
    ax1.set_title("Monthly Realized Default Rate vs CRIS Macro Stress Score (2007–2018)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Month of Loan Issuance")
    ax1.set_ylabel("Realized Default Rate (%)")
    ax2.set_ylabel("Macro Stress Score")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "default_rate_vs_macro_stress.png", dpi=150)
    plt.close()
    
    # ── PLOT 4: Stealth Default Rate vs Macro Stress (Scatter / Time Series) ──
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    ax1.plot(merged_monthly['month'], merged_monthly['stealth_default_rate'] * 100, label="Stealth Default Rate (FN / Defaults)", color=STEALTH_COLOR, linewidth=2)
    ax2.plot(merged_monthly['month'], merged_monthly['macro_stress_score'], label="CRIS Macro Stress Score (Right Axis)", color=PRIMARY_COLOR, linewidth=2, alpha=0.7)
    
    ax1.set_title("Stealth Default Rate (FN / Defaults) vs CRIS Macro Stress Score (2007–2018)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Month of Loan Issuance")
    ax1.set_ylabel("Stealth Default Rate (%)")
    ax2.set_ylabel("Macro Stress Score")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stealth_rate_vs_macro_stress.png", dpi=150)
    plt.close()

def main():
    setup_plot_style()
    
    # Load and clean LC loan dataset and get PD predictions
    df_all, opt_thr, features_underscores = load_lending_club_data()
    
    # Load and align macroeconomic data sources
    df_macro = load_and_preprocess_macro_variables()
    
    # Aggregate loan performance metrics monthly
    monthly_stats = aggregate_monthly_loan_metrics(df_all)
    
    # Merge monthly default rates with monthly macro data
    merged_monthly = pd.merge(monthly_stats, df_macro, on='month', how='inner').sort_values(by='month').reset_index(drop=True)
    merged_monthly.to_csv(DATA_DIR / "merged_monthly_data.csv", index=False)
    
    # 1. Do correlation analysis (Pearson, Spearman, rolling correlation, bootstrap CIs)
    run_correlation_analysis(merged_monthly)
    
    # 2. Compare Model A vs Model B (Explanatory stealth classifier)
    run_predictive_model_comparison(df_all, df_macro, opt_thr, features_underscores)
    
    # 3. Perform Regime Analysis (Low, Medium, High Stress)
    run_regime_analysis(merged_monthly)
    
    # 4. Profile Economic Shock Sensitivity (GFC aftermath, Fed hiking cycles)
    run_economic_shock_sensitivity(merged_monthly)
    
    # 5. Generate remaining required visualizations (total of 8)
    generate_remaining_figures(merged_monthly)
    
    logger.info("Macro Default Analysis complete. All figures, tables, and data are saved.")

if __name__ == "__main__":
    main()
