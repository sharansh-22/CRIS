"""
borrower_segmentation.py — Hidden Segment Discovery using PCA, KMeans, and DBSCAN.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# Configure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from configs.credit_config import SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("CRA.BorrowerSegmentation")

# Setup output folders
AN_DIR = PROJECT_ROOT / "systems" / "credit_risk" / "cr_analysis"
TABLES_DIR = AN_DIR / "outputs" / "tables"
FIGURES_DIR = AN_DIR / "outputs" / "figures"
DATA_DIR = AN_DIR / "outputs" / "data"

def run_segmentation():
    logger.info("Loading labeled stealth test dataset...")
    labeled_path = DATA_DIR / "labeled_stealth_test_df.parquet"
    if not labeled_path.exists():
        raise FileNotFoundError(f"Missing labeled stealth dataset: {labeled_path}")
    test_df = pd.read_parquet(labeled_path)
    
    # Exclude lender variables
    group_b_patterns = ["int_rate", "term_months", "installment", "grade"]
    
    # Filter features (borrower-only, numeric/boolean)
    features = [c for c in test_df.select_dtypes(include=[np.number, bool]).columns 
                if c not in ["target", "year", "pred_pd", "pred_target", "policy_code"] 
                and not any(pat in c for pat in group_b_patterns)]
    
    logger.info(f"Using {len(features)} borrower-only features for segmentation.")
    
    # Get Stealth Defaulters (Group C)
    stealth_df = test_df[test_df["archetype"] == "Group C: Stealth Defaulters"].copy()
    captured_df = test_df[test_df["archetype"] == "Group B: Captured Defaulters"].copy()
    
    if len(stealth_df) == 0:
        logger.error("No stealth defaulters found in dataset.")
        return
        
    X_stealth = stealth_df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_stealth)
    
    # ── STEP 1: PCA DIMENSIONALITY REDUCTION ──
    logger.info("Running PCA dimensionality reduction...")
    pca = PCA(n_components=5, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_var = pca.explained_variance_ratio_
    logger.info(f"PCA Explained Variance Ratio: {explained_var}")
    
    # Save PCA results to stealth_df for visualization
    stealth_df["PC1"] = X_pca[:, 0]
    stealth_df["PC2"] = X_pca[:, 1]
    
    # ── STEP 2: KMEANS CLUSTERING ──
    # We choose k=3 clusters for the hidden borrower segments (FICO/debt/income splits)
    best_k = 3
    logger.info(f"Running KMeans with k={best_k} clusters...")
    kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
    stealth_df["Cluster"] = kmeans.fit_predict(X_scaled)
    
    # ── STEP 3: DBSCAN FOR ANOMALY / DENSE SUBGROUP DETECTION ──
    logger.info("Running DBSCAN to identify core dense regions...")
    dbscan = DBSCAN(eps=3.0, min_samples=10)
    stealth_df["DBSCAN_Label"] = dbscan.fit_predict(X_scaled)
    noise_ratio = (stealth_df["DBSCAN_Label"] == -1).mean()
    logger.info(f"DBSCAN Noise Ratio (Unclustered outliers): {noise_ratio:.2%}")
    
    # ── STEP 4: CLUSTER PROFILING ──
    logger.info("Profiling KMeans clusters...")
    profile_features = [
        "fico_range_low", "dti", "annual_inc", "revol_util", "loan_amnt", 
        "cr_hist_years", "delinq_2yrs", "tot_hi_cred_lim"
    ]
    
    cluster_profiles = []
    for c in range(best_k):
        c_df = stealth_df[stealth_df["Cluster"] == c]
        profile = {
            "Cluster": f"Cluster {c}",
            "Count": len(c_df),
            "Share": len(c_df) / len(stealth_df)
        }
        for f in profile_features:
            if f in stealth_df.columns:
                profile[f"{f}_Mean"] = c_df[f].mean()
                profile[f"{f}_Median"] = c_df[f].median()
        cluster_profiles.append(profile)
        
    df_profiles = pd.DataFrame(cluster_profiles)
    df_profiles.to_csv(TABLES_DIR / "stealth_cluster_profiles.csv", index=False)
    logger.info(f"Cluster profiles saved to {TABLES_DIR / 'stealth_cluster_profiles.csv'}")
    
    # Save the updated stealth dataset with cluster labels
    stealth_df.to_parquet(DATA_DIR / "clustered_stealth_df.parquet", index=False)
    
    # ── STEP 5: VISUALIZATIONS ──
    logger.info("Generating clustering charts...")
    
    # Plot 1: KMeans clusters in PCA space
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=stealth_df, x="PC1", y="PC2", hue="Cluster", 
        palette="Set1", alpha=0.6, edgecolor=None
    )
    plt.title(f"PCA Projection of Stealth Defaulters (KMeans k={best_k})", fontsize=12, fontweight="bold")
    plt.xlabel(f"PC1 ({explained_var[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({explained_var[1]:.1%} variance)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stealth_pca_clusters.png", dpi=150)
    plt.close()
    
    # Plot 2: Stealth vs Captured Defaulters in PCA space
    # Scale combined for PCA comparison
    combined_defaults = pd.concat([captured_df, stealth_df]).copy()
    X_comb = scaler.fit_transform(combined_defaults[features].fillna(0))
    pca_comb = PCA(n_components=2, random_state=SEED)
    X_pca_comb = pca_comb.fit_transform(X_comb)
    combined_defaults["PC1"] = X_pca_comb[:, 0]
    combined_defaults["PC2"] = X_pca_comb[:, 1]
    
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=combined_defaults, x="PC1", y="PC2", hue="archetype",
        palette=["#da3637", "#f0883e"], alpha=0.5, edgecolor=None
    )
    plt.title("PCA Projection: Captured Defaulters vs Stealth Defaulters", fontsize=12, fontweight="bold")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stealth_vs_captured_pca.png", dpi=150)
    plt.close()
    
    logger.info("Clustering plots generated.")

if __name__ == "__main__":
    run_segmentation()
