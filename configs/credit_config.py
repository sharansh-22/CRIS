"""
CRIS Credit Risk Configuration (Refactored)
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "credit_risk"
OUTPUT_DIR = BASE_DIR / "outputs" / "credit_risk"
MODEL_DIR = BASE_DIR / "systems" / "credit_risk" / "models" / "saved_models"

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset
RAW_DATA_PATH = DATA_DIR / "accepted_2007_to_2018Q4.csv.gz"

# Features to drop (Leakage or unnecessary)
LEAKAGE_COLS = [
    "id", "member_id", "url", "desc", "funded_amnt", "funded_amnt_inv",
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low", "hardship_flag",
    "hardship_type", "hardship_reason", "hardship_status", "deferral_term",
    "hardship_amount", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length", "hardship_dpd",
    "hardship_loan_status", "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage",
    "settlement_term", "pymnt_plan", "title", "zip_code"
]

# Target mapping
TARGET_COL = "loan_status"
GOOD_STATUS = ["Fully Paid"]
BAD_STATUS = ["Charged Off", "Default"]

# Random seed
SEED = 42

# Research Hierarchy Configuration
# Primary research dataset is LendingClub.
# GMC and American Bankruptcy are isolated from the active research pipeline by default.
RUN_EXTERNAL_VALIDATION = False

