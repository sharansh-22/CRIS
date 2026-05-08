"""
CRIS Governance Overlay Configuration (Refactored)
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "governance"

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Governance Constraints
MONTHLY_REVIEW_CAPACITY = 8000
PORTFOLIO_CAP_DEFENSIVE = 0.40
RESERVE_FLAG_THRESHOLD = 0.35
