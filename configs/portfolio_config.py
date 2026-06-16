"""
portfolio_config.py - Institutional Configuration for the CRIS Portfolio Intelligence System.
"""

# Risk Parameter Thresholds
VAR_CONFIDENCE_LEVEL = 0.95
STRESS_TEST_DRAWDOWNS = [-0.10, -0.20, -0.30]  # Standard shock scenarios

# Rolling Window Horizons
REASSESSMENT_WINDOW_DAYS = 252  # 1-year lookback for baseline stats
FORECAST_HORIZONS = [5, 21, 63]  # Week, Month, Quarter

# Portfolio Composition (Placeholder for User Definition)
USER_PORTFOLIO = {
    "tickers": ["SPY", "QQQ", "TLT", "GLD"],
    "weights": [0.4, 0.3, 0.2, 0.1]
}

# Reporting Settings
REPORT_OUTPUT_DIR = "outputs/portfolio_reports"
PLOT_OUTPUT_DIR = "outputs/plots/portfolio"
