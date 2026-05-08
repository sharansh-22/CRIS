# Explainability Report - LightGBM

## Global Feature Importance
The SHAP summary plot (saved as `shap_summary_lightgbm.png`) shows the top features contributing to default risk.

### Key Drivers of Default:
- **dti**: Higher debt-to-income ratio typically increases risk.
- **int_rate**: Higher interest rates are strongly correlated with default (risk premium).
- **term**: 60-month loans are generally riskier than 36-month loans.
- **revol_util**: High utilization of revolving credit is a key indicator of financial stress.
- **annual_inc**: Lower annual income increases the probability of default.
