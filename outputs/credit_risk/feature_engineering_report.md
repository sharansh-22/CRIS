# Feature Engineering Report

## Transformations
- **emp_length**: Converted from string (e.g., '10+ years') to numeric (0-10).
- **earliest_cr_line**: Converted to `cr_hist_years` relative to `issue_d`.
- **term**: Converted to numeric months (36 or 60).
- **Categorical Variables**: One-hot encoded (grade, home_ownership, verification_status, purpose, application_type).
- **emp_title**: Dropped due to high cardinality.

## Final Feature Set
Total Features: 171 (excluding target and issue_d)
Top 10 columns by name:

- acc_now_delinq
- acc_open_past_24mths
- addr_state_AL
- addr_state_AR
- addr_state_AZ
- addr_state_CA
- addr_state_CO
- addr_state_CT
- addr_state_DC
- addr_state_DE
