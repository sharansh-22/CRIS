# Data Quality Report - Credit Risk Phase 1

**Total Records Processed:** 1345350
**Target Distribution:**
- Good Loans (0): 80.04%
- Bad Loans (1): 19.96%

## Removed Features (Leakage Prevention)
The following features were removed because they contain post-issuance information or are identifiers:

- all_util
- annual_inc_joint
- collection_recovery_fee
- debt_settlement_flag
- debt_settlement_flag_date
- deferral_term
- desc
- dti_joint
- funded_amnt
- funded_amnt_inv
- hardship_amount
- hardship_dpd
- hardship_end_date
- hardship_flag
- hardship_last_payment_amount
- hardship_length
- hardship_loan_status
- hardship_payoff_balance_amount
- hardship_reason
- hardship_start_date
- hardship_status
- hardship_type
- id
- il_util
- inq_fi
- inq_last_12m
- last_credit_pull_d
- last_fico_range_high
- last_fico_range_low
- last_pymnt_amnt
- last_pymnt_d
- loan_status
- max_bal_bc
- member_id
- mths_since_last_delinq
- mths_since_last_major_derog
- mths_since_last_record
- mths_since_rcnt_il
- mths_since_recent_bc_dlq
- mths_since_recent_revol_delinq
- next_pymnt_d
- open_acc_6m
- open_act_il
- open_il_12m
- open_il_24m
- open_rv_12m
- open_rv_24m
- orig_projected_additional_accrued_interest
- out_prncp
- out_prncp_inv
- payment_plan_start_date
- pymnt_plan
- recoveries
- revol_bal_joint
- sec_app_chargeoff_within_12_mths
- sec_app_collections_12_mths_ex_med
- sec_app_earliest_cr_line
- sec_app_fico_range_high
- sec_app_fico_range_low
- sec_app_inq_last_6mths
- sec_app_mort_acc
- sec_app_mths_since_last_major_derog
- sec_app_num_rev_accts
- sec_app_open_acc
- sec_app_open_act_il
- sec_app_revol_util
- settlement_amount
- settlement_date
- settlement_percentage
- settlement_status
- settlement_term
- title
- total_bal_il
- total_cu_tl
- total_pymnt
- total_pymnt_inv
- total_rec_int
- total_rec_late_fee
- total_rec_prncp
- url
- verification_status_joint
- zip_code

## Missing Value Strategy
- Columns with > 50% missing values were dropped.
- Remaining numerical columns were imputed with median.
- Remaining categorical columns were imputed with 'Unknown'.
