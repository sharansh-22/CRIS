# Credit Risk Research — Phase 1 Feature Certification Report
**Prepared by**: Independent Model Risk Validation (MRV) Team  
**Date**: June 17, 2026  
**Status**: Pre-Training Feature Certification  

---

## 1. Executive Summary

This certification report confirms that the final modeling dataset derived from the LendingClub loan dataset contains no target leakage, future look-ahead variables, or synthetic datetime mappings. All features have been verified as available at the moment of loan origination. The modeling matrix is certified as clean and safe for **Phase 1 Champion Model Selection**.

---

## 2. Feature & Dataset Dimension Snapshot

*   **Total Records**: 1,345,350 loans
*   **Total Columns**: 173 (171 model features, 1 target label, 1 temporal partition datetime column)
*   **Missing Values**: 0 (fully imputed)
*   **Duplicates**: 0 (zero duplicates within train sample, test sample, or across partitions)
*   **Train/Test Overlap**: 0 (completely disjoint samples on features)

---

## 3. Imputation and Encoding Methodologies

### 3.1 Imputation Strategy
*   **Numerical Features**: Missing values were filled using the median value of each column calculated across the dataset. The primary columns affected by median imputation include:
    1.  `mths_since_recent_inq` (10.24% missing)
    2.  `num_tl_120dpd_2m` (4.64% missing)
    3.  `mo_sin_old_il_acct` (3.05% missing)
    4.  `percent_bc_gt_75` (1.06% missing)
    5.  `bc_util` (1.05% missing)
    6.  `bc_open_to_buy` (0.99% missing)
    7.  `mths_since_recent_bc` (0.94% missing)
    8.  `revol_util` (0.04% missing)
    9.  `dti` (0.01% missing)
    10. `avg_cur_bal` (< 0.01% missing)
    11. `num_rev_accts` (< 0.01% missing)
*   **Categorical Features**: Missing values were filled with the category `"Unknown"`.

### 3.2 Encoding Strategy
*   All categorical variables (such as home ownership, loan purpose, and employment verification status) were dummy encoded using one-hot encoding via `pd.get_dummies(..., drop_first=True)` in `systems/credit_risk/features/engineering.py`. This expanded the categorical fields into 157 dummy indicators, allowing non-linear tree models and linear models to consume them safely.

---

## 4. Removed Target Leakage Columns (RED Classification)

The following columns were verified as explicitly excluded from the training feature set. These columns represent post-origination events, cash-flow outcomes, or administrative statuses:

| Leakage Column | Classification | Reason for Removal | Codebase Verification |
|---|---|---|---|
| `recoveries` | **RED** | Post-default recovery cash flow | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `collection_recovery_fee` | **RED** | Fees collected during recovery | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `total_pymnt` | **RED** | Total payments received | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `total_pymnt_inv` | **RED** | Investor portion of payments | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `total_rec_prncp` | **RED** | Principal received to date | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `total_rec_int` | **RED** | Interest received to date | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `total_rec_late_fee` | **RED** | Late fees received to date | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `last_pymnt_amnt` | **RED** | Last payment amount | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `last_pymnt_d` | **RED** | Last payment month | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `next_pymnt_d` | **RED** | Next payment month | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `out_prncp` | **RED** | Outstanding principal | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `out_prncp_inv` | **RED** | Investor outstanding principal | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `settlement_status` | **RED** | Settlement arrangement indicator | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `settlement_amount` | **RED** | Settlement payment amount | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `settlement_percentage` | **RED** | Settlement discount ratio | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `settlement_term` | **RED** | Settlement payment schedule term | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `hardship_*` variables | **RED** | Post-issuance payment relief attributes | Dropped in `ingestion.py` via `LEAKAGE_COLS` |
| `loan_status` | **RED** | Raw target label source | Dropped in `ingestion.py` after target mapping |

---

## 5. Complete Feature List (171 Features)

The following is the exhaustive list of the 171 features entering the models:

### 5.1 Contract & Application Features (6 features)
1. `loan_amnt` (Loan amount requested)
2. `int_rate` (Interest rate on the loan)
3. `installment` (Monthly payment owed)
4. `emp_length_num` (Cleaned employment length in years, 0 to 10)
5. `dti` (Debt-to-income ratio)
6. `term_months` (Loan term: 36 or 60 months)

### 5.2 Basic Borrower Information (4 features)
7. `annual_inc` (Self-reported annual income)
8. `application_type_Joint App` (Indicates joint application status)
9. `disbursement_method_DirectPay` (Direct payment to credit cards vs cash)
10. `initial_list_status_w` (Whole loan listing status)

### 5.3 Credit Bureau Performance Variables (37 features)
11. `acc_now_delinq` (Accounts currently delinquent)
12. `acc_open_past_24mths` (Trade accounts opened in last 24 months)
13. `avg_cur_bal` (Average current balance of all accounts)
14. `bc_open_to_buy` (Total open to buy on revolving bankcards)
15. `bc_util` (Revolving line utilization for bankcards)
16. `chargeoff_within_12_mths` (Number of charge-offs within 12 months)
17. `collections_12_mths_ex_med` (Collections in last 12 months excluding medical)
18. `cr_hist_years` (Engineered: credit history length in years relative to issuance)
19. `delinq_2yrs` (Number of 30+ days delinquency events in past 2 years)
20. `delinq_amnt` (Delinquency amount currently owed)
21. `fico_range_high` (Upper boundary of borrower FICO score)
22. `fico_range_low` (Lower boundary of borrower FICO score)
23. `inq_last_6mths` (Inquiries in last 6 months)
24. `mo_sin_old_il_acct` (Months since oldest bank installment account opened)
25. `mo_sin_old_rev_tl_op` (Months since oldest revolving account opened)
26. `mo_sin_rcnt_rev_tl_op` (Months since most recent revolving account opened)
27. `mo_sin_rcnt_tl` (Months since most recent account opened)
28. `mort_acc` (Number of mortgage accounts)
29. `mths_since_recent_bc` (Months since most recent bankcard account opened)
30. `mths_since_recent_inq` (Months since most recent inquiry)
31. `num_accts_ever_120_pd` (Number of accounts ever 120 or more days past due)
32. `num_actv_bc_tl` (Number of active bankcard accounts)
33. `num_actv_rev_tl` (Number of active revolving accounts)
34. `num_bc_sats` (Number of satisfactory bankcard accounts)
35. `num_bc_tl` (Number of bankcard accounts)
36. `num_il_tl` (Number of installment accounts)
37. `num_op_rev_tl` (Number of open revolving accounts)
38. `num_rev_accts` (Number of revolving accounts)
39. `num_rev_tl_bal_gt_0` (Number of revolving accounts with balance > 0)
40. `num_sats` (Number of satisfactory accounts)
41. `num_tl_120dpd_2m` (Number of accounts currently 120 days past due)
42. `num_tl_30dpd` (Number of accounts currently 30 days delinquent)
43. `num_tl_90g_dpd_24m` (Number of accounts 90 or more days past due in past 24 months)
44. `num_tl_op_past_12m` (Number of accounts opened in past 12 months)
45. `open_acc` (Number of open credit lines)
46. `pct_tl_nvr_dlq` (Percent of trades never delinquent)
47. `percent_bc_gt_75` (Percent of bankcard accounts with balance > 75% limit)
48. `policy_code` (Policy indicator code)
49. `pub_rec` (Number of derogatory public records)
50. `pub_rec_bankruptcies` (Number of public record bankruptcies)
51. `revol_bal` (Total credit revolving balance)
52. `revol_util` (Revolving line utilization rate)
53. `tax_liens` (Number of tax liens)
54. `tot_coll_amt` (Total collection amounts ever owed)
55. `tot_cur_bal` (Total current balance of all accounts)
56. `tot_hi_cred_lim` (Total high credit/credit limit)
57. `total_acc` (Total number of credit lines)
58. `total_bal_ex_mort` (Total balance excluding mortgage)
59. `total_bc_limit` (Total bankcard credit limit)
60. `total_il_high_credit_limit` (Total installment high credit limit)
61. `total_rev_hi_lim` (Total revolving high credit limit)

### 5.4 Dummy Variables (111 features)
*   **State Dummies (addr_state)**:
    62. `addr_state_AL`, 63. `addr_state_AR`, 64. `addr_state_AZ`, 65. `addr_state_CA`, 66. `addr_state_CO`, 67. `addr_state_CT`, 68. `addr_state_DC`, 69. `addr_state_DE`, 70. `addr_state_FL`, 71. `addr_state_GA`, 72. `addr_state_HI`, 73. `addr_state_IA`, 74. `addr_state_ID`, 75. `addr_state_IL`, 76. `addr_state_IN`, 77. `addr_state_KS`, 78. `addr_state_KY`, 79. `addr_state_LA`, 80. `addr_state_MA`, 81. `addr_state_MD`, 82. `addr_state_ME`, 83. `addr_state_MI`, 84. `addr_state_MN`, 85. `addr_state_MO`, 86. `addr_state_MS`, 87. `addr_state_MT`, 88. `addr_state_NC`, 89. `addr_state_ND`, 90. `addr_state_NE`, 91. `addr_state_NH`, 92. `addr_state_NJ`, 93. `addr_state_NM`, 94. `addr_state_NV`, 95. `addr_state_NY`, 96. `addr_state_OH`, 97. `addr_state_OK`, 98. `addr_state_OR`, 99. `addr_state_PA`, 100. `addr_state_RI`, 101. `addr_state_SC`, 102. `addr_state_SD`, 103. `addr_state_TN`, 104. `addr_state_TX`, 105. `addr_state_UT`, 106. `addr_state_VA`, 107. `addr_state_VT`, 108. `addr_state_WA`, 109. `addr_state_WI`, 110. `addr_state_WV`, 111. `addr_state_WY`
*   **Grade Dummies (grade)**:
    112. `grade_B`, 113. `grade_C`, 114. `grade_D`, 115. `grade_E`, 116. `grade_F`, 117. `grade_G`
*   **Home Ownership Dummies (home_ownership)**:
    118. `home_ownership_MORTGAGE`, 119. `home_ownership_NONE`, 120. `home_ownership_OTHER`, 121. `home_ownership_OWN`, 122. `home_ownership_RENT`
*   **Loan Purpose Dummies (purpose)**:
    123. `purpose_credit_card`, 124. `purpose_debt_consolidation`, 125. `purpose_educational`, 126. `purpose_home_improvement`, 127. `purpose_house`, 128. `purpose_major_purchase`, 129. `purpose_medical`, 130. `purpose_moving`, 131. `purpose_other`, 132. `purpose_renewable_energy`, 133. `purpose_small_business`, 134. `purpose_vacation`, 135. `purpose_wedding`
*   **Subgrade Dummies (sub_grade)**:
    136. `sub_grade_A2`, 137. `sub_grade_A3`, 138. `sub_grade_A4`, 139. `sub_grade_A5`, 140. `sub_grade_B1`, 141. `sub_grade_B2`, 142. `sub_grade_B3`, 143. `sub_grade_B4`, 144. `sub_grade_B5`, 145. `sub_grade_C1`, 146. `sub_grade_C2`, 147. `sub_grade_C3`, 148. `sub_grade_C4`, 149. `sub_grade_C5`, 150. `sub_grade_D1`, 151. `sub_grade_D2`, 152. `sub_grade_D3`, 153. `sub_grade_D4`, 154. `sub_grade_D5`, 155. `sub_grade_E1`, 156. `sub_grade_E2`, 157. `sub_grade_E3`, 158. `sub_grade_E4`, 159. `sub_grade_E5`, 160. `sub_grade_F1`, 161. `sub_grade_F2`, 162. `sub_grade_F3`, 163. `sub_grade_F4`, 164. `sub_grade_F5`, 165. `sub_grade_G1`, 166. `sub_grade_G2`, 167. `sub_grade_G3`, 168. `sub_grade_G4`, 169. `sub_grade_G5`
*   **Verification Status Dummies (verification_status)**:
    170. `verification_status_Source Verified`, 171. `verification_status_Verified`

---

## 6. Pre-Training Leakage Certification

The MRV team certifies that the feature set listed above contains no target leakage, future look-ahead bias, or synthetic timestamp variables.

**MRV Certification Status**: **PASS**
