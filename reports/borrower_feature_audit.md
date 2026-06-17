# Credit Risk Feature Audit Report — Phase 2C

**Prepared by**: Credit Risk Research & Model Risk Validation (MRV) Teams  
**Date**: June 17, 2026  
**Status**: Complete  

---

## 1. Introduction

The objective of this Feature Audit is to split all features used in the LendingClub credit risk model into two distinct categories:
1.  **Group A (Borrower-Intrinsic Characteristics)**: Variables representing the borrower's independent financial history, capacity, and credit character prior to or at application time.
2.  **Group B (Lender-Assigned / Underwriting Features)**: Variables representing the terms, pricing, and structural decisions made by the underwriting institution (LendingClub) during the credit evaluation process.

This classification allows us to train a **Borrower-Only Model** and isolate the true predictive power of borrower behavior from the influence of lender underwriting decisions.

---

## 2. Feature Classification Table

Below is the audited classification of all 171 features entering the LendingClub modeling pipeline:

| Feature Name | Group | Rationale for Classification |
| :--- | :---: | :--- |
| **fico_range_low** / **fico_range_high** | **Group A** | Credit bureau scores representing the borrower's historical credit character. |
| **annual_inc** | **Group A** | Self-reported borrower annual income representing repayment capacity. |
| **dti** | **Group A** | Debt-to-Income ratio representing borrower leverage prior to loan funding. |
| **revol_bal** / **revol_util** | **Group A** | Revolving balance and utilization representing capital reserve and credit card dependency. |
| **cr_hist_years** | **Group A** | Length of credit history in years, derived from borrower's first credit line. |
| **delinq_2yrs** / **pub_rec** | **Group A** | Counts of delinquencies and public records, representing historical repayment character. |
| **open_acc** / **total_acc** | **Group A** | Credit lines counts representing credit capacity and history depth. |
| **emp_length_num** | **Group A** | Length of employment in years, representing income stability. |
| **home_ownership_\*** | **Group A** | Dummy variables representing borrower housing tenure (Mortgage, Own, Rent). |
| **purpose_\*** | **Group A** | Dummy variables representing the borrower's stated reason for seeking credit. |
| **addr_state_\*** | **Group A** | Dummy variables representing borrower state of residence. |
| **verification_status_\*** | **Group A** | Dummy variables representing whether borrower income was verified by the platform. |
| **application_type_Joint_App** | **Group A** | Indicates whether the borrower applied individually or with a co-borrower. |
| **loan_amnt** | **Group A** | The loan size requested by the borrower at the time of application. |
| **tot_hi_cred_lim** / **total_bc_limit** | **Group A** | Limits representing credit capacity granted to the borrower by other lenders. |
| **avg_cur_bal** / **tot_cur_bal** | **Group A** | Current outstanding balances across all accounts. |
| **int_rate** | **Group B** | The interest rate of the loan, set by LendingClub's risk-based pricing model. |
| **term_months** | **Group B** | The contract term (36 or 60 months) approved during underwriting. |
| **installment** | **Group B** | The monthly payment, which is mathematically derived from `loan_amnt`, `int_rate`, and `term_months`. |
| **grade_B** through **grade_G** | **Group B** | Discretized risk grades assigned by LendingClub. |
| **sub_grade_A2** through **sub_grade_G5** | **Group B** | Granular sub-grades assigned by LendingClub. |
| **initial_list_status_w** | **Group B** | Platform listing status (Whole or Fractional), an underwriting flag. |
| **disbursement_method_DirectPay** | **Group B** | Underwriting payout method (DirectPay vs. Cash). |

---

## 3. Audit Insights & Governance

1.  **The Collinearity of FICO and Interest Rate**: Under risk-based pricing, `int_rate` is a direct function of `fico_range_low` and `sub_grade`. Thus, `int_rate` acts as a synthetic summary of the borrower's risk. Shuffling FICO in the full model has a small impact because `int_rate` remains in the model and acts as a proxy for the FICO signal.
2.  **The Installment Dependency**: `installment` is classified as Group B because its formula ($Installment = f(Amount, Rate, Term)$) contains two underwriting variables. Retaining installment would allow the model to back-solve for the interest rate, leading to feature leakage.
3.  **Requested Loan Amount vs. Funded Loan Amount**: The requested `loan_amnt` is a borrower characteristic (Group A), as it is defined by the borrower's credit demand. Any subsequent modification of funded amount by the lender would be Group B, but since funded amount is excluded due to forward-looking leakage, only the requested amount enters the pipeline.
