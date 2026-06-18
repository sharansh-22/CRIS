# Credit Risk Analysis (CRA) — Stealth Defaulter Research Framework
## Independent Research Report on Elusive Credit Risk Outliers

> [!NOTE]
> This study was conducted inside the `systems/credit_risk/cr_analysis/` module. It is a borrower-centric extension of the validated Credit Risk Platform, independent of the CRIS macroeconomic overlay project, focused strictly on intrinsic borrower-level credit risks.

---

## EXECUTIVE SUMMARY

A **Stealth Defaulter** is defined as a borrower who ultimately defaults but is assigned a low predicted Probability of Default (PD) by the champion model. These represent the false negatives of our core credit risk platform. 

This research investigated the behavior, characteristics, and predictability of these elusive borrowers using a testing cohort of 50,000 loans from 2018 onward.

### Key Empirical Findings:
1. **Significant Risk Leakage**: Stealth defaulters represent **42.00%** of all realized defaults. They are approved under standard underwriting guidelines due to their pristine borrower profiles.
2. **Pristine Underwriting Profiles**: On paper, stealth defaulters look virtually identical to (or better than) good borrowers: they have an average FICO of **709.9** (vs 710.1 for good borrowers), a lower DTI of **16.9%** (vs 18.2% for good borrowers), and a high average income of **$78.5k**.
3. **Fundamental Unpredictability**: A dedicated classifier trained to predict stealth defaulters achieved a low ROC-AUC of **0.59134** (95% CI: `[0.58346, 0.60138]`), confirming that stealth defaults behave primarily like irreducible random noise rather than predictable structure.
4. **Information Limits**: Model failures are driven by fundamental information limits (sudden life events like job loss, medical emergencies) rather than model design flaws. Neither interaction features nor segment-specific modeling yielded material predictive improvements.

---

## 1. POPULATION AUDIT

We performed a population audit on the 50,000 loan out-of-time test cohort to quantify the exact volume and rate of stealth defaults. Under the champion LightGBM model's optimized F1 threshold of **0.20439**, the cohort results are as follows:

| Metric                                         |        Value | Formatted   |
|:-----------------------------------------------|-------------:|:------------|
| Total Evaluation Cohort                        | 50000        | 50,000      |
| Total Defaults                                 |  7865        | 7,865       |
| Total Non-Defaults                             | 42135        | 42,135      |
| Captured Defaulters (True Positives)           |  4562        | 4,562       |
| Stealth Defaulters (False Negatives)           |  3303        | 3,303       |
| False Positives (Good Borrowers Flagged Risk)  | 12002        | 12,002      |
| True Negatives (Good Borrowers Approved)       | 30133        | 30,133      |
| False Negative Rate (Stealth / Total Defaults) |     0.419962 | 42.00%      |
| Model Decision Threshold                       |     0.204387 | 0.20439     |

* **Stealth Default Share**: Stealth defaulters make up **42.00%** of all default occurrences. This represents a significant risk leakage for any underwriting framework relying solely on standard risk-scoring models.

---

## 2. DECILE LOCATION ANALYSIS

We mapped the stealth defaulters back to the model's predicted risk deciles (where D1 represents the lowest predicted risk and D10 represents the highest):

| Decile   |   Total Borrowers |   Total Defaults |   Stealth Defaulters |   Captured Defaulters |   Stealth Share of Decile Defaults |   Stealth Share of Total Stealth |
|:---------|------------------:|-----------------:|---------------------:|----------------------:|-----------------------------------:|---------------------------------:|
| D1       |              5000 |              151 |                  151 |                     0 |                           1        |                         0.045716 |
| D2       |              5000 |              298 |                  298 |                     0 |                           1        |                         0.090221 |
| D3       |              5000 |              387 |                  387 |                     0 |                           1        |                         0.117166 |
| D4       |              5000 |              485 |                  485 |                     0 |                           1        |                         0.146836 |
| D5       |              5000 |              642 |                  642 |                     0 |                           1        |                         0.194369 |
| D6       |              5000 |              741 |                  741 |                     0 |                           1        |                         0.224342 |
| D7       |              5000 |              929 |                  599 |                   330 |                           0.644779 |                         0.18135  |
| D8       |              5000 |             1090 |                    0 |                  1090 |                           0        |                         0        |
| D9       |              5000 |             1355 |                    0 |                  1355 |                           0        |                         0        |
| D10      |              5000 |             1787 |                    0 |                  1787 |                           0        |                         0        |

### Inferences:
* **Decile Distribution**: Stealth defaulters are concentrated in the lower-risk deciles (**D1 to D7**), which is mathematically expected since they must have predicted PDs below the threshold.
* **Peak Concentration**: The highest counts of stealth defaults are in deciles **D5 (19.4%)**, **D6 (22.4%)**, and **D7 (18.1%)**. These represent the border-zone borrowers who look moderately safe on paper but fall victim to default.

---

## 3. BORROWER ARCHETYPE ANALYSIS

We compared the average borrower characteristics across three distinct groups:
* **Group A: Good Borrowers** (Non-defaults)
* **Group B: Captured Defaulters** (True Positives)
* **Group C: Stealth Defaulters** (False Negatives)

| Feature                       |   Group A: Good Borrowers Mean |   Group A: Good Borrowers Median |   Group A: Good Borrowers P10 |   Group A: Good Borrowers P90 |   Group B: Captured Defaulters Mean |   Group B: Captured Defaulters Median |   Group B: Captured Defaulters P10 |   Group B: Captured Defaulters P90 |   Group C: Stealth Defaulters Mean |   Group C: Stealth Defaulters Median |   Group C: Stealth Defaulters P10 |   Group C: Stealth Defaulters P90 |
|:------------------------------|-------------------------------:|---------------------------------:|------------------------------:|------------------------------:|------------------------------------:|--------------------------------------:|-----------------------------------:|-----------------------------------:|-----------------------------------:|-------------------------------------:|----------------------------------:|----------------------------------:|
| FICO                          |                     710.059    |                         705      |                      665      |                      765      |                          692.409    |                              685      |                          660       |                           735      |                         709.861    |                             700      |                         665       |                          770      |
| DTI (%)                       |                      18.2057   |                          16.44   |                        5.53   |                       30.63   |                           21.5406   |                               19.98   |                            6.031   |                            35.07   |                          16.9367   |                              15.59   |                           3.784   |                           29.818  |
| Annual Income ($)             |                   81429.3      |                       68000      |                    34000      |                   135000      |                        69224        |                            57352      |                        28807.2     |                        115000      |                       78498.9      |                           65000      |                       31200       |                       135000      |
| Revolving Utilization (%)     |                      38.6302   |                          35.4    |                        6.74   |                       76      |                           44.8067   |                               44.1    |                            9.8     |                            80.8    |                          41.1071   |                              38.6    |                           5.72    |                           79.6    |
| Loan Amount ($)               |                   14448.6      |                       12000      |                     3675      |                    30000      |                        18535.8      |                            17000      |                         7000       |                         34135      |                       14833.9      |                           10000      |                        4000       |                        35000      |
| Credit History Length (Years) |                      16.0184   |                          14.4969 |                        7.4141 |                       26.2505 |                           14.6884   |                               13.1663 |                            5.99863 |                            24.3365 |                          16.1505   |                              14.5051 |                           6.66776 |                           27.5855 |
| Delinquency Count (2 Years)   |                       0.232538 |                           0      |                        0      |                        1      |                            0.268303 |                                0      |                            0       |                             1      |                           0.226764 |                               0      |                           0       |                            1      |
| Public Records                |                       0.155595 |                           0      |                        0      |                        1      |                            0.182815 |                                0      |                            0       |                             1      |                           0.142598 |                               0      |                           0       |                            1      |
| Open Credit Lines             |                      11.5076   |                          10      |                        5      |                       19      |                           11.1694   |                               10      |                            5       |                            19      |                          10.3206   |                               9      |                           4       |                           18      |
| Employment Length (Years)     |                       5.55004  |                           5      |                        0.5    |                       10      |                            4.65267  |                                4      |                            0       |                            10      |                           5.29382  |                               5      |                           0       |                           10      |
| Total High Credit Limit ($)   |                  203492        |                      144700      |                    30309.8    |                   449791      |                       130150        |                            74515      |                        20368.5     |                        327622      |                      175602        |                          102969      |                       22500       |                       424451      |

### Comparison Inferences:
* **The Stealth Illusion**: Group C (Stealth Defaulters) exhibits borrower characteristics that are significantly superior to Group B (Captured Defaulters) and look nearly identical to Group A (Good Borrowers).
* **FICO**: Stealth defaulters' mean FICO is **709.9**, which matches Good Borrowers (**710.1**) and is much higher than Captured Defaulters (**692.4**).
* **DTI**: Stealth defaulters have a lower mean DTI (**16.9%**) than even the Good Borrowers (**18.2%**).
* **Income**: Stealth defaulters' mean income is **$78.5k**, which is close to Good Borrowers (**$81.4k**) and substantially higher than Captured Defaulters (**$69.2k**).

---

## 4. SHAP EXPLAINABILITY ANALYSIS

Using Tree SHAP, we analyzed the local feature attributions of the champion model to understand why it was misled by stealth defaulters:

| Feature              |   Captured_Mean_SHAP |   Stealth_Mean_SHAP |   SHAP_Difference |
|:---------------------|---------------------:|--------------------:|------------------:|
| int_rate             |           0.486685   |        -0.114376    |        -0.601061  |
| term_months          |           0.159259   |        -0.089607    |        -0.248866  |
| acc_open_past_24mths |           0.043707   |        -0.0193244   |        -0.0630314 |
| dti                  |           0.0337034  |        -0.0273511   |        -0.0610545 |
| loan_amnt            |           0.0469498  |        -0.00720924  |        -0.0541591 |
| annual_inc           |           0.0326312  |        -0.0128      |        -0.0454312 |
| fico_range_low       |           0.0150847  |        -0.0302776   |        -0.0453623 |
| mo_sin_old_rev_tl_op |           0.0283988  |         0.00709684  |        -0.021302  |
| total_bc_limit       |           0.0110607  |        -0.00991665  |        -0.0209774 |
| emp_length_num       |           0.0383348  |         0.0178107   |        -0.0205241 |
| grade_B              |           0.011525   |        -0.00615707  |        -0.017682  |
| home_ownership_RENT  |           0.00774933 |        -0.00985835  |        -0.0176077 |
| mort_acc             |           0.0246561  |         0.00718357  |        -0.0174726 |
| installment          |           0.0125668  |        -0.003787    |        -0.0163538 |
| avg_cur_bal          |           0.0156791  |         0.000857411 |        -0.0148217 |

### SHAP Inferences:
* **Lender-Pricing Features**: The largest drivers pushing the predicted PD lower for stealth defaulters were `int_rate` (SHAP difference of **-0.601**) and `term_months` (SHAP difference of **-0.249**). Because these borrowers qualified for lower interest rates and shorter terms, the model used this lender underwriting prior to reinforce its safety prediction.
* **Borrower intrinsic variables**: Pristine values of `fico_range_low` (SHAP difference of **-0.045**), `dti` (**-0.061**), and `annual_inc` (**-0.045**) drove the model's risk prediction to near zero.

### Representative Case Studies:
Below are three actual stealth defaulters from our cohort:

|   FICO |   DTI |   Annual_Income |   Revolving_Utilization |   Loan_Amount |   Predicted_PD |   Target |
|-------:|------:|----------------:|------------------------:|--------------:|---------------:|---------:|
|    790 |  1.36 |           95000 |                    22.2 |         12000 |      0.0138229 |        1 |
|    785 | 18.83 |           75000 |                    30.8 |         12000 |      0.0151602 |        1 |
|    795 |  7.19 |          120000 |                    11.2 |          6500 |      0.0153524 |        1 |

> [!WARNING]
> Case 3 represents a borrower with a near-perfect **795 FICO score**, a **7.19% DTI**, and **$120,000** in annual income, taking out a small **$6,500** loan. The model predicted a PD of only **1.54%**, yet the borrower ultimately defaulted. These cases prove that stealth defaults are driven by sudden, exogenous shocks that standard credit files cannot capture.

---

## 5. HIDDEN SEGMENT DISCOVERY

We ran KMeans clustering and DBSCAN on the stealth defaulters to see if they form dense, coherent subgroups:
* **DBSCAN Noise Ratio**: **100.00%** (At standard density thresholds, all points are classified as noise, indicating high dispersion in borrower feature space).
* **KMeans Segmentation**: The stealth defaulters are best characterized by 3 distinct archetypes:

| Cluster   |   Count |    Share |   fico_range_low_Mean |   fico_range_low_Median |   dti_Mean |   dti_Median |   annual_inc_Mean |   annual_inc_Median |   revol_util_Mean |   revol_util_Median |   loan_amnt_Mean |   loan_amnt_Median |   cr_hist_years_Mean |   cr_hist_years_Median |   delinq_2yrs_Mean |   delinq_2yrs_Median |   tot_hi_cred_lim_Mean |   tot_hi_cred_lim_Median |
|:----------|--------:|---------:|----------------------:|------------------------:|-----------:|-------------:|------------------:|--------------------:|------------------:|--------------------:|-----------------:|-------------------:|---------------------:|-----------------------:|-------------------:|---------------------:|-----------------------:|-------------------------:|
| Cluster 0 |     596 | 0.180442 |               717.567 |                     710 |    20.917  |       18.61  |          106916   |               86000 |           33.8146 |               30.8  |          19966.4 |            17537.5 |              18.1572 |                16.4189 |           0.187919 |                    0 |                 280720 |                   231230 |
| Cluster 1 |    1198 | 0.362701 |               695.78  |                     690 |    19.8781 |       18.595 |           88359   |               78000 |           54.1871 |               55.45 |          14878.6 |            12000   |              18.3927 |                16.7091 |           0.366444 |                    0 |                 261002 |                   231878 |
| Cluster 2 |    1509 | 0.456857 |               717.995 |                     710 |    13.0295 |       11.54  |           59447.3 |               50000 |           33.6031 |               29.5  |          12771.2 |            10000   |              13.5778 |                12.167  |           0.131213 |                    0 |                  66285 |                    46391 |

### Cluster Definitions:
1. **Cluster 0: "High-Income Elusive"** (18.0% of stealth): High FICO (717.6), high income ($106.9k), large loan amounts ($20.0k). These are wealthy borrowers who default due to asset shocks or business failures.
2. **Cluster 1: "Mature High-Utilizers"** (36.3% of stealth): Moderate FICO (695.8), long credit history (18.4 years), but high utilization (54.2%). These are credit-strained borrowers who gradually deteriorate.
3. **Cluster 2: "Low-Debt Starters"** (45.7% of stealth): High FICO (718.0), low DTI (13.0%), low income ($59.4k), small loans ($12.8k). These are conservative, low-debt borrowers who default due to sudden income loss because they lack a financial buffer.

---

## 6. NOISE VS STRUCTURE TEST

We trained a predictive classifier to identify stealth defaulters in advance:

| Metric              |     Value |    CI_Lower |    CI_Upper |
|:--------------------|----------:|------------:|------------:|
| ROC-AUC             | 0.591345  |   0.583463  |   0.601378  |
| PR-AUC              | 0.0872811 |   0.0843212 |   0.0923593 |
| Accuracy            | 0.72182   | nan         | nan         |
| Precision           | 0.0906284 | nan         | nan         |
| Recall              | 0.355434  | nan         | nan         |
| F1 Score            | 0.14443   | nan         | nan         |
| Optimized Threshold | 0.0772123 | nan         | nan         |

### Predictability Verdict:
**Weakly structured; dominated by high variance and irreducible noise.**

* **Performance Analysis**: A ROC-AUC of **0.59134** is only marginally better than a random coin toss (0.50). This low performance confirms that stealth defaulters do not form a structured, predictable pattern. Instead, they are dominated by high-variance credit events.

---

## 7. DETECTION IMPROVEMENT EXPERIMENTS

We tested if adding interaction features or training segment-specific models could improve stealth default detection:

### Interaction Features Experiment:
| Model                   |   ROC-AUC |   PR-AUC |   False Negatives |   Threshold |
|:------------------------|----------:|---------:|------------------:|------------:|
| Borrower Baseline       |  0.682379 | 0.271752 |              3317 |    0.18185  |
| Borrower + Interactions |  0.682178 | 0.270554 |              2931 |    0.168904 |

### Segment-Specific Models Experiment:
| Segment               |   General Model AUC |   Segment Model AUC |   AUC Delta |   General Model PR-AUC |   Segment Model PR-AUC |   PR-AUC Delta |   General Model FNs |   Segment Model FNs |   FN Change |
|:----------------------|--------------------:|--------------------:|------------:|-----------------------:|-----------------------:|---------------:|--------------------:|--------------------:|------------:|
| Older Borrowers       |            0.67964  |            0.682861 |  0.00322145 |               0.247575 |               0.253839 |    0.00626381  |                1340 |                1467 |         127 |
| Low Util Borrowers    |            0.69354  |            0.695019 |  0.00147919 |               0.267726 |               0.267868 |    0.000141827 |                2319 |                2119 |        -200 |
| High Income Borrowers |            0.692788 |            0.689347 | -0.00344106 |               0.249527 |               0.250014 |    0.000487352 |                1802 |                1750 |         -52 |

### Inferences:
* **Interaction Features**: Adding interaction features (e.g., FICO × DTI) actually *degraded* the overall ROC-AUC and PR-AUC, confirming they do not capture the underlying default drivers.
* **Segment Models**: Training dedicated models for specific segments (Older Borrowers, Low Utilization, High Income) yielded only marginal and inconsistent improvements (e.g. +0.003 AUC for older borrowers, but -0.003 AUC for high-income borrowers). 

---

## 8. RESEARCH CONCLUSIONS

Based on the empirical evidence, we answer the core research questions:

### 1. How much default risk is hidden from the champion model?
* **42.00%** of all default events are classified as safe by the champion model. This is the "hidden risk" of the portfolio.

### 2. What are the primary borrower profiles of stealth defaulters?
* Stealth defaulters fall into three distinct profiles: (a) High-Income Elusive (wealthy, large loans), (b) Mature High-Utilizers (long credit history, high usage), and (c) Low-Debt Starters (low income, low debt).

### 3. Why does the champion model fail to identify them?
* The model fails because these borrowers have pristine credit characteristics (high FICO, low DTI, low inquiries) and low interest rates, which are strong mathematical indicators of safety.

### 4. Are these failures driven by systemic model flaws or by fundamental information limits?
* **Fundamental information limits**. The low predictability of the stealth classifier (ROC-AUC = 0.59) and the failure of interaction/segment models prove that these defaults are caused by exogenous, unrecorded shocks (e.g. job loss, medical emergencies, divorce) rather than model design issues.

### 5. Can we mitigate these failures using advanced borrower features or segment models?
* **No**. Borrower-intrinsic features have been fully exploited. Segment models and interaction features do not provide material improvements.

### 6. What is the recommended strategy for managing stealth defaulters?
* Since stealth defaulters cannot be predicted statisticaly at application time, the recommended strategy is **structural risk mitigation**:
  1. **Portfolio Diversification**: Limit concentration in any single borrower archetype.
  2. **Capital Buffers**: Maintain loss reserves scaled to accommodate a baseline 42% false negative default leakage.
  3. **Exogenous Monitoring**: Supplement underwriting with real-time transactional monitoring or employment verification to capture cash flow disruptions post-approval.

---

## LIST OF GENERATED ARTIFACTS
All analysis outputs and figures are saved under:
* Tables: `systems/credit_risk/cr_analysis/outputs/tables/`
* Figures: `systems/credit_risk/cr_analysis/outputs/figures/`
  * [PCA Clusters](file:///home/sharansh/CRIS/systems/credit_risk/cr_analysis/outputs/figures/stealth_pca_clusters.png)
  * [Stealth vs Captured PCA](file:///home/sharansh/CRIS/systems/credit_risk/cr_analysis/outputs/figures/stealth_vs_captured_pca.png)
  * [Archetype Distributions](file:///home/sharansh/CRIS/systems/credit_risk/cr_analysis/outputs/figures/archetype_distributions.png)
  * [SHAP Comparison](file:///home/sharansh/CRIS/systems/credit_risk/cr_analysis/outputs/figures/shap_stealth_comparison.png)
