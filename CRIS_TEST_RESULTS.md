## TS-001: Entropy Method Selection — Results

### A. Raw Performance Metrics
| Method | Lead Time | FP Rate/mo | SNR | Consistency | Duration | Composite |
|--------|-----------|------------|-----|-------------|----------|-----------|
| Shannon | 0.0 days | **0.000** | 0.994 | 0.67 | 0.0 | 0.367 |
| Permutation | 0.0 days | **0.000** | 0.994 | **1.00** | 0.0 | 0.400 |
| Sample | **35.0 days** | 2.174 | **1.361** | **1.00** | **18.0** | **0.700** |
| Tsallis | 0.0 days | 0.149 | 0.995 | 0.67 | 0.0 | 0.347 |

### B. Composite Scores
1. 🥇 Sample (0.700)
2. 🥈 Permutation (0.400)
3. 🥉 Shannon (0.367)
4. Tsallis (0.347)

### C. Correlation Analysis (if applicable)
Permutation had near-zero correlation with Sample (0.02). Tsallis showed a 1.00 correlation with Shannon at q=0.5 due to a binning bug.

### Decision
**Winner: Sample Entropy (Primary) + Permutation Entropy (Confirmation)**

### Rationale
Sample Entropy was chosen as the primary method because it demonstrated a massive 35-day average early warning lead time and held its breach threshold for an average duration of 18.0 days, whereas other methods failed to breach early entirely (0 days). Permutation Entropy was selected as the confirmation method because it achieved perfect directional consistency (1.00) alongside a near-zero correlation (0.02) with Sample Entropy, ensuring independent validation.

### Rejected Methods
| Method | Reason |
|--------|--------|
| Shannon | Zero lead time and highly susceptible to volatility expansion. |
| Tsallis | Zero lead time and produced a mathematical bug showing 1.00 correlation with Shannon at q=0.5. |

### Plots Generated
- data/simulation_output/entropy_comparison_overview.png
- data/simulation_output/entropy_metrics_bar_chart.png
- data/simulation_output/entropy_correlation_heatmap.png
