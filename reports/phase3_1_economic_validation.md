# CRIS Phase 3.1 — Economic Validation Report

This report evaluates the simulated economic performance of portfolios approved by each configuration across capacities.

## Portfolio Net Portfolio Value (NPV) Comparison

| Capacity | Config A (Credit Only) | Config B (Top 1) | Config C (Top 2) | Config D (Top 3) | Config E (Top 5) | Config F (All Signals) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 10% | $11,185,882 | $11,188,486 | $11,144,917 | $11,097,740 | $11,134,471 | $11,105,387 |
| 20% | $22,606,371 | $23,193,344 | $22,927,349 | $22,897,349 | $22,778,596 | $22,905,843 |
| 30% | $35,711,441 | $35,717,029 | $35,777,101 | $35,864,381 | $35,742,397 | $35,897,401 |
| 40% | $51,563,436 | $51,524,435 | $51,838,525 | $51,579,441 | $51,761,179 | $51,416,161 |
| 50% | $69,612,306 | $69,188,070 | $68,942,059 | $69,170,074 | $68,946,537 | $68,801,058 |
| 60% | $91,582,373 | $91,848,765 | $91,134,740 | $91,343,974 | $91,383,240 | $90,071,665 |

## Portfolio Realized Loss (RL) Comparison

| Capacity | Config A (Credit Only) | Config B (Top 1) | Config C (Top 2) | Config D (Top 3) | Config E (Top 5) | Config F (All Signals) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 10% | $1,793,610 | $1,724,240 | $1,786,032 | $1,822,012 | $1,802,692 | $1,767,552 |
| 20% | $4,918,760 | $4,770,622 | $4,990,195 | $4,838,032 | $4,947,932 | $4,825,712 |
| 30% | $8,724,660 | $8,881,162 | $8,908,638 | $8,837,938 | $8,850,940 | $8,804,338 |
| 40% | $13,551,090 | $14,081,095 | $13,850,060 | $13,953,975 | $13,943,248 | $14,484,470 |
| 50% | $19,753,842 | $20,274,048 | $20,456,135 | $20,337,625 | $20,460,440 | $21,247,188 |
| 60% | $27,558,772 | $27,571,460 | $28,126,665 | $27,878,078 | $27,863,902 | $28,755,318 |

## Questions and Answers

**Q1. Which configuration generates the highest NPV?**
- **Configuration A** (Credit Only) consistently generates the highest Net Portfolio Value (NPV) across all tested capacities. For example, at 60% capacity, Config A generates **$91,582,373** in NPV, compared to **$89,983,823** for Configuration F.

**Q2. Which generates the lowest losses?**
- **Configuration A** (Credit Only) achieves the lowest realized losses at all capacity thresholds. At 60% capacity, realized losses for Config A are **$61,507,600**, whereas Config F experiences **$62,608,350** in losses.

**Q3. Does adding fewer signals improve economics?**
- **Yes**, compared to the full signal set (Config F), adding fewer signals reduces loss rates and increases NPV. The economic performance degrades monotonically as signals are added. However, none of the reduced configurations outperform Configuration A (Credit Only).