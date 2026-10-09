# Evaluation plan

Written in step 3.2, **before any model was trained**. Every model in Phase 3 is scored by `evaluate()` in `src/mkopoguard/models/metrics.py`, and these rules decide which one wins.

## Metrics

| Metric | Measures | No skill | Ceiling* | Role |
| --- | --- | --- | --- | --- |
| **PR-AUC** | Finding defaulters without false alarms | 0.199 (= default rate) | 0.614 | **Primary: picks the model** |
| ROC-AUC | Ranking risky above safe | 0.500 | 0.846 | Reported; leakage alarm |
| KS | Largest gap between defaulter and non-defaulter scores | 0.000 | 0.543 | Reported (credit-industry standard) |
| Brier / Brier skill | Accuracy of the probabilities | 0.160 / 0.000 | 0.113 / 0.294 | Checks calibration (step 3.10) |
| ECE | Predicted vs actual default rate across 10 bins | 0.000† | 0.013 | Checks calibration |
| Recall, precision, false-positive rate, approval rate | What happens to applicants at the threshold | — | — | Reported at the threshold |
| Accuracy | Share of correct decisions | **0.801** | 0.772 | Reported, never used to choose |

\* *Ceiling* = scoring with each applicant's true hidden default probability, on the validation set. No real model can beat it.
† Guessing the average is perfectly calibrated but useless, which is why calibration is never judged alone.

## Rules

1. **Model choice:** highest PR-AUC in 5-fold cross-validation on the training set, confirmed on the validation set. A model must beat logistic regression to justify extra complexity.
2. **Threshold:** decision metrics use a provisional threshold of 0.20 (about the default rate) until step 3.11 picks one from the cost of each kind of mistake.
3. **Leakage alarm:** any ROC-AUC above 0.86 or PR-AUC above 0.65 is treated as a leak, not a success, and is investigated before anything else.
4. **Groups:** every result is also reported for Kenya, statement holders and thin-file applicants. Groups under 500 applicants (e.g. ~141 Kenyans in validation, ~144 in test) are reported with a 95% bootstrap confidence interval (1,000 resamples).
5. **Test set:** opened once, in step 3.12, after every choice is final. Its scores are logged with `evaluated_on=test` and never used to change the model.

## Why not accuracy?

80% of applicants repay, so approving everyone is 80% accurate while catching zero defaulters. On the validation set, the do-nothing model's accuracy (0.801) beats even perfect knowledge (0.772). Accuracy rewards ignoring the minority class, which is exactly the class a lender cares about.
